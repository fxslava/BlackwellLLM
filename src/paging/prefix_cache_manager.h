#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "paged_kv_cache.h"
#include "radix_tree.h"
#include "tiered_memory_pager.h"
#include "kv_branch_serializer.h"

// ============================================================================
// PrefixCacheManager — SequenceManager + RadixTreeIndex + TieredMemoryPager
//                      + KVBranchSerializer, behind one facade
// ============================================================================
// Tiered evolution of the Phase-1 facade. The radix tree now stores VIRTUAL
// page ids (VPids) owned by the TieredMemoryPager; block tables keep physical
// pages. Request lifecycle:
//
//   auto a = pc.acquire(tokens, n, compute_stream);
//         // match -> PAGE FAULT the matched chain back into VRAM (RAM/NVMe
//         // promotions ride the transfer stream; compute_stream yields on an
//         // event, or pass nullptr to block) -> pin -> seed the sequence.
//         // If VRAM can't be freed the cached prefix is TRUNCATED, not
//         // failed: a.cached_tokens just comes back smaller.
//   prefill(a.seq, tokens + a.cached_tokens, ...); decode; ...
//   pc.commit(a.seq, all_tokens, total);   // wrap new pages in VPids + index
//   pc.release(a.seq);                     // unpin + unlock + destroy
//
// Reference/pin model (see tiered_memory_pager.h for the full contract):
//   block table  -> physical page refs (SequenceManager, unchanged)
//   radix tree   -> VPid refs (retain/release; backing freed at zero)
//   live acquire -> VPid pins (VRAM-locked; excluded from demotion)
//
// Memory pressure: ensure_free_pages() first DEMOTES cold tree pages down the
// waterfall (contents survive, index intact); only if the lower tiers are
// full does it fall back to true LRU tree eviction (index + contents die).
// Tree eviction remains the ONLY destruction path — the pager never drops a
// page behind the index's back.
//
// Still single-threaded host control plane, like the whole substrate.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

class PrefixCacheManager {
public:
    PrefixCacheManager(SequenceManager& sm, TieredMemoryPager& pager,
                       uint64_t model_hash)
        : m_sm(sm), m_pager(pager),
          m_tree(PAGE_SIZE,
                 [&pager](PageId v) { pager.retain(v); },
                 [&pager](PageId v) { pager.release(v); }),
          m_io(sm, model_hash), m_model_hash(model_hash) {}

    struct Acquired {
        SeqId seq = -1;
        int   cached_tokens = 0;   // KV already in VRAM; prefill starts here
    };

    // -----------------------------------------------------------------------
    // Create a sequence for tokens[0..n) reusing the longest cached prefix
    // that could be made VRAM-resident. Faulted pages are pinned until
    // release(). cached_tokens is page-aligned and STRICTLY < n (the last
    // token always re-runs so attention produces sampling logits).
    // compute_stream: non-null => promotions gate that stream GPU-side
    // (decode overlaps the DMA); null => fault blocks the host.
    // -----------------------------------------------------------------------
    Acquired acquire(const TokenId* tokens, int n,
                     cudaStream_t compute_stream = nullptr) {
        auto m = m_tree.match_prefix(tokens, n);
        int want = std::min(m.matched_tokens, ((n - 1) / PAGE_SIZE) * PAGE_SIZE);
        if (want < 0) want = 0;
        const int resident = m_pager.fault_in(m.pages.data(), want / PAGE_SIZE,
                                              compute_stream, /*pin=*/true);
        std::vector<PageId> phys(resident);
        for (int i = 0; i < resident; ++i) phys[i] = m_pager.phys_of(m.pages[i]);

        Acquired a;
        a.cached_tokens = resident * PAGE_SIZE;
        a.seq = m_sm.create_sequence_with_pages(phys, a.cached_tokens);
        m_tree.lock(m.node);
        m_locks[a.seq] = m.node;
        m_pins[a.seq].assign(m.pages.begin(), m.pages.begin() + resident);
        return a;
    }

    // -----------------------------------------------------------------------
    // Publish the sequence's completed pages for tokens[0..n) into the tree
    // (rounded down to full pages; the partial tail stays private). Each page
    // is wrapped in a fresh VPid; spans already indexed keep the existing
    // VPids and the duplicate registrations are gc'd on the spot. Re-pins the
    // deeper node.
    // -----------------------------------------------------------------------
    void commit(SeqId seq, const TokenId* tokens, int n) {
        int full = std::min(n, m_sm.length(seq));
        full -= full % PAGE_SIZE;
        if (full <= 0) return;
        std::vector<VPid> vp(full / PAGE_SIZE);
        for (int b = 0; b < (int)vp.size(); ++b)
            vp[b] = m_pager.adopt_vram_page(m_sm.block_page(seq, b));
        auto r = m_tree.insert(tokens, full, vp.data());
        for (VPid v : vp) m_pager.gc(v);   // reap the never-adopted duplicates
        auto it = m_locks.find(seq);
        if (it != m_locks.end()) m_tree.unlock(it->second);
        m_tree.lock(r.node);
        m_locks[seq] = r.node;
    }

    // Unpin + unlock the sequence's branch and destroy the sequence. Pages
    // survive at whatever tier iff the tree (or another holder) references
    // them; freshly unpinned pages re-enter the VRAM LRU as MRU.
    void release(SeqId seq) {
        auto pit = m_pins.find(seq);
        if (pit != m_pins.end()) {
            m_pager.unpin(pit->second.data(), (int)pit->second.size());
            m_pins.erase(pit);
        }
        auto it = m_locks.find(seq);
        if (it != m_locks.end()) { m_tree.unlock(it->second); m_locks.erase(it); }
        m_sm.destroy_sequence(seq);
    }

    // -----------------------------------------------------------------------
    // Make room BEFORE a prefill/load needing `needed` physical pages.
    // Ladder: (1) demote cold tree pages down the waterfall — cache contents
    // survive; (2) if the lower tiers are full, LRU-evict tree branches
    // (releases VPids, freeing backing at every tier) and demote again.
    // False = true OOM even with a drained tree (caller degrades).
    // -----------------------------------------------------------------------
    bool ensure_free_pages(int needed) {
        if (m_pager.ensure_vram(needed)) return true;
        m_tree.evict(needed - m_sm.free_pages());
        return m_pager.ensure_vram(needed);
    }

    // -----------------------------------------------------------------------
    // Disk persistence (.bkv archives — distinct from the pager's spill file,
    // though page records share the same byte layout).
    // dump(): the branch must be fully indexed (commit() first). Cold pages
    // are faulted back for the read in v1 (an any-tier read path is the
    // documented upgrade); the branch is pinned for the I/O duration.
    // -----------------------------------------------------------------------
    void dump(const TokenId* tokens, int n, const std::string& path,
              cudaStream_t stream = nullptr) {
        const int full = n - n % PAGE_SIZE;
        auto m = m_tree.match_prefix(tokens, full);
        if (m.matched_tokens < full)
            throw std::runtime_error("PrefixCacheManager::dump: prefix not fully "
                                     "cached (commit it first)");
        const int np = full / PAGE_SIZE;
        const int resident = m_pager.fault_in(m.pages.data(), np,
                                              /*compute_stream=*/nullptr,
                                              /*pin=*/true);
        if (resident < np) {
            m_pager.unpin(m.pages.data(), resident);
            throw std::runtime_error("PrefixCacheManager::dump: cannot restore "
                                     "branch to VRAM (pin pressure)");
        }
        std::vector<PageId> phys(np);
        for (int i = 0; i < np; ++i) phys[i] = m_pager.phys_of(m.pages[i]);
        m_tree.lock(m.node);
        try {
            m_io.dump(path, tokens, full, phys, stream);
        } catch (...) {
            m_tree.unlock(m.node);
            m_pager.unpin(m.pages.data(), np);
            throw;
        }
        m_tree.unlock(m.node);
        m_pager.unpin(m.pages.data(), np);
    }

    // Where a restored branch should live until first use.
    enum class LoadPolicy {
        Hot,       // stay VRAM-resident (about to be used)
        ColdRam,   // demote to pinned RAM immediately — warm-start many
                   // branches without hogging VRAM; first acquire faults them
                   // up at DMA speed
    };

    // Restore a branch and graft it into the tree; returns the tokens now
    // servable with zero prefill. Duplicate spans dedup exactly as commit().
    int load(const std::string& path, LoadPolicy policy = LoadPolicy::Hot,
             cudaStream_t stream = nullptr) {
        auto r = m_io.load(path, stream);     // we own one phys ref per page
        const int n = (int)r.tokens.size();
        std::vector<VPid> vp(r.pages.size());
        for (size_t i = 0; i < r.pages.size(); ++i)
            vp[i] = m_pager.adopt_vram_page(r.pages[i]);
        m_tree.insert(r.tokens.data(), n, vp.data());
        for (PageId p : r.pages) m_sm.release_page(p);   // drop the loader ref
        for (VPid v : vp)
            if (!m_pager.gc(v) && policy == LoadPolicy::ColdRam)
                m_pager.demote(v, Tier::RAM);
        return n;
    }

    // -----------------------------------------------------------------------
    // Inactivity lifecycle, stage 1: cold-spill the whole cache down the tier
    // waterfall (unpinned pages only — release live sequences first for a full
    // spill). The index survives; pages fault back in on the next acquire().
    // Returns the number of page demotions performed.
    // -----------------------------------------------------------------------
    int spill_all() { return m_pager.spill_all(); }

    // -- introspection --------------------------------------------------------
    size_t   indexed_pages() const { return m_tree.total_pages(); }
    uint64_t model_hash()    const { return m_model_hash; }
    RadixTreeIndex&          tree()        { return m_tree; }
    const RadixTreeIndex&    tree()  const { return m_tree; }
    TieredMemoryPager&       pager()       { return m_pager; }
    const TieredMemoryPager& pager() const { return m_pager; }

private:
    SequenceManager&   m_sm;
    TieredMemoryPager& m_pager;
    RadixTreeIndex     m_tree;    // stores VPids
    KVBranchSerializer m_io;
    uint64_t           m_model_hash;
    std::unordered_map<SeqId, RadixTreeIndex::Node*> m_locks;
    std::unordered_map<SeqId, std::vector<VPid>>     m_pins;
};

}} // namespace blackwell::paging
