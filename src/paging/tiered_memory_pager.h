#pragma once
#include <cassert>
#include <cstdint>
#include <vector>

// ============================================================================
// TieredMemoryPager — 3-tier residency (VRAM ⇄ pinned RAM ⇄ NVMe) for KV pages
// ============================================================================
// The pager introduces one level of indirection over the physical VRAM page
// space: a VIRTUAL PAGE ID (VPid). The radix tree stores VPids (its PageId
// slots are opaque int32s bound through retain/release callbacks, so the tree
// code is unchanged); block tables and CUDA kernels keep using physical VRAM
// PageIds. The pager owns the mapping
//
//     VPid  ->  { VRAM phys page | pinned-RAM slot | disk (spill) slot }
//
// and migrates cold pages DOWN (demotion waterfall: VRAM→RAM→DISK, LRU at
// each tier) and requested pages UP (page fault: RAM/DISK→VRAM) on demand.
//
// Fault semantics (graceful, never-fail):
//   fault_in() restores pages FRONT TO BACK and returns how many are now
//   VRAM-resident. If VRAM cannot be freed (everything pinned) the count is
//   simply smaller — the caller (PrefixCacheManager::acquire) uses the longer
//   prefill instead of failing the request. Block-vs-yield is the fence's
//   business: promotions ride the backend's transfer stream, and fence(cs)
//   either blocks the host (cs == nullptr) or makes the caller's compute
//   stream wait GPU-side (cudaStreamWaitEvent) so decoding overlaps the DMA.
//
// Pin model:
//   pins  — live block tables (and in-flight serializer I/O) referencing the
//           page. Pinned pages are VRAM-locked: excluded from the LRU lists,
//           never demoted. fault_in(pin=true) pins; unpin() releases.
//   refs  — radix-tree adoptions (retain/release). refs==0 && pins==0 frees
//           the backing at whatever tier it sits (release() or gc()).
//   A page referenced ONLY by a live sequence (post-CoW, pre-commit) is not
//   known to the pager at all — the PageAllocator refcount covers it exactly
//   as before.
//
// Layering (mirrors radix_tree.h): this header is PURE HOST CONTROL PLANE —
// no CUDA includes. Byte movement goes through TierBackend and VRAM slot
// allocation through VramPool; the real implementations live in
// cuda_tier_backend.h, and host unit tests inject fakes
// (tests/standalone/test_tiered_pager.cpp). The flat Phase-1 behaviour is the
// degenerate Config{0, 0}: demotion impossible, fault_in a pure pin/translate.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

using PageId = int32_t;   // physical VRAM page (same alias as paged_kv_cache.h)
using VPid   = int32_t;   // pager-owned virtual page handle
constexpr VPid kNoVPid = -1;

enum class Tier : uint8_t { VRAM = 0, RAM = 1, DISK = 2 };

// ---------------------------------------------------------------------------
// VramPool — how the pager allocates/releases physical pages. Bound to
// SequenceManager's PageAllocator in production (see cuda_tier_backend.h).
// ---------------------------------------------------------------------------
class VramPool {
public:
    virtual ~VramPool() = default;
    virtual PageId try_allocate() = 0;         // -1 when exhausted (never throws)
    virtual void   retain(PageId)  = 0;
    virtual void   release(PageId) = 0;
    virtual int    free_pages() const = 0;
};

// ---------------------------------------------------------------------------
// TierBackend — slot-addressed byte movement between tiers. The pager decides
// WHAT moves WHERE; the backend only moves page records. Slot spaces (RAM,
// DISK) are allocated by the pager from [0, Config::*_slots). Promotions
// (…_to_vram) may be asynchronous on an internal transfer stream; the pager
// issues exactly one fence() after each fault_in batch that promoted.
// Demotions must be complete when the call returns (their slot may be reused
// or written to disk immediately after).
// ---------------------------------------------------------------------------
class TierBackend {
public:
    virtual ~TierBackend() = default;
    virtual void vram_to_ram (PageId phys, int ram_slot)  = 0;
    virtual void ram_to_vram (int ram_slot, PageId phys)  = 0;
    virtual void ram_to_disk (int ram_slot, int disk_slot) = 0;
    virtual void disk_to_vram(int disk_slot, PageId phys) = 0;
    // Order all promotions issued since the last fence. compute_stream is an
    // opaque cudaStream_t (this header stays CUDA-free): non-null => GPU-side
    // wait (yield); null => host-side block.
    virtual void fence(void* compute_stream) = 0;
};

// ---------------------------------------------------------------------------
// The pager.
// ---------------------------------------------------------------------------
class TieredMemoryPager {
public:
    struct Config {
        int ram_slots  = 0;   // pinned host pool capacity, in pages
        int disk_slots = 0;   // spill file capacity, in pages
    };
    struct Stats {
        int      vram = 0, ram = 0, disk = 0;          // live pages per tier
        uint64_t faults_ram = 0, faults_disk = 0;      // promotions by source
        uint64_t demotions_ram = 0, demotions_disk = 0;// demotions by target
    };

    TieredMemoryPager(VramPool& pool, TierBackend& io, Config cfg)
        : m_pool(pool), m_io(io), m_cfg(cfg) {
        for (int s = cfg.ram_slots  - 1; s >= 0; --s) m_free_ram.push_back(s);
        for (int s = cfg.disk_slots - 1; s >= 0; --s) m_free_disk.push_back(s);
    }
    TieredMemoryPager(const TieredMemoryPager&) = delete;
    TieredMemoryPager& operator=(const TieredMemoryPager&) = delete;

    // -- registration / reference counting ----------------------------------

    // Wrap a physical VRAM page into a VPid. The pager takes its OWN reference
    // on the page (VramPool::retain) for as long as the VPid is VRAM-resident;
    // demotion trades that reference for a lower-tier slot. Starts with
    // refs == 0: adopt via retain() (radix-tree insert) or reap via gc().
    VPid adopt_vram_page(PageId phys) {
        const VPid v = alloc_meta();
        PageMeta& m = m_meta[v];
        m.tier = Tier::VRAM;
        m.phys = phys;
        m.slot = -1;
        m.refs = 0;
        m.pins = 0;
        m.tick = ++m_tick;
        m_pool.retain(phys);
        lru_push_front(m_vram_lru, v);
        ++m_stats.vram;
        return v;
    }

    void retain(VPid v)  { ++meta(v).refs; }
    void release(VPid v) {
        PageMeta& m = meta(v);
        assert(m.refs > 0);
        if (--m.refs == 0 && m.pins == 0) destroy(v);
        // pins > 0: zombie — reaped by the final unpin().
    }
    // Reap a never-adopted VPid (commit/load registration that turned out to
    // duplicate an already-indexed span). True if freed.
    bool gc(VPid v) {
        if (v < 0 || v >= (int)m_meta.size() || !m_meta[v].live) return false;
        if (m_meta[v].refs != 0 || m_meta[v].pins != 0) return false;
        destroy(v);
        return true;
    }

    // -- the page fault ------------------------------------------------------

    // Make vp[0..n) VRAM-resident, front to back, pinning each restored page
    // when `pin` (the acquire path always pins — the pages are about to enter
    // a block table). Returns the RESIDENT PREFIX COUNT k <= n: pages [0,k)
    // are in VRAM (and pinned if requested); restoration stopped at the first
    // page for which no VRAM could be freed. One backend fence is issued if
    // anything was promoted (see class comment for block-vs-yield).
    int fault_in(const VPid* vp, int n, void* compute_stream = nullptr,
                 bool pin = true) {
        int  resident = 0;
        bool promoted = false;
        for (int i = 0; i < n; ++i) {
            PageMeta& m = meta(vp[i]);
            m.tick = ++m_tick;
            if (m.tier == Tier::VRAM) {
                if (m.pins == 0) lru_move_front(m_vram_lru, vp[i]);
            } else {
                const PageId phys = acquire_phys();
                if (phys < 0) break;               // graceful truncation
                if (m.tier == Tier::RAM) {
                    lru_remove(m_ram_lru, vp[i]);
                    m_io.ram_to_vram(m.slot, phys);
                    m_free_ram.push_back(m.slot);
                    --m_stats.ram; ++m_stats.faults_ram;
                } else {
                    m_io.disk_to_vram(m.slot, phys);
                    m_free_disk.push_back(m.slot);
                    --m_stats.disk; ++m_stats.faults_disk;
                }
                m.tier = Tier::VRAM;
                m.phys = phys;
                m.slot = -1;
                ++m_stats.vram;
                lru_push_front(m_vram_lru, vp[i]);
                promoted = true;
            }
            if (pin) pin_one(vp[i]);
            ++resident;
        }
        if (promoted) m_io.fence(compute_stream);
        return resident;
    }

    void unpin(const VPid* vp, int n) { for (int i = 0; i < n; ++i) unpin_one(vp[i]); }
    void unpin_one(VPid v) {
        PageMeta& m = meta(v);
        assert(m.pins > 0 && m.tier == Tier::VRAM);
        if (--m.pins == 0) {
            if (m.refs == 0) destroy(v);                  // zombie reap
            else lru_push_front(m_vram_lru, v);           // MRU on unpin
        }
    }

    // -- demotion ------------------------------------------------------------

    // Free VRAM until the pool has `pages_needed` free pages, demoting LRU
    // unpinned pages down the waterfall. May return false with partial
    // progress (everything demotable already demoted, or lower tiers full) —
    // the caller escalates to radix-tree eviction, which releases refs and
    // frees lower-tier slots, then retries. NOTE: demoting a page whose
    // physical slot is still shared with a live sequence releases only the
    // pager's reference — VRAM frees when the sequence ends; the loop still
    // terminates because each step shrinks the VRAM LRU list.
    bool ensure_vram(int pages_needed) {
        while (m_pool.free_pages() < pages_needed)
            if (!demote_lru_vram()) return false;
        return true;
    }

    // Explicit tier hint (cold-load a deserialized branch straight out of
    // VRAM, pre-spill an idle branch before a big batch, …). Fails (false) if
    // the page is pinned or the target tier has no room after cascading.
    bool demote(VPid v, Tier target) {
        PageMeta& m = meta(v);
        if (m.pins > 0) return false;
        if (m.tier == Tier::VRAM && (int)target >= (int)Tier::RAM)
            if (!demote_vram_vpid(v)) return false;
        if (m.tier == Tier::RAM && target == Tier::DISK)
            if (!demote_ram_vpid(v)) return false;
        return (int)m.tier >= (int)target;
    }

    // -- introspection --------------------------------------------------------
    Tier   tier_of(VPid v) const { return cmeta(v).tier; }
    // Physical page; valid only while VRAM-resident (assert otherwise).
    PageId phys_of(VPid v) const {
        const PageMeta& m = cmeta(v);
        assert(m.tier == Tier::VRAM);
        return m.phys;
    }
    bool  is_pinned(VPid v) const { return cmeta(v).pins > 0; }
    int   ref_count(VPid v) const { return cmeta(v).refs; }
    void  touch(VPid v) {
        PageMeta& m = meta(v);
        m.tick = ++m_tick;
        if (m.tier == Tier::VRAM && m.pins == 0) lru_move_front(m_vram_lru, v);
        else if (m.tier == Tier::RAM)            lru_move_front(m_ram_lru, v);
    }
    const Stats&  stats()  const { return m_stats; }
    const Config& config() const { return m_cfg; }
    int free_ram_slots()  const { return (int)m_free_ram.size(); }
    int free_disk_slots() const { return (int)m_free_disk.size(); }

private:
    struct PageMeta {
        Tier     tier = Tier::VRAM;
        PageId   phys = -1;    // valid when tier == VRAM
        int32_t  slot = -1;    // RAM or DISK slot otherwise
        int32_t  refs = 0;     // radix-tree adoptions
        int32_t  pins = 0;     // VRAM locks (block tables / serializer I/O)
        uint64_t tick = 0;     // LRU clock
        VPid     prev = kNoVPid, next = kNoVPid;   // intrusive LRU links
        bool     live = false;
    };
    struct LruList { VPid head = kNoVPid, tail = kNoVPid; };

    PageMeta&       meta(VPid v)        { assert(m_meta[v].live); return m_meta[v]; }
    const PageMeta& cmeta(VPid v) const { assert(m_meta[v].live); return m_meta[v]; }

    VPid alloc_meta() {
        VPid v;
        if (!m_free_meta.empty()) { v = m_free_meta.back(); m_free_meta.pop_back(); }
        else { v = (VPid)m_meta.size(); m_meta.emplace_back(); }
        m_meta[v] = PageMeta{};
        m_meta[v].live = true;
        return v;
    }

    // Free the backing at whatever tier + retire the VPid.
    void destroy(VPid v) {
        PageMeta& m = meta(v);
        assert(m.pins == 0);
        switch (m.tier) {
            case Tier::VRAM:
                lru_remove(m_vram_lru, v);
                m_pool.release(m.phys);
                --m_stats.vram;
                break;
            case Tier::RAM:
                lru_remove(m_ram_lru, v);
                m_free_ram.push_back(m.slot);
                --m_stats.ram;
                break;
            case Tier::DISK:
                m_free_disk.push_back(m.slot);
                --m_stats.disk;
                break;
        }
        m.live = false;
        m_free_meta.push_back(v);
    }

    // A free physical page, demoting LRU VRAM pages as needed. -1 on failure.
    PageId acquire_phys() {
        PageId phys = m_pool.try_allocate();
        while (phys < 0) {
            if (!demote_lru_vram()) return -1;
            phys = m_pool.try_allocate();
        }
        return phys;
    }

    bool demote_lru_vram() {
        return m_vram_lru.tail != kNoVPid && demote_vram_vpid(m_vram_lru.tail);
    }

    bool demote_vram_vpid(VPid v) {
        const int rslot = acquire_ram_slot();
        if (rslot < 0) return false;
        PageMeta& m = meta(v);
        assert(m.tier == Tier::VRAM && m.pins == 0);
        lru_remove(m_vram_lru, v);
        m_io.vram_to_ram(m.phys, rslot);
        m_pool.release(m.phys);
        m.tier = Tier::RAM;
        m.phys = -1;
        m.slot = rslot;
        lru_push_front(m_ram_lru, v);
        --m_stats.vram; ++m_stats.ram; ++m_stats.demotions_ram;
        return true;
    }

    int acquire_ram_slot() {
        if (m_free_ram.empty()) {
            const VPid victim = m_ram_lru.tail;
            if (victim == kNoVPid || !demote_ram_vpid(victim)) return -1;
        }
        const int s = m_free_ram.back();
        m_free_ram.pop_back();
        return s;
    }

    bool demote_ram_vpid(VPid v) {
        if (m_free_disk.empty()) return false;
        const int dslot = m_free_disk.back();
        m_free_disk.pop_back();
        PageMeta& m = meta(v);
        assert(m.tier == Tier::RAM);
        lru_remove(m_ram_lru, v);
        m_io.ram_to_disk(m.slot, dslot);
        m_free_ram.push_back(m.slot);
        m.tier = Tier::DISK;
        m.slot = dslot;
        --m_stats.ram; ++m_stats.disk; ++m_stats.demotions_disk;
        return true;
    }

    void pin_one(VPid v) {
        PageMeta& m = meta(v);
        assert(m.tier == Tier::VRAM);
        if (m.pins++ == 0) lru_remove(m_vram_lru, v);
    }

    // -- intrusive doubly-linked LRU (head = MRU, tail = LRU victim) ----------
    void lru_push_front(LruList& l, VPid v) {
        PageMeta& m = m_meta[v];
        m.prev = kNoVPid;
        m.next = l.head;
        if (l.head != kNoVPid) m_meta[l.head].prev = v;
        l.head = v;
        if (l.tail == kNoVPid) l.tail = v;
    }
    void lru_remove(LruList& l, VPid v) {
        PageMeta& m = m_meta[v];
        if (m.prev != kNoVPid) m_meta[m.prev].next = m.next; else l.head = m.next;
        if (m.next != kNoVPid) m_meta[m.next].prev = m.prev; else l.tail = m.prev;
        m.prev = m.next = kNoVPid;
    }
    void lru_move_front(LruList& l, VPid v) {
        if (l.head == v) return;
        lru_remove(l, v);
        lru_push_front(l, v);
    }

    VramPool&   m_pool;
    TierBackend& m_io;
    Config      m_cfg;
    Stats       m_stats;

    std::vector<PageMeta> m_meta;
    std::vector<VPid>     m_free_meta;
    std::vector<int>      m_free_ram, m_free_disk;
    LruList  m_vram_lru, m_ram_lru;
    uint64_t m_tick = 0;
};

}} // namespace blackwell::paging
