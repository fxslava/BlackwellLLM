#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <stack>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

#include "kernels/paged_flash_attention.cuh"
#include "common.h"   // CUDA_CHECK

// ============================================================================
// Paged KV cache + Copy-on-Write sequence manager (host control plane)
// ============================================================================
// The GPU sees only two things per step: the bf16 page pools (one contiguous
// arena, indexed [layer][page][kv_head][slot][dim]) and, per active sequence,
// a flat int32 block table + seq_len. Everything below — page ref counting,
// fork/rewind, CoW — is pure host bookkeeping plus the occasional one-page
// device copy. Pages are reference-counted and shared across sequences; a page
// is physically duplicated only when a sharer first writes into it.
namespace blackwell { namespace paging {

using SeqId  = int32_t;
using PageId = int32_t;
constexpr PageId kNullPage = -1;

// One fixed-size KV slab (data lives in the device pool; this is just metadata).
struct PhysicalPage {
    int32_t ref_count = 0;   // # of block-table entries pointing here; 0 == free
};

// Per-sequence logical->physical map.
struct BlockTable {
    std::vector<PageId> pages;   // logical block i -> physical page id
    int32_t length = 0;          // valid tokens (<= pages.size() * PAGE_SIZE)
};

// Result of reserving the slot for the token about to be written.
struct AppendSlot { PageId page; int slot; };

// ---------------------------------------------------------------------------
// Free-list page allocator with reference counting.
// ---------------------------------------------------------------------------
class PageAllocator {
public:
    explicit PageAllocator(int total_pages) : m_pages(total_pages) {
        for (PageId p = total_pages - 1; p >= 0; --p) m_free.push(p);
    }
    PageId allocate() {
        if (m_free.empty())
            throw std::runtime_error("PagedKVCache: out of physical pages (OOM)");
        PageId p = m_free.top(); m_free.pop();
        m_pages[p].ref_count = 1;
        return p;
    }
    void incref(PageId p) { ++m_pages[p].ref_count; }
    // Returns true if the page was freed (ref hit zero).
    bool decref(PageId p) {
        if (--m_pages[p].ref_count == 0) { m_free.push(p); return true; }
        return false;
    }
    int  ref_count(PageId p) const { return m_pages[p].ref_count; }
    int  free_pages()        const { return (int)m_free.size(); }

private:
    std::vector<PhysicalPage> m_pages;
    std::stack<PageId>        m_free;
};

// ---------------------------------------------------------------------------
// SequenceManager: owns the device pools, the allocator and every sequence's
// block table. The KV pool is shared across all layers (a page id indexes the
// same physical slot in every layer), so fork/rewind/CoW operate once per page
// regardless of layer count.
// ---------------------------------------------------------------------------
class SequenceManager {
public:
    // num_gpu_layers: how many LEADING layers keep their KV pages resident in the
    // device pool. The remaining (num_layers - num_gpu_layers) layers are "KV
    // offloaded": their pages live in a pinned host mirror and are streamed into a
    // small set of device staging slabs on demand (stage_in_layer / spill_out_page).
    // A negative value (the default) or a value >= num_layers keeps EVERY layer
    // resident -- byte-for-byte the original all-VRAM behaviour (no host mirror,
    // no staging slots), so existing all-resident call sites are unchanged.
    SequenceManager(int num_layers, int num_kv_heads, int head_dim,
                    int total_pages, int max_blocks_per_seq, int num_gpu_layers = -1)
        : m_num_layers(num_layers), m_num_kv_heads(num_kv_heads),
          m_head_dim(head_dim), m_total_pages(total_pages),
          m_max_blocks(max_blocks_per_seq), m_alloc(total_pages) {
        m_resident  = (num_gpu_layers < 0 || num_gpu_layers > num_layers)
                          ? num_layers : num_gpu_layers;
        m_offloaded = num_layers - m_resident;
        m_slab_elems = (size_t)total_pages * per_page_elems();   // one layer's pool

        // The device pool holds the resident layers' slabs, plus -- only when KV
        // offloading is active -- kNumSlots transient staging slabs that offloaded
        // layers cycle through (slot = layer % kNumSlots). This is what bounds VRAM:
        // device KV scales with m_resident, not num_layers.
        const int device_slabs = m_resident + (m_offloaded > 0 ? kNumSlots : 0);
        const size_t pool_elems = (size_t)device_slabs * m_slab_elems;
        CUDA_CHECK_THROW(cudaMalloc(&m_d_k_pool, pool_elems * sizeof(kv_t)));
        CUDA_CHECK_THROW(cudaMalloc(&m_d_v_pool, pool_elems * sizeof(kv_t)));
        CUDA_CHECK_THROW(cudaMemset(m_d_k_pool, 0, pool_elems * sizeof(kv_t)));
        CUDA_CHECK_THROW(cudaMemset(m_d_v_pool, 0, pool_elems * sizeof(kv_t)));
        CUDA_CHECK_THROW(cudaMalloc(&m_d_block_scratch, max_blocks_per_seq * sizeof(int32_t)));

        // Pinned host mirror for the offloaded layers (page-indexed, identical
        // per-layer layout). cudaHostAlloc so stage/spill are true DMA transfers
        // with no pageable bounce buffer -- the same requirement PinnedHostPool
        // enforces for the continuous cache.
        if (m_offloaded > 0) {
            const size_t mirror_bytes = (size_t)m_offloaded * m_slab_elems * sizeof(kv_t);
            CUDA_CHECK_THROW(cudaHostAlloc(&m_h_k_mirror, mirror_bytes, cudaHostAllocDefault));
            CUDA_CHECK_THROW(cudaHostAlloc(&m_h_v_mirror, mirror_bytes, cudaHostAllocDefault));
            std::memset(m_h_k_mirror, 0, mirror_bytes);
            std::memset(m_h_v_mirror, 0, mirror_bytes);
        }
    }
    ~SequenceManager() {
        cudaFree(m_d_k_pool); cudaFree(m_d_v_pool); cudaFree(m_d_block_scratch);
        if (m_h_k_mirror) cudaFreeHost(m_h_k_mirror);
        if (m_h_v_mirror) cudaFreeHost(m_h_v_mirror);
    }
    SequenceManager(const SequenceManager&) = delete;
    SequenceManager& operator=(const SequenceManager&) = delete;

    // -- lifecycle ----------------------------------------------------------
    SeqId create_sequence() {
        SeqId id = m_next_id++;
        m_seqs.emplace(id, BlockTable{});
        return id;
    }
    void destroy_sequence(SeqId s) {
        auto& bt = seq(s);
        for (PageId p : bt.pages) m_alloc.decref(p);
        m_seqs.erase(s);
    }

    // Reserve the physical slot for the NEXT token of `seq`, resolving CoW.
    // Call exactly once per generated token, BEFORE the per-layer append; the
    // returned (page, slot) is then valid for every layer of that step.
    AppendSlot reserve_append_slot(SeqId s, cudaStream_t stream = 0) {
        BlockTable& bt = seq(s);
        const int pos   = bt.length;
        const int block = pos / PAGE_SIZE;
        const int slot  = pos % PAGE_SIZE;
        if ((int)bt.pages.size() == block) {
            bt.pages.push_back(m_alloc.allocate());          // fresh page, ref==1
        } else {
            PageId pid = bt.pages[block];
            if (m_alloc.ref_count(pid) > 1) {                // shared -> CoW
                PageId np = m_alloc.allocate();
                // Resident layers occupy the first m_resident device slabs in the
                // SAME [layer][page] layout the kernel expects, so the device copy
                // runs over exactly those layers. (m_resident == m_num_layers in
                // the all-resident default -> identical to the original call.)
                if (m_resident > 0)
                    launch_cow_copy_page(m_d_k_pool, m_d_v_pool, pid, np,
                                         m_resident, m_total_pages,
                                         m_num_kv_heads, m_head_dim, stream);
                // Offloaded layers live in the host mirror; duplicate the page there
                // too so the forked child carries their full history when re-staged.
                cow_copy_page_host(pid, np, stream);
                m_alloc.decref(pid);
                bt.pages[block] = np;
            }
        }
        ++bt.length;
        return { bt.pages[block], slot };
    }

    // Branch: child shares all of parent's pages (ref++). O(num_blocks), no copy.
    SeqId fork(SeqId parent) {
        SeqId child = create_sequence();
        BlockTable& c = seq(child);
        const BlockTable& p = seq(parent);
        c.pages = p.pages;
        c.length = p.length;
        for (PageId pg : c.pages) m_alloc.incref(pg);
        return child;
    }

    // -- micro-rewind (Continuous Speculative Tracking) ----------------------
    // Truncate the sequence's tail so exactly keep_tokens remain. Trailing
    // pages whose LAST reference this block table held are returned to the
    // free list; pages shared with a fork / the radix tree merely lose this
    // sequence's reference and live on for the other holders. The boundary
    // page may remain shared after truncation — the next append CoWs it
    // (reserve_append_slot's existing rule), so a diverging re-append can
    // never corrupt a cached or forked copy of the old tail.
    // Returns the number of physical pages actually freed. keep_tokens >=
    // length is a no-op returning 0: truncation never grows a sequence.
    int truncate(SeqId s, int keep_tokens) {
        BlockTable& bt = seq(s);
        if (keep_tokens < 0) keep_tokens = 0;
        if (keep_tokens >= bt.length) return 0;
        const int keep_blocks = (keep_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
        int freed = 0;
        while ((int)bt.pages.size() > keep_blocks) {
            if (m_alloc.decref(bt.pages.back())) ++freed;
            bt.pages.pop_back();
        }
        bt.length = keep_tokens;
        return freed;
    }

    // Roll back to target_len tokens; drop now-unreachable trailing pages.
    void rewind(SeqId s, int target_len) { (void)truncate(s, target_len); }

    // -- introspection (read-only; tests / debugging) -----------------------
    int    length(SeqId s)            const { return m_seqs.at(s).length; }
    int    num_blocks(SeqId s)        const { return (int)m_seqs.at(s).pages.size(); }
    PageId block_page(SeqId s, int b) const { return m_seqs.at(s).pages.at(b); }
    int    page_ref_count(PageId p)   const { return m_alloc.ref_count(p); }

    // -- prefix-cache / serialization plumbing -------------------------------
    // Used by RadixTreeIndex, KVBranchSerializer and PrefixCacheManager
    // (src/paging/radix_tree.h, kv_branch_serializer.h, prefix_cache_manager.h).
    // The tree and the serializer hold page references OUTSIDE any sequence via
    // retain/release; a page stays allocated while EITHER a block table or the
    // radix tree points at it — same ref_count, one owner rule.
    int    num_layers()   const { return m_num_layers; }
    int    num_kv_heads() const { return m_num_kv_heads; }
    int    head_dim()     const { return m_head_dim; }
    int    total_pages()  const { return m_total_pages; }
    size_t page_elems()   const { return per_page_elems(); }   // per layer, K or V

    // New sequence whose leading blocks are pre-populated shared pages (radix
    // prefix hit or a deserialized branch). Increfs every page — the sequence
    // owns its reference exactly as if it had been fork()ed. `length` must be
    // <= pages.size() * PAGE_SIZE; appends past it CoW shared pages as usual.
    SeqId create_sequence_with_pages(const std::vector<PageId>& pages, int length) {
        SeqId id = create_sequence();
        BlockTable& bt = seq(id);
        bt.pages  = pages;
        bt.length = length;
        for (PageId p : bt.pages) m_alloc.incref(p);
        return id;
    }

    // Raw page ownership for non-sequence holders (deserializer fills a page
    // before any block table exists). allocate_raw_page returns ref==1 owned by
    // the caller; balance with release_page (or transfer via retain elsewhere).
    PageId allocate_raw_page()      { return m_alloc.allocate(); }
    void   retain_page(PageId p)    { m_alloc.incref(p); }
    void   release_page(PageId p)   { m_alloc.decref(p); }

    // Copy ONE layer's K/V of one physical page to/from host memory (h_k / h_v
    // hold page_elems() kv_t each). Resident layers move device<->host on
    // `stream` (use pinned buffers + sync before touching the bytes); offloaded
    // layers hit the pinned host mirror synchronously — the mirror IS the truth
    // for them, no staging slab round-trip.
    void read_page(int layer, PageId p, kv_t* h_k, kv_t* h_v,
                   cudaStream_t stream = 0) const {
        const size_t pp = per_page_elems(), bytes = pp * sizeof(kv_t);
        const size_t off = (size_t)p * pp;
        if (layer_is_offloaded(layer)) {
            std::memcpy(h_k, host_k_base(layer) + off, bytes);
            std::memcpy(h_v, host_v_base(layer) + off, bytes);
        } else {
            CUDA_CHECK_THROW(cudaMemcpyAsync(h_k, layer_k_pool(layer) + off, bytes,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_THROW(cudaMemcpyAsync(h_v, layer_v_pool(layer) + off, bytes,
                                       cudaMemcpyDeviceToHost, stream));
        }
    }
    void write_page(int layer, PageId p, const kv_t* h_k, const kv_t* h_v,
                    cudaStream_t stream = 0) {
        const size_t pp = per_page_elems(), bytes = pp * sizeof(kv_t);
        const size_t off = (size_t)p * pp;
        if (layer_is_offloaded(layer)) {
            std::memcpy(host_k_base(layer) + off, h_k, bytes);
            std::memcpy(host_v_base(layer) + off, h_v, bytes);
        } else {
            CUDA_CHECK_THROW(cudaMemcpyAsync(layer_k_pool(layer) + off, h_k, bytes,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_THROW(cudaMemcpyAsync(layer_v_pool(layer) + off, h_v, bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
    }

    // -- device views handed to the kernel launchers ------------------------

    // Upload this sequence's block table; returns a device pointer valid until
    // the next call (single reusable scratch buffer — fine for one active seq).
    const int32_t* device_block_table(SeqId s, cudaStream_t stream = 0) {
        const BlockTable& bt = seq(s);
        CUDA_CHECK_THROW(cudaMemcpyAsync(m_d_block_scratch, bt.pages.data(),
                                   bt.pages.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        return m_d_block_scratch;
    }

    bool layer_is_offloaded(int layer) const { return layer >= m_resident; }

    // Device base of a layer's KV pool. Resident layers map to their own slab;
    // offloaded layers map to the staging slab they cycle through (layer % slots).
    // For an offloaded layer the returned pointer is only valid AFTER a matching
    // stage_in_layer() (the kernels read/write through it, exactly as for resident
    // layers -- callers never branch on residency).
    kv_t* layer_k_pool(int layer) const { return m_d_k_pool + (size_t)slab_of(layer) * m_slab_elems; }
    kv_t* layer_v_pool(int layer) const { return m_d_v_pool + (size_t)slab_of(layer) * m_slab_elems; }
    kv_t* k_pool() const { return m_d_k_pool; }
    kv_t* v_pool() const { return m_d_v_pool; }

    // ---- KV offloading: page-aware stage-in / spill-out -------------------
    // Bring an offloaded layer's live KV pages back into its device staging slab
    // BEFORE the layer's append/attention. Driven by the sequence's block table,
    // so only the pages it actually occupies are transferred (host -> device).
    // No-op for resident layers. Stream-ordered against the kernels (same stream).
    void stage_in_layer(int layer, SeqId s, cudaStream_t stream = 0) {
        if (!layer_is_offloaded(layer)) return;
        const BlockTable& bt = seq(s);
        kv_t* d_k = layer_k_pool(layer);
        kv_t* d_v = layer_v_pool(layer);
        const kv_t* h_k = host_k_base(layer);
        const kv_t* h_v = host_v_base(layer);
        const size_t pp = per_page_elems(), page_bytes = pp * sizeof(kv_t);
        for (PageId p : bt.pages) {
            const size_t off = (size_t)p * pp;
            CUDA_CHECK_THROW(cudaMemcpyAsync(d_k + off, h_k + off, page_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_THROW(cudaMemcpyAsync(d_v + off, h_v + off, page_bytes, cudaMemcpyHostToDevice, stream));
        }
    }

    // Spill the single page just appended (device staging slab -> host mirror)
    // AFTER the layer's attention, so the freshly written column survives the slab
    // being reused by the next offloaded layer. No-op for resident layers.
    void spill_out_page(int layer, PageId page, cudaStream_t stream = 0) {
        if (!layer_is_offloaded(layer)) return;
        const size_t pp = per_page_elems(), off = (size_t)page * pp, page_bytes = pp * sizeof(kv_t);
        CUDA_CHECK_THROW(cudaMemcpyAsync(host_k_base(layer) + off, layer_k_pool(layer) + off,
                                   page_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(host_v_base(layer) + off, layer_v_pool(layer) + off,
                                   page_bytes, cudaMemcpyDeviceToHost, stream));
    }

    int  num_gpu_layers() const { return m_resident; }
    int  free_pages()     const { return m_alloc.free_pages(); }

private:
    static constexpr int kNumSlots = 2;   // device staging slabs for offloaded layers

    size_t per_page_elems() const {
        return (size_t)m_num_kv_heads * PAGE_SIZE * m_head_dim;
    }
    BlockTable&       seq(SeqId s)       { return m_seqs.at(s); }
    const BlockTable& seq(SeqId s) const { return m_seqs.at(s); }

    // Device slab backing a layer's KV pool: its own slab if resident, else the
    // staging slab it cycles through (slot = layer % kNumSlots, after the resident
    // slabs). For the all-resident default this is just `layer`.
    int slab_of(int layer) const {
        return layer_is_offloaded(layer) ? (m_resident + (layer % kNumSlots)) : layer;
    }
    // Base of an offloaded layer's pages in the pinned host mirror.
    kv_t* host_k_base(int layer) const { return m_h_k_mirror + (size_t)(layer - m_resident) * m_slab_elems; }
    kv_t* host_v_base(int layer) const { return m_h_v_mirror + (size_t)(layer - m_resident) * m_slab_elems; }

    // Host-side CoW for offloaded layers' mirror pages. Waits for any in-flight
    // spill (async D2H on `stream`) to land, then duplicates src->dst for every
    // offloaded layer. CoW is a rare fork-boundary event, so the sync is cheap.
    void cow_copy_page_host(PageId src, PageId dst, cudaStream_t stream = 0) {
        if (m_offloaded <= 0) return;
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        const size_t pp = per_page_elems(), page_bytes = pp * sizeof(kv_t);
        for (int oi = 0; oi < m_offloaded; ++oi) {
            const size_t base = (size_t)oi * m_slab_elems;
            std::memcpy(m_h_k_mirror + base + (size_t)dst * pp,
                        m_h_k_mirror + base + (size_t)src * pp, page_bytes);
            std::memcpy(m_h_v_mirror + base + (size_t)dst * pp,
                        m_h_v_mirror + base + (size_t)src * pp, page_bytes);
        }
    }

    int m_num_layers, m_num_kv_heads, m_head_dim, m_total_pages, m_max_blocks;
    PageAllocator m_alloc;
    std::unordered_map<SeqId, BlockTable> m_seqs;
    SeqId m_next_id = 0;

    kv_t*    m_d_k_pool = nullptr;
    kv_t*    m_d_v_pool = nullptr;
    int32_t* m_d_block_scratch = nullptr;

    // KV offloading state (m_offloaded == 0 => everything resident, all no-ops).
    int    m_resident = 0;      // leading layers kept in the device pool
    int    m_offloaded = 0;     // num_layers - m_resident, mirrored to host
    size_t m_slab_elems = 0;    // elements in one layer's pool (total_pages*per_page)
    kv_t*  m_h_k_mirror = nullptr;   // pinned host mirror for offloaded layers' K pages
    kv_t*  m_h_v_mirror = nullptr;   // ... and V pages
};

}} // namespace blackwell::paging
