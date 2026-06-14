#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <stack>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

#include "../kernels/paged_flash_attention.cuh"
#include "../common.h"   // CUDA_CHECK

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
    SequenceManager(int num_layers, int num_kv_heads, int head_dim,
                    int total_pages, int max_blocks_per_seq)
        : m_num_layers(num_layers), m_num_kv_heads(num_kv_heads),
          m_head_dim(head_dim), m_total_pages(total_pages),
          m_max_blocks(max_blocks_per_seq), m_alloc(total_pages) {
        const size_t per_page = per_page_elems();
        const size_t pool_elems = (size_t)num_layers * total_pages * per_page;
        CUDA_CHECK(cudaMalloc(&m_d_k_pool, pool_elems * sizeof(kv_t)));
        CUDA_CHECK(cudaMalloc(&m_d_v_pool, pool_elems * sizeof(kv_t)));
        CUDA_CHECK(cudaMemset(m_d_k_pool, 0, pool_elems * sizeof(kv_t)));
        CUDA_CHECK(cudaMemset(m_d_v_pool, 0, pool_elems * sizeof(kv_t)));
        CUDA_CHECK(cudaMalloc(&m_d_block_scratch, max_blocks_per_seq * sizeof(int32_t)));
    }
    ~SequenceManager() {
        cudaFree(m_d_k_pool); cudaFree(m_d_v_pool); cudaFree(m_d_block_scratch);
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
                launch_cow_copy_page(m_d_k_pool, m_d_v_pool, pid, np,
                                     m_num_layers, m_total_pages,
                                     m_num_kv_heads, m_head_dim, stream);
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

    // Roll back to target_len tokens; drop now-unreachable trailing pages.
    void rewind(SeqId s, int target_len) {
        BlockTable& bt = seq(s);
        if (target_len >= bt.length) return;
        const int keep_blocks = (target_len + PAGE_SIZE - 1) / PAGE_SIZE;
        while ((int)bt.pages.size() > keep_blocks) {
            m_alloc.decref(bt.pages.back());   // frees iff ref hits zero
            bt.pages.pop_back();
        }
        bt.length = target_len;
        // The boundary page may still be fork-shared; the next append CoWs it.
    }

    // -- introspection (read-only; tests / debugging) -----------------------
    int    length(SeqId s)            const { return m_seqs.at(s).length; }
    int    num_blocks(SeqId s)        const { return (int)m_seqs.at(s).pages.size(); }
    PageId block_page(SeqId s, int b) const { return m_seqs.at(s).pages.at(b); }
    int    page_ref_count(PageId p)   const { return m_alloc.ref_count(p); }

    // -- device views handed to the kernel launchers ------------------------

    // Upload this sequence's block table; returns a device pointer valid until
    // the next call (single reusable scratch buffer — fine for one active seq).
    const int32_t* device_block_table(SeqId s, cudaStream_t stream = 0) {
        const BlockTable& bt = seq(s);
        CUDA_CHECK(cudaMemcpyAsync(m_d_block_scratch, bt.pages.data(),
                                   bt.pages.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        return m_d_block_scratch;
    }

    kv_t* layer_k_pool(int layer) const {
        return m_d_k_pool + (size_t)layer * m_total_pages * per_page_elems();
    }
    kv_t* layer_v_pool(int layer) const {
        return m_d_v_pool + (size_t)layer * m_total_pages * per_page_elems();
    }
    kv_t* k_pool() const { return m_d_k_pool; }
    kv_t* v_pool() const { return m_d_v_pool; }

    int free_pages() const { return m_alloc.free_pages(); }

private:
    size_t per_page_elems() const {
        return (size_t)m_num_kv_heads * PAGE_SIZE * m_head_dim;
    }
    BlockTable&       seq(SeqId s)       { return m_seqs.at(s); }
    const BlockTable& seq(SeqId s) const { return m_seqs.at(s); }

    int m_num_layers, m_num_kv_heads, m_head_dim, m_total_pages, m_max_blocks;
    PageAllocator m_alloc;
    std::unordered_map<SeqId, BlockTable> m_seqs;
    SeqId m_next_id = 0;

    kv_t*    m_d_k_pool = nullptr;
    kv_t*    m_d_v_pool = nullptr;
    int32_t* m_d_block_scratch = nullptr;
};

}} // namespace blackwell::paging
