#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "common.h"  // CUDA_CHECK_THROW (Hybrid error doctrine)
#include "paged_kv_cache.h"
#include "radix_tree.h"       // TokenId, hash helper

// ============================================================================
// KVBranchSerializer — zero-prefill KV state persistence (.bkv files)
// ============================================================================
// Dumps a page chain (a radix-tree branch: system prompt, few-shot header, a
// ToT parent…) to disk and reloads it into freshly allocated pages across
// process restarts, skipping prefill entirely: the reloaded pages slot
// straight into block tables / the radix tree and the attention kernels never
// know the difference.
//
// FILE LAYOUT (little-endian, x86-64 host assumed):
//   [0, 4096)                  BkvHeader (padded)
//   [4096, tokens_end)         int32 token ids, zero-padded to a 4 KiB multiple
//   [payload_off, EOF)         page-major KV payload:
//                                for page in [0, num_pages):
//                                  for layer in [0, num_layers):
//                                    K blob, V blob     (page_bytes each, bf16)
//
// Page-major ordering means a PREFIX of the file is itself a valid shorter
// branch — a truncated / partial load restores the first N pages with all
// layers intact. Every section starts 4 KiB-aligned and page_bytes is a 4 KiB
// multiple for all real geometries (kv_heads*16*head_dim*2), so the identical
// format later serves sector-aligned unbuffered I/O and the DirectStorage
// file->VRAM path (see docs/KV_PREFIX_CACHE.md) with no format change.
//
// v1 I/O is synchronous std::fstream through ONE pinned staging buffer per
// (page, layer) — correct and NVMe-sequential; double-buffering and
// DirectStorage are documented upgrades, not format changes.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

constexpr uint32_t kBkvVersion   = 1;
constexpr uint32_t kBkvDtypeBf16 = 0;
constexpr size_t   kBkvAlign     = 4096;

struct BkvHeader {
    char     magic[8];          // "BLKWLKV1"
    uint32_t version;
    uint32_t dtype;             // kBkvDtypeBf16
    uint32_t page_size;         // tokens per page (PAGE_SIZE)
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t num_tokens;        // multiple of page_size
    uint32_t num_pages;         // num_tokens / page_size
    uint64_t model_hash;        // caller-provided model identity guard
    uint64_t token_hash;        // FNV-1a over the token section (integrity)
    uint64_t token_section_off; // == 4096
    uint64_t payload_off;       // 4 KiB-aligned
    uint8_t  reserved[kBkvAlign - 72];
};
static_assert(sizeof(BkvHeader) == kBkvAlign, "BkvHeader must be one 4KiB sector");

class KVBranchSerializer {
public:
    // model_hash: any stable identity of (checkpoint x geometry) — reloading
    // a branch into a different model is rejected, not silently corrupted.
    KVBranchSerializer(SequenceManager& sm, uint64_t model_hash)
        : m_sm(sm), m_model_hash(model_hash) {
        m_page_bytes = m_sm.page_elems() * sizeof(kv_t);
        CUDA_CHECK_THROW(cudaHostAlloc(&m_stage_k, m_page_bytes, cudaHostAllocDefault));
        CUDA_CHECK_THROW(cudaHostAlloc(&m_stage_v, m_page_bytes, cudaHostAllocDefault));
    }
    ~KVBranchSerializer() {
        cudaFreeHost(m_stage_k);
        cudaFreeHost(m_stage_v);
    }
    KVBranchSerializer(const KVBranchSerializer&) = delete;
    KVBranchSerializer& operator=(const KVBranchSerializer&) = delete;

    // -----------------------------------------------------------------------
    // Dump `pages` (the chain backing tokens[0..num_tokens), num_tokens a
    // multiple of PAGE_SIZE) to `path`. The pages must stay live for the
    // call — PrefixCacheManager locks the branch around this.
    // -----------------------------------------------------------------------
    void dump(const std::string& path, const TokenId* tokens, int num_tokens,
              const std::vector<PageId>& pages, cudaStream_t stream = 0) {
        if (num_tokens % PAGE_SIZE != 0 ||
            (int)pages.size() != num_tokens / PAGE_SIZE)
            throw std::runtime_error("bkv dump: tokens/pages mismatch");

        BkvHeader h{};
        std::memcpy(h.magic, "BLKWLKV1", 8);
        h.version           = kBkvVersion;
        h.dtype             = kBkvDtypeBf16;
        h.page_size         = PAGE_SIZE;
        h.num_layers        = (uint32_t)m_sm.num_layers();
        h.num_kv_heads      = (uint32_t)m_sm.num_kv_heads();
        h.head_dim          = (uint32_t)m_sm.head_dim();
        h.num_tokens        = (uint32_t)num_tokens;
        h.num_pages         = (uint32_t)pages.size();
        h.model_hash        = m_model_hash;
        h.token_hash        = hash_page_tokens(tokens, num_tokens);
        h.token_section_off = sizeof(BkvHeader);
        h.payload_off       = h.token_section_off +
                              align_up((size_t)num_tokens * sizeof(TokenId));

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("bkv dump: cannot open " + path);
        f.write((const char*)&h, sizeof(h));
        f.write((const char*)tokens, (size_t)num_tokens * sizeof(TokenId));
        write_pad(f, align_up((size_t)num_tokens * sizeof(TokenId)) -
                     (size_t)num_tokens * sizeof(TokenId));

        for (PageId p : pages) {
            for (int layer = 0; layer < m_sm.num_layers(); ++layer) {
                m_sm.read_page(layer, p, m_stage_k, m_stage_v, stream);
                CUDA_CHECK_THROW(cudaStreamSynchronize(stream));  // D2H into staging landed
                f.write((const char*)m_stage_k, m_page_bytes);
                f.write((const char*)m_stage_v, m_page_bytes);
            }
        }
        if (!f) throw std::runtime_error("bkv dump: write failed for " + path);
    }

    // -----------------------------------------------------------------------
    // Load a branch from `path` into freshly allocated pages. On return the
    // CALLER owns one reference per page (allocate_raw_page); it hands them to
    // the radix tree via insert() and then drops its own refs (see
    // PrefixCacheManager::load). Throws (with full page rollback) on I/O
    // error, geometry/model mismatch, or page-pool exhaustion — call
    // ensure_free_pages() first.
    // -----------------------------------------------------------------------
    struct LoadResult {
        std::vector<TokenId> tokens;
        std::vector<PageId>  pages;
    };
    LoadResult load(const std::string& path, cudaStream_t stream = 0) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("bkv load: cannot open " + path);

        BkvHeader h{};
        f.read((char*)&h, sizeof(h));
        if (!f || std::memcmp(h.magic, "BLKWLKV1", 8) != 0)
            throw std::runtime_error("bkv load: bad magic in " + path);
        if (h.version != kBkvVersion || h.dtype != kBkvDtypeBf16)
            throw std::runtime_error("bkv load: unsupported version/dtype");
        if (h.model_hash != m_model_hash)
            throw std::runtime_error("bkv load: model_hash mismatch (foreign checkpoint)");
        if (h.page_size    != (uint32_t)PAGE_SIZE          ||
            h.num_layers   != (uint32_t)m_sm.num_layers()  ||
            h.num_kv_heads != (uint32_t)m_sm.num_kv_heads()||
            h.head_dim     != (uint32_t)m_sm.head_dim())
            throw std::runtime_error("bkv load: KV geometry mismatch");

        LoadResult r;
        r.tokens.resize(h.num_tokens);
        f.seekg((std::streamoff)h.token_section_off);
        f.read((char*)r.tokens.data(), (size_t)h.num_tokens * sizeof(TokenId));
        if (!f || hash_page_tokens(r.tokens.data(), (int)h.num_tokens) != h.token_hash)
            throw std::runtime_error("bkv load: token section corrupt");

        f.seekg((std::streamoff)h.payload_off);
        try {
            for (uint32_t pi = 0; pi < h.num_pages; ++pi) {
                const PageId p = m_sm.allocate_raw_page();   // throws on OOM
                r.pages.push_back(p);
                for (int layer = 0; layer < m_sm.num_layers(); ++layer) {
                    f.read((char*)m_stage_k, m_page_bytes);
                    f.read((char*)m_stage_v, m_page_bytes);
                    if (!f) throw std::runtime_error("bkv load: truncated payload");
                    m_sm.write_page(layer, p, m_stage_k, m_stage_v, stream);
                    // Staging is reused next iteration; the H2D must be done.
                    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
                }
            }
        } catch (...) {
            for (PageId p : r.pages) m_sm.release_page(p);   // full rollback
            throw;
        }
        return r;
    }

private:
    static size_t align_up(size_t v) { return (v + kBkvAlign - 1) & ~(kBkvAlign - 1); }
    static void write_pad(std::ofstream& f, size_t n) {
        static const char zeros[kBkvAlign] = {};
        f.write(zeros, (std::streamsize)n);
    }

    SequenceManager& m_sm;
    uint64_t m_model_hash;
    size_t   m_page_bytes = 0;   // one layer's K (or V) page blob
    kv_t*    m_stage_k = nullptr;   // pinned staging, page_bytes each
    kv_t*    m_stage_v = nullptr;
};

}} // namespace blackwell::paging
