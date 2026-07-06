#include "kv_cache/paged_kv_manager.h"
#include "kernels/rope.cuh"
#include "kernels/paged_flash_attention.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace blackwell {

PagedKVManager::PagedKVManager(const ModelConfig& config, size_t max_seq_len,
                               size_t num_gpu_layers, int branch_factor,
                               int min_total_pages)
    : m_config(config),
      m_num_q_heads(static_cast<int>(config.num_attention_heads)),
      m_num_kv_heads(static_cast<int>(config.num_key_value_heads)),
      m_head_dim(static_cast<int>(config.head_dim)),
      m_rope_theta(config.rope_theta)
{
    const int max_blocks =
        static_cast<int>((max_seq_len + paging::PAGE_SIZE - 1) / paging::PAGE_SIZE);
    // Headroom for concurrent fork branches (RuntimeConfig::paged_branch_factor).
    // Device-pool VRAM now scales with the RESIDENT layer count (offloaded layers
    // cost only host RAM + 2 staging slabs), so this factor inflates the host
    // mirror cheaply rather than VRAM.
    // min_total_pages (RuntimeConfig::kv_vram_cache_pages) floors the pool so
    // the prefix cache keeps VRAM residency headroom beyond the live sequences.
    const int total_pages = std::max(max_blocks * branch_factor, min_total_pages);

    // Resolve the GPU/CPU split: clamp SIZE_MAX / oversize to "all resident".
    const int gpu_layers =
        (num_gpu_layers == static_cast<size_t>(-1) || num_gpu_layers > config.num_layers)
            ? static_cast<int>(config.num_layers)
            : static_cast<int>(num_gpu_layers);

    m_seqmgr = std::make_unique<paging::SequenceManager>(
        static_cast<int>(config.num_layers), m_num_kv_heads, m_head_dim,
        total_pages, max_blocks, gpu_layers);

    // Engine sequence 0 always exists -- the default decode stream.
    const paging::SeqId s0 = m_seqmgr->create_sequence();
    m_id_map[0] = s0;
    m_residency[s0] = SeqResidency::Resident;
}

void PagedKVManager::ensure_resident(paging::SeqId internal) {
    auto it = m_residency.find(internal);
    if (it == m_residency.end())
        throw std::runtime_error("PagedKVManager: sequence has no residency record");
    if (it->second == SeqResidency::Swapped)
        // Offloading seam: swap this sequence's pages back from the host mirror
        // here. Until swap-in lands, refuse rather than read a stale block table
        // and fault the GPU.
        throw std::runtime_error(
            "PagedKVManager: sequence " + std::to_string(internal) +
            " is swapped to host; swap-in not yet implemented");
    // Resident: nothing to do (future: bump LRU recency for the eviction policy).
}

paging::SeqId PagedKVManager::internal_id(SeqId engine_seq) const {
    auto it = m_id_map.find(engine_seq);
    if (it == m_id_map.end())
        throw std::runtime_error("PagedKVManager: unknown sequence id " +
                                 std::to_string(engine_seq));
    return it->second;
}

void PagedKVManager::prepare_decode_step(SeqId seq, int pos) {
    const paging::SeqId internal = internal_id(seq);

    // Offloading-aware: a swapped/suspended sequence must be faulted in before
    // any device page is touched (today: throws if swapped; nothing swaps yet).
    ensure_resident(internal);

    // forward(token, pos) is POSITION-ADDRESSED, exactly like the continuous
    // cache: writing slot `pos` invalidates everything from `pos` on. Callers
    // legitimately drive us with pos <= length -- the adapter's incremental KV
    // reuse re-feeds the eos token, replays the last token on an identical
    // prompt, and resets to pos 0 on divergence. Reconcile by truncating the
    // sequence to `pos` (rewind drops/decrefs the now-stale tail pages) so the
    // reserve below lands exactly at `pos`. Only a forward gap is a real error.
    const int len = m_seqmgr->length(internal);
    if (pos > len)
        throw std::runtime_error(
            "PagedKVManager: decode pos " + std::to_string(pos) +
            " skips past sequence length " + std::to_string(len) + " (KV gap)");
    if (pos < len)
        m_seqmgr->rewind(internal, pos);

    // Resolve the physical slot (CoW the target page if it is fork-shared) and
    // stage the device block table; latch the context for the layer sweep.
    const paging::AppendSlot slot = m_seqmgr->reserve_append_slot(internal);
    m_append_page = slot.page;
    m_append_slot = slot.slot;
    m_seq_len     = m_seqmgr->length(internal);          // == pos + 1
    m_block_table = m_seqmgr->device_block_table(internal);
    m_active      = internal;
}

void PagedKVManager::attention_decode(int layer_idx, int pos,
                                      float* d_Q, float* d_K, float* d_V, float* d_O) {
    if (m_active < 0)
        throw std::runtime_error(
            "PagedKVManager::attention_decode called without a preceding prepare_decode_step");

    // RoPE in place on this layer's freshly projected Q and K (rotate_half).
    launch_rope_inplace(d_Q, pos, m_num_q_heads,  m_head_dim, m_rope_theta);
    launch_rope_inplace(d_K, pos, m_num_kv_heads, m_head_dim, m_rope_theta);

    // KV-offloaded layer: fault its live pages back from the pinned host mirror
    // into its device staging slab BEFORE the append/attention read it (no-op for
    // resident layers). Same stream as the kernels below => correctly ordered.
    m_seqmgr->stage_in_layer(layer_idx, m_active);

    paging::kv_t* k_pool = m_seqmgr->layer_k_pool(layer_idx);
    paging::kv_t* v_pool = m_seqmgr->layer_v_pool(layer_idx);

    // Scatter the rotated K / raw V for this token into the resolved page slot.
    launch_paged_kv_append(d_K, d_V, k_pool, v_pool,
                           m_append_page, m_append_slot, m_num_kv_heads, m_head_dim);

    // Exact attention over the sequence prefix via the block table.
    launch_paged_flash_attention_decode(d_Q, k_pool, v_pool, d_O, m_block_table,
                                        m_seq_len, m_num_q_heads, m_num_kv_heads, m_head_dim);

    // Spill the page just written back to the host mirror so it survives the
    // staging slab being reused by the next offloaded layer (no-op if resident).
    m_seqmgr->spill_out_page(layer_idx, m_append_page);
}

// Batched (BLOCK_M-tile) prefill needs multi-token activation buffers and a
// Tensor-Core prefill kernel that do not exist yet. Prompt prefill runs through
// EnginePrefillCoordinator (engine_prefill_coordinator.h), which sweeps the
// uncached delta token-by-token over the standard decode path; when the batched
// kernels land, the coordinator switches to these two calls and nothing above
// it changes.
void PagedKVManager::prepare_prefill_step(SeqId, int, int) {
    throw std::runtime_error(
        "PagedKVManager: batched prefill kernels are not implemented; drive prompts "
        "through EnginePrefillCoordinator (per-token sweep) instead");
}

void PagedKVManager::attention_prefill(int, int, int, float*, float*, float*, float*) {
    throw std::runtime_error(
        "PagedKVManager: batched prefill kernels are not implemented; drive prompts "
        "through EnginePrefillCoordinator (per-token sweep) instead");
}

void* PagedKVManager::get_layer_k_ptr(int layer_idx) {
    return static_cast<void*>(m_seqmgr->layer_k_pool(layer_idx));
}
void* PagedKVManager::get_layer_v_ptr(int layer_idx) {
    return static_cast<void*>(m_seqmgr->layer_v_pool(layer_idx));
}

SeqId PagedKVManager::bind_external(paging::SeqId internal) {
    const SeqId engine_seq = m_next_external--;
    m_id_map.emplace(engine_seq, internal);
    m_residency[internal] = SeqResidency::Resident;
    return engine_seq;
}

void PagedKVManager::unbind_external(SeqId engine_seq) {
    auto it = m_id_map.find(engine_seq);
    if (it == m_id_map.end())
        throw std::runtime_error("PagedKVManager::unbind_external: unknown id " +
                                 std::to_string(engine_seq));
    // Invalidate the latched per-token context if it points at this sequence:
    // its block table upload must not be consumed by a later layer sweep.
    if (m_active == it->second) {
        m_active      = -1;
        m_block_table = nullptr;
    }
    m_residency.erase(it->second);
    m_id_map.erase(it);
}

void PagedKVManager::fork(SeqId parent, SeqId child) {
    if (child < 0)
        throw std::runtime_error("PagedKVManager::fork: child sequence id " +
                                 std::to_string(child) +
                                 " is in the reserved external range (must be >= 0)");
    if (m_id_map.count(child))
        throw std::runtime_error("PagedKVManager::fork: child sequence id " +
                                 std::to_string(child) + " already exists");
    const paging::SeqId child_internal = m_seqmgr->fork(internal_id(parent));
    m_id_map[child] = child_internal;
    m_residency[child_internal] = SeqResidency::Resident;
}

int PagedKVManager::truncate_sequence(SeqId seq, size_t keep_tokens) {
    const paging::SeqId internal = internal_id(seq);
    ensure_resident(internal);

    const int len = m_seqmgr->length(internal);
    if (keep_tokens > static_cast<size_t>(len))
        throw std::invalid_argument(
            "PagedKVManager::truncate_sequence: keep_tokens " +
            std::to_string(keep_tokens) + " exceeds sequence length " +
            std::to_string(len) + " (truncation never grows a sequence)");

    // The latched context (block table upload, seq_len, append slot) belongs
    // to a layer sweep over the PRE-truncation state; drop it so a stale
    // attention_decode cannot read pages this truncation is about to free.
    if (m_active == internal) {
        m_active      = -1;
        m_block_table = nullptr;
    }
    return m_seqmgr->truncate(internal, static_cast<int>(keep_tokens));
}

void PagedKVManager::rewind(SeqId seq, int target_pos) {
    const paging::SeqId internal = internal_id(seq);
    if (m_active == internal) {
        m_active      = -1;
        m_block_table = nullptr;
    }
    // rewind keeps its historical lenient contract (target >= length no-ops),
    // unlike truncate_sequence which rejects forward "truncation".
    m_seqmgr->truncate(internal, target_pos);
}

} // namespace blackwell
