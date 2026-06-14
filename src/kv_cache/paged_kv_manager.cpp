#include "kv_cache/paged_kv_manager.h"
#include "kernels/rope.cuh"
#include "kernels/paged_flash_attention.cuh"
#include <stdexcept>
#include <string>

namespace blackwell {

PagedKVManager::PagedKVManager(const ModelConfig& config, size_t max_seq_len)
    : m_config(config),
      m_num_q_heads(static_cast<int>(config.num_attention_heads)),
      m_num_kv_heads(static_cast<int>(config.num_key_value_heads)),
      m_head_dim(static_cast<int>(config.head_dim)),
      m_rope_theta(config.rope_theta)
{
    const int max_blocks =
        static_cast<int>((max_seq_len + paging::PAGE_SIZE - 1) / paging::PAGE_SIZE);
    // Headroom for concurrent fork branches. Pool VRAM scales with this; the
    // engine passes max_seq_len, so size it modestly and tune later if needed.
    constexpr int kBranchFactor = 4;
    const int total_pages = max_blocks * kBranchFactor;

    m_seqmgr = std::make_unique<paging::SequenceManager>(
        static_cast<int>(config.num_layers), m_num_kv_heads, m_head_dim,
        total_pages, max_blocks);

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

    // The append slot index is the sequence's current length; keep the engine's
    // pos and the manager's length in lock-step so the token lands at pos.
    if (m_seqmgr->length(internal) != pos)
        throw std::runtime_error(
            "PagedKVManager: decode pos " + std::to_string(pos) +
            " out of sync with sequence length " +
            std::to_string(m_seqmgr->length(internal)));

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

    paging::kv_t* k_pool = m_seqmgr->layer_k_pool(layer_idx);
    paging::kv_t* v_pool = m_seqmgr->layer_v_pool(layer_idx);

    // Scatter the rotated K / raw V for this token into the resolved page slot.
    launch_paged_kv_append(d_K, d_V, k_pool, v_pool,
                           m_append_page, m_append_slot, m_num_kv_heads, m_head_dim);

    // Exact attention over the sequence prefix via the block table.
    launch_paged_flash_attention_decode(d_Q, k_pool, v_pool, d_O, m_block_table,
                                        m_seq_len, m_num_q_heads, m_num_kv_heads, m_head_dim);
}

void PagedKVManager::prepare_prefill_step(SeqId, int, int) {
    throw std::runtime_error(
        "PagedKVManager: chunked prefill is not yet wired into the engine (decode path only)");
}

void PagedKVManager::attention_prefill(int, int, int, float*, float*, float*, float*) {
    throw std::runtime_error(
        "PagedKVManager: chunked prefill is not yet wired into the engine (decode path only)");
}

void* PagedKVManager::get_layer_k_ptr(int layer_idx) {
    return static_cast<void*>(m_seqmgr->layer_k_pool(layer_idx));
}
void* PagedKVManager::get_layer_v_ptr(int layer_idx) {
    return static_cast<void*>(m_seqmgr->layer_v_pool(layer_idx));
}

void PagedKVManager::fork(SeqId parent, SeqId child) {
    if (m_id_map.count(child))
        throw std::runtime_error("PagedKVManager::fork: child sequence id " +
                                 std::to_string(child) + " already exists");
    const paging::SeqId child_internal = m_seqmgr->fork(internal_id(parent));
    m_id_map[child] = child_internal;
    m_residency[child_internal] = SeqResidency::Resident;
}

void PagedKVManager::rewind(SeqId seq, int target_pos) {
    m_seqmgr->rewind(internal_id(seq), target_pos);
}

} // namespace blackwell
