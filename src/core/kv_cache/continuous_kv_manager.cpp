#include "kv_cache/continuous_kv_manager.h"
#include "kernels/rope.cuh"
#include "kernels/attention.cuh"
#include "rope_config.h"   // rope_scaling_from(ModelConfig)
#include <stdexcept>

namespace blackwell {

void ContinuousKVManager::prepare_decode_step(SeqId seq, int /*pos*/) {
    // The legacy path is single-sequence; per-token staging is folded into
    // attention_decode (it is per-layer). Guard against branch misuse.
    if (seq != 0)
        throw std::runtime_error(
            "ContinuousKVManager: single-sequence only (seq id must be 0; branching needs the paged manager)");
}

void ContinuousKVManager::prepare_prefill_step(SeqId seq, int /*start_pos*/, int /*num_tokens*/) {
    if (seq != 0)
        throw std::runtime_error("ContinuousKVManager: single-sequence only (seq id must be 0)");
}

void ContinuousKVManager::attention_decode(int layer_idx, int pos,
                                           float* d_Q, float* d_K, float* d_V, float* d_O) {
    // EXACT legacy step_attention_math sequence (no behavior change):
    //   stage KV prefix -> fused RoPE+append -> decode attention -> spill column.
    m_arena.prepare_layer_kv(layer_idx, pos);

    float* d_layer_k_cache = m_arena.get_layer_k_cache(layer_idx);
    float* d_layer_v_cache = m_arena.get_layer_v_cache(layer_idx);

    launch_fused_rope_kv_kernel(
        d_Q, d_K, d_V, d_layer_k_cache, d_layer_v_cache, pos,
        m_config.num_attention_heads, m_config.num_key_value_heads, m_config.head_dim,
        m_arena.get_max_seq_len(), m_config.rope_theta, rope_scaling_from(m_config));

    launch_attention_decoding_kernel(
        d_Q, d_layer_k_cache, d_layer_v_cache, d_O, pos,
        m_config.num_attention_heads, m_config.num_key_value_heads, m_config.head_dim,
        m_arena.get_max_seq_len());

    m_arena.commit_layer_kv(layer_idx, pos);
}

void ContinuousKVManager::attention_prefill(int, int, int, float*, float*, float*, float*) {
    throw std::runtime_error(
        "ContinuousKVManager: chunked prefill is not implemented (the legacy path is decode-only)");
}

void ContinuousKVManager::fork(SeqId, SeqId) {
    throw std::runtime_error(
        "ContinuousKVManager::fork: Copy-on-Write is unsupported by the FP32 contiguous cache; "
        "construct the engine with the paged KV manager for branching");
}

void ContinuousKVManager::rewind(SeqId, int target_pos) {
    // Single-sequence KV rewind: the resident FP32 slabs are position-addressed
    // and overwritten in place, so rolling back is purely a bookkeeping step --
    // reconcile the arena's offload high-water marks (host mirror + staging slots)
    // with target_pos so no speculative column above it leaks across the rewind.
    // (CoW branch-rewind still needs the paged manager; this is the linear rewind.)
    m_arena.truncate_kv(target_pos);
}

} // namespace blackwell
