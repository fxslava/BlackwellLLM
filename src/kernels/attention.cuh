#pragma once
#include <cstddef>

// Single-token decode attention (online softmax over positions 0..pos).
//
// HARD LAUNCH CONTRACT (violations are NOT diagnosed at runtime):
//   * head_dim <= 128 (the fixed block size). Each thread owns one channel
//     of the head; channels >= blockDim.x would be SILENTLY DROPPED from
//     both the Q*K dot product and the output vector.
//   * q_heads % kv_heads == 0 (GQA ratio is derived as gridDim.x / kv_heads).
//   * K_cache / V_cache layout: float[kv_heads][max_seq_len][head_dim].
void launch_attention_decoding_kernel(
    const float* d_Q,
    const float* d_K_cache,
    const float* d_V_cache,
    float* d_O,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len);