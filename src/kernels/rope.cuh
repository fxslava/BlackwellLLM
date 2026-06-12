#pragma once
#include <cstddef>

// rotate_half RoPE on Q and K plus append of rotated K / raw V into the
// KV cache slot for `pos`.
//
// HARD LAUNCH CONTRACT (violations are NOT diagnosed at runtime):
//   * head_dim must be even (rotate_half pairs channel k with
//     k + head_dim/2) and head_dim/2 <= 1024 (one thread per pair,
//     blockDim.x = head_dim / 2).
//   * K_cache / V_cache layout: float[kv_heads][max_seq_len][head_dim].
void launch_fused_rope_kv_kernel(
    float* d_Q,
    float* d_K,
    const float* d_V,
    float* d_K_cache,
    float* d_V_cache,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    float rope_theta = 500000.0f);