#pragma once
#include <cstddef>

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