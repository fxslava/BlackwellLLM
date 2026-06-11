#pragma once
#include <cstddef>

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