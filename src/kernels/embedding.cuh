#pragma once
#include <cstddef>

void launch_embedding_kernel(const int* d_tokens, 
                             const float* d_embed_table, 
                             float* d_output, 
                             size_t seq_len, 
                             size_t hidden_dim);

void launch_bf16_embedding_kernel(
    const int* d_tokens,
    const void* d_embed_table,
    float* d_output,
    size_t seq_len,
    size_t hidden_dim
);