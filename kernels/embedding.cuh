#pragma once
#include <cstddef>

void launch_embedding_kernel(const int* d_tokens, 
                             const float* d_embed_table, 
                             float* d_output, 
                             size_t seq_len, 
                             size_t hidden_dim);