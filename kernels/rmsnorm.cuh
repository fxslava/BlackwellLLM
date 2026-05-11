#pragma once
#include <cstddef>

void launch_rmsnorm_residual_kernel(float* d_x, 
                                    float* d_residual, 
                                    const void* d_weight, 
                                    size_t seq_len, 
                                    size_t hidden_dim, 
                                    float eps = 1e-5f);

void launch_rmsnorm_kernel(const float* d_input, 
                           float* d_output, 
                           const void* d_weight, 
                           size_t seq_len, 
                           size_t hidden_dim, 
                           float eps = 1e-5f);