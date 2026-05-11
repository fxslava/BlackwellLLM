#pragma once
#include <cstddef>
#include <cstdint>

void launch_fp8_gemv_kernel(const void* d_W_fp8,
                            const float* d_X,
                            const void* d_weight_scales,
                            const void* d_input_scale,
                            float* d_Y,
                            size_t M,
                            size_t K,
                            int scale_stride);

void launch_fp8_gemv_residual_kernel(const void* d_W_fp8,
                                     const float* d_X,
                                     const void* d_weight_scales,
                                     const void* d_input_scale,
                                     float* d_Y_accum,
                                     size_t M,
                                     size_t K,
                                     int scale_stride);