#pragma once
#include <cstddef>
#include <cstdint>

// 🎯 Объявление ядра предварительного расчета динамического скейла токена
void launch_quantize_per_token_kernel(const float* d_X, float* d_token_scale, size_t K);

void launch_fp8_gemv_kernel(const void* d_W_fp8,
                            const float* d_X,
                            const void* d_weight_scales,
                            const void* d_input_scale,
                            const float* d_token_scale,
                            float* d_Y,
                            size_t M,
                            size_t K,
                            int scale_stride);

void launch_fp8_gemv_residual_kernel(const void* d_W_fp8,
                                     const float* d_X,
                                     const void* d_weight_scales,
                                     const void* d_input_scale,
                                     const float* d_token_scale,
                                     float* d_Y_accum,
                                     size_t M,
                                     size_t K,
                                     int scale_stride);