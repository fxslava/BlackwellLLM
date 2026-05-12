#pragma once
#include <cstddef>
#include <cstdint>
#include <cublasLt.h>

// Ядро предварительного расчета динамического скейла токена
void launch_quantize_per_token_kernel(const float* d_X, float* d_token_scale, size_t K);

// Унифицированный запуск FP8 GEMV/GEMM через аппаратный движок cuBLASLt
void launch_fp8_linear_cublaslt(cublasLtHandle_t cublaslt_handle,
                                const void* d_W_fp8,
                                const float* d_X,
                                void* d_X_fp8,
                                float* d_Y_raw,
                                const void* d_weight_scales,
                                const void* d_input_scale,
                                const float* d_token_scale,
                                float* d_Y,
                                size_t M,
                                size_t K,
                                int scale_stride,
                                bool accumulate);