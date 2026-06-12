#pragma once
#include <cstddef>

void launch_bf16_gemv_kernel(const void* d_W_bf16,
                             const float* d_X,
                             float* d_Y,
                             size_t M,
                             size_t K);

// Same warp-reduction GEMV with FP16 weights (AWQ/GPTQ checkpoints store
// lm_head.weight in half); accumulation stays in FP32.
void launch_fp16_gemv_kernel(const void* d_W_fp16,
                             const float* d_X,
                             float* d_Y,
                             size_t M,
                             size_t K);