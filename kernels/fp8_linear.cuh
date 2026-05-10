#pragma once
#include <cstddef>
#include <cstdint>

void launch_fp8_gemv_kernel(const uint8_t* d_W_fp8,
                            const float* d_X,
                            const float* d_scales,
                            float* d_Y,
                            size_t M,
                            size_t K);