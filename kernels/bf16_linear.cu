#include "bf16_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void bf16_gemv_warp_kernel(const __nv_bfloat16* __restrict__ W_bf16,
                                      const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      size_t M,
                                      size_t K)
{
    size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane   = threadIdx.x % 32;

    if (row >= M) return;

    const __nv_bfloat16* cur_W_row = W_bf16 + row * K;

    float dot = 0.0f;
    for (size_t col = lane; col < K; col += 32) {
        float w_val = __bfloat162float(cur_W_row[col]);
        // Входной буфер X уже идеально согласован с сеткой Bfloat16 на этапе нормализации
        dot += w_val * X[col];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    if (lane == 0) {
        Y[row] = dot;
    }
}

void launch_bf16_gemv_kernel(const void* d_W_bf16,
                             const float* d_X,
                             float* d_Y,
                             size_t M,
                             size_t K)
{
    int threads = 256;
    size_t total_threads = M * 32; 
    size_t blocks = (total_threads + threads - 1) / threads;

    bf16_gemv_warp_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_W_bf16),
        d_X, d_Y, M, K
    );
}