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

    // 🎯 Используем double для поглощения произведений каналов-выбросов без потери точности
    double dot = 0.0;
    for (size_t col = lane; col < K; col += 32) {
        double w_val = static_cast<double>(__bfloat162float(cur_W_row[col]));
        double x_val = static_cast<double>(X[col]);
        dot += w_val * x_val;
    }

    // Каскадная редукция варпа поддерживается аппаратно для double
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    if (lane == 0) {
        // 🎯 Золотая семантика: выход линейного слоя BF16 в PyTorch является тензором Bfloat16.
        // Защелкиваем итоговый логит в дискретную сетку
        float final_logit = static_cast<float>(dot);
        Y[row] = __bfloat162float(__float2bfloat16(final_logit));
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