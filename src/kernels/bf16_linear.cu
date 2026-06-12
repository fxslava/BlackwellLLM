#include "bf16_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Распаковка половинных типов весов в FP32 для шаблонного GEMV
__device__ __forceinline__ float gemv_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float gemv_to_fp32(__half v)        { return __half2float(v); }

// ============================================================================
// ЭТАЛОННОЕ ЯДРО GEMV ДЛЯ ПОЛОВИННЫХ ТИПОВ (Warp-Level Reduction) - МАКСИМАЛЬНАЯ ТОЧНОСТЬ
// ============================================================================
template <typename WT>
__global__ void half_gemv_warp_kernel(const WT* __restrict__ W_half,
                                      const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      size_t M,
                                      size_t K)
{
    // Каждый варп (32 потока) отвечает за одну строку матрицы
    size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane   = threadIdx.x % 32;

    if (row >= M) return;

    const WT* cur_W_row = W_half + row * K;

    // Нативное накопление во float32 (использует аппаратные FMA инструкции CUDA)
    float dot = 0.0f;

    for (size_t col = lane; col < K; col += 32) {
        // Декодируем вес из BF16/FP16 во Float32
        float w_val = gemv_to_fp32(cur_W_row[col]);

        // Вектор X поступает из нормализации в полной точности Float32
        float x_val = X[col];

        // Математика идет без потерь в 32 битах
        dot += w_val * x_val;
    }

    // Быстрая каскадная редукция суммы внутри варпа
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    // Нулевой поток варпа записывает финальный ответ
    if (lane == 0) {
        // 🎯 ИСПРАВЛЕНО: Пишем чистый FP32 без искусственного урезания до BF16!
        // Теперь Softmax механизма внимания получит идеальные, неискаженные данные.
        Y[row] = dot;
    }
}

// ============================================================================
// ФУНКЦИЯ ЗАПУСКА
// ============================================================================
void launch_bf16_gemv_kernel(const void* d_W_bf16,
                             const float* d_X,
                             float* d_Y,
                             size_t M,
                             size_t K)
{
    int threads = 256;
    // Общее количество потоков рассчитывается так, чтобы выделить 32 потока на строку
    size_t total_threads = M * 32;
    size_t blocks = (total_threads + threads - 1) / threads;

    half_gemv_warp_kernel<__nv_bfloat16><<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_W_bf16),
        d_X, d_Y, M, K
    );
}

// FP16 вариант: lm_head/embed у AWQ-чекпойнтов хранятся в half
void launch_fp16_gemv_kernel(const void* d_W_fp16,
                             const float* d_X,
                             float* d_Y,
                             size_t M,
                             size_t K)
{
    int threads = 256;
    size_t total_threads = M * 32;
    size_t blocks = (total_threads + threads - 1) / threads;

    half_gemv_warp_kernel<__half><<<blocks, threads>>>(
        reinterpret_cast<const __half*>(d_W_fp16),
        d_X, d_Y, M, K
    );
}