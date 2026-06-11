#include "bf16_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// ЭТАЛОННОЕ ЯДРО GEMV ДЛЯ BFLOAT16 (Warp-Level Reduction) - МАКСИМАЛЬНАЯ ТОЧНОСТЬ
// ============================================================================
__global__ void bf16_gemv_warp_kernel(const __nv_bfloat16* __restrict__ W_bf16,
                                      const float* __restrict__ X,
                                      float* __restrict__ Y,
                                      size_t M,
                                      size_t K)
{
    // Каждый варп (32 потока) отвечает за одну строку матрицы
    size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane   = threadIdx.x % 32;

    if (row >= M) return;

    const __nv_bfloat16* cur_W_row = W_bf16 + row * K;

    // Нативное накопление во float32 (использует аппаратные FMA инструкции CUDA)
    float dot = 0.0f;
    
    for (size_t col = lane; col < K; col += 32) {
        // Декодируем вес из Bfloat16 во Float32
        float w_val = __bfloat162float(cur_W_row[col]);
        
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

    bf16_gemv_warp_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_W_bf16),
        d_X, d_Y, M, K
    );
}