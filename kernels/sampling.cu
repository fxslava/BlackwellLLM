#include "sampling.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

#define ARGMAX_BLOCK_SIZE 512

// Нативная редукция максимума и его индекса внутри одного варпа (32 потока)
__inline__ __device__ void warp_reduce_argmax(float& max_val, int& max_idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Получаем значения из соседнего потока
        float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        int other_idx   = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
        
        // Обновляем максимум
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }
}

__global__ void argmax_kernel(const float* __restrict__ logits, 
                              int* __restrict__ out_token_id, 
                              size_t vocab_size) 
{
    int tid = threadIdx.x;
    
    // Локальные переменные потока
    float thread_max_val = -CUDART_INF_F;
    int thread_max_idx   = 0;

    // Каждый поток последовательно обрабатывает свой срез словаря (Grid-stride)
    for (size_t i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = static_cast<int>(i);
        }
    }

    // Буферы в Shared Memory для сбора итогов с 16 варпов
    __shared__ float s_max_vals[ARGMAX_BLOCK_SIZE / 32];
    __shared__ int   s_max_idxs[ARGMAX_BLOCK_SIZE / 32];

    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    // Шаг 1: Сворачиваем итоги внутри каждого варпа
    warp_reduce_argmax(thread_max_val, thread_max_idx);

    // Нулевой поток каждого варпа пишет результат в Shared Memory
    if (lane_id == 0) {
        s_max_vals[warp_id] = thread_max_val;
        s_max_idxs[warp_id] = thread_max_idx;
    }
    __syncthreads();

    // Шаг 2: Первый варп (warp_id == 0) сворачивает 16 промежуточных максимумов
    if (warp_id == 0) {
        // Потоки с 0 по 15 читают данные из Shared Memory, остальные берут -INF
        float warp_max_val = (lane_id < (ARGMAX_BLOCK_SIZE / 32)) ? s_max_vals[lane_id] : -CUDART_INF_F;
        int   warp_max_idx = (lane_id < (ARGMAX_BLOCK_SIZE / 32)) ? s_max_idxs[lane_id] : 0;

        warp_reduce_argmax(warp_max_val, warp_max_idx);

        // Итоговый абсолютный максимум оказывается в нулевом потоке блока
        if (tid == 0) {
            *out_token_id = warp_max_idx;
        }
    }
}

void launch_argmax_kernel(const float* d_logits, int* d_out_token_id, size_t vocab_size) {
    // Запускаем ровно 1 блок для мгновенной обработки без лишних накладных расходов
    argmax_kernel<<<1, ARGMAX_BLOCK_SIZE>>>(d_logits, d_out_token_id, vocab_size);
}