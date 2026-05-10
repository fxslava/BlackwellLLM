#include "rmsnorm.cuh"
#include <cuda_runtime.h>

// Фиксированный размер блока для сатурации SM
#define RMSNORM_BLOCK_SIZE 256

// 1. Нативная редукция суммы внутри одного варпа (32 потока)
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Потоки обмениваются значениями в регистрах без обращения к памяти
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 2. Редукция суммы на уровне всего блока (256 потоков = 8 варпов)
__inline__ __device__ float block_reduce_sum(float val, float* shared_warp_sums) {
    int warp_id = threadIdx.x >> 5; // Аналог threadIdx.x / 32
    int lane_id = threadIdx.x & 31; // Аналог threadIdx.x % 32

    // Шаг 1: Считаем сумму внутри каждого варпа
    val = warp_reduce_sum(val);

    // Шаг 2: Нулевой поток каждого варпа пишет свой итог в Shared Memory
    if (lane_id == 0) {
        shared_warp_sums[warp_id] = val;
    }
    __syncthreads(); // Ждем, пока все 8 варпов запишут данные

    // Шаг 3: Первый варп (warp_id == 0) считывает 8 промежуточных сумм и сворачивает их
    // Остальные потоки варпа (индексы 8..31) берут 0.0f, чтобы не влиять на сумму
    float warp_val = (threadIdx.x < (RMSNORM_BLOCK_SIZE / 32)) ? shared_warp_sums[lane_id] : 0.0f;
    
    if (warp_id == 0) {
        warp_val = warp_reduce_sum(warp_val);
    }
    
    return warp_val; // Итоговая сумма всего блока окажется в нулевом потоке (threadIdx.x == 0)
}

__global__ void rmsnorm_residual_kernel(float* x, 
                                        float* residual, 
                                        const float* weight, 
                                        size_t hidden_dim, 
                                        float eps) {
    size_t seq_idx = blockIdx.x;
    float* cur_x = x + seq_idx * hidden_dim;
    float* cur_res = residual + seq_idx * hidden_dim;

    // Аккумулируем сумму квадратов локально в регистрах потока
    float thread_sum_sq = 0.0f;
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float x_val = cur_x[h];
        float res_val = cur_res[h];
        
        // Fused Residual Add: складываем и сразу сохраняем в глобальную память
        res_val += x_val;
        cur_res[h] = res_val;
        
        thread_sum_sq += res_val * res_val;
    }

    // Буфер в Shared Memory для хранения сумм 8 варпов (32 байта всего)
    __shared__ float shared_warp_sums[RMSNORM_BLOCK_SIZE / 32];
    
    // Получаем полную сумму квадратов по всему токену
    float block_sum_sq = block_reduce_sum(thread_sum_sq, shared_warp_sums);

    // Нулевой поток считает 1/sqrt через быстрый аппаратный блок SFU
    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) {
        s_rsqrt = rsqrtf((block_sum_sq / static_cast<float>(hidden_dim)) + eps);
    }
    __syncthreads(); // Гарантируем, что s_rsqrt доступен всему блоку

    // Применяем нормализацию и масштабирование весами
    float rsqrt = s_rsqrt;
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        cur_x[h] = cur_res[h] * rsqrt * weight[h];
    }
}

void launch_rmsnorm_residual_kernel(float* d_x, 
                                    float* d_residual, 
                                    const float* d_weight, 
                                    size_t seq_len, 
                                    size_t hidden_dim, 
                                    float eps) {
    dim3 blocks(seq_len);
    dim3 threads(RMSNORM_BLOCK_SIZE);

    rmsnorm_residual_kernel<<<blocks, threads>>>(d_x, d_residual, d_weight, hidden_dim, eps);
}