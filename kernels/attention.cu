#include "attention.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

#define ATTN_BLOCK_SIZE 128

// Нативная редукция суммы внутри варпа
__inline__ __device__ float attn_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Нативная редукция суммы по всему блоку (128 потоков = 4 варпа)
// Возвращает итоговую сумму всем потокам блока (broadcast)
__inline__ __device__ float attn_block_reduce_sum(float val, float* shared_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = attn_warp_reduce_sum(val);

    if (lane_id == 0) {
        shared_sums[warp_id] = val;
    }
    __syncthreads();

    float warp_val = (threadIdx.x < (ATTN_BLOCK_SIZE / 32)) ? shared_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        warp_val = attn_warp_reduce_sum(warp_val);
    }

    // Броадкастим результат из нулевого потока на весь блок
    __shared__ float s_final_sum;
    if (threadIdx.x == 0) {
        s_final_sum = warp_val;
    }
    __syncthreads();

    return s_final_sum;
}

__global__ void attention_decoding_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    int pos,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    float scale) 
{
    // blockIdx.x - отвечает за конкретную голову Query (qh)
    // threadIdx.x - отвечает за конкретный элемент размерности (0 .. 127)
    int qh = blockIdx.x;
    int tid = threadIdx.x;

    int gqa_ratio = gridDim.x / kv_heads;
    int kvh = qh / gqa_ratio;

    // Загружаем наш элемент Query в быстрый регистр
    float q_val = (tid < head_dim) ? Q[qh * head_dim + tid] : 0.0f;

    // Переменные для Online Softmax
    float m = -CUDART_INF_F; // Текущий максимум
    float l = 0.0f;          // Текущая сумма экспонент
    float acc = 0.0f;        // Аккумулятор для выходного вектора O[tid]

    __shared__ float shared_sums[ATTN_BLOCK_SIZE / 32];

    // Итерируемся по всему кэшу от 0 до текущей позиции токена
    for (int t = 0; t <= pos; ++t) {
        // 1. Коалесцированное чтение ключа K
        size_t k_offset = (kvh * max_seq_len + t) * head_dim + tid;
        float k_val = (tid < head_dim) ? K_cache[k_offset] : 0.0f;

        // Считаем локальное произведение
        float dot_elem = q_val * k_val;

        // Сворачиваем сумму по всему блоку, чтобы получить скаляр Q * K^T
        float score = attn_block_reduce_sum(dot_elem, shared_sums) * scale;

        // 2. Магия Online Softmax
        float m_prev = m;
        if (score > m) {
            m = score;
        }

        // Инструкция expf аппаратно ускоряется блоками SFU
        float exp_score = expf(score - m);
        float exp_prev  = expf(m_prev - m);

        // Обновляем сумму экспонент с учетом сдвига максимума
        l = l * exp_prev + exp_score;

        // 3. Коалесцированное чтение значения V и пересчет аккумулятора
        size_t v_offset = (kvh * max_seq_len + t) * head_dim + tid;
        float v_val = (tid < head_dim) ? V_cache[v_offset] : 0.0f;

        acc = acc * exp_prev + exp_score * v_val;
    }

    // Финальное деление на полную сумму и запись ответа
    if (tid < head_dim) {
        O[qh * head_dim + tid] = acc / l;
    }
}

void launch_attention_decoding_kernel(
    const float* d_Q,
    const float* d_K_cache,
    const float* d_V_cache,
    float* d_O,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len) 
{
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Запускаем сетку: 1 блок = 1 голова Q
    dim3 blocks(q_heads);
    dim3 threads(ATTN_BLOCK_SIZE);

    attention_decoding_kernel<<<blocks, threads>>>(
        d_Q, d_K_cache, d_V_cache, d_O, pos, kv_heads, head_dim, max_seq_len, scale
    );
}