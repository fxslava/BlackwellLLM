#include "rope.cuh"
#include <cuda_runtime.h>

// Ядро обработки Query голов
__global__ void rope_q_kernel(float* __restrict__ Q, 
                              int pos, 
                              size_t head_dim, 
                              float rope_theta) 
{
    // blockIdx.x - индекс головы Q
    // threadIdx.x - индекс пары элементов внутри головы
    size_t head_idx = blockIdx.x;
    size_t pair_idx = threadIdx.x;
    size_t i = pair_idx * 2;

    if (i >= head_dim) return;

    float* cur_q = Q + head_idx * head_dim;

    // Быстрый расчет частоты
    float freq = __fdividef(1.0f, powf(rope_theta, __fdividef((float)i, (float)head_dim)));
    float angle = pos * freq;

    float sin_val, cos_val;
    // sincosf - аппаратная инструкция SFU (быстрее, чем отдельные sin и cos)
    sincosf(angle, &sin_val, &cos_val);

    float q0 = cur_q[i];
    float q1 = cur_q[i + 1];

    cur_q[i]     = q0 * cos_val - q1 * sin_val;
    cur_q[i + 1] = q0 * sin_val + q1 * cos_val;
}

// Ядро обработки KV голов + кэширование
__global__ void rope_kv_append_kernel(float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ K_cache,
                                      float* __restrict__ V_cache,
                                      int pos,
                                      size_t head_dim,
                                      size_t max_seq_len,
                                      float rope_theta)
{
    size_t head_idx = blockIdx.x;
    size_t pair_idx = threadIdx.x;
    size_t i = pair_idx * 2;

    if (i >= head_dim) return;

    float* cur_k = K + head_idx * head_dim;
    const float* cur_v = V + head_idx * head_dim;

    // Смещения для глобального кэша
    float* k_slot = K_cache + (head_idx * max_seq_len + pos) * head_dim;
    float* v_slot = V_cache + (head_idx * max_seq_len + pos) * head_dim;

    float freq = __fdividef(1.0f, powf(rope_theta, __fdividef((float)i, (float)head_dim)));
    float angle = pos * freq;

    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    float k0 = cur_k[i];
    float k1 = cur_k[i + 1];

    float k0_rot = k0 * cos_val - k1 * sin_val;
    float k1_rot = k0 * sin_val + k1 * cos_val;

    // Запись обновленных значений
    cur_k[i]     = k0_rot;
    cur_k[i + 1] = k1_rot;

    k_slot[i]     = k0_rot;
    k_slot[i + 1] = k1_rot;

    v_slot[i]     = cur_v[i];
    v_slot[i + 1] = cur_v[i + 1];
}

void launch_fused_rope_kv_kernel(
    float* d_Q,
    float* d_K,
    const float* d_V,
    float* d_K_cache,
    float* d_V_cache,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    float rope_theta)
{
    // Запускаем два параллельных ядра в одном дефолтном стриме
    // Количество потоков = количеству пар (head_dim / 2)
    dim3 threads(head_dim / 2);

    rope_q_kernel<<<q_heads, threads>>>(d_Q, pos, head_dim, rope_theta);
    rope_kv_append_kernel<<<kv_heads, threads>>>(d_K, d_V, d_K_cache, d_V_cache, pos, head_dim, max_seq_len, rope_theta);
}