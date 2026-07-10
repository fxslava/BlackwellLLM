#include "rope.cuh"
#include <cuda_runtime.h>

// ============================================================================
// 1. ВРАЩЕНИЕ QUERY ГОЛОВ (Стандарт Hugging Face / rotate_half)
// ============================================================================
__global__ void rope_q_kernel(float* __restrict__ Q, 
                              int pos, 
                              size_t head_dim, 
                              float rope_theta) 
{
    size_t head_idx = blockIdx.x;
    size_t k = threadIdx.x; // Индекс канала от 0 до (head_dim / 2 - 1)

    if (k >= head_dim / 2) return;

    float* cur_q = Q + head_idx * head_dim;

    // Базовая частота для пары k (эквивалентно исходному шагу 2 * k)
    float freq = __fdividef(1.0f, powf(rope_theta, __fdividef(static_cast<float>(2 * k), static_cast<float>(head_dim))));
    float angle = pos * freq;

    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    // 🎯 Золотой стандарт HF: считываем каналы из первой и второй половины размерности
    float q0 = cur_q[k];
    float q1 = cur_q[k + head_dim / 2];

    // Применяем вращение rotate_half
    cur_q[k]                = q0 * cos_val - q1 * sin_val;
    cur_q[k + head_dim / 2] = q0 * sin_val + q1 * cos_val;
}

// ============================================================================
// 2. ВРАЩЕНИЕ И КЭШИРОВАННАЯ ЗАПИСЬ KV ГОЛОВ (Стандарт Hugging Face)
// ============================================================================
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
    size_t k = threadIdx.x;

    if (k >= head_dim / 2) return;

    float* cur_k = K + head_idx * head_dim;
    const float* cur_v = V + head_idx * head_dim;

    // Линейные смещения для слотов глобального кэша
    float* k_slot = K_cache + (head_idx * max_seq_len + pos) * head_dim;
    float* v_slot = V_cache + (head_idx * max_seq_len + pos) * head_dim;

    float freq = __fdividef(1.0f, powf(rope_theta, __fdividef(static_cast<float>(2 * k), static_cast<float>(head_dim))));
    float angle = pos * freq;

    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    float k0 = cur_k[k];
    float k1 = cur_k[k + head_dim / 2];

    float k0_rot = k0 * cos_val - k1 * sin_val;
    float k1_rot = k0 * sin_val + k1 * cos_val;

    // Записываем результат обратно в буфер операнда
    cur_k[k]                = k0_rot;
    cur_k[k + head_dim / 2] = k1_rot;

    // Дублируем в линейный слот KV-кэша
    k_slot[k]                = k0_rot;
    k_slot[k + head_dim / 2] = k1_rot;

    v_slot[k]                = cur_v[k];
    v_slot[k + head_dim / 2] = cur_v[k + head_dim / 2];
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
    dim3 threads(head_dim / 2);

    rope_q_kernel<<<q_heads, threads>>>(d_Q, pos, head_dim, rope_theta);
    rope_kv_append_kernel<<<kv_heads, threads>>>(d_K, d_V, d_K_cache, d_V_cache, pos, head_dim, max_seq_len, rope_theta);
}

// In-place rotate_half over a [num_heads, head_dim] buffer (no cache write).
// rope_q_kernel is layout-agnostic -- it rotates any such buffer -- so it serves
// for both Q and K on the paged path.
void launch_rope_inplace(
    float* d_X,
    int pos,
    size_t num_heads,
    size_t head_dim,
    float rope_theta)
{
    dim3 threads(head_dim / 2);
    rope_q_kernel<<<num_heads, threads>>>(d_X, pos, head_dim, rope_theta);
}

// Batched rotate_half over [num_tokens, num_heads, head_dim]: token t (row
// blockIdx.y) rotates for position start_pos + t. Same rotate_half math as
// rope_q_kernel, one block per (head, token).
__global__ void rope_q_batched_kernel(float* __restrict__ X,
                                      int start_pos,
                                      size_t num_heads,
                                      size_t head_dim,
                                      float rope_theta)
{
    const size_t head_idx = blockIdx.x;
    const size_t t        = blockIdx.y;
    const size_t k        = threadIdx.x;
    if (k >= head_dim / 2) return;

    const int pos = start_pos + (int)t;
    float* cur = X + (t * num_heads + head_idx) * head_dim;

    float freq = __fdividef(1.0f, powf(rope_theta,
                    __fdividef(static_cast<float>(2 * k), static_cast<float>(head_dim))));
    float angle = pos * freq;
    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    float x0 = cur[k];
    float x1 = cur[k + head_dim / 2];
    cur[k]                = x0 * cos_val - x1 * sin_val;
    cur[k + head_dim / 2] = x0 * sin_val + x1 * cos_val;
}

void launch_rope_inplace_batched(
    float* d_X,
    int start_pos,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta)
{
    dim3 grid((unsigned)num_heads, (unsigned)num_tokens);
    dim3 threads((unsigned)(head_dim / 2));
    rope_q_batched_kernel<<<grid, threads>>>(d_X, start_pos, num_heads, head_dim, rope_theta);
}