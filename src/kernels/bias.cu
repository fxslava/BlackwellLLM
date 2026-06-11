#include "bias.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cassert>

#define BIAS_BLOCK_SIZE 256

// 🎯 Декодирование bias из формата чекпоинта: один fp32-результат без
// промежуточных округлений (bf16-parity: сложение выполняется один раз в fp32)
__inline__ __device__ float bias_to_fp32(const __nv_bfloat16* bias, size_t i) {
    return __bfloat162float(bias[i]);
}

__inline__ __device__ float bias_to_fp32(const __half* bias, size_t i) {
    return __half2float(bias[i]);
}

template <typename BiasT>
__global__ void bias_add_kernel(float* __restrict__ Y,
                                const BiasT* __restrict__ bias,
                                size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Y[i] += bias_to_fp32(bias, i);
}

template <typename BiasT>
__global__ void fused_qkv_bias_kernel(float* __restrict__ Q,
                                      float* __restrict__ K,
                                      float* __restrict__ V,
                                      const BiasT* __restrict__ bias_q,
                                      const BiasT* __restrict__ bias_k,
                                      const BiasT* __restrict__ bias_v,
                                      size_t q_dim,
                                      size_t kv_dim)
{
    size_t N_total = q_dim + 2 * kv_dim;
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_total) return;

    if (t < q_dim) {
        Q[t] += bias_to_fp32(bias_q, t);
    } else if (t < q_dim + kv_dim) {
        size_t i = t - q_dim;
        K[i] += bias_to_fp32(bias_k, i);
    } else {
        size_t i = t - q_dim - kv_dim;
        V[i] += bias_to_fp32(bias_v, i);
    }
}

void launch_bias_add_kernel(float* d_Y,
                            const void* d_bias,
                            size_t N,
                            BiasDType dtype)
{
    if (d_bias == nullptr || N == 0) return;

    unsigned int blocks = (N + BIAS_BLOCK_SIZE - 1) / BIAS_BLOCK_SIZE;

    dim3 grid(blocks);
    dim3 threads(BIAS_BLOCK_SIZE);

    switch (dtype) {
    case BiasDType::BF16:
        bias_add_kernel<<<grid, threads>>>(
            d_Y, static_cast<const __nv_bfloat16*>(d_bias), N);
        break;
    case BiasDType::FP16:
        bias_add_kernel<<<grid, threads>>>(
            d_Y, static_cast<const __half*>(d_bias), N);
        break;
    }
}

void launch_fused_qkv_bias_kernel(float* d_Q,
                                  float* d_K,
                                  float* d_V,
                                  const void* d_bias_q,
                                  const void* d_bias_k,
                                  const void* d_bias_v,
                                  size_t q_dim,
                                  size_t kv_dim,
                                  BiasDType dtype)
{
    if (d_bias_q == nullptr && d_bias_k == nullptr && d_bias_v == nullptr) return;

    // Qwen2: либо все три bias присутствуют, либо ни одного
    assert(d_bias_q != nullptr && d_bias_k != nullptr && d_bias_v != nullptr);

    size_t N_total = q_dim + 2 * kv_dim;
    if (N_total == 0) return;

    unsigned int blocks = (N_total + BIAS_BLOCK_SIZE - 1) / BIAS_BLOCK_SIZE;

    dim3 grid(blocks);
    dim3 threads(BIAS_BLOCK_SIZE);

    switch (dtype) {
    case BiasDType::BF16:
        fused_qkv_bias_kernel<<<grid, threads>>>(
            d_Q, d_K, d_V,
            static_cast<const __nv_bfloat16*>(d_bias_q),
            static_cast<const __nv_bfloat16*>(d_bias_k),
            static_cast<const __nv_bfloat16*>(d_bias_v),
            q_dim, kv_dim);
        break;
    case BiasDType::FP16:
        fused_qkv_bias_kernel<<<grid, threads>>>(
            d_Q, d_K, d_V,
            static_cast<const __half*>(d_bias_q),
            static_cast<const __half*>(d_bias_k),
            static_cast<const __half*>(d_bias_v),
            q_dim, kv_dim);
        break;
    }
}
