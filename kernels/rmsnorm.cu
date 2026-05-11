#include "rmsnorm.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define RMSNORM_BLOCK_SIZE 256

// 1. Нативная редукция суммы внутри одного варпа (32 потока)
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 2. Быстрая редукция суммы на уровне всего блока (без atomicAdd)
__inline__ __device__ float block_reduce_sum(float val, float* shared_warp_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        shared_warp_sums[warp_id] = val;
    }
    __syncthreads();

    float warp_val = (threadIdx.x < (RMSNORM_BLOCK_SIZE / 32)) ? shared_warp_sums[lane_id] : 0.0f;
    
    if (warp_id == 0) {
        warp_val = warp_reduce_sum(warp_val);
    }
    
    return warp_val; 
}

// ============================================================================
// ЯДРО 1: Fused Residual RMSNorm (С поддержкой Bfloat16 весов)
// ============================================================================
__global__ void rmsnorm_residual_bf16_weight_kernel(float* x, 
                                                    float* residual, 
                                                    const __nv_bfloat16* weight, 
                                                    size_t hidden_dim, 
                                                    float eps) {
    size_t seq_idx = blockIdx.x;
    float* cur_x = x + seq_idx * hidden_dim;
    float* cur_res = residual + seq_idx * hidden_dim;

    float thread_sum_sq = 0.0f;
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float x_val = cur_x[h];
        float res_val = cur_res[h];
        
        // Входной поток складывается с остаточным
        res_val += x_val;
        cur_res[h] = res_val;
        
        thread_sum_sq += res_val * res_val;
    }

    __shared__ float shared_warp_sums[RMSNORM_BLOCK_SIZE / 32];
    float block_sum_sq = block_reduce_sum(thread_sum_sq, shared_warp_sums);

    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) {
        s_rsqrt = rsqrtf((block_sum_sq / static_cast<float>(hidden_dim)) + eps);
    }
    __syncthreads(); 

    float rsqrt = s_rsqrt;
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        // Аппаратная распаковка Bfloat16 -> Float32 на лету
        float w_val = __bfloat162float(weight[h]);
        cur_x[h] = cur_res[h] * rsqrt * w_val;
    }
}

// ============================================================================
// ЯДРО 2: Прямая нормализация (Оптимизирована, убран atomicAdd)
// ============================================================================
__global__ void rmsnorm_bf16_weight_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           const __nv_bfloat16* __restrict__ weight,
                                           size_t hidden_dim,
                                           float eps) 
{
    size_t row_offset = blockIdx.x * hidden_dim;
    const float* cur_input = input + row_offset;
    float* cur_output = output + row_offset;

    float thread_sum_sq = 0.0f;
    for (size_t idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
        float val = cur_input[idx];
        thread_sum_sq += val * val;
    }

    // Используем ту же быструю варп-редукцию вместо медленных атомиков
    __shared__ float shared_warp_sums[RMSNORM_BLOCK_SIZE / 32];
    float block_sum_sq = block_reduce_sum(thread_sum_sq, shared_warp_sums);

    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) {
        s_rsqrt = rsqrtf((block_sum_sq / static_cast<float>(hidden_dim)) + eps);
    }
    __syncthreads();

    float rsqrt = s_rsqrt;
    for (size_t idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
        float w_val = __bfloat162float(weight[idx]);
        cur_output[idx] = cur_input[idx] * rsqrt * w_val;
    }
}

// ============================================================================
// ОБЕРТКИ ЗАПУСКА (Принимают const void* для совместимости с VRAMArena)
// ============================================================================
void launch_rmsnorm_residual_kernel(float* d_x, 
                                    float* d_residual, 
                                    const void* d_weight, 
                                    size_t seq_len, 
                                    size_t hidden_dim, 
                                    float eps) {
    dim3 blocks(seq_len);
    dim3 threads(RMSNORM_BLOCK_SIZE);
    
    const __nv_bfloat16* bf16_w = reinterpret_cast<const __nv_bfloat16*>(d_weight);
    rmsnorm_residual_bf16_weight_kernel<<<blocks, threads>>>(d_x, d_residual, bf16_w, hidden_dim, eps);
}

void launch_rmsnorm_kernel(const float* d_input, 
                           float* d_output, 
                           const void* d_weight, 
                           size_t seq_len, 
                           size_t hidden_dim, 
                           float eps) 
{
    const __nv_bfloat16* bf16_w = reinterpret_cast<const __nv_bfloat16*>(d_weight);
    rmsnorm_bf16_weight_kernel<<<seq_len, RMSNORM_BLOCK_SIZE>>>(d_input, d_output, bf16_w, hidden_dim, eps);
}