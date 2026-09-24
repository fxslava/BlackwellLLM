#include "rmsnorm.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Распаковка/защелкивание половинных типов для шаблонного RMSNorm
__device__ __forceinline__ float rms_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rms_to_fp32(__half v)        { return __half2float(v); }
__device__ __forceinline__ float rms_latch(float v, __nv_bfloat16*) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float rms_latch(float v, __half*)        { return __half2float(__float2half(v)); }

#define RMSNORM_BLOCK_SIZE 256

// Нативная редукция суммы внутри варпа в double
__inline__ __device__ double warp_reduce_sum_double(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Редукция суммы по всему блоку в double
__inline__ __device__ double block_reduce_sum_double(double val, double* shared_warp_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = warp_reduce_sum_double(val);

    if (lane_id == 0) {
        shared_warp_sums[warp_id] = val;
    }
    __syncthreads();

    double warp_val = (threadIdx.x < (RMSNORM_BLOCK_SIZE / 32)) ? shared_warp_sums[lane_id] : 0.0;
    if (warp_id == 0) {
        warp_val = warp_reduce_sum_double(warp_val);
    }
    return warp_val; 
}

__global__ void rmsnorm_residual_bf16_weight_kernel(float* x, 
                                                    float* residual, 
                                                    const __nv_bfloat16* weight, 
                                                    size_t hidden_dim, 
                                                    float eps) {
    size_t seq_idx = blockIdx.x;
    float* cur_x = x + seq_idx * hidden_dim;
    float* cur_res = residual + seq_idx * hidden_dim;

    // 🎯 Используем double для защиты от поглощения мантиссы на выбросах
    double thread_sum_sq = 0.0;
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float x_val = cur_x[h];
        float res_val = cur_res[h];
        
        res_val += x_val;
        // Защелкиваем остаточный поток в сетку Bfloat16
        res_val = __bfloat162float(__float2bfloat16(res_val));
        
        cur_res[h] = res_val;
        double d_res = static_cast<double>(res_val);
        thread_sum_sq += d_res * d_res;
    }

    __shared__ double shared_warp_sums[RMSNORM_BLOCK_SIZE / 32];
    double block_sum_sq = block_reduce_sum_double(thread_sum_sq, shared_warp_sums);

    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) {
        double variance = block_sum_sq / static_cast<double>(hidden_dim);
        s_rsqrt = static_cast<float>(1.0 / sqrt(variance + static_cast<double>(eps)));
    }
    __syncthreads(); 

    float rsqrt = s_rsqrt;
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        // 🎯 Золотая логика HF: сначала каст нормализованного входа к BF16, затем умножение на вес
        float norm_val = cur_res[h] * rsqrt;
        __nv_bfloat16 norm_bf16 = __float2bfloat16(norm_val);
        
        float mul_val = __bfloat162float(norm_bf16) * __bfloat162float(weight[h]);
        cur_x[h] = __bfloat162float(__float2bfloat16(mul_val));
    }
}

// Шаблон по типу весов: семантика HF (hidden_states.to(input_dtype)) требует,
// чтобы и нормализованный вход, и результат защелкивались в сетку ТОГО ЖЕ
// половинного типа, в котором хранится чекпойнт (BF16 у Llama, FP16 у AWQ).
// ACCUM == true turns the epilogue into `output += normalized * w` (fp32, exactly
// like the GEMV kernels' residual epilogue). That is the GLM-4-0414 sandwich norm:
// the sub-layer output is normalized and then JOINS the residual stream, so the
// projection kernels' fused residual-accumulate cannot be used for those layers.
template <typename WT, bool ACCUM = false>
__global__ void rmsnorm_half_weight_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           const WT* __restrict__ weight,
                                           size_t hidden_dim,
                                           float eps,
                                           bool add_unit_offset)
{
    size_t row_offset = blockIdx.x * hidden_dim;
    const float* cur_input = input + row_offset;
    float* cur_output = output + row_offset;

    // 🎯 Редукция суммы квадратов строго в double
    double thread_sum_sq = 0.0;
    for (size_t idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
        double val = static_cast<double>(cur_input[idx]);
        thread_sum_sq += val * val;
    }

    __shared__ double shared_warp_sums[RMSNORM_BLOCK_SIZE / 32];
    double block_sum_sq = block_reduce_sum_double(thread_sum_sq, shared_warp_sums);

    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) {
        double variance = block_sum_sq / static_cast<double>(hidden_dim);
        s_rsqrt = static_cast<float>(1.0 / sqrt(variance + static_cast<double>(eps)));
    }
    __syncthreads();

    float rsqrt = s_rsqrt;
    for (size_t idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
        float norm_val = cur_input[idx] * rsqrt;
        float out_val;
        if (add_unit_offset) {
            // Qwen3_5RMSNorm: (x * (1 + w)).to(dtype) -- multiply in fp32, latch once
            // at the end (zero-centered weight). See modeling_qwen3_5.Qwen3_5RMSNorm.
            float mul_val = norm_val * (1.0f + rms_to_fp32(weight[idx]));
            out_val = rms_latch(mul_val, static_cast<WT*>(nullptr));
        } else {
            // Llama / Qwen2.5 semantics: x.to(dtype) * w (latch the normalized input
            // BEFORE multiplying the weight).
            float norm_latched = rms_latch(norm_val, static_cast<WT*>(nullptr));
            float mul_val = norm_latched * rms_to_fp32(weight[idx]);
            out_val = rms_latch(mul_val, static_cast<WT*>(nullptr));
        }
        if (ACCUM) cur_output[idx] += out_val;   // fp32 residual join (see above)
        else       cur_output[idx]  = out_val;
    }
}

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
                           float eps,
                           bool add_unit_offset)
{
    const __nv_bfloat16* bf16_w = reinterpret_cast<const __nv_bfloat16*>(d_weight);
    rmsnorm_half_weight_kernel<__nv_bfloat16><<<seq_len, RMSNORM_BLOCK_SIZE>>>(d_input, d_output, bf16_w, hidden_dim, eps, add_unit_offset);
}

void launch_rmsnorm_fp16_kernel(const float* d_input,
                                float* d_output,
                                const void* d_weight,
                                size_t seq_len,
                                size_t hidden_dim,
                                float eps,
                                bool add_unit_offset)
{
    const __half* fp16_w = reinterpret_cast<const __half*>(d_weight);
    rmsnorm_half_weight_kernel<__half><<<seq_len, RMSNORM_BLOCK_SIZE>>>(d_input, d_output, fp16_w, hidden_dim, eps, add_unit_offset);
}

void launch_rmsnorm_accum_kernel(const float* d_input,
                                 float* d_accum,
                                 const void* d_weight,
                                 size_t seq_len,
                                 size_t hidden_dim,
                                 float eps,
                                 bool add_unit_offset)
{
    const __nv_bfloat16* bf16_w = reinterpret_cast<const __nv_bfloat16*>(d_weight);
    rmsnorm_half_weight_kernel<__nv_bfloat16, /*ACCUM=*/true><<<seq_len, RMSNORM_BLOCK_SIZE>>>(
        d_input, d_accum, bf16_w, hidden_dim, eps, add_unit_offset);
}

void launch_rmsnorm_accum_fp16_kernel(const float* d_input,
                                      float* d_accum,
                                      const void* d_weight,
                                      size_t seq_len,
                                      size_t hidden_dim,
                                      float eps,
                                      bool add_unit_offset)
{
    const __half* fp16_w = reinterpret_cast<const __half*>(d_weight);
    rmsnorm_half_weight_kernel<__half, /*ACCUM=*/true><<<seq_len, RMSNORM_BLOCK_SIZE>>>(
        d_input, d_accum, fp16_w, hidden_dim, eps, add_unit_offset);
}