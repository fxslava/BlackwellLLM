#include "fp8_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define GEMV_BLOCK_SIZE 128

__inline__ __device__ float gemv_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__inline__ __device__ float gemv_block_reduce_max(float val, float* shared_warp_maxes) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = gemv_warp_reduce_max(val);

    if (lane_id == 0) shared_warp_maxes[warp_id] = val;
    __syncthreads();

    float warp_val = (threadIdx.x < (GEMV_BLOCK_SIZE / 32)) ? shared_warp_maxes[lane_id] : 0.0f;
    if (warp_id == 0) warp_val = gemv_warp_reduce_max(warp_val);
    return warp_val;
}

__inline__ __device__ float gemv_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ float gemv_block_reduce_sum(float val, float* shared_warp_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = gemv_warp_reduce_sum(val);

    if (lane_id == 0) shared_warp_sums[warp_id] = val;
    __syncthreads();

    float warp_val = (threadIdx.x < (GEMV_BLOCK_SIZE / 32)) ? shared_warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) warp_val = gemv_warp_reduce_sum(warp_val);
    return warp_val;
}

__inline__ __device__ float device_unpack_fp8_e4m3(uint8_t byte_val) {
    if ((byte_val & 0x7F) == 0) return 0.0f;
    int sign = (byte_val & 0x80) ? -1 : 1;
    int exp  = (byte_val & 0x78) >> 3;
    int mant = byte_val & 0x07;

    if (exp == 0) return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
    return sign * ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

__inline__ __device__ float dynamic_quantize_e4m3(float x, float scale, float inv_scale) {
    float scaled = x * scale;
    if (fabsf(scaled) < 1e-4f) return 0.0f;

    scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);
    int exp;
    float mantissa = frexpf(scaled, &exp);
    mantissa = roundf(mantissa * 16.0f) / 16.0f;
    return ldexpf(mantissa, exp) * inv_scale;
}

__global__ void fp8_gemv_kernel(const uint8_t* __restrict__ W_fp8,
                                const float* __restrict__ X,
                                const __nv_bfloat16* __restrict__ weight_scales,
                                const __nv_bfloat16* __restrict__ input_scale,
                                float* __restrict__ Y,
                                size_t K,
                                int scale_stride) 
{
    size_t row_idx = blockIdx.x;
    const uint8_t* cur_W_row = W_fp8 + row_idx * K;
    
    __shared__ float shared_warp_acc[GEMV_BLOCK_SIZE / 32];
    __shared__ float s_dyn_scale;
    __shared__ float s_dyn_inv_scale;

    float thread_max = 0.0f;
    for (size_t col = threadIdx.x; col < K; col += blockDim.x) {
        thread_max = fmaxf(thread_max, fabsf(X[col]));
    }
    float block_max = gemv_block_reduce_max(thread_max, shared_warp_acc);

    if (threadIdx.x == 0) {
        if (input_scale == nullptr) {
            float max_val = fmaxf(block_max, 1e-5f);
            s_dyn_scale = 448.0f / max_val;
            s_dyn_inv_scale = max_val / 448.0f;
        } else {
            s_dyn_scale = 1.0f;
            s_dyn_inv_scale = 1.0f;
        }
    }
    __syncthreads();

    float d_scale = s_dyn_scale;
    float d_inv_scale = s_dyn_inv_scale;
    bool apply_quant = (input_scale == nullptr);

    float thread_acc = 0.0f;
    size_t num_vec_elems = K / 16;
    const uint4* W_vec = reinterpret_cast<const uint4*>(cur_W_row);

    for (size_t step = threadIdx.x; step < num_vec_elems; step += blockDim.x) {
        uint4 w_chunk = W_vec[step];
        const float* x_base = X + step * 16;

        float x[16];
        if (apply_quant) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                x[i] = dynamic_quantize_e4m3(x_base[i], d_scale, d_inv_scale);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) x[i] = x_base[i];
        }

        thread_acc += device_unpack_fp8_e4m3( w_chunk.x        & 0xFF) * x[0];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >>  8) & 0xFF) * x[1];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >> 16) & 0xFF) * x[2];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >> 24) & 0xFF) * x[3];

        thread_acc += device_unpack_fp8_e4m3( w_chunk.y        & 0xFF) * x[4];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >>  8) & 0xFF) * x[5];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >> 16) & 0xFF) * x[6];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >> 24) & 0xFF) * x[7];

        thread_acc += device_unpack_fp8_e4m3( w_chunk.z        & 0xFF) * x[8];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >>  8) & 0xFF) * x[9];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >> 16) & 0xFF) * x[10];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >> 24) & 0xFF) * x[11];

        thread_acc += device_unpack_fp8_e4m3( w_chunk.w        & 0xFF) * x[12];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >>  8) & 0xFF) * x[13];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >> 16) & 0xFF) * x[14];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >> 24) & 0xFF) * x[15];
    }

    float row_sum = gemv_block_reduce_sum(thread_acc, shared_warp_acc);

    if (threadIdx.x == 0) {
        float w_scale = __bfloat162float(weight_scales[row_idx * scale_stride]);
        float i_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : 1.0f;
        float res_val = row_sum * w_scale * i_scale;
        // 🎯 Эмулируем прохождение активации через тензор Bfloat16
        Y[row_idx] = __bfloat162float(__float2bfloat16(res_val));
    }
}

void launch_fp8_gemv_kernel(const void* d_W_fp8,
                            const float* d_X,
                            const void* d_weight_scales,
                            const void* d_input_scale,
                            float* d_Y,
                            size_t M,
                            size_t K,
                            int scale_stride) {
    dim3 blocks(M);
    dim3 threads(GEMV_BLOCK_SIZE);
    const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(d_W_fp8);
    const __nv_bfloat16* ws_ptr = reinterpret_cast<const __nv_bfloat16*>(d_weight_scales);
    const __nv_bfloat16* is_ptr = reinterpret_cast<const __nv_bfloat16*>(d_input_scale);
    fp8_gemv_kernel<<<blocks, threads>>>(w_ptr, d_X, ws_ptr, is_ptr, d_Y, K, scale_stride);
}

__global__ void fp8_gemv_residual_warp_kernel(const uint8_t* __restrict__ W_fp8,
                                              const float* __restrict__ X,
                                              const __nv_bfloat16* __restrict__ weight_scales,
                                              const __nv_bfloat16* __restrict__ input_scale,
                                              float* __restrict__ Y_accum,
                                              size_t M,
                                              size_t K,
                                              int scale_stride)
{
    size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane   = threadIdx.x % 32;
    if (row >= M) return;

    float warp_max = 0.0f;
    for (size_t col = lane; col < K; col += 32) {
        warp_max = fmaxf(warp_max, fabsf(X[col]));
    }
    warp_max = gemv_warp_reduce_max(warp_max);

    float d_scale = 1.0f;
    float d_inv_scale = 1.0f;
    bool apply_quant = (input_scale == nullptr);

    if (apply_quant) {
        float max_val = fmaxf(warp_max, 1e-5f);
        d_scale = 448.0f / max_val;
        d_inv_scale = max_val / 448.0f;
    }

    const uint8_t* cur_W_row = W_fp8 + row * K;
    float dot = 0.0f;

    for (size_t col = lane; col < K; col += 32) {
        float w_val = device_unpack_fp8_e4m3(cur_W_row[col]);
        float x_val = apply_quant ? dynamic_quantize_e4m3(X[col], d_scale, d_inv_scale) : X[col];
        dot += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    if (lane == 0) {
        float w_scale = __bfloat162float(weight_scales[row * scale_stride]);
        float i_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : 1.0f;
        
        // 🎯 Считываем текущий аккумулятор, прибавляем инкремент и жестко округляем до Bfloat16
        float new_acc = Y_accum[row] + (dot * w_scale * i_scale);
        Y_accum[row] = __bfloat162float(__float2bfloat16(new_acc));
    }
}

void launch_fp8_gemv_residual_kernel(const void* d_W_fp8,
                                     const float* d_X,
                                     const void* d_weight_scales,
                                     const void* d_input_scale,
                                     float* d_Y_accum,
                                     size_t M,
                                     size_t K,
                                     int scale_stride)
{
    int threads = 256;
    size_t total_threads = M * 32;
    size_t blocks = (total_threads + threads - 1) / threads;
    const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(d_W_fp8);
    const __nv_bfloat16* ws_ptr = reinterpret_cast<const __nv_bfloat16*>(d_weight_scales);
    const __nv_bfloat16* is_ptr = reinterpret_cast<const __nv_bfloat16*>(d_input_scale);
    fp8_gemv_residual_warp_kernel<<<blocks, threads>>>(w_ptr, d_X, ws_ptr, is_ptr, d_Y_accum, M, K, scale_stride);
}