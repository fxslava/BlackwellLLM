#include "fp8_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>

#define GEMV_BLOCK_SIZE 256
#define QUANT_BLOCK_SIZE 1024

// 🎯 ОТКЛЮЧАЕМ принудительную квантизацию активаций. 
// Теперь, если модель позволяет, мы будем умножать чистые FP32 активации на распакованные FP8 веса (Weight-Only режим)!
#define FORCE_ACTIVATION_QUANTIZATION false

// ============================================================================
// 1. РАСЧЕТ ДИНАМИЧЕСКОГО СКЕЙЛА ТОКЕНА
// ============================================================================
__global__ void quantize_per_token_kernel(const float* __restrict__ X, float* __restrict__ token_scale, size_t K) {
    __shared__ float s_max_vals[32];

    float thread_max = 0.0f;
    for (size_t i = threadIdx.x; i < K; i += blockDim.x) {
        thread_max = fmaxf(thread_max, fabsf(X[i]));
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    if (lane_id == 0) {
        s_max_vals[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane_id < (blockDim.x >> 5)) ? s_max_vals[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xFFFFFFFF, block_max, offset));
        }

        if (threadIdx.x == 0) {
            float max_val = fmaxf(block_max, 1e-12f);
            float scale = max_val / 448.0f;
            
            token_scale[0] = scale;        
            token_scale[1] = 1.0f / scale; 
        }
    }
}

void launch_quantize_per_token_kernel(const float* d_X, float* d_token_scale, size_t K) {
    quantize_per_token_kernel<<<1, QUANT_BLOCK_SIZE>>>(d_X, d_token_scale, K);
}

// ============================================================================
// ВНУТРЕННИЕ ФУНКЦИИ РЕДУКЦИИ И РАСПАКОВКИ
// ============================================================================
__inline__ __device__ float gemv_block_reduce_sum(float val, float* shared_warp_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // 1. Локальная редукция внутри варпа в чистом float32
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane_id == 0) {
        shared_warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 2. Нулевой варп дособирает ответ в чистом float32
    float warp_val = (threadIdx.x < (GEMV_BLOCK_SIZE / 32)) ? shared_warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xFFFFFFFF, warp_val, offset);
        }
    }
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

__inline__ __device__ float golden_hardware_quantize_e4m3(float x, float scale) {
    float scaled = x / scale;
    scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);
    __nv_fp8_e4m3 fp8_val(scaled);
    return static_cast<float>(fp8_val);
}

__inline__ __device__ float unpack_raw_weight(uint8_t byte_val) {
    return device_unpack_fp8_e4m3(byte_val);
}

// ============================================================================
// 2. ЧИСТОЕ ЯДРО УМНОЖЕНИЯ (Pure Unscaled Dot Product + Epilogue)
// ============================================================================

__global__ void fp8_gemv_splitk_kernel(const uint8_t* __restrict__ W_fp8,
                                       const float* __restrict__ X,
                                       const __nv_bfloat16* __restrict__ weight_scales,
                                       const __nv_bfloat16* __restrict__ input_scale,
                                       const float* __restrict__ token_scale,
                                       float* __restrict__ Y,
                                       size_t K,
                                       int scale_stride) 
{
    size_t row_idx = blockIdx.x;
    const uint8_t* cur_W_row = W_fp8 + row_idx * K;
    __shared__ float shared_warp_acc[GEMV_BLOCK_SIZE / 32];

    float w_scale = __bfloat162float(weight_scales[row_idx * scale_stride]);
    bool apply_quant = (input_scale != nullptr) || FORCE_ACTIVATION_QUANTIZATION;
    float d_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : token_scale[0];

    // 🎯 Аккумуляторы работают в полном 32-битном разрешении
    float acc_tiles[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t num_vec_elems = K / 16;
    const uint4* W_vec = reinterpret_cast<const uint4*>(cur_W_row);

    for (size_t step = threadIdx.x; step < num_vec_elems; step += blockDim.x) {
        uint4 w_chunk = W_vec[step];
        const float* x_base = X + step * 16;

        float x[16];
        if (apply_quant) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                x[i] = golden_hardware_quantize_e4m3(x_base[i], d_scale);
            }
        } else {
            // 🎯 Идеальная точность: читаем FP32 активации напрямую
            #pragma unroll
            for (int i = 0; i < 16; ++i) x[i] = x_base[i];
        }

        // Fused Multiply-Add (FMA) выполняется аппаратно в FP32
        acc_tiles[0] += unpack_raw_weight( w_chunk.x        & 0xFF) * x[0]
                      + unpack_raw_weight((w_chunk.x >>  8) & 0xFF) * x[1]
                      + unpack_raw_weight((w_chunk.x >> 16) & 0xFF) * x[2]
                      + unpack_raw_weight((w_chunk.x >> 24) & 0xFF) * x[3];

        acc_tiles[1] += unpack_raw_weight( w_chunk.y        & 0xFF) * x[4]
                      + unpack_raw_weight((w_chunk.y >>  8) & 0xFF) * x[5]
                      + unpack_raw_weight((w_chunk.y >> 16) & 0xFF) * x[6]
                      + unpack_raw_weight((w_chunk.y >> 24) & 0xFF) * x[7];

        acc_tiles[2] += unpack_raw_weight( w_chunk.z        & 0xFF) * x[8]
                      + unpack_raw_weight((w_chunk.z >>  8) & 0xFF) * x[9]
                      + unpack_raw_weight((w_chunk.z >> 16) & 0xFF) * x[10]
                      + unpack_raw_weight((w_chunk.z >> 24) & 0xFF) * x[11];

        acc_tiles[3] += unpack_raw_weight( w_chunk.w        & 0xFF) * x[12]
                      + unpack_raw_weight((w_chunk.w >>  8) & 0xFF) * x[13]
                      + unpack_raw_weight((w_chunk.w >> 16) & 0xFF) * x[14]
                      + unpack_raw_weight((w_chunk.w >> 24) & 0xFF) * x[15];
    }

    float thread_acc = acc_tiles[0] + acc_tiles[1] + acc_tiles[2] + acc_tiles[3];
    float row_sum = gemv_block_reduce_sum(thread_acc, shared_warp_acc);

    if (threadIdx.x == 0) {
        float combined_scale = w_scale * (apply_quant ? d_scale : 1.0f);
        // 🎯 Чистая запись в глобальную память без искажений
        Y[row_idx] = row_sum * combined_scale;
    }
}

void launch_fp8_gemv_kernel(const void* d_W_fp8,
                            const float* d_X,
                            const void* d_weight_scales,
                            const void* d_input_scale,
                            const float* d_token_scale,
                            float* d_Y,
                            size_t M, size_t K, int scale_stride) 
{
    dim3 blocks(M);
    dim3 threads(GEMV_BLOCK_SIZE);
    fp8_gemv_splitk_kernel<<<blocks, threads>>>(
        (const uint8_t*)d_W_fp8, d_X, (const __nv_bfloat16*)d_weight_scales,
        (const __nv_bfloat16*)d_input_scale, d_token_scale, d_Y, K, scale_stride);
}

// ============================================================================
// 3. ОСТАТОЧНОЕ ЯДРО (С идентичной чистой логикой на float)
// ============================================================================
__global__ void fp8_gemv_splitk_residual_kernel(const uint8_t* __restrict__ W_fp8,
                                                const float* __restrict__ X,
                                                const __nv_bfloat16* __restrict__ weight_scales,
                                                const __nv_bfloat16* __restrict__ input_scale,
                                                const float* __restrict__ token_scale,
                                                float* __restrict__ Y_accum,
                                                size_t K, int scale_stride)
{
    size_t row_idx = blockIdx.x;
    const uint8_t* cur_W_row = W_fp8 + row_idx * K;
    __shared__ float shared_warp_acc[GEMV_BLOCK_SIZE / 32];

    float w_scale = __bfloat162float(weight_scales[row_idx * scale_stride]);
    bool apply_quant = (input_scale != nullptr) || FORCE_ACTIVATION_QUANTIZATION;
    float d_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : token_scale[0];

    float acc_tiles[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t num_vec_elems = K / 16;
    const uint4* W_vec = reinterpret_cast<const uint4*>(cur_W_row);

    for (size_t step = threadIdx.x; step < num_vec_elems; step += blockDim.x) {
        uint4 w_chunk = W_vec[step];
        const float* x_base = X + step * 16;

        float x[16];
        if (apply_quant) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                x[i] = golden_hardware_quantize_e4m3(x_base[i], d_scale);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) x[i] = x_base[i];
        }

        acc_tiles[0] += unpack_raw_weight( w_chunk.x        & 0xFF) * x[0]
                      + unpack_raw_weight((w_chunk.x >>  8) & 0xFF) * x[1]
                      + unpack_raw_weight((w_chunk.x >> 16) & 0xFF) * x[2]
                      + unpack_raw_weight((w_chunk.x >> 24) & 0xFF) * x[3];

        acc_tiles[1] += unpack_raw_weight( w_chunk.y        & 0xFF) * x[4]
                      + unpack_raw_weight((w_chunk.y >>  8) & 0xFF) * x[5]
                      + unpack_raw_weight((w_chunk.y >> 16) & 0xFF) * x[6]
                      + unpack_raw_weight((w_chunk.y >> 24) & 0xFF) * x[7];

        acc_tiles[2] += unpack_raw_weight( w_chunk.z        & 0xFF) * x[8]
                      + unpack_raw_weight((w_chunk.z >>  8) & 0xFF) * x[9]
                      + unpack_raw_weight((w_chunk.z >> 16) & 0xFF) * x[10]
                      + unpack_raw_weight((w_chunk.z >> 24) & 0xFF) * x[11];

        acc_tiles[3] += unpack_raw_weight( w_chunk.w        & 0xFF) * x[12]
                      + unpack_raw_weight((w_chunk.w >>  8) & 0xFF) * x[13]
                      + unpack_raw_weight((w_chunk.w >> 16) & 0xFF) * x[14]
                      + unpack_raw_weight((w_chunk.w >> 24) & 0xFF) * x[15];
    }

    // 🎯 Чистое сложение без обрезания до BF16
    float thread_acc = acc_tiles[0] + acc_tiles[1] + acc_tiles[2] + acc_tiles[3];
    float row_sum = gemv_block_reduce_sum(thread_acc, shared_warp_acc);

    if (threadIdx.x == 0) {
        float combined_scale = w_scale * (apply_quant ? d_scale : 1.0f);
        float final_res = row_sum * combined_scale;

        // 🎯 Прямое накопление остатка (Residual Accumulation) в 32 битах
        Y_accum[row_idx] += final_res;
    }
}

void launch_fp8_gemv_residual_kernel(const void* d_W_fp8,
                                     const float* d_X,
                                     const void* d_weight_scales,
                                     const void* d_input_scale,
                                     const float* d_token_scale,
                                     float* d_Y_accum,
                                     size_t M, size_t K, int scale_stride)
{
    dim3 blocks(M);
    dim3 threads(GEMV_BLOCK_SIZE);
    fp8_gemv_splitk_residual_kernel<<<blocks, threads>>>(
        (const uint8_t*)d_W_fp8, d_X, (const __nv_bfloat16*)d_weight_scales,
        (const __nv_bfloat16*)d_input_scale, d_token_scale, d_Y_accum, K, scale_stride);
}