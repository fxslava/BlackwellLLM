#include "fp8_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define GEMV_BLOCK_SIZE 128

// Аппаратный декодер FP8 E4M3 на стороне GPU
__inline__ __device__ float device_unpack_fp8_e4m3(uint8_t byte_val) {
    if ((byte_val & 0x7F) == 0) {
        return (byte_val & 0x80) ? -0.0f : 0.0f;
    }
    int sign = (byte_val & 0x80) ? -1 : 1;
    int exp  = (byte_val & 0x78) >> 3;
    int mant = byte_val & 0x07;

    if (exp == 0) {
        return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

// Нативная редукция суммы внутри варпа
__inline__ __device__ float gemv_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Нативная редукция суммы по всему блоку (128 потоков = 4 варпа)
__inline__ __device__ float gemv_block_reduce_sum(float val, float* shared_warp_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = gemv_warp_reduce_sum(val);

    if (lane_id == 0) {
        shared_warp_sums[warp_id] = val;
    }
    __syncthreads();

    float warp_val = (threadIdx.x < (GEMV_BLOCK_SIZE / 32)) ? shared_warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        warp_val = gemv_warp_reduce_sum(warp_val);
    }
    return warp_val;
}

__global__ void fp8_gemv_kernel(const uint8_t* __restrict__ W_fp8,
                                const float* __restrict__ X,
                                const __nv_bfloat16* __restrict__ weight_scales,
                                const __nv_bfloat16* __restrict__ input_scale,
                                float* __restrict__ Y,
                                size_t K,
                                int scale_stride) {
    size_t row_idx = blockIdx.x;
    const uint8_t* cur_W_row = W_fp8 + row_idx * K;
    
    float thread_acc = 0.0f;

    size_t num_vec_elems = K / 16;
    const uint4* W_vec = reinterpret_cast<const uint4*>(cur_W_row);

    for (size_t step = threadIdx.x; step < num_vec_elems; step += blockDim.x) {
        uint4 w_chunk = W_vec[step];
        const float* x_base = X + step * 16;

        // Распаковка 16 байт
        thread_acc += device_unpack_fp8_e4m3( w_chunk.x        & 0xFF) * x_base[0];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >>  8) & 0xFF) * x_base[1];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >> 16) & 0xFF) * x_base[2];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >> 24) & 0xFF) * x_base[3];

        thread_acc += device_unpack_fp8_e4m3( w_chunk.y        & 0xFF) * x_base[4];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >>  8) & 0xFF) * x_base[5];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >> 16) & 0xFF) * x_base[6];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >> 24) & 0xFF) * x_base[7];

        thread_acc += device_unpack_fp8_e4m3( w_chunk.z        & 0xFF) * x_base[8];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >>  8) & 0xFF) * x_base[9];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >> 16) & 0xFF) * x_base[10];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >> 24) & 0xFF) * x_base[11];

        thread_acc += device_unpack_fp8_e4m3( w_chunk.w        & 0xFF) * x_base[12];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >>  8) & 0xFF) * x_base[13];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >> 16) & 0xFF) * x_base[14];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >> 24) & 0xFF) * x_base[15];
    }

    __shared__ float shared_warp_sums[GEMV_BLOCK_SIZE / 32];
    float row_sum = gemv_block_reduce_sum(thread_acc, shared_warp_sums);

    if (threadIdx.x == 0) {
        // Аппаратная распаковка скейлов из Bfloat16 во Float32
        float w_scale = __bfloat162float(weight_scales[row_idx * scale_stride]);
        float i_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : 1.0f;
        Y[row_idx] = row_sum * w_scale * i_scale;
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

// ----------------------------------------------------------------------------
// Остаточное ядро (Residual GEMV)
// ----------------------------------------------------------------------------
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

    const uint8_t* cur_W_row = W_fp8 + row * K;
    
    float dot = 0.0f;
    for (size_t col = lane; col < K; col += 32) {
        float w_val = device_unpack_fp8_e4m3(cur_W_row[col]);
        dot += w_val * X[col];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    if (lane == 0) {
        float w_scale = __bfloat162float(weight_scales[row * scale_stride]);
        float i_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : 1.0f;
        Y_accum[row] += dot * w_scale * i_scale;
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