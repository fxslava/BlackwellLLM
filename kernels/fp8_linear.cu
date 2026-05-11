#include "fp8_linear.cuh"
#include <cuda_runtime.h>
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
                                const float* __restrict__ scales,
                                float* __restrict__ Y,
                                size_t K) {
    // blockIdx.x соответствует индексу строки (M)
    size_t row_idx = blockIdx.x;
    
    // Сдвигаем указатель весов на начало текущей строки
    const uint8_t* cur_W_row = W_fp8 + row_idx * K;
    
    float thread_acc = 0.0f;

    // Векторизованное чтение: кастим сырые указатели в 128-битные uint4
    // 1 транзакция uint4 = 16 байт = 16 элементов FP8
    size_t num_vec_elems = K / 16;
    const uint4* W_vec = reinterpret_cast<const uint4*>(cur_W_row);
    const uint4* X_vec = reinterpret_cast<const uint4*>(X);

    for (size_t step = threadIdx.x; step < num_vec_elems; step += blockDim.x) {
        // Читаем 16 весов и 16 активаций (активации пока читаются как float, 
        // поэтому берем 4 транзакции float4, чтобы покрыть 16 элементов)
        uint4 w_chunk = W_vec[step];
        
        const float* x_base = X + step * 16;

        // Распаковываем 16 байт из регистров w_chunk через побайтовые маски и сдвиги
        // Обработка первых 4 байт (из w_chunk.x)
        thread_acc += device_unpack_fp8_e4m3( w_chunk.x        & 0xFF) * x_base[0];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >>  8) & 0xFF) * x_base[1];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >> 16) & 0xFF) * x_base[2];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.x >> 24) & 0xFF) * x_base[3];

        // Обработка вторых 4 байт (из w_chunk.y)
        thread_acc += device_unpack_fp8_e4m3( w_chunk.y        & 0xFF) * x_base[4];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >>  8) & 0xFF) * x_base[5];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >> 16) & 0xFF) * x_base[6];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.y >> 24) & 0xFF) * x_base[7];

        // Обработка третьих 4 байт (из w_chunk.z)
        thread_acc += device_unpack_fp8_e4m3( w_chunk.z        & 0xFF) * x_base[8];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >>  8) & 0xFF) * x_base[9];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >> 16) & 0xFF) * x_base[10];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.z >> 24) & 0xFF) * x_base[11];

        // Обработка четвертых 4 байт (из w_chunk.w)
        thread_acc += device_unpack_fp8_e4m3( w_chunk.w        & 0xFF) * x_base[12];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >>  8) & 0xFF) * x_base[13];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >> 16) & 0xFF) * x_base[14];
        thread_acc += device_unpack_fp8_e4m3((w_chunk.w >> 24) & 0xFF) * x_base[15];
    }

    // Буфер для 4 варпов
    __shared__ float shared_warp_sums[GEMV_BLOCK_SIZE / 32];
    
    // Сворачиваем сумму по строке
    float row_sum = gemv_block_reduce_sum(thread_acc, shared_warp_sums);

    // Нулевой поток умножает итог на скейлинг-фактор и пишет ответ в VRAM
    if (threadIdx.x == 0) {
        Y[row_idx] = row_sum * scales[row_idx];
    }
}

void launch_fp8_gemv_kernel(const uint8_t* d_W_fp8,
                            const float* d_X,
                            const float* d_scales,
                            float* d_Y,
                            size_t M,
                            size_t K) {
    // 1 блок на каждую строку выходного вектора
    dim3 blocks(M);
    dim3 threads(GEMV_BLOCK_SIZE);

    fp8_gemv_kernel<<<blocks, threads>>>(d_W_fp8, d_X, d_scales, d_Y, K);
}

// Детерминированная инлайн-распаковка FP8 (E4M3) во float
__device__ __inline__ float unpack_fp8_e4m3(uint8_t byte_val) {
    if ((byte_val & 0x7F) == 0) return 0.0f;
    int sign = (byte_val & 0x80) ? -1 : 1;
    int exp  = (byte_val & 0x78) >> 3;
    int mant = byte_val & 0x07;
    
    if (exp == 0) {
        return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

__global__ void fp8_gemv_residual_warp_kernel(const uint8_t* __restrict__ W_fp8,
                                              const float* __restrict__ X,
                                              const float* __restrict__ scales,
                                              float* __restrict__ Y_accum,
                                              size_t M,
                                              size_t K)
{
    // 1 варп (32 потока) полностью обрабатывает 1 строку матрицы
    size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane   = threadIdx.x % 32;

    if (row >= M) return;

    const uint8_t* cur_W_row = W_fp8 + row * K;
    float row_scale = scales[row];

    float dot = 0.0f;
    for (size_t col = lane; col < K; col += 32) {
        float w_val = unpack_fp8_e4m3(cur_W_row[col]);
        dot += w_val * X[col];
    }

    // Быстрая редукция суммы внутри варпа
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    // Нулевой поток варпа прибавляет результат к главному накопителю
    if (lane == 0) {
        Y_accum[row] += dot * row_scale;
    }
}

void launch_fp8_gemv_residual_kernel(const uint8_t* d_W_fp8,
                                     const float* d_X,
                                     const float* d_scales,
                                     float* d_Y_accum,
                                     size_t M,
                                     size_t K)
{
    int threads = 256;
    size_t total_threads = M * 32; // По 32 потока на каждую из M строк
    size_t blocks = (total_threads + threads - 1) / threads;

    fp8_gemv_residual_warp_kernel<<<blocks, threads>>>(d_W_fp8, d_X, d_scales, d_Y_accum, M, K);
}