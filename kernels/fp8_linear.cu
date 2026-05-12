#include "fp8_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>

#define QUANT_BLOCK_SIZE 1024

// ============================================================================
// 1. РАСЧЕТ ДИНАМИЧЕСКОГО СКЕЙЛА ТОКЕНА (Float32 мантисса)
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
// 2. ЯДРО ПОЭЛЕМЕНТНОГО КВАНТОВАНИЯ В FP8 E4M3
// ============================================================================
__global__ void quantize_tensor_to_fp8_kernel(const float* __restrict__ X,
                                              __nv_fp8_e4m3* __restrict__ X_fp8,
                                              const float* __restrict__ token_scale,
                                              size_t K) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        float inv_scale = token_scale[1];
        float scaled = X[idx] * inv_scale;
        // Зажимаем в границы FP8 и аппаратно кастуем с RNE-округлением
        scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);
        X_fp8[idx] = __nv_fp8_e4m3(scaled);
    }
}

// ============================================================================
// 3. ЯДРО ПРИМЕНЕНИЯ СКЕЙЛОВ И ОСТАТОЧНОГО СЛИЯНИЯ
// ============================================================================
__global__ void apply_scales_and_residual_kernel(const float* __restrict__ Y_raw,
                                                 const __nv_bfloat16* __restrict__ weight_scales,
                                                 const __nv_bfloat16* __restrict__ input_scale,
                                                 const float* __restrict__ token_scale,
                                                 float* __restrict__ Y,
                                                 size_t M,
                                                 int scale_stride,
                                                 bool accumulate)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        float w_scale = __bfloat162float(weight_scales[idx * scale_stride]);
        float d_scale = (input_scale != nullptr) ? __bfloat162float(*input_scale) : token_scale[0];

        // Точное слияние мантисс: итоговый скейл операндов
        double scaled_res = static_cast<double>(Y_raw[idx]) * static_cast<double>(w_scale) * static_cast<double>(d_scale);
        float bf16_val = __bfloat162float(__float2bfloat16(static_cast<float>(scaled_res)));

        if (accumulate) {
            double cur_acc = static_cast<double>(Y[idx]);
            Y[idx] = __bfloat162float(__float2bfloat16(static_cast<float>(cur_acc + static_cast<double>(bf16_val))));
        } else {
            Y[idx] = bf16_val;
        }
    }
}

// ============================================================================
// 4. ГЛАВНЫЙ ИНТЕРФЕЙС ЗАПУСКА CUBLASLT
// ============================================================================
void launch_fp8_linear_cublaslt(cublasLtHandle_t cublaslt_handle,
                                const void* d_W_fp8,
                                const float* d_X,
                                void* d_X_fp8,
                                float* d_Y_raw,
                                const void* d_weight_scales,
                                const void* d_input_scale,
                                const float* d_token_scale,
                                float* d_Y,
                                size_t M,
                                size_t K,
                                int scale_stride,
                                bool accumulate)
{
    size_t threads = 256;
    
    // Шаг 1: Динамическое квантование входного вектора активаций
    size_t blocks_x = (K + threads - 1) / threads;
    quantize_tensor_to_fp8_kernel<<<blocks_x, threads>>>((const float*)d_X, (__nv_fp8_e4m3*)d_X_fp8, d_token_scale, K);
    
    // Шаг 2: Настройка дескрипторов для тензорных ядер
    // Матрица W_fp8 в памяти Row-Major (M x K). Для cuBLASLt описываем её как Col-Major (K x M) с транспонированием (transA = T).
    // Вектор X_fp8 описываем как Col-Major (K x 1). Результат Y_raw будет (M x 1).
    cublasLtMatrixLayout_t matA_desc, matB_desc, matC_desc;
    cublasLtMatrixLayoutCreate(&matA_desc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&matB_desc, CUDA_R_8F_E4M3, K, 1, K);
    cublasLtMatrixLayoutCreate(&matC_desc, CUDA_R_32F, M, 1, M);

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Шаг 3: Аппаратное перемножение на Tensor Cores
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(cublaslt_handle, matmul_desc,
                   &alpha, d_W_fp8, matA_desc,
                   d_X_fp8, matB_desc,
                   &beta, d_Y_raw, matC_desc,
                   d_Y_raw, matC_desc,
                   nullptr, nullptr, 0, 0);

    cublasLtMatrixLayoutDestroy(matA_desc);
    cublasLtMatrixLayoutDestroy(matB_desc);
    cublasLtMatrixLayoutDestroy(matC_desc);
    cublasLtMatmulDescDestroy(matmul_desc);

    // Шаг 4: Применение скейлов и фиксация в шину Bfloat16
    size_t blocks_y = (M + threads - 1) / threads;
    apply_scales_and_residual_kernel<<<blocks_y, threads>>>(d_Y_raw, 
                                                            (const __nv_bfloat16*)d_weight_scales, 
                                                            (const __nv_bfloat16*)d_input_scale, 
                                                            d_token_scale, 
                                                            d_Y, M, scale_stride, accumulate);
}