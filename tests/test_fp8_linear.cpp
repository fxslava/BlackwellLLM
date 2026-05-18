#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "common.h"
#include "fp8_linear.cuh"
#include "cpu_models.h"

class Fp8LinearTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[CUDA ERROR]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(Fp8LinearTests, GemvDecodingCorrectness) {
    const size_t M = 64;   
    const size_t K = 4096; 

    std::vector<uint8_t> h_W_fp8(M * K);
    std::vector<float> h_X(K);
    std::vector<float> h_scales(M);
    std::vector<__nv_bfloat16> h_scales_bf16(M);
    std::vector<float> h_cpu_Y(M, 0.0f);

    for (size_t i = 0; i < h_W_fp8.size(); ++i) h_W_fp8[i] = static_cast<uint8_t>(i % 120 + 1); 
    for (size_t i = 0; i < K; ++i)              h_X[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    
    for (size_t i = 0; i < M; ++i) {
        h_scales[i] = 0.5f + static_cast<float>(i % 4) * 0.25f;
        h_scales_bf16[i] = __float2bfloat16(h_scales[i]);
    }

    // CPU baseline calculation expects float scales
    cpu_fp8_gemv(h_W_fp8.data(), h_X.data(), h_scales.data(), h_cpu_Y.data(), M, K);

    // Upload via RAII wrappers ensuring clean BF16 scale pointers
    CudaVector<uint8_t>       d_W_fp8(M * K); d_W_fp8.upload(h_W_fp8);
    CudaVector<float>         d_X(K);         d_X.upload(h_X);
    CudaVector<__nv_bfloat16> d_scales(M);    d_scales.upload(h_scales_bf16);
    CudaVector<float>         d_Y(M);

    launch_fp8_gemv_kernel(d_W_fp8, d_X, d_scales, nullptr, nullptr, d_Y, M, K, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_Y(M);
    d_Y.download(h_gpu_Y);

    for (size_t i = 0; i < M; ++i) {
        ASSERT_NEAR(h_cpu_Y[i], h_gpu_Y[i], 5e-3f) 
            << "Linear GEMV mismatch detected at row index: " << i;
    }
}

TEST_F(Fp8LinearTests, Fp8GemvResidualCorrectness) {
    const size_t M = 128;  
    const size_t K = 4096; 

    std::vector<uint8_t> h_W_fp8(M * K);
    std::vector<float> h_X(K);
    std::vector<float> h_scales(M);
    std::vector<__nv_bfloat16> h_scales_bf16(M);
    std::vector<float> h_Y_cpu(M); 

    for (size_t i = 0; i < M; ++i) {
        h_Y_cpu[i] = static_cast<float>(i) * 0.5f; 
        h_scales[i] = 0.02f; 
        h_scales_bf16[i] = __float2bfloat16(h_scales[i]);
    }
    for (size_t j = 0; j < K; ++j)     h_X[j] = static_cast<float>(j % 50) * 0.05f - 1.0f;
    for (size_t i = 0; i < M * K; ++i) h_W_fp8[i] = static_cast<uint8_t>((i % 120) + 1);

    for (size_t i = 0; i < M; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < K; ++j) {
            dot += cpu_unpack_fp8_e4m3(h_W_fp8[i * K + j]) * h_X[j];
        }
        h_Y_cpu[i] += static_cast<float>(dot) * h_scales[i]; 
    }

    CudaVector<uint8_t>       d_W_fp8(M * K); d_W_fp8.upload(h_W_fp8);
    CudaVector<float>         d_X(K);         d_X.upload(h_X);
    CudaVector<__nv_bfloat16> d_scales(M);    d_scales.upload(h_scales_bf16);
    CudaVector<float>         d_Y_accum(M);   
    
    std::vector<float> h_Y_initial(M);
    for (size_t i = 0; i < M; ++i) h_Y_initial[i] = static_cast<float>(i) * 0.5f;
    d_Y_accum.upload(h_Y_initial);

    launch_fp8_gemv_residual_kernel(d_W_fp8, d_X, d_scales, nullptr, nullptr, d_Y_accum, M, K, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(M);
    d_Y_accum.download(h_gpu_output);

    for (size_t i = 0; i < M; ++i) {
        ASSERT_NEAR(h_Y_cpu[i], h_gpu_output[i], 1.0f) 
            << "Residual GEMV mismatch detected at row index: " << i;
    }
}