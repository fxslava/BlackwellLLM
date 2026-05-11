#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "fp8_linear.cuh"
#include "cpu_models.h"

class Fp8LinearTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
    }
};

TEST_F(Fp8LinearTests, GemvDecodingCorrectness) {
    const size_t M = 64;   
    const size_t K = 4096; 

    std::vector<uint8_t> h_W_fp8(M * K);
    std::vector<float> h_X(K);
    std::vector<float> h_scales(M);
    std::vector<float> h_cpu_Y(M, 0.0f);

    for (size_t i = 0; i < h_W_fp8.size(); ++i) h_W_fp8[i] = static_cast<uint8_t>(i % 120 + 1); 
    for (size_t i = 0; i < K; ++i)              h_X[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    for (size_t i = 0; i < M; ++i)              h_scales[i] = 0.5f + static_cast<float>(i % 4) * 0.25f;

    cpu_fp8_gemv(h_W_fp8.data(), h_X.data(), h_scales.data(), h_cpu_Y.data(), M, K);

    // 🚨 Элегантное выделение памяти через обертку
    CudaVector<uint8_t> d_W_fp8(M * K); d_W_fp8.upload(h_W_fp8);
    CudaVector<float>   d_X(K);         d_X.upload(h_X);
    CudaVector<float>   d_scales(M);    d_scales.upload(h_scales);
    CudaVector<float>   d_Y(M);

    launch_fp8_gemv_kernel(d_W_fp8, d_X, d_scales, d_Y, M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_Y(M);
    d_Y.download(h_gpu_Y);

    for (size_t i = 0; i < M; ++i) {
        ASSERT_NEAR(h_cpu_Y[i], h_gpu_Y[i], 5e-3f) << "Расхождение в Linear GEMV по строке " << i;
    }
}

TEST_F(Fp8LinearTests, Fp8GemvResidualCorrectness) {
    const size_t M = 128;  
    const size_t K = 4096; 

    std::vector<uint8_t> h_W_fp8(M * K);
    std::vector<float> h_X(K);
    std::vector<float> h_scales(M);
    std::vector<float> h_Y_cpu(M); 

    for (size_t i = 0; i < M; ++i) {
        h_Y_cpu[i] = static_cast<float>(i) * 0.5f; 
        h_scales[i] = 0.02f; 
    }
    for (size_t j = 0; j < K; ++j)     h_X[j] = static_cast<float>(j % 50) * 0.05f - 1.0f;
    for (size_t i = 0; i < M * K; ++i) h_W_fp8[i] = static_cast<uint8_t>((i % 120) + 1);

    // Эталонный расчет на CPU: Y_accum += (W * X) * scale
    for (size_t i = 0; i < M; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < K; ++j) {
            dot += cpu_unpack_fp8_e4m3(h_W_fp8[i * K + j]) * h_X[j];
        }
        h_Y_cpu[i] += static_cast<float>(dot) * h_scales[i]; 
    }

    CudaVector<uint8_t> d_W_fp8(M * K); d_W_fp8.upload(h_W_fp8);
    CudaVector<float>   d_X(K);         d_X.upload(h_X);
    CudaVector<float>   d_scales(M);    d_scales.upload(h_scales);
    CudaVector<float>   d_Y_accum(M);   
    
    // Загружаем начальное состояние накопителя
    std::vector<float> h_Y_initial(M);
    for (size_t i = 0; i < M; ++i) h_Y_initial[i] = static_cast<float>(i) * 0.5f;
    d_Y_accum.upload(h_Y_initial);

    launch_fp8_gemv_residual_kernel(d_W_fp8, d_X, d_scales, d_Y_accum, M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(M);
    d_Y_accum.download(h_gpu_output);

    for (size_t i = 0; i < M; ++i) {
        ASSERT_NEAR(h_Y_cpu[i], h_gpu_output[i], 5e-3f) << "Ошибка Residual GEMV на строке " << i;
    }
}