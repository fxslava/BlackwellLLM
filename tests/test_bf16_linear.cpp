#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "common.h"
#include "cpu_models.h"
#include "bf16_linear.cuh"

class Bf16LinearTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(Bf16LinearTests, Bf16GemvCorrectness) {
    const size_t M = 256;  
    const size_t K = 4096; 

    std::vector<__nv_bfloat16> h_W_bf16(M * K);
    std::vector<float> h_X(K);
    std::vector<float> h_Y_cpu(M, 0.0f);

    for (size_t j = 0; j < K; ++j) {
        h_X[j] = static_cast<float>(j % 100) * 0.01f - 0.5f;
    }

    for (size_t i = 0; i < M; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < K; ++j) {
            float orig_w = static_cast<float>((i + j) % 50) * 0.05f - 1.2f;
            __nv_bfloat16 bf16_w = __float2bfloat16(orig_w);
            
            h_W_bf16[i * K + j] = bf16_w;
            dot += __bfloat162float(bf16_w) * h_X[j];
        }
        h_Y_cpu[i] = static_cast<float>(dot);
    }

    CudaVector<__nv_bfloat16> d_W_bf16(M * K); d_W_bf16.upload(h_W_bf16);
    CudaVector<float>         d_X(K);          d_X.upload(h_X);
    CudaVector<float>         d_Y(M);

    launch_bf16_gemv_kernel(d_W_bf16, d_X, d_Y, M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(M);
    d_Y.download(h_gpu_output);

    for (size_t i = 0; i < M; ++i) {
        ASSERT_NEAR(h_Y_cpu[i], h_gpu_output[i], 1e-3f) << "Расхождение в логитах BF16 на токене " << i;
    }
}