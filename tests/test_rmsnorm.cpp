#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "rmsnorm.cuh"
#include "cpu_models.h"

class RmsnormTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
    }
};

TEST_F(RmsnormTests, FusedRmsnormResidualCorrectness) {
    const size_t seq_len = 4;
    const size_t hidden_dim = 4096;
    const float eps = 1e-5f;

    std::vector<float> h_x(seq_len * hidden_dim);
    std::vector<float> h_residual(seq_len * hidden_dim);
    std::vector<float> h_weight(hidden_dim);

    for (size_t i = 0; i < h_x.size(); ++i) {
        h_x[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        h_residual[i] = static_cast<float>(i % 7) * 0.2f + 0.1f;
    }
    for (size_t i = 0; i < hidden_dim; ++i) h_weight[i] = 1.0f + static_cast<float>(i % 5) * 0.01f;

    std::vector<float> h_cpu_x = h_x;
    std::vector<float> h_cpu_residual = h_residual;
    cpu_rmsnorm_residual(h_cpu_x.data(), h_cpu_residual.data(), h_weight.data(), seq_len, hidden_dim, eps);

    // 🚨 МАГИЯ RAII: Выделение и загрузка в 3 строчки вместо 12
    CudaVector<float> d_x(h_x.size());           d_x.upload(h_x);
    CudaVector<float> d_residual(h_residual.size()); d_residual.upload(h_residual);
    CudaVector<float> d_weight(h_weight.size());     d_weight.upload(h_weight);

    launch_rmsnorm_residual_kernel(d_x, d_residual, d_weight, seq_len, hidden_dim, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_x(h_x.size());           d_x.download(h_gpu_x);
    std::vector<float> h_gpu_residual(h_residual.size()); d_residual.download(h_gpu_residual);

    for (size_t i = 0; i < h_gpu_residual.size(); ++i) {
        ASSERT_NEAR(h_cpu_residual[i], h_gpu_residual[i], 1e-4f) << "Ошибка в Residual по индексу " << i;
        ASSERT_NEAR(h_cpu_x[i], h_gpu_x[i], 1e-4f)               << "Ошибка в X_norm по индексу " << i;
    }
    // cudaFree вызывается автоматически деструкторами d_x, d_residual и d_weight!
}

TEST_F(RmsnormTests, DirectRmsnormCorrectness) {
    const size_t hidden_dim = 4096;
    const size_t seq_len = 1;
    const float eps = 1e-5f;

    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weight(hidden_dim);
    std::vector<float> h_cpu_output(hidden_dim, 0.0f);

    for (size_t i = 0; i < hidden_dim; ++i) {
        h_input[i] = static_cast<float>(i % 100) * 0.01f - 0.5f;
        h_weight[i] = 1.0f + static_cast<float>(i % 10) * 0.05f;
    }

    // Используем вынесенный эталон
    cpu_rmsnorm_direct(h_input.data(), h_cpu_output.data(), h_weight.data(), seq_len, hidden_dim, eps);

    CudaVector<float> d_input(hidden_dim);  d_input.upload(h_input);
    CudaVector<float> d_weight(hidden_dim); d_weight.upload(h_weight);
    CudaVector<float> d_output(hidden_dim); // Инициализируется нулями внутри конструктора

    launch_rmsnorm_kernel(d_input, d_output, d_weight, seq_len, hidden_dim, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(hidden_dim);
    d_output.download(h_gpu_output);

    for (size_t i = 0; i < hidden_dim; ++i) {
        ASSERT_NEAR(h_cpu_output[i], h_gpu_output[i], 1e-5f) << "Сбой RMSNorm по индексу " << i;
    }

    std::vector<float> h_input_check(hidden_dim);
    d_input.download(h_input_check);
    ASSERT_EQ(h_input, h_input_check) << "Критическая ошибка: ядро перезаписало входной буфер!";
}