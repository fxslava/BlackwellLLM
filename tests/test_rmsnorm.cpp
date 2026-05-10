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
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(RmsnormTests, FusedRmsnormResidualCorrectness) {
    const size_t seq_len = 4;
    const size_t hidden_dim = 4096;
    const float eps = 1e-5f;

    // 1. Подготовка данных на Host
    std::vector<float> h_x(seq_len * hidden_dim);
    std::vector<float> h_residual(seq_len * hidden_dim);
    std::vector<float> h_weight(hidden_dim);

    // Заполняем псевдослучайными, но детерминированными числами
    for (size_t i = 0; i < h_x.size(); ++i) {
        h_x[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        h_residual[i] = static_cast<float>(i % 7) * 0.2f + 0.1f;
    }
    for (size_t i = 0; i < hidden_dim; ++i) {
        h_weight[i] = 1.0f + static_cast<float>(i % 5) * 0.01f; // Веса близки к 1.0
    }

    // Создаем копии для CPU расчетов
    std::vector<float> h_cpu_x = h_x;
    std::vector<float> h_cpu_residual = h_residual;

    // 2. Вычисление Golden Reference на CPU
    cpu_rmsnorm_residual(h_cpu_x.data(), h_cpu_residual.data(), h_weight.data(), seq_len, hidden_dim, eps);

    // 3. Выделение памяти на GPU
    float *d_x, *d_residual, *d_weight;
    CUDA_CHECK(cudaMalloc(&d_x, seq_len * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, seq_len * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Запуск ядра
    launch_rmsnorm_residual_kernel(d_x, d_residual, d_weight, seq_len, hidden_dim, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Проверка результатов
    std::vector<float> h_gpu_x(seq_len * hidden_dim);
    std::vector<float> h_gpu_residual(seq_len * hidden_dim);
    
    CUDA_CHECK(cudaMemcpy(h_gpu_x.data(), d_x, seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_residual.data(), d_residual, seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // Сравниваем Residual буфер
    for (size_t i = 0; i < h_gpu_residual.size(); ++i) {
        ASSERT_NEAR(h_cpu_residual[i], h_gpu_residual[i], 1e-4f) << "Ошибка в Residual по индексу " << i;
    }

    // Сравниваем нормализованный X
    for (size_t i = 0; i < h_gpu_x.size(); ++i) {
        ASSERT_NEAR(h_cpu_x[i], h_gpu_x[i], 1e-4f) << "Ошибка в X_norm по индексу " << i;
    }

    // 6. Очистка
    cudaFree(d_x);
    cudaFree(d_residual);
    cudaFree(d_weight);
}