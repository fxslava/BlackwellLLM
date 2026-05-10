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
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(Fp8LinearTests, GemvDecodingCorrectness) {
    // K обязано быть кратно 16 для нашей векторизованной оптимизации uint4
    const size_t M = 64;   // Количество выходных признаков (строк)
    const size_t K = 4096; // Входная скрытая размерность (столбцов)

    // 1. Подготовка данных на Host
    std::vector<uint8_t> h_W_fp8(M * K);
    std::vector<float> h_X(K);
    std::vector<float> h_scales(M);
    std::vector<float> h_cpu_Y(M, 0.0f);

    // Заполняем веса безопасными паттернами FP8 (избегаем NaN/Inf, которые в E4M3 равны 0x7F/0xFF)
    for (size_t i = 0; i < h_W_fp8.size(); ++i) {
        // Генерируем разнообразные нормальные и субнормальные мантиссы/экспоненты
        uint8_t raw = static_cast<uint8_t>(i % 120 + 1); 
        h_W_fp8[i] = raw;
    }

    for (size_t i = 0; i < K; ++i) {
        h_X[i] = static_cast<float>(i % 5) * 0.1f - 0.2f; // Вектор активаций от -0.2 до 0.2
    }

    for (size_t i = 0; i < M; ++i) {
        h_scales[i] = 0.5f + static_cast<float>(i % 4) * 0.25f; // Скейлинг-факторы
    }

    // 2. Вычисление Golden Reference на CPU
    cpu_fp8_gemv(h_W_fp8.data(), h_X.data(), h_scales.data(), h_cpu_Y.data(), M, K);

    // 3. Выделение памяти на GPU
    uint8_t* d_W_fp8;
    float *d_X, *d_scales, *d_Y;

    CUDA_CHECK(cudaMalloc(&d_W_fp8, M * K * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_X, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W_fp8, h_W_fp8.data(), M * K * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), M * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Запуск тестируемого ядра
    launch_fp8_gemv_kernel(d_W_fp8, d_X, d_scales, d_Y, M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Верификация результатов
    std::vector<float> h_gpu_Y(M);
    CUDA_CHECK(cudaMemcpy(h_gpu_Y.data(), d_Y, M * sizeof(float), cudaMemcpyDeviceToHost));

    // Сравниваем выходные векторы
    for (size_t i = 0; i < M; ++i) {
        // Используем чуть более мягкий эпсилон из-за разницы в порядке 
        // суммации чисел (дерево редукции GPU против линейного цикла CPU)
        ASSERT_NEAR(h_cpu_Y[i], h_gpu_Y[i], 5e-3f) 
            << "Расхождение в Linear GEMV по строке " << i;
    }

    // 6. Очистка
    cudaFree(d_W_fp8);
    cudaFree(d_X);
    cudaFree(d_scales);
    cudaFree(d_Y);
}