#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "swiglu.cuh"
#include "cpu_models.h"

class SwigluTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(SwigluTests, ActivationCorrectness) {
    // Размерность intermediate_dim для Llama 3 8B
    const size_t num_elements = 14336; 

    // 1. Подготовка данных на Host
    std::vector<float> h_gate(num_elements);
    std::vector<float> h_up(num_elements);
    std::vector<float> h_cpu_output(num_elements, 0.0f);

    // Заполняем реалистичным диапазоном значений (от -3.0 до 3.0)
    for (size_t i = 0; i < num_elements; ++i) {
        h_gate[i] = static_cast<float>(i % 60) * 0.1f - 3.0f;
        h_up[i]   = static_cast<float>(i % 40) * 0.1f - 2.0f;
    }

    // 2. Вычисление Golden Reference на CPU
    cpu_fused_swiglu(h_gate.data(), h_up.data(), h_cpu_output.data(), num_elements);

    // 3. Выделение памяти на GPU
    float *d_gate, *d_up, *d_output;
    CUDA_CHECK(cudaMalloc(&d_gate, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Запуск тестируемого ядра
    launch_fused_swiglu_kernel(d_gate, d_up, d_output, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Проверка результатов
    std::vector<float> h_gpu_output(num_elements);
    CUDA_CHECK(cudaMemcpy(h_gpu_output.data(), d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < num_elements; ++i) {
        ASSERT_NEAR(h_cpu_output[i], h_gpu_output[i], 1e-4f) 
            << "Расхождение в SwiGLU по абсолютному индексу " << i;
    }

    // 6. Очистка
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);
}