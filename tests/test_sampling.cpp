#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "sampling.cuh"
#include "cpu_models.h"

class SamplingTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(SamplingTests, ArgmaxGreedyCorrectness) {
    const size_t vocab_size = 128256; // Реальный размер словаря Llama 3
    const int target_max_idx = 84042; // Секретный индекс, куда мы спрячем максимум

    // 1. Подготовка данных на Host
    std::vector<float> h_logits(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        // Заполняем фоновыми вероятностями
        h_logits[i] = static_cast<float>(i % 100) * 0.01f - 5.0f; 
    }
    
    // Внедряем гарантированный максимум
    h_logits[target_max_idx] = 500.0f;

    int cpu_token_id = -1;
    int gpu_token_id = -2;

    // 2. Вычисление Golden Reference на CPU
    cpu_argmax(h_logits.data(), &cpu_token_id, vocab_size);

    // 3. Выделение памяти на GPU
    float* d_logits;
    int* d_out_token_id;
    CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_token_id, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Запуск ядра
    launch_argmax_kernel(d_logits, d_out_token_id, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Проверка результата
    CUDA_CHECK(cudaMemcpy(&gpu_token_id, d_out_token_id, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cpu_token_id, target_max_idx) << "Сбой в логике CPU эталона!";
    ASSERT_EQ(gpu_token_id, cpu_token_id)   << "GPU Argmax выбрал неверный токен!";

    // 6. Очистка
    cudaFree(d_logits);
    cudaFree(d_out_token_id);
}