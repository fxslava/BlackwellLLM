#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "embedding.cuh"
#include "cpu_models.h"

// 1. АВТОМАТИЧЕСКАЯ ПОЧИНКА КИРИЛЛИЦЫ ДЛЯ WINDOWС
#ifdef _WIN32
#include <windows.h>
static struct ConsoleInitializer {
    ConsoleInitializer() {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
} console_init;
#endif

// 2. ИСПОЛЬЗУЕМ ФИКСТУРУ ДЛЯ ГАРАНТИРОВАННОЙ ОЧИСТКИ И ПРОВЕРОК
class KernelTests : public ::testing::Test {
protected:
    void TearDown() override {
        // Гарантирует, что даже при падении теста мы увидим зависшие ошибки CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[КРИТИЧЕСКАЯ ОШИБКА CUDA в TearDown]: " 
                      << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(KernelTests, EmbeddingLookupCorrectness) {
    const size_t vocab_size = 32000; 
    const size_t hidden_dim = 4096;  
    const size_t seq_len = 8;        

    // Подготовка данных на Host
    std::vector<float> h_embed_table(vocab_size * hidden_dim);
    for (size_t i = 0; i < h_embed_table.size(); ++i) {
        h_embed_table[i] = static_cast<float>(i % 100) * 0.01f;
    }

    std::vector<int> h_tokens = {12, 450, 0, 31999, 5, 88, 42, 10};
    std::vector<float> h_cpu_output(seq_len * hidden_dim, 0.0f);

    // Вычисление на CPU (Golden Reference)
    cpu_embedding_lookup(h_tokens.data(), h_embed_table.data(), h_cpu_output.data(), seq_len, hidden_dim);

    // Выделение памяти и пересылка на GPU
    float* d_embed_table = nullptr;
    int* d_tokens = nullptr;
    float* d_gpu_output = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_embed_table, vocab_size * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tokens, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpu_output, seq_len * hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_embed_table, h_embed_table.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));

    // Запуск ядра
    launch_embedding_kernel(d_tokens, d_embed_table, d_gpu_output, seq_len, hidden_dim);

    // 3. ЖЕЛЕЗОБЕТОННАЯ ПРОВЕРКА ЗАПУСКА ЯДРА (Именно она покажет, почему результат 0)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копирование результата обратно и проверка
    std::vector<float> h_gpu_output(seq_len * hidden_dim);
    CUDA_CHECK(cudaMemcpy(h_gpu_output.data(), d_gpu_output, seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_gpu_output.size(); ++i) {
        ASSERT_NEAR(h_cpu_output[i], h_gpu_output[i], 1e-5f) 
            << "Ошибка по абсолютному индексу " << i << "\n"
            << "Ожидалось (CPU): " << h_cpu_output[i] << ", Получено (GPU): " << h_gpu_output[i];
    }

    // Очистка
    cudaFree(d_embed_table);
    cudaFree(d_tokens);
    cudaFree(d_gpu_output);
}