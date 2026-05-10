#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "attention.cuh"
#include "cpu_models.h"

class AttentionTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(AttentionTests, OnlineSoftmaxDecodingCorrectness) {
    const size_t q_heads = 32;
    const size_t kv_heads = 8;     // GQA ratio = 4
    const size_t head_dim = 128;   // Обязано быть <= ATTN_BLOCK_SIZE (128)
    const size_t max_seq_len = 200;
    const int pos = 85;            // В кэше лежит 86 токенов (от 0 до 85)

    // 1. Подготовка данных на Host
    std::vector<float> h_Q(q_heads * head_dim);
    std::vector<float> h_K_cache(kv_heads * max_seq_len * head_dim);
    std::vector<float> h_V_cache(kv_heads * max_seq_len * head_dim);
    std::vector<float> h_cpu_O(q_heads * head_dim, 0.0f);

    // Заполняем детерминированным разнородным мусором
    for (size_t i = 0; i < h_Q.size(); ++i) {
        h_Q[i] = static_cast<float>(i % 13) * 0.05f - 0.3f;
    }
    for (size_t i = 0; i < h_K_cache.size(); ++i) {
        h_K_cache[i] = static_cast<float>(i % 17) * 0.02f - 0.16f;
        h_V_cache[i] = static_cast<float>(i % 7)  * 0.05f - 0.1f;
    }

    // 2. Вычисление Golden Reference на CPU
    cpu_attention_decoding(
        h_Q.data(), h_K_cache.data(), h_V_cache.data(), h_cpu_O.data(),
        pos, q_heads, kv_heads, head_dim, max_seq_len
    );

    // 3. Выделение памяти на GPU
    float *d_Q, *d_K_cache, *d_V_cache, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, q_heads * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_cache, kv_heads * max_seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_cache, kv_heads * max_seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, q_heads * head_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_cache, h_K_cache.data(), h_K_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_cache, h_V_cache.data(), h_V_cache.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Запуск тестируемого ядра
    launch_attention_decoding_kernel(
        d_Q, d_K_cache, d_V_cache, d_O, pos, q_heads, kv_heads, head_dim, max_seq_len
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Проверка результатов
    std::vector<float> h_gpu_O(q_heads * head_dim);
    CUDA_CHECK(cudaMemcpy(h_gpu_O.data(), d_O, h_gpu_O.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Сравниваем выходные векторы O
    for (size_t i = 0; i < h_gpu_O.size(); ++i) {
        // Устанавливаем разумный эпсилон, так как экспоненты и суммация 
        // 86 элементов в разном порядке дают небольшую погрешность мантиссы
        ASSERT_NEAR(h_cpu_O[i], h_gpu_O[i], 1e-3f) 
            << "Ошибка в Attention Output по абсолютному индексу " << i;
    }

    // 6. Очистка
    cudaFree(d_Q);
    cudaFree(d_K_cache);
    cudaFree(d_V_cache);
    cudaFree(d_O);
}