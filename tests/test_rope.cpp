#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "rope.cuh"
#include "cpu_models.h"

class RopeTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(RopeTests, FusedRopeKvAppendCorrectness) {
    const size_t q_heads = 32;
    const size_t kv_heads = 8;
    const size_t head_dim = 128;
    const size_t max_seq_len = 100;
    const int pos = 42; // Текущая позиция токена
    const float theta = 500000.0f;

    // 1. Подготовка данных на Host
    std::vector<float> h_Q(q_heads * head_dim);
    std::vector<float> h_K(kv_heads * head_dim);
    std::vector<float> h_V(kv_heads * head_dim);
    
    // Глобальные буферы кэша
    std::vector<float> h_K_cache(kv_heads * max_seq_len * head_dim, 0.0f);
    std::vector<float> h_V_cache(kv_heads * max_seq_len * head_dim, 0.0f);

    // Заполнение исходных векторов
    for (size_t i = 0; i < h_Q.size(); ++i) h_Q[i] = static_cast<float>(i % 10) * 0.1f;
    for (size_t i = 0; i < h_K.size(); ++i) {
        h_K[i] = static_cast<float>(i % 7) * 0.1f;
        h_V[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    }

    // Копии для CPU
    std::vector<float> h_cpu_Q = h_Q;
    std::vector<float> h_cpu_K = h_K;
    std::vector<float> h_cpu_K_cache = h_K_cache;
    std::vector<float> h_cpu_V_cache = h_V_cache;

    // 2. Вычисление Golden Reference на CPU
    cpu_fused_rope_kv_append(h_cpu_Q.data(), h_cpu_K.data(), h_V.data(),
                             h_cpu_K_cache.data(), h_cpu_V_cache.data(),
                             pos, q_heads, kv_heads, head_dim, max_seq_len, theta);

    // 3. Выделение памяти на GPU
    float *d_Q, *d_K, *d_V, *d_K_cache, *d_V_cache;
    CUDA_CHECK(cudaMalloc(&d_Q, q_heads * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, kv_heads * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, kv_heads * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_cache, kv_heads * max_seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_cache, kv_heads * max_seq_len * head_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_cache, h_K_cache.data(), h_K_cache.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_cache, h_V_cache.data(), h_V_cache.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Запуск ядра
    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, d_K_cache, d_V_cache,
                                pos, q_heads, kv_heads, head_dim, max_seq_len, theta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Проверка результатов
    std::vector<float> h_gpu_Q(h_Q.size());
    std::vector<float> h_gpu_K_cache(h_K_cache.size());
    std::vector<float> h_gpu_V_cache(h_V_cache.size());

    CUDA_CHECK(cudaMemcpy(h_gpu_Q.data(), d_Q, h_Q.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_K_cache.data(), d_K_cache, h_K_cache.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_V_cache.data(), d_V_cache, h_V_cache.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Проверка Q
    for (size_t i = 0; i < h_gpu_Q.size(); ++i) {
        ASSERT_NEAR(h_cpu_Q[i], h_gpu_Q[i], 1e-4f) << "Расхождение в Q по индексу " << i;
    }

    // Проверка слота в K_cache
    size_t cache_slot_size = kv_heads * head_dim;
    for (size_t h = 0; h < kv_heads; ++h) {
        for (size_t i = 0; i < head_dim; ++i) {
            size_t idx = (h * max_seq_len + pos) * head_dim + i;
            ASSERT_NEAR(h_cpu_K_cache[idx], h_gpu_K_cache[idx], 1e-4f) << "Расхождение в K_cache";
            ASSERT_NEAR(h_cpu_V_cache[idx], h_gpu_V_cache[idx], 1e-4f) << "Расхождение в V_cache";
        }
    }

    // 6. Очистка
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_K_cache); cudaFree(d_V_cache);
}