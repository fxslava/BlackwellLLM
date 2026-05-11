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
    const size_t kv_heads = 8;     
    const size_t head_dim = 128;   
    const size_t max_seq_len = 200;
    const int pos = 85;            

    std::vector<float> h_Q(q_heads * head_dim);
    std::vector<float> h_K_cache(kv_heads * max_seq_len * head_dim);
    std::vector<float> h_V_cache(kv_heads * max_seq_len * head_dim);
    std::vector<float> h_cpu_O(q_heads * head_dim, 0.0f);

    for (size_t i = 0; i < h_Q.size(); ++i) {
        h_Q[i] = static_cast<float>(i % 13) * 0.05f - 0.3f;
    }
    for (size_t i = 0; i < h_K_cache.size(); ++i) {
        h_K_cache[i] = static_cast<float>(i % 17) * 0.02f - 0.16f;
        h_V_cache[i] = static_cast<float>(i % 7)  * 0.05f - 0.1f;
    }

    cpu_attention_decoding(
        h_Q.data(), h_K_cache.data(), h_V_cache.data(), h_cpu_O.data(),
        pos, q_heads, kv_heads, head_dim, max_seq_len
    );

    // Элегантная инициализация через RAII
    CudaVector<float> d_Q(h_Q.size());             d_Q.upload(h_Q);
    CudaVector<float> d_K_cache(h_K_cache.size()); d_K_cache.upload(h_K_cache);
    CudaVector<float> d_V_cache(h_V_cache.size()); d_V_cache.upload(h_V_cache);
    CudaVector<float> d_O(h_cpu_O.size());

    launch_attention_decoding_kernel(
        d_Q, d_K_cache, d_V_cache, d_O, pos, q_heads, kv_heads, head_dim, max_seq_len
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_O(h_cpu_O.size());
    d_O.download(h_gpu_O);

    for (size_t i = 0; i < h_gpu_O.size(); ++i) {
        ASSERT_NEAR(h_cpu_O[i], h_gpu_O[i], 1e-3f) 
            << "Ошибка в Attention Output по абсолютному индексу " << i;
    }
}