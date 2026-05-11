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
    const int pos = 42; 
    const float theta = 500000.0f;

    std::vector<float> h_Q(q_heads * head_dim);
    std::vector<float> h_K(kv_heads * head_dim);
    std::vector<float> h_V(kv_heads * head_dim);
    std::vector<float> h_K_cache(kv_heads * max_seq_len * head_dim, 0.0f);
    std::vector<float> h_V_cache(kv_heads * max_seq_len * head_dim, 0.0f);

    for (size_t i = 0; i < h_Q.size(); ++i) h_Q[i] = static_cast<float>(i % 10) * 0.1f;
    for (size_t i = 0; i < h_K.size(); ++i) {
        h_K[i] = static_cast<float>(i % 7) * 0.1f;
        h_V[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    }

    std::vector<float> h_cpu_Q = h_Q;
    std::vector<float> h_cpu_K = h_K;
    std::vector<float> h_cpu_K_cache = h_K_cache;
    std::vector<float> h_cpu_V_cache = h_V_cache;

    cpu_fused_rope_kv_append(h_cpu_Q.data(), h_cpu_K.data(), h_V.data(),
                             h_cpu_K_cache.data(), h_cpu_V_cache.data(),
                             pos, q_heads, kv_heads, head_dim, max_seq_len, theta);

    CudaVector<float> d_Q(h_Q.size());             d_Q.upload(h_Q);
    CudaVector<float> d_K(h_K.size());             d_K.upload(h_K);
    CudaVector<float> d_V(h_V.size());             d_V.upload(h_V);
    CudaVector<float> d_K_cache(h_K_cache.size()); d_K_cache.upload(h_K_cache);
    CudaVector<float> d_V_cache(h_V_cache.size()); d_V_cache.upload(h_V_cache);

    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, d_K_cache, d_V_cache,
                                pos, q_heads, kv_heads, head_dim, max_seq_len, theta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_Q(h_Q.size());
    std::vector<float> h_gpu_K_cache(h_K_cache.size());
    std::vector<float> h_gpu_V_cache(h_V_cache.size());

    d_Q.download(h_gpu_Q);
    d_K_cache.download(h_gpu_K_cache);
    d_V_cache.download(h_gpu_V_cache);

    for (size_t i = 0; i < h_gpu_Q.size(); ++i) {
        ASSERT_NEAR(h_cpu_Q[i], h_gpu_Q[i], 1e-4f) << "Расхождение в Q по индексу " << i;
    }

    for (size_t h = 0; h < kv_heads; ++h) {
        for (size_t i = 0; i < head_dim; ++i) {
            size_t idx = (h * max_seq_len + pos) * head_dim + i;
            ASSERT_NEAR(h_cpu_K_cache[idx], h_gpu_K_cache[idx], 1e-4f) << "Расхождение в K_cache";
            ASSERT_NEAR(h_cpu_V_cache[idx], h_gpu_V_cache[idx], 1e-4f) << "Расхождение в V_cache";
        }
    }
}