#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "kernels/sampling.cuh"
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
    const size_t vocab_size = 128256; 
    const int target_max_idx = 84042; 

    std::vector<float> h_logits(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 100) * 0.01f - 5.0f; 
    }
    h_logits[target_max_idx] = 500.0f;

    int cpu_token_id = -1;
    cpu_argmax(h_logits.data(), &cpu_token_id, vocab_size);

    CudaVector<float> d_logits(vocab_size); d_logits.upload(h_logits);
    CudaVector<int>   d_out_token_id(1);

    launch_argmax_kernel(d_logits, d_out_token_id, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_gpu_token(1);
    d_out_token_id.download(h_gpu_token);
    int gpu_token_id = h_gpu_token[0];

    ASSERT_EQ(cpu_token_id, target_max_idx) << "Сбой в логике CPU эталона!";
    ASSERT_EQ(gpu_token_id, cpu_token_id)   << "GPU Argmax выбрал неверный токен!";
}