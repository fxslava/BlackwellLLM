// Greedy sampling (argmax over logits) vs the CPU reference at both target
// vocab sizes.
#include <gtest/gtest.h>
#include <vector>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/sampling.cuh"

namespace {

class SamplingValidation : public test_utils::CudaTest {};

void expect_argmax_finds_peak(size_t vocab_size, int target_idx, unsigned seed) {
    std::vector<float> h_logits =
        test_utils::random_uniform(vocab_size, seed, -5.0f, 5.0f);
    h_logits[target_idx] = 500.0f;  // unambiguous global maximum

    int ref_token = -1;
    cpu_argmax(h_logits.data(), &ref_token, vocab_size);
    ASSERT_EQ(ref_token, target_idx) << "CPU reference logic is broken";

    CudaVector<float> d_logits(vocab_size); d_logits.upload(h_logits);
    CudaVector<int>   d_token(1);

    launch_argmax_kernel(d_logits, d_token, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> gpu_token(1);
    d_token.download(gpu_token);
    EXPECT_EQ(gpu_token[0], ref_token) << "GPU argmax picked the wrong token";
}

}  // namespace

TEST_F(SamplingValidation, LlamaVocabArgmax) {
    expect_argmax_finds_peak(128256, 84042, 121);
}

TEST_F(SamplingValidation, QwenVocabArgmax) {
    expect_argmax_finds_peak(152064, 151000, 122);
}

// Peaks at the very first and very last index are the classic off-by-one
// blind spots of block-reduction argmax implementations.
TEST_F(SamplingValidation, BoundaryIndexPeaks) {
    expect_argmax_finds_peak(128256, 0, 123);
    expect_argmax_finds_peak(128256, 128255, 124);
}
