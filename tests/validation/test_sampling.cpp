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

// Replays the exact greedy-decode scenario observed in the integration test
// (BOS step of Llama-3-8B): vocab 128256 (not a power of two, not a multiple
// of the 512-thread block), winner at token 128006 with realistic tight
// margins over the runner-ups, and BOS (128000) far down the distribution —
// the token the integration test once wrongly reported as the argmax.
TEST_F(SamplingValidation, KnownLogitsLlamaDecodeStep) {
    const size_t vocab = 128256;
    std::vector<float> h_logits(vocab);
    for (size_t i = 0; i < vocab; ++i)
        h_logits[i] = -8.0f + 0.0001f * static_cast<float>(i % 1000);

    // Top of the real distribution from logits_comparison.csv.
    h_logits[128006] = 11.6819f;   // expected winner
    h_logits[78191]  = 11.05f;     // close runner-up in a different block stride
    h_logits[9125]   = 10.73f;
    h_logits[128000] = 1.74295f;   // BOS

    CudaVector<float> d_logits(vocab); d_logits.upload(h_logits);
    CudaVector<int>   d_token(1);

    launch_argmax_kernel(d_logits, d_token, vocab);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> gpu_token(1);
    d_token.download(gpu_token);
    EXPECT_EQ(gpu_token[0], 128006);
    EXPECT_NE(gpu_token[0], 128000) << "argmax echoed the BOS/input token";
}

// The last block stride covers only indices [128000, 128256), so half the
// 512-thread block has no element there. Sweeping the peak across the final
// two strides makes every thread, lane and warp the winner at least once,
// covering the grid-coverage and both reduction stages exhaustively.
TEST_F(SamplingValidation, PeakSweepAcrossFinalBlockStrides) {
    const size_t vocab = 128256;
    const float  kPeak = 50.0f;
    std::vector<float> h_logits =
        test_utils::random_uniform(vocab, /*seed=*/125, -5.0f, 5.0f);

    CudaVector<float> d_logits(vocab); d_logits.upload(h_logits);
    CudaVector<int>   d_token(1);

    for (size_t peak = vocab - 1024; peak < vocab; ++peak) {
        const float original = h_logits[peak];
        CUDA_CHECK(cudaMemcpy(d_logits.d_ptr + peak, &kPeak, sizeof(float),
                              cudaMemcpyHostToDevice));

        launch_argmax_kernel(d_logits, d_token, vocab);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> gpu_token(1);
        d_token.download(gpu_token);
        ASSERT_EQ(gpu_token[0], static_cast<int>(peak))
            << "GPU argmax missed the peak at index " << peak;

        CUDA_CHECK(cudaMemcpy(d_logits.d_ptr + peak, &original, sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}
