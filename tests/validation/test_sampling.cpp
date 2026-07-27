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

// ---------------------------------------------------------------------------
// Repetition penalty (the degenerate-loop brake the live translator's decode
// loop applies before sampling). Contract: sampling.cuh.
// ---------------------------------------------------------------------------

// Sign-aware shaping: a POSITIVE logit is divided by the penalty, a NEGATIVE one
// multiplied (dividing a negative would make it LARGER, i.e. reward the repeat).
// Untouched ids must be bit-identical.
TEST_F(SamplingValidation, RepetitionPenaltyIsSignAware) {
    const size_t vocab = 1024;
    const float  penalty = 1.15f;
    std::vector<float> h_logits =
        test_utils::random_uniform(vocab, /*seed=*/131, -5.0f, 5.0f);
    h_logits[10] = 4.0f;    // positive -> divided
    h_logits[20] = -4.0f;   // negative -> multiplied
    h_logits[30] = 0.0f;    // the v > 0 boundary: takes the multiply branch

    const std::vector<int> h_ids{10, 20, 30};
    CudaVector<float> d_logits(vocab); d_logits.upload(h_logits);
    CudaVector<int>   d_ids(h_ids.size()); d_ids.upload(h_ids);

    launch_repetition_penalty_kernel(d_logits, vocab, d_ids,
                                     static_cast<int>(h_ids.size()), penalty);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(vocab);
    d_logits.download(out);
    EXPECT_FLOAT_EQ(out[10], 4.0f / penalty);
    EXPECT_FLOAT_EQ(out[20], -4.0f * penalty);
    EXPECT_FLOAT_EQ(out[30], 0.0f);
    for (size_t i = 0; i < vocab; ++i) {
        if (i == 10 || i == 20 || i == 30) continue;
        ASSERT_FLOAT_EQ(out[i], h_logits[i]) << "untouched logit " << i << " moved";
    }
}

// A rolling window legitimately holds the SAME id many times (that is what a
// degenerate loop looks like). The penalty must still be applied exactly ONCE
// per distinct id — compounding it would be both non-deterministic (a
// read-modify-write race) and far too aggressive.
TEST_F(SamplingValidation, RepetitionPenaltyAppliedOncePerDistinctId) {
    const size_t vocab = 1024;
    const float  penalty = 1.15f;
    std::vector<float> h_logits(vocab, 1.0f);
    h_logits[7] = 8.0f;

    // id 7 repeated 40 times, as an exact phrase loop would produce.
    const std::vector<int> h_ids(40, 7);
    CudaVector<float> d_logits(vocab); d_logits.upload(h_logits);
    CudaVector<int>   d_ids(h_ids.size()); d_ids.upload(h_ids);

    launch_repetition_penalty_kernel(d_logits, vocab, d_ids,
                                     static_cast<int>(h_ids.size()), penalty);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(vocab);
    d_logits.download(out);
    EXPECT_FLOAT_EQ(out[7], 8.0f / penalty)
        << "penalty compounded across duplicate window entries";
}

// The documented no-ops (penalty <= 1, empty window) and the out-of-range-id
// guard must all leave the logits untouched rather than corrupt memory.
TEST_F(SamplingValidation, RepetitionPenaltyNoOpsAndRangeGuard) {
    const size_t vocab = 256;
    std::vector<float> h_logits =
        test_utils::random_uniform(vocab, /*seed=*/132, -5.0f, 5.0f);
    // -1 is the window's "unwritten slot" fill; the large id is out of vocab.
    const std::vector<int> h_ids{-1, 5, 999999};

    CudaVector<float> d_logits(vocab); d_logits.upload(h_logits);
    CudaVector<int>   d_ids(h_ids.size()); d_ids.upload(h_ids);

    // penalty == 1.0 -> disabled.
    launch_repetition_penalty_kernel(d_logits, vocab, d_ids, 3, 1.0f);
    // empty window -> nothing to penalize.
    launch_repetition_penalty_kernel(d_logits, vocab, d_ids, 0, 1.15f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(vocab);
    d_logits.download(out);
    for (size_t i = 0; i < vocab; ++i)
        ASSERT_FLOAT_EQ(out[i], h_logits[i]) << "no-op call modified logit " << i;

    // Now a live call: only the in-range id 5 may move; -1 and 999999 are skipped
    // (an unguarded kernel would write outside the 256-element buffer here).
    launch_repetition_penalty_kernel(d_logits, vocab, d_ids, 3, 1.15f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_logits.download(out);
    const float expect5 =
        h_logits[5] > 0.0f ? h_logits[5] / 1.15f : h_logits[5] * 1.15f;
    EXPECT_FLOAT_EQ(out[5], expect5);
    for (size_t i = 0; i < vocab; ++i) {
        if (i == 5) continue;
        ASSERT_FLOAT_EQ(out[i], h_logits[i]) << "out-of-range id touched logit " << i;
    }
}

// Realistic geometry: the live window (64 ids) against the Llama vocab, mixed
// with untouched neighbours, so the multi-block launch path is exercised.
TEST_F(SamplingValidation, RepetitionPenaltyLlamaVocabFullWindow) {
    const size_t vocab = 128256;
    const float  penalty = 1.15f;
    std::vector<float> h_logits =
        test_utils::random_uniform(vocab, /*seed=*/133, -5.0f, 5.0f);

    std::vector<int> h_ids(64);
    for (int i = 0; i < 64; ++i) h_ids[i] = 1000 + i * 977;   // spread across blocks

    CudaVector<float> d_logits(vocab); d_logits.upload(h_logits);
    CudaVector<int>   d_ids(h_ids.size()); d_ids.upload(h_ids);

    launch_repetition_penalty_kernel(d_logits, vocab, d_ids, 64, penalty);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(vocab);
    d_logits.download(out);
    std::vector<char> penalized(vocab, 0);
    for (const int id : h_ids) penalized[static_cast<size_t>(id)] = 1;
    for (size_t i = 0; i < vocab; ++i) {
        const float want = penalized[i]
            ? (h_logits[i] > 0.0f ? h_logits[i] / penalty : h_logits[i] * penalty)
            : h_logits[i];
        ASSERT_FLOAT_EQ(out[i], want) << "logit " << i << " wrong after penalty";
    }
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
