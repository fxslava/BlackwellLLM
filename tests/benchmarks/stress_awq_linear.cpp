// AWQ int4 GEMV stress suite: pathological shapes, ragged/malformed
// parameters, non-finite inputs and rapid sequential launches. Target-model
// shapes are covered in validation/test_awq_linear.cpp.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "common/awq_reference.h"

using test_utils::AwqProblem;
using test_utils::AwqDevice;
using test_utils::expect_awq_matches_reference;
using test_utils::make_awq_problem;
using test_utils::max_rel_error;
using test_utils::kAwqRelTolerance;

class AwqStress : public test_utils::CudaTest {};

// Minimum legal width: a single packed column, reductions maximally
// distributed across split-K blocks.
TEST_F(AwqStress, ExtremeNarrowOutputOC8) {
    expect_awq_matches_reference(8192, 8, 128, 21);
}

// Extreme GQA split-K: the Qwen MLP reduction depth (18944) feeding a single
// packed column, then the GQA KV width (512) fed by the same deep reduction.
// Both maximize the number of split-K partial sums per output element.
TEST_F(AwqStress, ExtremeGqaSplitKDeepReduction) {
    expect_awq_matches_reference(18944, 8, 128, 31);
    expect_awq_matches_reference(18944, 512, 128, 32);
}

TEST_F(AwqStress, RaggedTailGroup) {
    expect_awq_matches_reference(200, 64, 64, 22);    // IC % gs = 8
    expect_awq_matches_reference(1000, 128, 128, 23); // IC % gs = 104
}

// Non-aligned reduction: IC=333 is odd and far from any tile multiple, with
// a group size that does not divide it (333 % 96 = 45).
TEST_F(AwqStress, NonAlignedInFeaturesAndGroupSize) {
    expect_awq_matches_reference(333, 64, 96, 33);
}

TEST_F(AwqStress, GroupLargerThanInFeatures) {
    expect_awq_matches_reference(96, 128, 128, 24);
}

// group_size <= 0 must fall back to per-channel quantization (one group
// spanning all of in_features) instead of crashing or dividing by zero.
TEST_F(AwqStress, MalformedGroupSizeFallsBackToPerChannel) {
    expect_awq_matches_reference(4096, 256, -1, 25);
    expect_awq_matches_reference(512, 128, 0, 26);
}

// A NaN activation must poison every output element (each output is a dot
// product over all of in_features); no value may pass through untouched.
TEST_F(AwqStress, NaNActivationPropagates) {
    AwqProblem p = make_awq_problem(256, 64, 128, 28);
    p.x[123] = std::nanf("");
    AwqDevice dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_TRUE(std::isnan(out[i])) << "output " << i << " not NaN: " << out[i];
}

TEST_F(AwqStress, InfActivationProducesNonFinite) {
    AwqProblem p = make_awq_problem(256, 64, 128, 29);
    p.x[7] = INFINITY;
    AwqDevice dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_FALSE(std::isfinite(out[i])) << "output " << i << " finite: " << out[i];
}

// Back-to-back launches without intermediate synchronization. The split-K
// path interleaves cudaMemsetAsync with atomicAdd kernels on the default
// stream; any ordering violation or leftover partial sum shows up as a
// wrong result in one of the rounds.
TEST_F(AwqStress, RapidSequentialSplitKLaunches) {
    const AwqProblem p = make_awq_problem(3584, 512, 128, 30);
    AwqDevice a(p);
    AwqDevice b(p);

    for (int i = 0; i < 32; ++i)
        (i % 2 ? b : a).launch(p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(p.out_features);
    a.y.download(out);
    EXPECT_LE(max_rel_error(out, p.ref), kAwqRelTolerance) << "buffer A corrupted";
    b.y.download(out);
    EXPECT_LE(max_rel_error(out, p.ref), kAwqRelTolerance) << "buffer B corrupted";

    // Repeated rounds into one buffer: each result must be independently correct.
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 16; ++i) a.launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        a.y.download(out);
        EXPECT_LE(max_rel_error(out, p.ref), kAwqRelTolerance) << "round " << round;
    }
}
