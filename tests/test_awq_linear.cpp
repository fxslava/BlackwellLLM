// AWQ int4 GEMV kernel tests, split into two categories:
//   AWQValidation -- mathematical correctness on exact Qwen2.5-Coder-7B
//                    decode shapes against an FP64 CPU reference.
//   AWQStress     -- pathological shapes, ragged/malformed parameters,
//                    non-finite inputs and rapid sequential launches.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.h"
#include "test_utils.h"
#include "kernels/awq_linear.cuh"

using test_utils::AwqProblem;
using test_utils::make_awq_problem;
using test_utils::make_awq_constant_problem;
using test_utils::max_rel_error;

namespace {

// FP16 scales bound the achievable precision; 4.2e-3 was the worst case
// observed across shapes, 5e-3 leaves headroom without masking real bugs.
constexpr float kRelTolerance = 5e-3f;

struct AwqDevice {
    CudaVector<uint32_t> qweight;
    CudaVector<uint32_t> qzeros;
    CudaVector<half> scales;
    CudaVector<float> x;
    CudaVector<float> y;

    explicit AwqDevice(const AwqProblem& p)
        : qweight(p.qweight.size()),
          qzeros(p.qzeros.size()),
          scales(p.scales.size()),
          x(p.x.size()),
          y(p.out_features) {
        qweight.upload(p.qweight);
        qzeros.upload(p.qzeros);
        scales.upload(p.scales);
        x.upload(p.x);
        // Poison the output so a kernel that silently writes nothing fails loudly.
        CUDA_CHECK(cudaMemset(y.d_ptr, 0xCC, p.out_features * sizeof(float)));
    }

    void launch(const AwqProblem& p) {
        launch_awq_gemv_kernel(qweight, scales, qzeros, x, y,
                               p.out_features, p.in_features, p.group_size);
    }

    std::vector<float> run(const AwqProblem& p) {
        launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(p.out_features);
        y.download(out);
        return out;
    }
};

void expect_awq_matches_reference(int in_features, int out_features,
                                  int group_size, unsigned seed) {
    const AwqProblem p = make_awq_problem(in_features, out_features, group_size, seed);
    AwqDevice dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref);
    EXPECT_LE(max_rel, kRelTolerance)
        << "AWQ GEMV mismatch: IC=" << in_features << " OC=" << out_features
        << " group_size=" << group_size << " max_rel=" << max_rel;
}

}  // namespace

// ============================================================================
// Category A: Validation (Qwen2.5-Coder-7B: hidden=3584, intermediate=18944,
// GQA KV dim = 4 heads * 128 = 512, group_size=128)
// ============================================================================

TEST(AWQValidation, QwenAttentionQProj) {
    expect_awq_matches_reference(3584, 3584, 128, 11);
}

// OC=512 -> only 64 packed columns: the launcher distributes the reduction
// dimension across blocks (split-K), so GQA K/V projections take the
// atomic-accumulation path.
TEST(AWQValidation, QwenGqaKvProjSplitK) {
    expect_awq_matches_reference(3584, 512, 128, 12);
}

TEST(AWQValidation, QwenMlpGateUpProj) {
    expect_awq_matches_reference(3584, 18944, 128, 13);
}

TEST(AWQValidation, QwenMlpDownProj) {
    expect_awq_matches_reference(18944, 3584, 128, 14);
}

// ============================================================================
// Category B: Stress (edge shapes, malformed parameters, hardware limits)
// ============================================================================

// Minimum legal width: a single packed column, reductions maximally
// distributed across split-K blocks.
TEST(AWQStress, ExtremeNarrowOutputOC8) {
    expect_awq_matches_reference(8192, 8, 128, 21);
}

TEST(AWQStress, RaggedTailGroup) {
    expect_awq_matches_reference(200, 64, 64, 22);   // IC % gs = 8
    expect_awq_matches_reference(1000, 128, 128, 23); // IC % gs = 104
}

TEST(AWQStress, GroupLargerThanInFeatures) {
    expect_awq_matches_reference(96, 128, 128, 24);
}

// group_size <= 0 must fall back to per-channel quantization (one group
// spanning all of in_features) instead of crashing or dividing by zero.
TEST(AWQStress, MalformedGroupSizeFallsBackToPerChannel) {
    expect_awq_matches_reference(4096, 256, -1, 25);
    expect_awq_matches_reference(512, 128, 0, 26);
}

// Vocab-sized projection (Qwen lm_head): exercises grid sizing and 64-bit
// indexing at the largest shape the engine launches. Constant-pattern
// weights keep the CPU reference O(IC) instead of O(IC*OC).
TEST(AWQStress, QwenVocabSizedLmHead) {
    const AwqProblem p = make_awq_constant_problem(3584, 152064, 128, 27);
    AwqDevice dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, kRelTolerance) << "lm_head-sized GEMV mismatch, max_rel=" << max_rel;
}

// A NaN activation must poison every output element (each output is a dot
// product over all of in_features); no value may pass through untouched.
TEST(AWQStress, NaNActivationPropagates) {
    AwqProblem p = make_awq_problem(256, 64, 128, 28);
    p.x[123] = std::nanf("");
    AwqDevice dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_TRUE(std::isnan(out[i])) << "output " << i << " not NaN: " << out[i];
}

TEST(AWQStress, InfActivationProducesNonFinite) {
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
TEST(AWQStress, RapidSequentialSplitKLaunches) {
    const AwqProblem p = make_awq_problem(3584, 512, 128, 30);
    AwqDevice a(p);
    AwqDevice b(p);

    for (int i = 0; i < 32; ++i)
        (i % 2 ? b : a).launch(p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(p.out_features);
    a.y.download(out);
    EXPECT_LE(max_rel_error(out, p.ref), kRelTolerance) << "buffer A corrupted";
    b.y.download(out);
    EXPECT_LE(max_rel_error(out, p.ref), kRelTolerance) << "buffer B corrupted";

    // Repeated rounds into one buffer: each result must be independently correct.
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 16; ++i) a.launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        a.y.download(out);
        EXPECT_LE(max_rel_error(out, p.ref), kRelTolerance) << "round " << round;
    }
}
