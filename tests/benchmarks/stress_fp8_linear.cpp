// FP8 E4M3 GEMV stress suite: pathological shapes, vector-width boundaries,
// non-finite/degenerate inputs and rapid launches. Target-model shapes are
// covered in validation/test_fp8_linear.cpp.
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "common/fp8_reference.h"

using test_utils::Fp8Problem;
using test_utils::Fp8Device;
using test_utils::expect_fp8_matches_reference;
using test_utils::make_fp8_problem;
using test_utils::max_rel_error;
using test_utils::kFp8RelTolerance;
using test_utils::kFp8DenomFloor;

class Fp8Stress : public test_utils::CudaTest {};

// One block per row: tiny M leaves almost the whole GPU idle but must still
// be exact.
TEST_F(Fp8Stress, ExtremeNarrowOutputM4) {
    expect_fp8_matches_reference(4, 4096, 51);
}

TEST_F(Fp8Stress, SingleRowM1) {
    expect_fp8_matches_reference(1, 4096, 52);
}

// Extreme GQA-style aspect ratio: a KV-narrow output fed by the deepest
// reduction the engine launches (the MLP intermediate width).
TEST_F(Fp8Stress, NarrowOutputDeepReductionK14336) {
    expect_fp8_matches_reference(8, 14336, 60);
}

// K=16 is the smallest legal reduction (one uint4 weight chunk per row).
TEST_F(Fp8Stress, MinimumVectorWidthK16) {
    expect_fp8_matches_reference(64, 16, 53);
}

// K = 4112 = 257 * 16: not a multiple of the 4096-element block tile, so the
// last vector iteration is handled by a partial wave of threads.
TEST_F(Fp8Stress, NonTileAlignedK) {
    expect_fp8_matches_reference(64, 4112, 54);
    expect_fp8_matches_reference(64, 4112, 55, /*act_scale=*/0.1f);
}

// All-zero activations: every output must be exactly zero (no scale or
// epilogue term may leak in), and the dynamic scale kernel must hit its
// 1e-12 floor instead of producing 0, Inf or NaN.
TEST_F(Fp8Stress, ZeroActivations) {
    Fp8Problem p = make_fp8_problem(256, 4096, 56);
    std::fill(p.x.begin(), p.x.end(), 0.0f);
    Fp8Device dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_EQ(0.0f, std::fabs(out[i])) << "row " << i;

    CudaVector<float> d_scale(2);
    launch_quantize_per_token_kernel(dev.x, d_scale, p.K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> scale(2);
    d_scale.download(scale);
    EXPECT_TRUE(std::isfinite(scale[0]) && scale[0] > 0.0f) << scale[0];
    EXPECT_TRUE(std::isfinite(scale[1]) && scale[1] > 0.0f) << scale[1];
}

// A NaN activation must poison every row of the weight-only path.
TEST_F(Fp8Stress, NaNActivationPropagates) {
    Fp8Problem p = make_fp8_problem(128, 1024, 57);
    p.x[511] = std::nanf("");
    Fp8Device dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_TRUE(std::isnan(out[i])) << "row " << i << " not NaN: " << out[i];
}

TEST_F(Fp8Stress, InfActivationProducesNonFinite) {
    Fp8Problem p = make_fp8_problem(128, 1024, 58);
    p.x[0] = INFINITY;
    Fp8Device dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_FALSE(std::isfinite(out[i])) << "row " << i << " finite: " << out[i];
}

// Back-to-back launches without intermediate synchronization, alternating
// between two output buffers. The kernel is deterministic (fixed-order block
// reduction, no atomics), so both buffers must agree bitwise and match the
// reference.
TEST_F(Fp8Stress, RapidSequentialLaunches) {
    const Fp8Problem p = make_fp8_problem(1024, 4096, 59);
    Fp8Device a(p);
    Fp8Device b(p);

    for (int i = 0; i < 32; ++i)
        (i % 2 ? b : a).launch(p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out_a(p.M), out_b(p.M);
    a.y.download(out_a);
    b.y.download(out_b);
    EXPECT_LE(max_rel_error(out_a, p.ref, kFp8DenomFloor), kFp8RelTolerance);
    EXPECT_EQ(0, std::memcmp(out_a.data(), out_b.data(), p.M * sizeof(float)))
        << "deterministic kernel produced different results across launches";
}
