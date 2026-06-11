// FP8 E4M3 GEMV kernel tests, split into two categories:
//   FP8Validation -- mathematical correctness on exact Llama 3 8B decode
//                    shapes (row-wise weight scales, weight-only and
//                    quantized-activation paths, per-token scale kernel,
//                    residual accumulation) against FP32/FP64 CPU math.
//   FP8Stress     -- pathological shapes, vector-width boundaries,
//                    non-finite/degenerate inputs and rapid launches.
//
// Kernel contract notes (see fp8_linear.cu): K must be a multiple of 16
// (uint4/float4 vectorized loads), and token_scale is read unconditionally,
// so every launch here passes a valid 2-float token_scale buffer the way the
// engine does -- never nullptr.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "common.h"
#include "test_utils.h"
#include "kernels/fp8_linear.cuh"

using test_utils::Fp8Problem;
using test_utils::make_fp8_problem;
using test_utils::max_rel_error;

namespace {

// The reference replicates the kernel's quantization exactly (E4M3 RNE,
// bf16-rounded scales), so the only error source is FP32 accumulation order.
constexpr float kRelTolerance = 1e-3f;
constexpr float kDenomFloor = 1e-2f;

struct Fp8Device {
    CudaVector<uint8_t> weights;
    CudaVector<__nv_bfloat16> w_scales;
    CudaVector<__nv_bfloat16> input_scale;
    CudaVector<float> token_scale;
    CudaVector<float> x;
    CudaVector<float> y;

    explicit Fp8Device(const Fp8Problem& p)
        : weights(p.weights.size()),
          w_scales(p.w_scales.size()),
          input_scale(1),
          token_scale(2),
          x(p.x.size()),
          y(p.M) {
        weights.upload(p.weights);
        w_scales.upload(p.w_scales);
        x.upload(p.x);
        token_scale.upload({1.0f, 1.0f});
        if (p.act_scale > 0.0f)
            input_scale.upload({__float2bfloat16(p.act_scale)});
        CUDA_CHECK(cudaMemset(y.d_ptr, 0xCC, p.M * sizeof(float)));
    }

    const __nv_bfloat16* input_scale_arg(const Fp8Problem& p) const {
        return p.act_scale > 0.0f ? input_scale.d_ptr : nullptr;
    }

    void launch(const Fp8Problem& p) {
        launch_fp8_gemv_kernel(weights, x, w_scales, input_scale_arg(p),
                               token_scale, y, p.M, p.K, 1);
    }

    std::vector<float> run(const Fp8Problem& p) {
        launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(p.M);
        y.download(out);
        return out;
    }
};

void expect_fp8_matches_reference(size_t M, size_t K, unsigned seed,
                                  float act_scale = 0.0f) {
    const Fp8Problem p = make_fp8_problem(M, K, seed, act_scale);
    Fp8Device dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, kDenomFloor);
    EXPECT_LE(max_rel, kRelTolerance)
        << "FP8 GEMV mismatch: M=" << M << " K=" << K
        << " act_scale=" << act_scale << " max_rel=" << max_rel;
}

}  // namespace

// ============================================================================
// Category A: Validation (Llama 3 8B: hidden=4096, intermediate=14336,
// GQA KV dim = 8 heads * 128 = 1024)
// ============================================================================

TEST(FP8Validation, LlamaAttnOutProj) {
    expect_fp8_matches_reference(4096, 4096, 41);
}

TEST(FP8Validation, LlamaGqaKvProj) {
    expect_fp8_matches_reference(1024, 4096, 42);
}

TEST(FP8Validation, LlamaMlpUpProj) {
    expect_fp8_matches_reference(14336, 4096, 43);
}

TEST(FP8Validation, LlamaMlpDownProj) {
    expect_fp8_matches_reference(4096, 14336, 44);
}

// Per-tensor activation quantization: activations are converted to E4M3 in
// x/scale domain on the fly and the epilogue multiplies the scale back.
TEST(FP8Validation, QuantizedActivationPath) {
    expect_fp8_matches_reference(4096, 4096, 45, /*act_scale=*/0.02f);
    expect_fp8_matches_reference(1024, 4096, 46, /*act_scale=*/0.5f);
}

// Per-token dynamic scale kernel: scale = max(|x|, 1e-12) / 448 plus its
// reciprocal, both finite. fast-math division on the device allows a few ulp.
TEST(FP8Validation, PerTokenScaleComputation) {
    const size_t K = 4096;
    const std::vector<float> h_x = test_utils::random_normal(K, 47, 3.0f);

    CudaVector<float> d_x(K);
    d_x.upload(h_x);
    CudaVector<float> d_scale(2);

    launch_quantize_per_token_kernel(d_x, d_scale, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> scale(2);
    d_scale.download(scale);

    float max_abs = 0.0f;
    for (float v : h_x) max_abs = std::max(max_abs, std::fabs(v));
    const float expected = std::max(max_abs, 1e-12f) / 448.0f;

    EXPECT_NEAR(scale[0], expected, expected * 1e-5f);
    EXPECT_NEAR(scale[0] * scale[1], 1.0f, 1e-5f);
}

// Residual variant: Y_accum += gemv result, accumulation stays in FP32.
TEST(FP8Validation, ResidualAccumulation) {
    const Fp8Problem p = make_fp8_problem(1024, 4096, 48);
    const std::vector<float> initial = test_utils::random_normal(p.M, 49, 10.0f);

    std::vector<float> expected(p.M);
    for (size_t i = 0; i < p.M; ++i) expected[i] = initial[i] + p.ref[i];

    Fp8Device dev(p);
    dev.y.upload(initial);
    launch_fp8_gemv_residual_kernel(dev.weights, dev.x, dev.w_scales,
                                    nullptr, dev.token_scale, dev.y,
                                    p.M, p.K, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(p.M);
    dev.y.download(out);
    EXPECT_LE(max_rel_error(out, expected, kDenomFloor), kRelTolerance);
}

// ============================================================================
// Category B: Stress (edge shapes, degenerate inputs, hardware limits)
// ============================================================================

// One block per row: tiny M leaves almost the whole GPU idle but must still
// be exact.
TEST(FP8Stress, ExtremeNarrowOutputM4) {
    expect_fp8_matches_reference(4, 4096, 51);
}

TEST(FP8Stress, SingleRowM1) {
    expect_fp8_matches_reference(1, 4096, 52);
}

// K=16 is the smallest legal reduction (one uint4 weight chunk per row).
TEST(FP8Stress, MinimumVectorWidthK16) {
    expect_fp8_matches_reference(64, 16, 53);
}

// K = 4112 = 257 * 16: not a multiple of the 4096-element block tile, so the
// last vector iteration is handled by a partial wave of threads.
TEST(FP8Stress, NonTileAlignedK) {
    expect_fp8_matches_reference(64, 4112, 54);
    expect_fp8_matches_reference(64, 4112, 55, /*act_scale=*/0.1f);
}

// All-zero activations: every output must be exactly zero (no scale or
// epilogue term may leak in), and the dynamic scale kernel must hit its
// 1e-12 floor instead of producing 0, Inf or NaN.
TEST(FP8Stress, ZeroActivations) {
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
TEST(FP8Stress, NaNActivationPropagates) {
    Fp8Problem p = make_fp8_problem(128, 1024, 57);
    p.x[511] = std::nanf("");
    Fp8Device dev(p);
    const std::vector<float> out = dev.run(p);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_TRUE(std::isnan(out[i])) << "row " << i << " not NaN: " << out[i];
}

TEST(FP8Stress, InfActivationProducesNonFinite) {
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
TEST(FP8Stress, RapidSequentialLaunches) {
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
    EXPECT_LE(max_rel_error(out_a, p.ref, kDenomFloor), kRelTolerance);
    EXPECT_EQ(0, std::memcmp(out_a.data(), out_b.data(), p.M * sizeof(float)))
        << "deterministic kernel produced different results across launches";
}
