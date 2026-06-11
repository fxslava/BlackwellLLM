// FP8 E4M3 GEMV mathematical correctness on the exact Llama 3 8B decode
// shapes (hidden=4096, intermediate=14336, GQA KV dim = 8 heads * 128 = 1024):
// row-wise weight scales, weight-only and quantized-activation paths, the
// per-token dynamic scale kernel and residual accumulation, all asserted
// against FP32/FP64 CPU math. Edge shapes and degenerate inputs live in
// benchmarks/stress_fp8_linear.cpp.
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "common/fp8_reference.h"

using test_utils::Fp8Problem;
using test_utils::Fp8Device;
using test_utils::expect_fp8_matches_reference;
using test_utils::make_fp8_problem;
using test_utils::max_rel_error;
using test_utils::kFp8RelTolerance;
using test_utils::kFp8DenomFloor;

class Fp8Validation : public test_utils::CudaTest {};

TEST_F(Fp8Validation, LlamaAttnOutProj) {
    expect_fp8_matches_reference(4096, 4096, 41);
}

TEST_F(Fp8Validation, LlamaGqaKvProj) {
    expect_fp8_matches_reference(1024, 4096, 42);
}

TEST_F(Fp8Validation, LlamaMlpUpProj) {
    expect_fp8_matches_reference(14336, 4096, 43);
}

TEST_F(Fp8Validation, LlamaMlpDownProj) {
    expect_fp8_matches_reference(4096, 14336, 44);
}

// Per-tensor activation quantization: activations are converted to E4M3 in
// x/scale domain on the fly and the epilogue multiplies the scale back.
TEST_F(Fp8Validation, QuantizedActivationPath) {
    expect_fp8_matches_reference(4096, 4096, 45, /*act_scale=*/0.02f);
    expect_fp8_matches_reference(1024, 4096, 46, /*act_scale=*/0.5f);
}

// Per-token dynamic scale kernel: scale = max(|x|, 1e-12) / 448 plus its
// reciprocal, both finite. fast-math division on the device allows a few ulp.
TEST_F(Fp8Validation, PerTokenScaleComputation) {
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
TEST_F(Fp8Validation, ResidualAccumulation) {
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
    EXPECT_LE(max_rel_error(out, expected, kFp8DenomFloor), kFp8RelTolerance);
}
