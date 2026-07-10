// FP8 E4M3 GEMV mathematical correctness on the exact Llama 3 8B decode
// shapes (hidden=4096, intermediate=14336, GQA KV dim = 8 heads * 128 = 1024):
// row-wise weight scales, weight-only and quantized-activation paths, the
// per-token dynamic scale kernel and residual accumulation, all asserted
// against FP32/FP64 CPU math. Edge shapes and degenerate inputs live in
// benchmarks/stress_fp8_linear.cpp.
#include <gtest/gtest.h>
#include <cmath>
#include <random>
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

namespace {

float rel_frobenius(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return (float)std::sqrt(num / std::max(den, 1e-30));
}

// A multi-token FP8 problem: T activation rows against E4M3 weights + row-wise
// scales, with the exact quantization the batched kernel / GEMV apply. act_scale
// > 0 selects the per-tensor activation-quantization path (a static scale applied
// uniformly across the batch).
struct Fp8Batched {
    size_t M, K, T;
    float act_scale;
    std::vector<uint8_t> weights;         // [M, K]
    std::vector<__nv_bfloat16> w_scales;  // [M]
    std::vector<float> x;                 // [T, K]
    std::vector<float> ref;               // [T, M]
};

Fp8Batched make_fp8_batched(size_t M, size_t K, size_t T, unsigned seed, float act_scale) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    Fp8Batched p{M, K, T, 0.0f, {}, {}, {}, {}};
    p.weights.resize(M * K);
    p.w_scales.resize(M);
    if (act_scale > 0.0f) p.act_scale = __bfloat162float(__float2bfloat16(act_scale));
    for (auto& v : p.weights) v = (uint8_t)byte(rng);
    for (auto& v : p.w_scales) v = __float2bfloat16(0.01f + 0.005f * std::fabs(nrm(rng)));
    p.x = test_utils::random_normal(T * K, seed ^ 0x2468ACEu);

    // Effective activations, quantized ONCE per (t,k) (not per output row — the
    // 256-candidate E4M3 rounding is expensive) and reused across all M rows.
    std::vector<float> x_eff = p.x;
    if (p.act_scale > 0.0f)
        for (auto& v : x_eff) v = test_utils::cpu_quantize_e4m3(v / p.act_scale);

    // Unpack the weights once (also 256-independent, but avoids repeated decode).
    std::vector<float> w_deq(M * K);
    for (size_t i = 0; i < M * K; ++i) w_deq[i] = cpu_unpack_fp8_e4m3(p.weights[i]);

    p.ref.assign(T * M, 0.0f);
    const float post = (p.act_scale > 0.0f ? p.act_scale : 1.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t row = 0; row < M; ++row) {
            const float* w = w_deq.data() + row * K;
            const float* xe = x_eff.data() + t * K;
            double dot = 0.0;
            for (size_t k = 0; k < K; ++k) dot += (double)w[k] * (double)xe[k];
            p.ref[t * M + row] = (float)dot * __bfloat162float(p.w_scales[row]) * post;
        }
    return p;
}

std::vector<float> run_fp8_batched(const Fp8Batched& p) {
    CudaVector<uint8_t> w(p.weights.size());
    CudaVector<__nv_bfloat16> ws(p.w_scales.size()), is(1);
    CudaVector<float> ts(2), x(p.T * p.K), y(p.T * p.M);
    w.upload(p.weights); ws.upload(p.w_scales); x.upload(p.x);
    ts.upload({1.0f, 1.0f});
    const __nv_bfloat16* is_arg = nullptr;
    if (p.act_scale > 0.0f) { is.upload({__float2bfloat16(p.act_scale)}); is_arg = is.d_ptr; }
    launch_batched_fp8_gemm(w, x, ws, is_arg, ts, y, p.M, p.K, /*scale_stride=*/1, p.T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(p.T * p.M);
    y.download(out);
    return out;
}

}  // namespace

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

// TF32 Tensor-Core accuracy on random data: a few 1e-3 relative Frobenius.
constexpr float kFp8BatchTol = 6e-3f;

// Batched FP8 GEMM vs the FP64 reference, weight-only and quantized-activation
// paths, ragged token counts (zero-padded tail tiles).
TEST_F(Fp8Validation, BatchedMatchesReference) {
    for (int T : {1, 5, 16, 33}) {
        const Fp8Batched wo = make_fp8_batched(4096, 1024, T, 61 + T, /*act_scale=*/0.0f);
        EXPECT_LE(rel_frobenius(run_fp8_batched(wo), wo.ref), kFp8BatchTol)
            << "weight-only T=" << T;
        const Fp8Batched aq = make_fp8_batched(4096, 1024, T, 91 + T, /*act_scale=*/0.02f);
        EXPECT_LE(rel_frobenius(run_fp8_batched(aq), aq.ref), kFp8BatchTol)
            << "act-quant T=" << T;
    }
}

// Batched FP8 GEMM reproduces the batch=1 GEMV run row-by-row.
TEST_F(Fp8Validation, BatchedAgreesWithGemvRowByRow) {
    const size_t M = 1024, K = 4096, T = 4;
    const Fp8Batched p = make_fp8_batched(M, K, T, 77, /*act_scale=*/0.02f);
    const std::vector<float> batched = run_fp8_batched(p);

    CudaVector<uint8_t> w(p.weights.size());
    CudaVector<__nv_bfloat16> ws(p.w_scales.size()), is(1);
    CudaVector<float> ts(2), xrow(K), yrow(M);
    w.upload(p.weights); ws.upload(p.w_scales); ts.upload({1.0f, 1.0f});
    is.upload({__float2bfloat16(p.act_scale)});
    for (size_t t = 0; t < T; ++t) {
        std::vector<float> row(p.x.begin() + t * K, p.x.begin() + (t + 1) * K);
        xrow.upload(row);
        launch_fp8_gemv_kernel(w, xrow, ws, is.d_ptr, ts, yrow, M, K, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gemv(M);
        yrow.download(gemv);
        std::vector<float> brow(batched.begin() + t * M, batched.begin() + (t + 1) * M);
        EXPECT_LE(rel_frobenius(brow, gemv), kFp8BatchTol) << "row " << t;
    }
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
