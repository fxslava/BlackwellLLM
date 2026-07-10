// Batched BF16 linear projection (Tensor-Core TF32 GEMM) — the num_tokens > 1
// path that mirrors the batch=1 GEMV broadcast across T activation rows. Checked
// against an FP64-accumulated CPU reference, and against the existing GEMV run
// row-by-row (the two paths must agree to within TF32 rounding).
#include <gtest/gtest.h>
#include <vector>
#include <cuda_bf16.h>

#include "common/cuda_test_utils.h"
#include "common.h"
#include "kernels/batched_bf16_gemm.cuh"
#include "kernels/bf16_linear.cuh"

using test_utils::random_normal;

namespace {

// Relative Frobenius-norm error ||a-b|| / ||b|| — the standard GEMM correctness
// metric. Unlike a per-element relative error it is not blown up by individual
// near-zero (cancellation) outputs, which is what a reduced-precision Tensor-Core
// path (TF32) produces on random data.
float rel_frobenius(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return (float)std::sqrt(num / std::max(den, 1e-30));
}

struct GemmProblem {
    size_t M, K, T;
    std::vector<__nv_bfloat16> W;   // [M, K]
    std::vector<float> X;           // [T, K]
    std::vector<float> ref;         // [T, M]
};

GemmProblem make_problem(size_t M, size_t K, size_t T, unsigned seed) {
    GemmProblem p{M, K, T, {}, {}, {}};
    const std::vector<float> w = random_normal(M * K, seed, 0.05f);
    p.W.resize(M * K);
    for (size_t i = 0; i < w.size(); ++i) p.W[i] = __float2bfloat16(w[i]);
    p.X = random_normal(T * K, seed ^ 0xC2B2AE35u);
    p.ref.resize(T * M);
    for (size_t t = 0; t < T; ++t)
        for (size_t m = 0; m < M; ++m) {
            double dot = 0.0;
            for (size_t k = 0; k < K; ++k)
                dot += (double)__bfloat162float(p.W[m * K + k]) * (double)p.X[t * K + k];
            p.ref[t * M + m] = (float)dot;
        }
    return p;
}

std::vector<float> run_batched(const GemmProblem& p) {
    CudaVector<__nv_bfloat16> dW(p.W.size());
    CudaVector<float> dX(p.X.size());
    CudaVector<float> dY(p.T * p.M);
    dW.upload(p.W);
    dX.upload(p.X);
    launch_bf16_gemm_batched(dW, dX, dY, p.M, p.K, p.T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(p.T * p.M);
    dY.download(out);
    return out;
}

}  // namespace

class BatchedGemmValidation : public test_utils::CudaTest {};

// TF32 Tensor-Core accuracy bound: ~10 mantissa bits into the multiply, FP32
// accumulate. A relative Frobenius error of a few 1e-3 is the expected quality
// on random data; a layout/transpose bug would instead land near 1.0.
constexpr float kTf32Tol = 5e-3f;

// Odd token count exercises the zero-padded tail tile (T not a multiple of 16).
TEST_F(BatchedGemmValidation, MatchesReferenceRaggedTokens) {
    const auto p = make_problem(/*M=*/256, /*K=*/4096, /*T=*/5, 61);
    const float rel = rel_frobenius(run_batched(p), p.ref);
    EXPECT_LE(rel, kTf32Tol) << "batched GEMM mismatch, rel_frobenius=" << rel;
}

// Deep reduction (MLP intermediate width) + a full 16-row tile.
TEST_F(BatchedGemmValidation, DeepReductionFullTile) {
    const auto p = make_problem(/*M=*/64, /*K=*/14336, /*T=*/16, 62);
    const float rel = rel_frobenius(run_batched(p), p.ref);
    EXPECT_LE(rel, kTf32Tol) << "batched GEMM mismatch, rel_frobenius=" << rel;
}

// M/K/T all non-multiples of the 16/16/8 tile: every edge is zero-padded.
TEST_F(BatchedGemmValidation, RaggedAllDims) {
    const auto p = make_problem(/*M=*/70, /*K=*/300, /*T=*/9, 64);
    const float rel = rel_frobenius(run_batched(p), p.ref);
    EXPECT_LE(rel, kTf32Tol) << "batched GEMM mismatch, rel_frobenius=" << rel;
}

// The batched path over distinct rows reproduces the batch=1 GEMV on each row:
// same weights, same input -> same output (both close to the FP32 reference).
TEST_F(BatchedGemmValidation, AgreesWithGemvRowByRow) {
    const size_t M = 128, K = 4096, T = 4;
    const auto p = make_problem(M, K, T, 63);
    const std::vector<float> batched = run_batched(p);

    CudaVector<__nv_bfloat16> dW(p.W.size());
    CudaVector<float> dRow(K), dY(M);
    dW.upload(p.W);
    for (size_t t = 0; t < T; ++t) {
        std::vector<float> row(p.X.begin() + t * K, p.X.begin() + (t + 1) * K);
        dRow.upload(row);
        launch_bf16_gemv_kernel(dW, dRow, dY, M, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gemv(M);
        dY.download(gemv);
        std::vector<float> brow(batched.begin() + t * M, batched.begin() + (t + 1) * M);
        // Batched TF32 GEMM row t vs FP32 GEMV on the same row: differ only by the
        // activation's TF32 rounding.
        const float rel = rel_frobenius(brow, gemv);
        EXPECT_LE(rel, kTf32Tol) << "row " << t << " batched-vs-GEMV rel=" << rel;
    }
}
