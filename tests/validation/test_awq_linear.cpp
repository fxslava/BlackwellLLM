// AWQ int4 GEMV mathematical correctness on the exact Qwen2.5-Coder-7B decode
// shapes (hidden=3584, intermediate=18944, GQA KV dim = 4 heads * 128 = 512,
// group_size=128), asserted against an FP64 CPU reference within FP16-scale
// tolerance. Edge shapes and malformed parameters live in
// benchmarks/stress_awq_linear.cpp.
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "common/awq_reference.h"

using test_utils::expect_awq_matches_reference;

class AwqValidation : public test_utils::CudaTest {};

namespace {

// Relative Frobenius-norm error — the GEMM correctness metric (robust to the
// near-zero cancellation outputs that inflate a per-element relative error under
// reduced-precision TF32 Tensor Cores).
float rel_frobenius(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return (float)std::sqrt(num / std::max(den, 1e-30));
}

// A multi-token AWQ problem in canonical AutoAWQ packing: T activation rows, an
// FP64 reference, and the packed int4 weights the batched GEMM consumes.
struct AwqBatched {
    int in, out, gs, T;
    std::vector<uint32_t> qweight, qzeros;
    std::vector<half> scales;
    std::vector<float> x;    // [T, in]
    std::vector<float> ref;  // [T, out]
};

AwqBatched make_awq_batched(int in, int out, int gs, int T, unsigned seed) {
    using test_utils::kAwqOrder;
    const int num_groups = (in + gs - 1) / gs;
    const int packed_cols = out / 8;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q4(0, 15);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    AwqBatched p{in, out, gs, T, {}, {}, {}, {}, {}};
    std::vector<int> w((size_t)in * out), z((size_t)num_groups * out);
    p.scales.resize((size_t)num_groups * out);
    for (auto& v : w) v = q4(rng);
    for (auto& v : z) v = q4(rng);
    for (auto& v : p.scales) v = __float2half(nrm(rng) * 0.05f);

    p.qweight.assign((size_t)in * packed_cols, 0);
    p.qzeros.assign((size_t)num_groups * packed_cols, 0);
    for (int k = 0; k < in; ++k)
        for (int pc = 0; pc < packed_cols; ++pc)
            for (int n = 0; n < 8; ++n)
                p.qweight[(size_t)k * packed_cols + pc] |=
                    (uint32_t)w[(size_t)k * out + pc * 8 + kAwqOrder[n]] << (4 * n);
    for (int g = 0; g < num_groups; ++g)
        for (int pc = 0; pc < packed_cols; ++pc)
            for (int n = 0; n < 8; ++n)
                p.qzeros[(size_t)g * packed_cols + pc] |=
                    (uint32_t)z[(size_t)g * out + pc * 8 + kAwqOrder[n]] << (4 * n);

    p.x = test_utils::random_normal((size_t)T * in, seed ^ 0x1234567u);
    p.ref.assign((size_t)T * out, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int oc = 0; oc < out; ++oc) {
            double acc = 0.0;
            for (int k = 0; k < in; ++k) {
                const int g = k / gs;
                acc += (double)(w[(size_t)k * out + oc] - z[(size_t)g * out + oc]) *
                       (double)__half2float(p.scales[(size_t)g * out + oc]) *
                       (double)p.x[(size_t)t * in + k];
            }
            p.ref[(size_t)t * out + oc] = (float)acc;
        }
    return p;
}

std::vector<float> run_awq_batched(const AwqBatched& p) {
    CudaVector<uint32_t> qw(p.qweight.size()), qz(p.qzeros.size());
    CudaVector<half> sc(p.scales.size());
    CudaVector<float> x((size_t)p.T * p.in), y((size_t)p.T * p.out);
    qw.upload(p.qweight); qz.upload(p.qzeros); sc.upload(p.scales); x.upload(p.x);
    launch_batched_awq_gemm(qw, sc, qz, x, y, p.out, p.in, p.gs, p.T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)p.T * p.out);
    y.download(out);
    return out;
}

}  // namespace

// TF32 Tensor-Core accuracy on random data: a few 1e-3 relative Frobenius. A
// layout/dequant bug would instead land near 1.0.
constexpr float kAwqBatchTol = 6e-3f;

// Batched AWQ GEMM vs the FP64 reference, on Qwen attention/MLP shapes. Ragged
// token counts exercise the zero-padded tail tile.
TEST_F(AwqValidation, BatchedMatchesReference) {
    for (int T : {1, 5, 16, 33}) {
        const AwqBatched p = make_awq_batched(3584, 512, 128, T, 71 + T);
        const float rel = rel_frobenius(run_awq_batched(p), p.ref);
        EXPECT_LE(rel, kAwqBatchTol) << "T=" << T << " rel_frobenius=" << rel;
    }
}

// Batched AWQ GEMM reproduces the batch=1 GEMV run row-by-row (both approximate
// the same dequantized weights; they differ only by the activation's TF32 round).
TEST_F(AwqValidation, BatchedAgreesWithGemvRowByRow) {
    const int in = 3584, out = 512, gs = 128, T = 4;
    const AwqBatched p = make_awq_batched(in, out, gs, T, 88);
    const std::vector<float> batched = run_awq_batched(p);

    CudaVector<uint32_t> qw(p.qweight.size()), qz(p.qzeros.size());
    CudaVector<half> sc(p.scales.size());
    CudaVector<float> xrow(in), yrow(out);
    qw.upload(p.qweight); qz.upload(p.qzeros); sc.upload(p.scales);
    for (int t = 0; t < T; ++t) {
        std::vector<float> row(p.x.begin() + (size_t)t * in, p.x.begin() + (size_t)(t + 1) * in);
        xrow.upload(row);
        launch_awq_gemv_kernel(qw, sc, qz, xrow, yrow, out, in, gs);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gemv(out);
        yrow.download(gemv);
        std::vector<float> brow(batched.begin() + (size_t)t * out,
                                batched.begin() + (size_t)(t + 1) * out);
        const float rel = rel_frobenius(brow, gemv);
        EXPECT_LE(rel, kAwqBatchTol) << "row " << t << " batched-vs-GEMV rel=" << rel;
    }
}

TEST_F(AwqValidation, QwenAttentionQProj) {
    expect_awq_matches_reference(3584, 3584, 128, 11);
}

// OC=512 -> only 64 packed columns: the launcher distributes the reduction
// dimension across blocks (split-K), so GQA K/V projections take the
// atomic-accumulation path.
TEST_F(AwqValidation, QwenGqaKvProjSplitK) {
    expect_awq_matches_reference(3584, 512, 128, 12);
}

TEST_F(AwqValidation, QwenMlpGateUpProj) {
    expect_awq_matches_reference(3584, 18944, 128, 13);
}

TEST_F(AwqValidation, QwenMlpDownProj) {
    expect_awq_matches_reference(18944, 3584, 128, 14);
}

// Vocab-sized projection (Qwen lm_head): exercises grid sizing and 64-bit
// indexing at the largest shape the engine launches. Constant-pattern
// weights keep the CPU reference O(IC) instead of O(IC*OC).
TEST_F(AwqValidation, QwenVocabSizedLmHead) {
    const test_utils::AwqProblem p =
        test_utils::make_awq_constant_problem(3584, 152064, 128, 15);
    test_utils::AwqDevice dev(p);
    const float max_rel = test_utils::max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, test_utils::kAwqRelTolerance)
        << "lm_head-sized GEMV mismatch, max_rel=" << max_rel;
}
