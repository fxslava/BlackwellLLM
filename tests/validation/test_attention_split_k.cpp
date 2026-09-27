// Split-K (Flash-Decoding) decode attention vs. the CPU reference AND vs. the
// single-block kernel it replaces, across the split factors and context lengths
// the runtime ladder actually picks.
//
// ON TOLERANCES. Both decode kernels end with `cast_to_bf16_and_back` -- a
// deliberate mantissa truncation that buys PyTorch bf16 parity (see
// src/kernels/attention.cu). It quantizes every output to 8 significant bits, so
// no comparison against an FP32 reference can be tighter than one bf16 bucket
// (~4e-3 RELATIVE), no matter how exact the softmax is. The interesting
// question is therefore not "is split-K within 1e-4 of FP32" -- nothing here is --
// but "does splitting the context add error on top of what the shipping kernel
// already has". Every case below asserts all three:
//   1. split-K is within one bf16 bucket of the FP32 CPU reference;
//   2. split-K is within one bf16 bucket of the single-block kernel;
//   3. split-K's error vs. the reference exceeds the single-block kernel's own
//      error by less than 1e-4 -- the real accuracy claim.
// plus a hard no-NaN / no-Inf sweep, which is what a mis-handled empty slice or a
// -inf/-inf log-sum-exp would produce.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/attention.cuh"

namespace {

class AttentionSplitKValidation : public test_utils::CudaTest {};

// One bf16 bucket, relative: bf16 keeps 8 significant bits, so the spacing at a
// value of magnitude v is at most v * 2^-7. Two fp32 results that round into
// adjacent buckets differ by that much -- the floor on every comparison here.
constexpr float kBf16RelUlp = 1.0f / 128.0f;

float max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float e : v) m = std::max(m, std::fabs(e));
    return m;
}

void expect_all_finite(const std::vector<float>& v, const std::string& what) {
    for (size_t i = 0; i < v.size(); ++i)
        ASSERT_TRUE(std::isfinite(v[i])) << what << ": non-finite value " << v[i]
                                         << " at index " << i;
}

struct Geometry {
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    const char* name;
};

// Runs one (geometry, context length, split factor) case through both kernels and
// asserts the three-way agreement documented at the top of this file.
void expect_split_k_matches(const Geometry& geo, int pos, int splits,
                            size_t max_seq_len, unsigned seed) {
    SCOPED_TRACE(std::string(geo.name) + " pos=" + std::to_string(pos) +
                 " splits=" + std::to_string(splits));

    const size_t q_elems  = geo.q_heads * geo.head_dim;
    const size_t kv_elems = geo.kv_heads * max_seq_len * geo.head_dim;

    const std::vector<float> h_Q = test_utils::random_uniform(q_elems, seed, -0.3f, 0.3f);
    const std::vector<float> h_K = test_utils::random_uniform(kv_elems, seed + 1, -0.2f, 0.2f);
    const std::vector<float> h_V = test_utils::random_uniform(kv_elems, seed + 2, -0.2f, 0.2f);

    std::vector<float> h_ref(q_elems, 0.0f);
    cpu_attention_decoding(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), pos,
                           geo.q_heads, geo.kv_heads, geo.head_dim, max_seq_len);

    CudaVector<float> d_Q(q_elems);  d_Q.upload(h_Q);
    CudaVector<float> d_K(kv_elems); d_K.upload(h_K);
    CudaVector<float> d_V(kv_elems); d_V.upload(h_V);
    CudaVector<float> d_O_split(q_elems);
    CudaVector<float> d_O_legacy(q_elems);
    // The scratchpad is sized for the MAXIMUM split factor exactly as the arena
    // sizes it, so one buffer serves every `splits` in the sweep -- the property
    // that makes the decode loop allocation-free.
    CudaVector<float> d_scratch(
        blackwell::attn::split_k_scratch_floats(geo.q_heads, geo.head_dim));

    // Poison both outputs: a kernel that silently writes nothing must fail loudly
    // rather than compare leftover zeros.
    test_utils::poison(d_O_split);
    test_utils::poison(d_O_legacy);

    launch_attention_decoding_split_k(d_Q, d_K, d_V, d_O_split, d_scratch, splits,
                                      pos, geo.q_heads, geo.kv_heads, geo.head_dim,
                                      max_seq_len);
    launch_attention_decoding_kernel(d_Q, d_K, d_V, d_O_legacy, pos,
                                     geo.q_heads, geo.kv_heads, geo.head_dim,
                                     max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_split(q_elems), h_legacy(q_elems);
    d_O_split.download(h_split);
    d_O_legacy.download(h_legacy);

    expect_all_finite(h_split, "split-K output");

    const float tol = max_abs(h_ref) * kBf16RelUlp;
    ASSERT_GT(tol, 0.0f) << "degenerate reference (all zeros) -- the test proves nothing";

    // (1) and (2): within one bf16 bucket of FP32 and of the shipping kernel.
    test_utils::expect_allclose(h_ref, h_split, tol, "split-K vs CPU reference");
    test_utils::expect_allclose(h_legacy, h_split, tol, "split-K vs single-block kernel");

    // (3) the real claim: splitting the context adds < 1e-4 of error.
    const float err_split  = test_utils::max_abs_error(h_split, h_ref);
    const float err_legacy = test_utils::max_abs_error(h_legacy, h_ref);
    EXPECT_LT(err_split, err_legacy + 1e-4f)
        << "split-K error " << err_split << " vs single-block " << err_legacy;
}

// GLM-4-9B is the motivating geometry (32 query heads on a 56-SM part), at both
// of its GQA widths. Llama 3 8B and Qwen2.5-Coder-7B ride along because the
// single-block kernel's own suite covers them and the split must not regress them.
constexpr Geometry kGeometries[] = {
    {32,  2, 128, "GLM-4-9B 32q/2kv"},
    {32,  4, 128, "GLM-4-9B 32q/4kv"},
    {32,  8, 128, "Llama-3-8B 32q/8kv"},
    {28,  4, 128, "Qwen2.5-7B 28q/4kv"},
};

}  // namespace

// ---------------------------------------------------------------------------
// Split-factor x context-length sweep
// ---------------------------------------------------------------------------

// Every split factor the ladder can emit, at the context length band it emits it
// for. Each must reproduce the single-block kernel.
TEST_F(AttentionSplitKValidation, EverySplitFactorMatchesSingleBlockKernel) {
    unsigned seed = 900;
    for (const Geometry& geo : kGeometries) {
        for (int splits : {1, 2, 4, 8, 16}) {
            for (int pos : {255, 1023, 2047}) {
                expect_split_k_matches(geo, pos, splits,
                                       /*max_seq_len=*/2048, seed);
                seed += 3;
            }
        }
    }
}

// The context lengths the benchmark reports, at the split factor the runtime
// ladder actually picks for each -- i.e. the exact configurations that ship.
TEST_F(AttentionSplitKValidation, LadderSelectedSplitMatchesAtEveryBenchLength) {
    unsigned seed = 1500;
    for (const Geometry& geo : kGeometries) {
        for (int n : {256, 512, 1024, 2048, 4096}) {
            const int splits = blackwell::attn::select_split_k(n);
            expect_split_k_matches(geo, /*pos=*/n - 1, splits,
                                   /*max_seq_len=*/4096, seed);
            seed += 3;
        }
    }
}

// Short and RAGGED contexts: N=1 (softmax over one token must reproduce V[0]),
// and lengths that do not divide by the split factor, which is what exercises the
// launcher's re-derivation of the split count (a naive ceil(N/S) would launch
// blocks with nothing to do). Splits are deliberately over-requested.
TEST_F(AttentionSplitKValidation, ShortAndRaggedContexts) {
    unsigned seed = 2100;
    for (const Geometry& geo : kGeometries) {
        for (int pos : {0, 1, 2, 4, 6, 13, 30, 127}) {
            for (int splits : {4, 8, 16}) {
                expect_split_k_matches(geo, pos, splits, /*max_seq_len=*/256, seed);
                seed += 3;
            }
        }
    }
}

// head_dim below the block width: only the first `head_dim` threads own a channel,
// and the dot-product striping (lane, lane+32, lane+64, lane+96) must zero the
// out-of-range channels rather than read past the head.
TEST_F(AttentionSplitKValidation, NarrowHeadDim) {
    unsigned seed = 2700;
    for (size_t head_dim : {size_t{32}, size_t{64}, size_t{80}, size_t{96}}) {
        const Geometry geo{32, 4, head_dim, "narrow head_dim"};
        for (int splits : {4, 8}) {
            expect_split_k_matches(geo, /*pos=*/1023, splits, /*max_seq_len=*/1024, seed);
            seed += 3;
        }
    }
}

// The two documented degenerations: a resolved split factor of 1 and a null
// scratchpad both fall through to the single-block kernel, which is the same
// launch -- so the outputs must be BITWISE identical, not merely close.
TEST_F(AttentionSplitKValidation, DegenerateRequestsFallBackBitwise) {
    const Geometry geo = kGeometries[0];
    const int pos = 1023;
    const size_t max_seq_len = 2048;
    const size_t q_elems  = geo.q_heads * geo.head_dim;
    const size_t kv_elems = geo.kv_heads * max_seq_len * geo.head_dim;

    const std::vector<float> h_Q = test_utils::random_uniform(q_elems, 3300, -0.3f, 0.3f);
    const std::vector<float> h_K = test_utils::random_uniform(kv_elems, 3301, -0.2f, 0.2f);
    const std::vector<float> h_V = test_utils::random_uniform(kv_elems, 3302, -0.2f, 0.2f);

    CudaVector<float> d_Q(q_elems);  d_Q.upload(h_Q);
    CudaVector<float> d_K(kv_elems); d_K.upload(h_K);
    CudaVector<float> d_V(kv_elems); d_V.upload(h_V);
    CudaVector<float> d_O(q_elems);
    CudaVector<float> d_scratch(
        blackwell::attn::split_k_scratch_floats(geo.q_heads, geo.head_dim));

    std::vector<float> h_legacy(q_elems), h_splits_one(q_elems), h_no_scratch(q_elems);

    launch_attention_decoding_kernel(d_Q, d_K, d_V, d_O, pos, geo.q_heads,
                                     geo.kv_heads, geo.head_dim, max_seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_O.download(h_legacy);

    launch_attention_decoding_split_k(d_Q, d_K, d_V, d_O, d_scratch, /*splits=*/1, pos,
                                      geo.q_heads, geo.kv_heads, geo.head_dim, max_seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_O.download(h_splits_one);

    launch_attention_decoding_split_k(d_Q, d_K, d_V, d_O, /*d_scratch=*/nullptr,
                                      /*splits=*/8, pos, geo.q_heads, geo.kv_heads,
                                      geo.head_dim, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_O.download(h_no_scratch);

    for (size_t i = 0; i < q_elems; ++i) {
        ASSERT_FLOAT_EQ(h_legacy[i], h_splits_one[i]) << "splits=1 diverged at " << i;
        ASSERT_FLOAT_EQ(h_legacy[i], h_no_scratch[i]) << "null scratch diverged at " << i;
    }
}

// Long context at the maximum split factor: the case with the most reduction
// terms and the widest spread of per-slice softmax maxima, i.e. where a broken
// global log-sum-exp shows up first.
TEST_F(AttentionSplitKValidation, LongContextMaxSplits) {
    expect_split_k_matches(kGeometries[0], /*pos=*/4095, /*splits=*/16,
                           /*max_seq_len=*/4096, /*seed=*/3900);
    expect_split_k_matches(kGeometries[1], /*pos=*/4095, /*splits=*/16,
                           /*max_seq_len=*/4096, /*seed=*/3910);
}

// A Q with a single dominant K row makes the softmax nearly one-hot, so all but
// one slice contributes exp(m_s - M) ~ 0. If the reduction mishandled those
// slices (0 * inf, or a -inf that escaped), it would surface as NaN here.
TEST_F(AttentionSplitKValidation, NearOneHotSoftmaxStaysFinite) {
    const Geometry geo = kGeometries[0];
    const int pos = 2047;
    const size_t max_seq_len = 2048;
    const size_t q_elems  = geo.q_heads * geo.head_dim;
    const size_t kv_elems = geo.kv_heads * max_seq_len * geo.head_dim;

    // Large-magnitude Q/K so the score spread is tens of nats: every slice but the
    // one holding the argmax underflows to exactly zero weight.
    std::vector<float> h_Q = test_utils::random_uniform(q_elems, 4500, -4.0f, 4.0f);
    std::vector<float> h_K = test_utils::random_uniform(kv_elems, 4501, -4.0f, 4.0f);
    std::vector<float> h_V = test_utils::random_uniform(kv_elems, 4502, -1.0f, 1.0f);

    std::vector<float> h_ref(q_elems, 0.0f);
    cpu_attention_decoding(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), pos,
                           geo.q_heads, geo.kv_heads, geo.head_dim, max_seq_len);

    CudaVector<float> d_Q(q_elems);  d_Q.upload(h_Q);
    CudaVector<float> d_K(kv_elems); d_K.upload(h_K);
    CudaVector<float> d_V(kv_elems); d_V.upload(h_V);
    CudaVector<float> d_O(q_elems);
    CudaVector<float> d_scratch(
        blackwell::attn::split_k_scratch_floats(geo.q_heads, geo.head_dim));
    test_utils::poison(d_O);

    launch_attention_decoding_split_k(d_Q, d_K, d_V, d_O, d_scratch, /*splits=*/16, pos,
                                      geo.q_heads, geo.kv_heads, geo.head_dim, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_split(q_elems);
    d_O.download(h_split);
    expect_all_finite(h_split, "near-one-hot split-K output");
    test_utils::expect_allclose(h_ref, h_split, max_abs(h_ref) * kBf16RelUlp,
                                "near-one-hot split-K vs CPU reference");
}

// ---------------------------------------------------------------------------
// Dispatch policy (CPU-only: no kernel, no device memory)
// ---------------------------------------------------------------------------

TEST(AttentionSplitKPolicy, LadderMatchesTheDocumentedBands) {
    using blackwell::attn::select_split_k;
    EXPECT_EQ(select_split_k(1), 1);
    EXPECT_EQ(select_split_k(255), 1);
    EXPECT_EQ(select_split_k(256), 4);
    EXPECT_EQ(select_split_k(1023), 4);
    EXPECT_EQ(select_split_k(1024), 8);
    EXPECT_EQ(select_split_k(4095), 8);
    EXPECT_EQ(select_split_k(4096), 16);
    EXPECT_EQ(select_split_k(1 << 20), 16);
}

TEST(AttentionSplitKPolicy, CeilingAndContextBothClampTheSplit) {
    using blackwell::attn::select_split_k;
    // max_splits == 1 is how the runtime plan turns split-K off.
    EXPECT_EQ(select_split_k(8192, /*max_splits=*/1), 1);
    EXPECT_EQ(select_split_k(8192, /*max_splits=*/4), 4);
    EXPECT_EQ(select_split_k(8192, /*max_splits=*/64), blackwell::attn::kSplitKMaxSplits);
    // Never more slices than tokens, and never fewer than one.
    EXPECT_EQ(select_split_k(2, /*max_splits=*/16), 1);   // below the 256 band anyway
    EXPECT_GE(select_split_k(300, /*max_splits=*/16), 1);
    EXPECT_EQ(select_split_k(0), 1);
}

TEST(AttentionSplitKPolicy, ScratchSizingCoversTheWholeLayout) {
    using blackwell::attn::split_k_scratch_floats;
    // acc [q][S][D] + m [q][S] + l [q][S].
    EXPECT_EQ(split_k_scratch_floats(32, 128, 16), 32u * 16u * 128u + 2u * 32u * 16u);
    EXPECT_EQ(split_k_scratch_floats(32, 128, 1), 32u * 128u + 2u * 32u);
    // A smaller split factor must fit inside the max-sized allocation -- the
    // property that makes the per-step dispatch allocation-free.
    for (int s = 1; s <= blackwell::attn::kSplitKMaxSplits; ++s)
        EXPECT_LE(split_k_scratch_floats(32, 128, s), split_k_scratch_floats(32, 128));
}
