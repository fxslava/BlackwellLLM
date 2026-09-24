// GLM-4 partial INTERLEAVED RoPE (kernels/full_attention.cu) vs the CPU
// reference, at the GLM-4-9B geometry: 32 query heads, 2 KV heads, head_dim 128,
// rotary_dim 64 (partial_rotary_factor 0.5).
//
// Three properties, in increasing strength:
//   1. the rotation matches the reference at positions spanning the trained range
//      (and the pass-through channels [rotary_dim, head_dim) are bit-identical);
//   2. pos == 0 is the identity, bitwise -- the cheapest canary for a wrong
//      frequency ladder or an off-by-one in the pair index;
//   3. THE DECISIVE ONE: interleaved and half-split RoPE are the same operator up
//      to a permutation of the rotary channels, so
//        half_split(pi(x)) == pi(interleaved(x))
//      and therefore q.k -- the only thing attention ever computes from Q and K --
//      is IDENTICAL under the two pairings. This pins the claim that GLM-4 could
//      equally be served by permuting the q_proj/k_proj rows, and would catch an
//      interleaved kernel that used a subtly different ladder from the half-split
//      one it is supposed to mirror (see docs/GLM4_TURBOQUANT_INTEGRATION.md §1.2).
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/full_attention.cuh"

namespace {

class RopeInterleavedValidation : public test_utils::CudaTest {};

// GLM-4-9B attention geometry.
struct Geom {
    size_t q_heads    = 32;
    size_t kv_heads   = 2;
    size_t head_dim   = 128;
    size_t rotary_dim = 64;

    size_t q_elems()  const { return q_heads * head_dim; }
    size_t kv_elems() const { return kv_heads * head_dim; }
};

// Runs the fused Q+K launcher and returns the two rotated buffers.
struct QK { std::vector<float> q, k; };

QK run_gpu(const Geom& g, const std::vector<float>& q_in, const std::vector<float>& k_in,
           int pos, float theta) {
    CudaVector<float> d_Q(q_in.size());  d_Q.upload(q_in);
    CudaVector<float> d_K(k_in.size());  d_K.upload(k_in);

    launch_rope_interleaved_partial_inplace(d_Q, d_K, pos, g.q_heads, g.kv_heads,
                                            g.head_dim, g.rotary_dim, theta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    QK out;
    out.q.resize(q_in.size());
    out.k.resize(k_in.size());
    d_Q.download(out.q);
    d_K.download(out.k);
    return out;
}

// pi: interleaved channel 2j -> j, 2j+1 -> j + rotary_dim/2, identity above the
// rotary span. Applied per head to a [num_heads, head_dim] buffer.
std::vector<float> permute_to_half_split(const Geom& g, const std::vector<float>& x,
                                         size_t num_heads) {
    const size_t half = g.rotary_dim / 2;
    std::vector<float> out(x.size());
    for (size_t h = 0; h < num_heads; ++h) {
        const float* src = x.data() + h * g.head_dim;
        float* dst = out.data() + h * g.head_dim;
        for (size_t j = 0; j < half; ++j) {
            dst[j]        = src[2 * j];
            dst[j + half] = src[2 * j + 1];
        }
        for (size_t c = g.rotary_dim; c < g.head_dim; ++c) dst[c] = src[c];
    }
    return out;
}

// GQA-aware dot products: every (query head, its KV head) pair, summed over the
// whole head_dim. This is the quantity attention is built from.
std::vector<float> qk_dots(const Geom& g, const std::vector<float>& q,
                           const std::vector<float>& k) {
    const size_t group = g.q_heads / g.kv_heads;
    std::vector<float> dots(g.q_heads, 0.0f);
    for (size_t h = 0; h < g.q_heads; ++h) {
        const float* qh = q.data() + h * g.head_dim;
        const float* kh = k.data() + (h / group) * g.head_dim;
        double acc = 0.0;
        for (size_t c = 0; c < g.head_dim; ++c) acc += double(qh[c]) * double(kh[c]);
        dots[h] = static_cast<float>(acc);
    }
    return dots;
}

constexpr float kTheta = 10000.0f;   // GLM-4 base; the ladder is theta-agnostic

} // namespace

// Absolute tolerance for a host-vs-device comparison at position `pos`.
//
// The rotation angle is pos*freq with freq <= 1, so it is carried in fp32 with an
// absolute error of about eps_f32 * pos. At pos = 131071 that is ~0.016 rad of
// phase: five of fp32's seven significant digits are spent on the integer part of
// the angle, and the host's double-precision argument reduction and the device's
// sincosf then disagree in what is left. The output error is bounded by
// |x| * phase_error, which for unit-magnitude inputs is the expression below.
//
// This is a statement about fp32 trig, not about the kernel -- which is why the
// large positions ALSO get the position-independent invariant asserted after this
// loop (a wrong ladder or a wrong pairing fails that one at any position).
float host_vs_device_tol(int pos) {
    return 1e-5f + 2.0e-7f * static_cast<float>(pos);
}

// Each rotated pair is a planar rotation, so its 2-norm is preserved exactly --
// independently of the position, the frequency and the argument reduction. A
// kernel that paired the wrong channels, applied the wrong ladder or scaled
// instead of rotating fails this even at pos = 131071.
void expect_pairs_are_rotations(const Geom& g, const std::vector<float>& before,
                                const std::vector<float>& after, size_t num_heads,
                                const char* what) {
    const size_t pairs = g.rotary_dim / 2;
    for (size_t h = 0; h < num_heads; ++h) {
        const float* b = before.data() + h * g.head_dim;
        const float* a = after.data()  + h * g.head_dim;
        for (size_t j = 0; j < pairs; ++j) {
            const double nb = std::hypot(b[2 * j], b[2 * j + 1]);
            const double na = std::hypot(a[2 * j], a[2 * j + 1]);
            EXPECT_NEAR(nb, na, 1e-5 * std::max(1.0, nb))
                << what << ": head " << h << " pair " << j << " changed magnitude";
        }
    }
}

// 1. Q and K against the CPU reference across the trained positional range.
TEST_F(RopeInterleavedValidation, MatchesCpuReferenceAcrossPositions) {
    const Geom g;
    for (int pos : {0, 1, 7, 4095, 131071}) {
        const auto q_in = test_utils::random_uniform(g.q_elems(),  7u, -1.0f, 1.0f);
        const auto k_in = test_utils::random_uniform(g.kv_elems(), 8u, -1.0f, 1.0f);

        std::vector<float> q_ref = q_in, k_ref = k_in;
        cpu_rope_interleaved_partial(q_ref.data(), pos, g.q_heads,  g.head_dim,
                                     g.rotary_dim, kTheta);
        cpu_rope_interleaved_partial(k_ref.data(), pos, g.kv_heads, g.head_dim,
                                     g.rotary_dim, kTheta);

        const QK got = run_gpu(g, q_in, k_in, pos, kTheta);

        SCOPED_TRACE("pos = " + std::to_string(pos));
        const float tol = host_vs_device_tol(pos);
        test_utils::expect_allclose(q_ref, got.q, tol, "interleaved RoPE Q");
        test_utils::expect_allclose(k_ref, got.k, tol, "interleaved RoPE K");

        // Position-independent: holds to 1e-5 even where the elementwise
        // comparison above has to loosen to ~1e-2.
        expect_pairs_are_rotations(g, q_in, got.q, g.q_heads,  "Q");
        expect_pairs_are_rotations(g, k_in, got.k, g.kv_heads, "K");
    }
}

// The pass-through half of every head must come back untouched, BITWISE -- a
// kernel that rotated the full head would still pass a loose allclose at small
// positions, so this is asserted exactly.
TEST_F(RopeInterleavedValidation, NonRotaryChannelsAreBitIdentical) {
    const Geom g;
    const int pos = 1234;
    const auto q_in = test_utils::random_uniform(g.q_elems(),  11u, -1.0f, 1.0f);
    const auto k_in = test_utils::random_uniform(g.kv_elems(), 12u, -1.0f, 1.0f);

    const QK got = run_gpu(g, q_in, k_in, pos, kTheta);

    for (size_t h = 0; h < g.q_heads; ++h)
        for (size_t c = g.rotary_dim; c < g.head_dim; ++c) {
            const size_t i = h * g.head_dim + c;
            EXPECT_FLOAT_EQ(q_in[i], got.q[i]) << "Q head " << h << " channel " << c;
        }
    for (size_t h = 0; h < g.kv_heads; ++h)
        for (size_t c = g.rotary_dim; c < g.head_dim; ++c) {
            const size_t i = h * g.head_dim + c;
            EXPECT_FLOAT_EQ(k_in[i], got.k[i]) << "K head " << h << " channel " << c;
        }
}

// 2. pos == 0: sin == 0, cos == 1, so the whole buffer is untouched exactly.
TEST_F(RopeInterleavedValidation, PositionZeroIsIdentity) {
    const Geom g;
    const auto q_in = test_utils::random_uniform(g.q_elems(),  21u, -1.0f, 1.0f);
    const auto k_in = test_utils::random_uniform(g.kv_elems(), 22u, -1.0f, 1.0f);

    const QK got = run_gpu(g, q_in, k_in, /*pos=*/0, kTheta);

    for (size_t i = 0; i < q_in.size(); ++i) EXPECT_FLOAT_EQ(q_in[i], got.q[i]) << "Q[" << i << "]";
    for (size_t i = 0; i < k_in.size(); ++i) EXPECT_FLOAT_EQ(k_in[i], got.k[i]) << "K[" << i << "]";
}

// 3. The permutation identity, run through the two REAL kernels, plus the
// consequence that actually matters: identical attention scores.
TEST_F(RopeInterleavedValidation, EquivalentToHalfSplitUnderChannelPermutation) {
    const Geom g;
    const int pos = 777;
    const auto q_in = test_utils::random_uniform(g.q_elems(),  31u, -1.0f, 1.0f);
    const auto k_in = test_utils::random_uniform(g.kv_elems(), 32u, -1.0f, 1.0f);

    // Interleaved kernel on the original layout, then permute the result.
    const QK inter = run_gpu(g, q_in, k_in, pos, kTheta);
    const auto q_inter_perm = permute_to_half_split(g, inter.q, g.q_heads);
    const auto k_inter_perm = permute_to_half_split(g, inter.k, g.kv_heads);

    // Half-split kernel on the permuted layout (what a weight-permuting loader
    // would feed it).
    const auto q_perm = permute_to_half_split(g, q_in, g.q_heads);
    const auto k_perm = permute_to_half_split(g, k_in, g.kv_heads);
    CudaVector<float> d_Q(q_perm.size()); d_Q.upload(q_perm);
    CudaVector<float> d_K(k_perm.size()); d_K.upload(k_perm);
    launch_rope_partial_inplace(d_Q, pos, static_cast<int>(g.q_heads),
                                static_cast<int>(g.head_dim),
                                static_cast<int>(g.rotary_dim), kTheta);
    launch_rope_partial_inplace(d_K, pos, static_cast<int>(g.kv_heads),
                                static_cast<int>(g.head_dim),
                                static_cast<int>(g.rotary_dim), kTheta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> q_half(q_perm.size()), k_half(k_perm.size());
    d_Q.download(q_half);
    d_K.download(k_half);

    // Both kernels compute the same sincosf from the same ladder, so this is
    // bit-exact up to the multiply order -- a tight tolerance is the point.
    test_utils::expect_allclose(q_half, q_inter_perm, 1e-6f, "pi(interleaved Q) vs half-split Q");
    test_utils::expect_allclose(k_half, k_inter_perm, 1e-6f, "pi(interleaved K) vs half-split K");

    // The consequence: attention sees identical scores either way.
    const auto dots_inter = qk_dots(g, inter.q, inter.k);
    const auto dots_half  = qk_dots(g, q_half,  k_half);
    for (size_t h = 0; h < dots_inter.size(); ++h)
        EXPECT_NEAR(dots_inter[h], dots_half[h], 1e-3f) << "q.k for head " << h;
}

// The batched launcher (prefill) must agree with the per-token one (decode) row
// by row -- if they diverge, cached keys disagree with the queries that read them.
TEST_F(RopeInterleavedValidation, BatchedMatchesPerTokenRows) {
    const Geom g;
    const int start_pos = 500;
    const size_t num_tokens = 5;

    const auto chunk = test_utils::random_uniform(num_tokens * g.kv_elems(), 41u, -1.0f, 1.0f);

    CudaVector<float> d_X(chunk.size());
    d_X.upload(chunk);
    launch_rope_interleaved_partial_inplace_batched(
        d_X, start_pos, num_tokens, g.kv_heads, g.head_dim, g.rotary_dim, kTheta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> batched(chunk.size());
    d_X.download(batched);

    for (size_t t = 0; t < num_tokens; ++t) {
        const std::vector<float> row(chunk.begin() + t * g.kv_elems(),
                                     chunk.begin() + (t + 1) * g.kv_elems());
        // Rotate this row alone at its own logical position (Q side unused: the
        // launcher rotates Q heads first, so pass the row as the K buffer).
        const std::vector<float> dummy_q(g.head_dim, 0.0f);
        const QK single = run_gpu(Geom{1, g.kv_heads, g.head_dim, g.rotary_dim},
                                  dummy_q, row, start_pos + static_cast<int>(t), kTheta);
        const std::vector<float> got(batched.begin() + t * g.kv_elems(),
                                     batched.begin() + (t + 1) * g.kv_elems());
        SCOPED_TRACE("token " + std::to_string(t));
        test_utils::expect_allclose(single.k, got, 1e-6f, "batched vs per-token row");
    }
}
