// RoPE-aware KV head eviction (kv_evict.cu) — the sliding window for continuous
// streaming, docs/CONTINUOUS_STREAMING.md T3.
//
// THE DECISIVE PROPERTY, and the reason this file exists:
//
//   a cache built at positions [0..N), then evicted by delta
//     ==
//   a cache built from scratch with those rows already absent, at the shifted
//   positions
//
// Anything weaker (spot-checking a few rotations, eyeballing norms) would pass
// for a transform that is subtly off by a constant phase, which is exactly the
// bug that would make the model quietly worse rather than obviously broken.
// Building the reference by REPLAYING the append kernel — rather than by
// reimplementing the rotation on the CPU — is deliberate: it means the test
// cannot agree with a shared misunderstanding of the frequency ladder.
//
// Equality is exact in the reals but not in float: the evicted path computes
// cos(p*f)cos(d*f) + sin(p*f)sin(d*f) where the reference computes cos((p-d)*f).
// The tolerance below is IEEE arithmetic, not slack in the transform. V carries
// no phase, so V comparisons ARE bit-exact and are asserted as such.
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "common/cuda_test_utils.h"
#include "kernels/full_attention.cuh"   // partial RoPE (both pairings) + kv_append
#include "kernels/kv_evict.cuh"
#include "kernels/rope.cuh"

namespace {

class KvEvictValidation : public test_utils::CudaTest {};

// Geometry small enough to be quick, wide enough to exercise the real layout
// (multiple kv heads, an even head_dim with a rotate_half split).
struct Geom {
    size_t q_heads  = 4;
    size_t kv_heads = 2;
    size_t head_dim = 64;
    size_t max_seq  = 512;
    // Rotary span and pairing. 0 == full rotary (follows head_dim), which keeps
    // every pre-existing case on the fused full-head half-split append path.
    size_t rotary_dim  = 0;
    bool   interleaved = false;   // true: GLM-4 adjacent-pair partial RoPE

    size_t rot() const { return rotary_dim ? rotary_dim : head_dim; }
    bool   partial() const { return rot() != head_dim || interleaved; }
    size_t row_elems() const { return kv_heads * head_dim; }
    size_t cache_elems() const { return kv_heads * max_seq * head_dim; }
};

// One logical row's K and V payload. The CONTENT is keyed to the row's identity,
// never to the position it happens to be written at — that is what lets the same
// row be replayed at a shifted position in the reference build.
struct Rows {
    std::vector<std::vector<float>> k, v;
};

Rows make_rows(const Geom& g, int count, unsigned seed) {
    Rows r;
    for (int i = 0; i < count; ++i) {
        r.k.push_back(test_utils::random_uniform(g.row_elems(),
                                                 seed + static_cast<unsigned>(2 * i), -1.0f, 1.0f));
        r.v.push_back(test_utils::random_uniform(
            g.row_elems(), seed + static_cast<unsigned>(2 * i) + 1, -1.0f, 1.0f));
    }
    return r;
}

struct Cache {
    std::vector<float> k, v;
};

// Appends rows[row_ids[i]] at positions[i], through the SAME launcher the engine
// uses, so the cache under test and the reference are produced by one code path.
Cache build_cache(const Geom& g, const Rows& rows, const std::vector<int>& row_ids,
                  const std::vector<int>& positions, float theta, RopeScaling scaling) {
    CudaVector<float> d_K_cache(g.cache_elems());
    CudaVector<float> d_V_cache(g.cache_elems());
    const std::vector<float> zeros(g.cache_elems(), 0.0f);
    d_K_cache.upload(zeros);
    d_V_cache.upload(zeros);

    CudaVector<float> d_Q(g.q_heads * g.head_dim);
    CudaVector<float> d_K(g.row_elems());
    CudaVector<float> d_V(g.row_elems());
    const std::vector<float> q_scratch(g.q_heads * g.head_dim, 0.5f);

    for (size_t i = 0; i < row_ids.size(); ++i) {
        d_Q.upload(q_scratch);
        d_K.upload(rows.k[static_cast<size_t>(row_ids[i])]);
        d_V.upload(rows.v[static_cast<size_t>(row_ids[i])]);
        if (!g.partial()) {
            launch_fused_rope_kv_kernel(d_Q, d_K, d_V, d_K_cache, d_V_cache, positions[i],
                                        g.q_heads, g.kv_heads, g.head_dim, g.max_seq, theta, scaling);
        } else if (g.interleaved) {
            // The GLM-4 append path: adjacent-pair partial RoPE, then a plain
            // append into the very same contiguous slot the fused kernel writes.
            launch_rope_interleaved_partial_inplace(d_Q, d_K, positions[i], g.q_heads,
                                                    g.kv_heads, g.head_dim, g.rot(),
                                                    theta, scaling);
            launch_kv_append(d_K, d_V, d_K_cache, d_V_cache, positions[i],
                             static_cast<int>(g.kv_heads), static_cast<int>(g.head_dim),
                             static_cast<int>(g.max_seq));
        } else {
            launch_rope_partial_inplace(d_K, positions[i], static_cast<int>(g.kv_heads),
                                        static_cast<int>(g.head_dim),
                                        static_cast<int>(g.rot()), theta, scaling);
            launch_kv_append(d_K, d_V, d_K_cache, d_V_cache, positions[i],
                             static_cast<int>(g.kv_heads), static_cast<int>(g.head_dim),
                             static_cast<int>(g.max_seq));
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Cache out;
    out.k.resize(g.cache_elems());
    out.v.resize(g.cache_elems());
    d_K_cache.download(out.k);
    d_V_cache.download(out.v);
    return out;
}

// Applies eviction to a cache that was built by build_cache().
Cache evict(const Geom& g, const Cache& in, int keep_from, int delta, int cache_len,
            float theta, RopeScaling scaling) {
    CudaVector<float> d_K(g.cache_elems());
    CudaVector<float> d_V(g.cache_elems());
    d_K.upload(in.k);
    d_V.upload(in.v);

    launch_kv_evict_head(d_K, d_V, keep_from, delta, cache_len, g.kv_heads, g.head_dim,
                         g.rot(), g.interleaved, g.max_seq, theta, scaling);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Cache out;
    out.k.resize(g.cache_elems());
    out.v.resize(g.cache_elems());
    d_K.download(out.k);
    d_V.download(out.v);
    return out;
}

// Compares only the LIVE region [0, live_rows) — everything above the new
// sequence length is deliberately left as garbage by the kernel (it sits past
// the tail, attention never reads it, the next prefill overwrites it).
void expect_live_region_matches(const Geom& g, const Cache& got, const Cache& want,
                                int live_rows, float tol, const char* what) {
    std::vector<float> a, b, av, bv;
    for (size_t h = 0; h < g.kv_heads; ++h) {
        for (int r = 0; r < live_rows; ++r) {
            const size_t off = (h * g.max_seq + static_cast<size_t>(r)) * g.head_dim;
            a.insert(a.end(), got.k.begin() + static_cast<std::ptrdiff_t>(off),
                     got.k.begin() + static_cast<std::ptrdiff_t>(off + g.head_dim));
            b.insert(b.end(), want.k.begin() + static_cast<std::ptrdiff_t>(off),
                     want.k.begin() + static_cast<std::ptrdiff_t>(off + g.head_dim));
            av.insert(av.end(), got.v.begin() + static_cast<std::ptrdiff_t>(off),
                      got.v.begin() + static_cast<std::ptrdiff_t>(off + g.head_dim));
            bv.insert(bv.end(), want.v.begin() + static_cast<std::ptrdiff_t>(off),
                      want.v.begin() + static_cast<std::ptrdiff_t>(off + g.head_dim));
        }
    }
    test_utils::expect_allclose(b, a, tol, (std::string("K cache: ") + what).c_str());
    // V is a pure copy through both paths, so it must be BIT-exact.
    test_utils::expect_allclose(bv, av, 0.0f, (std::string("V cache: ") + what).c_str());
}

// THE equivalence check, parameterised. Builds the full sequence, evicts, and
// compares against the same content prefilled from scratch without the dropped
// rows.
void expect_eviction_is_transparent(const Geom& g, int total, int keep_from, int delta,
                                    float theta, RopeScaling scaling, float tol,
                                    const char* what) {
    ASSERT_GT(delta, 0);
    ASSERT_LE(keep_from + delta, total);
    const Rows rows = make_rows(g, total, /*seed=*/1234u);

    // (1) the full sequence at its natural positions...
    std::vector<int> ids, pos;
    for (int i = 0; i < total; ++i) {
        ids.push_back(i);
        pos.push_back(i);
    }
    const Cache full = build_cache(g, rows, ids, pos, theta, scaling);

    // (2) ...evicted.
    const Cache evicted = evict(g, full, keep_from, delta, total, theta, scaling);

    // (3) the identical truncated content, prefilled from scratch: the frozen
    //     prefix keeps its absolute positions, everything above the cut slides
    //     down by delta.
    std::vector<int> ref_ids, ref_pos;
    for (int i = 0; i < keep_from; ++i) {
        ref_ids.push_back(i);
        ref_pos.push_back(i);
    }
    for (int i = keep_from + delta; i < total; ++i) {
        ref_ids.push_back(i);
        ref_pos.push_back(i - delta);
    }
    const Cache scratch = build_cache(g, rows, ref_ids, ref_pos, theta, scaling);

    expect_live_region_matches(g, evicted, scratch, total - delta, tol, what);
}

// ---- the equivalence property -----------------------------------------------

TEST_F(KvEvictValidation, EvictionMatchesAFromScratchPrefillOfTheSurvivors) {
    const Geom g{};
    expect_eviction_is_transparent(g, /*total=*/96, /*keep_from=*/8, /*delta=*/32,
                                   500000.0f, RopeScaling{}, 5e-4f, "baseline");
}

TEST_F(KvEvictValidation, TransparentAcrossPrefixAndDeltaCombinations) {
    const Geom g{};
    struct Case { int total, keep_from, delta; const char* what; };
    const Case cases[] = {
        {64, 0, 16, "no frozen prefix"},
        {64, 1, 1, "minimal prefix, minimal delta"},
        {128, 16, 64, "half the sequence"},
        {128, 32, 96, "everything above the prefix"},   // rows to move == 0
        {100, 7, 13, "odd sizes"},
    };
    for (const Case& c : cases) {
        expect_eviction_is_transparent(g, c.total, c.keep_from, c.delta, 500000.0f,
                                       RopeScaling{}, 5e-4f, c.what);
    }
}

// The overlap split is the one piece of the launcher that is not obviously
// correct: src == dst + delta, so when more rows move than delta, a single
// launch would have blocks overwriting rows other blocks still have to read.
// delta=4 against 120 moving rows forces 30 stream-ordered chunks.
TEST_F(KvEvictValidation, TransparentWhenTheShiftOverlapsItsOwnSource) {
    const Geom g{};
    expect_eviction_is_transparent(g, /*total=*/128, /*keep_from=*/4, /*delta=*/4,
                                   500000.0f, RopeScaling{}, 5e-4f, "delta << rows");
    expect_eviction_is_transparent(g, /*total=*/200, /*keep_from=*/0, /*delta=*/1,
                                   500000.0f, RopeScaling{}, 5e-4f, "single-row shift");
}

// apply_rope_scaling is position-independent, so the composition argument holds
// verbatim under llama3 rope_scaling — but that is a claim about someone else's
// function, so it gets asserted rather than reasoned about.
TEST_F(KvEvictValidation, TransparentUnderLlama3RopeScaling) {
    const Geom g{};
    RopeScaling s{};
    s.enabled = 1;
    s.factor = 8.0f;
    s.low_freq_factor = 1.0f;
    s.high_freq_factor = 4.0f;
    s.orig_ctx = 8192.0f;
    expect_eviction_is_transparent(g, /*total=*/96, /*keep_from=*/8, /*delta=*/32,
                                   500000.0f, s, 5e-4f, "llama3 scaling");
}

TEST_F(KvEvictValidation, TransparentAtProductionHeadGeometry) {
    Geom g{};
    g.q_heads = 32;      // Llama-3.1-8B: 32 q heads over 8 kv heads, head_dim 128
    g.kv_heads = 8;
    g.head_dim = 128;
    g.max_seq = 256;
    expect_eviction_is_transparent(g, /*total=*/192, /*keep_from=*/16, /*delta=*/64,
                                   500000.0f, RopeScaling{}, 5e-4f, "8B geometry");
}

// PARTIAL ROTARY. A model that rotates only head_dim/2 channels (GLM-4) leaves
// the upper half phase-free. A full-head evict would re-phase those channels
// against a frequency they never carried, so they are the interesting ones here:
// expect_eviction_is_transparent compares the WHOLE row, tail included.
TEST_F(KvEvictValidation, TransparentAtPartialRotaryHalfSplit) {
    Geom g{};
    g.head_dim = 128;
    g.rotary_dim = 64;       // partial_rotary_factor 0.5, half-split pairing
    g.max_seq = 256;
    expect_eviction_is_transparent(g, /*total=*/160, /*keep_from=*/8, /*delta=*/32,
                                   500000.0f, RopeScaling{}, 5e-4f, "partial half-split");
}

// The GLM-4 configuration end to end: adjacent-pair rotation over 64 of 128
// channels, appended by the interleaved kernel and evicted by the matching
// pairing. Pins the claim that the composition argument is about the ANGLE, not
// about which two channels happen to hold it.
TEST_F(KvEvictValidation, TransparentAtGlm4PartialInterleaved) {
    Geom g{};
    g.q_heads = 8;
    g.kv_heads = 2;          // GLM-4-9B GQA ratio 16 shrunk to keep the test quick
    g.head_dim = 128;
    g.rotary_dim = 64;
    g.interleaved = true;
    g.max_seq = 256;
    expect_eviction_is_transparent(g, /*total=*/160, /*keep_from=*/4, /*delta=*/48,
                                   10000.0f, RopeScaling{}, 5e-4f, "GLM-4 interleaved");
    expect_eviction_is_transparent(g, /*total=*/128, /*keep_from=*/0, /*delta=*/4,
                                   10000.0f, RopeScaling{}, 5e-4f,
                                   "GLM-4 interleaved, overlapping shift");
}

// Evictions compose: a window that has already slid can slide again, which is
// the actual steady state of a long session.
TEST_F(KvEvictValidation, RepeatedEvictionsStayTransparent) {
    const Geom g{};
    const int total = 128, keep = 8, delta = 24;
    const Rows rows = make_rows(g, total, /*seed=*/77u);

    std::vector<int> ids, pos;
    for (int i = 0; i < total; ++i) {
        ids.push_back(i);
        pos.push_back(i);
    }
    Cache live = build_cache(g, rows, ids, pos, 500000.0f, RopeScaling{});

    int len = total;
    int dropped = 0;
    for (int round = 0; round < 3; ++round) {
        live = evict(g, live, keep, delta, len, 500000.0f, RopeScaling{});
        len -= delta;
        dropped += delta;
    }

    // The survivors are the prefix plus everything from keep+dropped onward,
    // sitting at positions shifted down by the TOTAL dropped.
    std::vector<int> ref_ids, ref_pos;
    for (int i = 0; i < keep; ++i) {
        ref_ids.push_back(i);
        ref_pos.push_back(i);
    }
    for (int i = keep + dropped; i < total; ++i) {
        ref_ids.push_back(i);
        ref_pos.push_back(i - dropped);
    }
    const Cache scratch = build_cache(g, rows, ref_ids, ref_pos, 500000.0f, RopeScaling{});
    // Looser: three composed rotations accumulate three roundings.
    expect_live_region_matches(g, live, scratch, len, 1e-3f, "3 evictions");
}

// ---- the frozen prefix ------------------------------------------------------

// The sink is the whole reason the re-phase works: it must keep absolute phase
// 0..S-1, untouched, not merely "close".
TEST_F(KvEvictValidation, FrozenPrefixIsBitIdentical) {
    const Geom g{};
    const int total = 96, keep = 16, delta = 32;
    const Rows rows = make_rows(g, total, /*seed=*/5u);
    std::vector<int> ids, pos;
    for (int i = 0; i < total; ++i) {
        ids.push_back(i);
        pos.push_back(i);
    }
    const Cache before = build_cache(g, rows, ids, pos, 500000.0f, RopeScaling{});
    const Cache after = evict(g, before, keep, delta, total, 500000.0f, RopeScaling{});

    for (size_t h = 0; h < g.kv_heads; ++h) {
        for (int r = 0; r < keep; ++r) {
            const size_t off = (h * g.max_seq + static_cast<size_t>(r)) * g.head_dim;
            for (size_t c = 0; c < g.head_dim; ++c) {
                ASSERT_EQ(after.k[off + c], before.k[off + c])
                    << "prefix K moved at head " << h << " row " << r << " chan " << c;
                ASSERT_EQ(after.v[off + c], before.v[off + c])
                    << "prefix V moved at head " << h << " row " << r << " chan " << c;
            }
        }
    }
}

// ---- argument guards --------------------------------------------------------

// The launcher is a runtime-tier entry point on the engine thread: an illegal
// request must be a no-op, not a stomp past the end of the cache.
TEST_F(KvEvictValidation, IllegalArgumentsLeaveTheCacheUntouched) {
    const Geom g{};
    const int total = 64;
    const Rows rows = make_rows(g, total, /*seed=*/9u);
    std::vector<int> ids, pos;
    for (int i = 0; i < total; ++i) {
        ids.push_back(i);
        pos.push_back(i);
    }
    const Cache base = build_cache(g, rows, ids, pos, 500000.0f, RopeScaling{});

    struct Bad { int keep, delta, len; const char* what; };
    const Bad bad[] = {
        {8, 0, total, "delta == 0"},
        {8, -4, total, "negative delta"},
        {-1, 8, total, "negative keep_from"},
        {8, 100, total, "keep_from + delta past the tail"},
        {total, 8, total, "prefix already at the tail"},
    };
    for (const Bad& b : bad) {
        const Cache got = evict(g, base, b.keep, b.delta, b.len, 500000.0f, RopeScaling{});
        test_utils::expect_allclose(base.k, got.k, 0.0f, b.what);
        test_utils::expect_allclose(base.v, got.v, 0.0f, b.what);
    }
}

}  // namespace
