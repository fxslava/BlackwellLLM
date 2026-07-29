#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#include "rope.cuh"   // RopeScaling (POD; the device helper is nvcc-guarded)

// ---------------------------------------------------------------------------
// RoPE-AWARE KV HEAD EVICTION — the sliding window for continuous streaming
// (docs/CONTINUOUS_STREAMING.md, T3).
//
// Drops `delta` rows from the FRONT of the live region of the KV cache and
// slides everything above them down, so a session can run indefinitely without
// ever resetting. Rows below `keep_from` are the frozen prefix and are never
// touched: they keep absolute RoPE phase 0..keep_from-1 and remain the attention
// sink anchored at true position 0.
//
// WHY THIS IS NOT A MEMMOVE. rope_kv_append_kernel rotates K by `pos * freq`
// BEFORE writing it to the cache (src/kernels/rope.cu), so cached keys carry
// ABSOLUTE positional phase. Sliding a key from row p to row p-delta without
// touching its contents would leave it claiming position p while sitting at
// p-delta: every relative distance the attention dot product measures against it
// would be wrong by delta. (Leaving a positional hole instead is no better --
// absolute positions then grow without bound and the sink-to-current distance
// degrades exactly as it does past the trained context, which is the failure
// this whole mechanism exists to escape.)
//
// WHY THE FIX IS EXACT. `freq` is a pure function of (channel, head_dim,
// rope_theta, scaling) -- there is no position in it, and apply_rope_scaling
// does not introduce one. RoPE is therefore a rotation by `p * freq_j` in each
// 2-D channel pair, and rotations about the same axis COMPOSE by adding angles.
// Rotating a key cached at p by a further `-delta * freq_j` yields precisely the
// key that would have been cached at p - delta. V carries no phase at all and is
// a pure copy.
//
// Equality is exact in the reals, not in float: cos(p*f)cos(d*f) + sin(p*f)sin(d*f)
// and cos((p-d)*f) differ in the last ulps. The equivalence test asserts a tight
// tolerance rather than bit-identity, and that is a property of IEEE arithmetic,
// not a weakness in the transform.
//
// HARD LAUNCH CONTRACT (violations are NOT diagnosed at runtime):
//   * K_cache / V_cache layout float[kv_heads][max_seq_len][head_dim] -- the
//     same layout rope_kv_append_kernel writes.
//   * head_dim even, head_dim/2 <= 1024 (one thread per channel pair).
//   * 0 <= keep_from, delta > 0, keep_from + delta <= cache_len <= max_seq_len.
//   * rope_theta and scaling MUST match what the append kernel was called with,
//     or the composed angle is measured against a different frequency ladder.
//
// AFTER THE CALL the live region is [0, cache_len - delta): rows
// [keep_from, cache_len - delta) hold what used to be at [keep_from + delta,
// cache_len), re-phased. The vacated tail is left as-is rather than zeroed --
// it sits above the new sequence length, so attention never reads it and the
// next prefill overwrites it.
//
// OVERLAP. This is a downward in-place shift, so source and destination overlap
// whenever delta < rows-to-move. The launcher splits the work into stream-
// ordered chunks of at most `delta` rows, which is what makes the general case
// correct; the common case (a large eviction) is a single launch.
//
// Not thread-safe with respect to the cache: this is the single engine thread's
// operation, like every other KV mutation (CLAUDE.md doctrine).
// ---------------------------------------------------------------------------
void launch_kv_evict_head(
    float* d_K_cache,
    float* d_V_cache,
    int keep_from,       // first row that survives (the frozen prefix length, S)
    int delta,           // rows dropped from [keep_from, keep_from + delta)
    int cache_len,       // live rows before the call (the sequence length)
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    float rope_theta = 500000.0f,
    RopeScaling scaling = {},
    cudaStream_t stream = 0);
