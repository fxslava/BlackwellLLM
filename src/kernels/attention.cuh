#pragma once
#include <cstddef>

#include <cuda_runtime.h>

// Single-token decode attention (online softmax over positions 0..pos).
//
// HARD LAUNCH CONTRACT (violations are NOT diagnosed at runtime):
//   * head_dim <= 128 (the fixed block size). Each thread owns one channel
//     of the head; channels >= blockDim.x would be SILENTLY DROPPED from
//     both the Q*K dot product and the output vector.
//   * q_heads % kv_heads == 0 (GQA ratio is derived as gridDim.x / kv_heads).
//   * K_cache / V_cache layout: float[kv_heads][max_seq_len][head_dim].
void launch_attention_decoding_kernel(
    const float* d_Q,
    const float* d_K_cache,
    const float* d_V_cache,
    float* d_O,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    cudaStream_t stream = 0);

// ============================================================================
// Split-K decode attention (Flash-Decoding)
// ============================================================================
// The single-block kernel above launches exactly q_heads blocks. At GLM-4-9B
// geometry that is 32 blocks on a 56-SM part: ~43% of the device is idle for
// the whole of attention, and the latency is linear in the context length with
// nothing to hide it behind (batch = 1 -- there is no other work in flight).
//
// Split-K partitions the CONTEXT dimension instead of the head dimension, so
// the grid becomes [q_heads, S] and fills the device at any head count:
//
//   Phase 1 (split_k_attention_partial_kernel, grid [q_heads, S]): block (h, s)
//     runs the same exact online softmax over its own token slice
//     t in [s*L, min((s+1)*L, N)), L = ceil(N/S), and writes the slice's
//     partial context vector plus its LOCAL softmax statistics (m, l) to a
//     scratchpad.
//   Phase 2 (split_k_attention_reduce_kernel, grid [q_heads]): rescales the S
//     partials by the GLOBAL log-sum-exp and writes the final output.
//
// The result is mathematically the same attention, so the two phases together
// are a drop-in replacement for the single-block kernel (they even reproduce
// its deliberate bf16 output truncation); they differ only in fp32 summation
// order, hence agree with it to within one bf16 ULP.
namespace blackwell {
namespace attn {

// Threads per block, both phases. Phase 1 gives one thread one output channel,
// which is what caps head_dim at 128 (same contract as the single-block kernel).
constexpr int kSplitKBlockSize = 128;

// Upper bound on the split factor. Phase 2 reduces the S slices inside ONE warp,
// so this must stay <= 32. It also sizes the scratchpad (see below), so raising
// it costs VRAM in the arena.
// MIRRORED by blackwell::KernelLimits::kAttentionSplitKMax (runtime_config.h) --
// the validator clamps the tier-2 knob against that copy; a static_assert in
// kv_cache/continuous_kv_manager.cpp keeps the two honest.
constexpr int kSplitKMaxSplits = 16;
static_assert(kSplitKMaxSplits <= 32,
              "phase 2 reduces the splits within a single warp");

// Floats of scratchpad one decode step needs for `max_splits` splits. ONE
// allocation, three regions laid out back to back:
//   [0,               q*S*D)          partial accumulators, [q_head][split][channel]
//   [q*S*D,           q*S*D + q*S)    per-slice running max   m
//   [q*S*D + q*S,     q*S*D + 2*q*S)  per-slice running sum   l
// The launcher derives the SAME layout from the split factor it actually picks,
// which is <= max_splits, so any allocation sized from this helper serves every
// step without reallocation. ~266 KB at q=32, D=128, S=16 -- deliberately small
// enough to stay L2-resident across the phase boundary.
inline size_t split_k_scratch_floats(size_t q_heads, size_t head_dim,
                                     int max_splits = kSplitKMaxSplits) {
    const size_t slices = q_heads * static_cast<size_t>(max_splits);
    return slices * head_dim + 2 * slices;
}

// Split factor for a context of `context_length` tokens, clamped to max_splits.
//
// A ladder rather than a formula of the SM count: the win is bounded below by
// the per-block fixed cost (Q load + the two-kernel launch pair), so short
// contexts must NOT be split at all, and above ~4k tokens the grid is already
// wide enough that more slices only add reduction work. max_splits == 1 is how
// the runtime plan disables split-K entirely (parity / debugging).
inline int select_split_k(int context_length, int max_splits = kSplitKMaxSplits) {
    if (max_splits < 1) max_splits = 1;
    if (max_splits > kSplitKMaxSplits) max_splits = kSplitKMaxSplits;

    int splits;
    if (context_length < 256)       splits = 1;
    else if (context_length < 1024) splits = 4;
    else if (context_length < 4096) splits = 8;
    else                            splits = 16;

    if (splits > max_splits)     splits = max_splits;
    if (splits > context_length) splits = context_length;
    return splits < 1 ? 1 : splits;
}

// Register / shared-memory / spill footprint of the three decode-attention
// kernels, as the driver reports them for the CURRENT device. Queried by the
// benchmark suite; `cudaFuncGetAttributes` needs the kernel symbols, which only
// attention.cu sees.
struct SplitKKernelStats {
    int    partial_num_regs     = 0;
    size_t partial_shared_bytes = 0;
    size_t partial_local_bytes  = 0;   // > 0 means register spills
    int    reduce_num_regs      = 0;
    size_t reduce_shared_bytes  = 0;
    size_t reduce_local_bytes   = 0;
    int    legacy_num_regs      = 0;   // the single-block kernel, for comparison
    size_t legacy_shared_bytes  = 0;
    size_t legacy_local_bytes   = 0;
};
SplitKKernelStats query_split_k_kernel_stats();

} // namespace attn
} // namespace blackwell

// Two-phase split-K decode attention. Same inputs, same output buffer and the
// same HARD LAUNCH CONTRACT as launch_attention_decoding_kernel, plus:
//   * d_scratch holds at least
//     blackwell::attn::split_k_scratch_floats(q_heads, head_dim, splits) floats.
//     It is pure scratch: written before it is read every step, never read
//     across steps, so it needs no initialization and survives no state.
//   * splits is a REQUEST. The launcher clamps it to
//     [1, min(kSplitKMaxSplits, context_length)] and then re-derives it from the
//     resulting slice width, so no launched slice is ever empty.
//
// Degenerates to launch_attention_decoding_kernel -- the parity reference --
// when the resolved split factor is 1 or d_scratch is null, so a caller may
// hand it an unconditional `splits` and let the ladder decide.
void launch_attention_decoding_split_k(
    const float* d_Q,
    const float* d_K_cache,
    const float* d_V_cache,
    float* d_O,
    float* d_scratch,
    int splits,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    cudaStream_t stream = 0);
