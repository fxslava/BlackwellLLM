#pragma once
#include <cstddef>

// ---------------------------------------------------------------------------
// RoPE frequency rescaling (Llama-3 "rope_scaling"). Default-constructed => OFF,
// so every existing launcher call is byte-identical vanilla rotate_half. Plain
// POD (no CUDA types) so core .cpp TUs compiled by MSVC can include this header;
// the device-side apply_rope_scaling helper is guarded to nvcc only.
//
// enabled==0  -> inv_freq unchanged (vanilla).
// enabled==1  -> Llama-3 wavelength-band rescaling (transformers
//                _compute_llama3_parameters): high-freq band unchanged, low-freq
//                band divided by `factor`, medium band smoothly interpolated.
// ---------------------------------------------------------------------------
struct RopeScaling {
    int   enabled = 0;
    float factor = 1.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 1.0f;
    float orig_ctx = 0.0f;   // original_max_position_embeddings
};

#ifdef __CUDACC__
__device__ __forceinline__ float apply_rope_scaling(float inv_freq, RopeScaling s) {
    if (!s.enabled) return inv_freq;
    const float kTwoPi = 6.28318530717958647692f;
    const float low_wavelen  = s.orig_ctx / s.low_freq_factor;   // long-wavelength bound
    const float high_wavelen = s.orig_ctx / s.high_freq_factor;  // short-wavelength bound
    const float wavelen = __fdividef(kTwoPi, inv_freq);
    if (wavelen < high_wavelen)                       // high-frequency band: unchanged
        return inv_freq;
    if (wavelen > low_wavelen)                         // low-frequency band: /factor
        return __fdividef(inv_freq, s.factor);
    // medium band: smooth interpolation between /factor and unchanged
    const float smooth = (s.orig_ctx / wavelen - s.low_freq_factor) /
                         (s.high_freq_factor - s.low_freq_factor);
    return (1.0f - smooth) * __fdividef(inv_freq, s.factor) + smooth * inv_freq;
}
#endif

// rotate_half RoPE on Q and K plus append of rotated K / raw V into the
// KV cache slot for `pos`.
//
// HARD LAUNCH CONTRACT (violations are NOT diagnosed at runtime):
//   * head_dim must be even (rotate_half pairs channel k with
//     k + head_dim/2) and head_dim/2 <= 1024 (one thread per pair,
//     blockDim.x = head_dim / 2).
//   * K_cache / V_cache layout: float[kv_heads][max_seq_len][head_dim].
void launch_fused_rope_kv_kernel(
    float* d_Q,
    float* d_K,
    const float* d_V,
    float* d_K_cache,
    float* d_V_cache,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    float rope_theta = 500000.0f,
    RopeScaling scaling = {});

// rotate_half RoPE applied IN PLACE to a [num_heads, head_dim] buffer, with no
// cache write. Used by the paged KV path, which appends the rotated K/V into a
// scattered physical page rather than a contiguous slab (so it cannot use the
// fused rope+cache-write kernel above). Apply once to Q and once to K.
//   HARD CONTRACT: head_dim even, head_dim/2 <= 1024.
void launch_rope_inplace(
    float* d_X,
    int pos,
    size_t num_heads,
    size_t head_dim,
    float rope_theta = 500000.0f,
    RopeScaling scaling = {});

// Batched rotate_half RoPE over a [num_tokens, num_heads, head_dim] buffer: row
// t is rotated for logical position (start_pos + t). The num_tokens == 1 case is
// exactly launch_rope_inplace(pos=start_pos). Batched-prefill counterpart of the
// per-token in-place RoPE; the rotated K is then scatter-appended into the pages.
//   HARD CONTRACT: head_dim even, head_dim/2 <= 1024.
void launch_rope_inplace_batched(
    float* d_X,
    int start_pos,
    size_t num_tokens,
    size_t num_heads,
    size_t head_dim,
    float rope_theta = 500000.0f,
    RopeScaling scaling = {});

// TRUE-batch (multi-sequence) rotate_half RoPE over a [batch_size, num_heads,
// head_dim] buffer: row b is rotated for its OWN logical position d_positions[b].
// This is the sequence-aware sibling of launch_rope_inplace_batched — the batched
// decode path packs one query row per INDEPENDENT sequence, and sequences sit at
// different generation steps, so the position is per-row rather than start_pos+t.
// d_positions is a device array of batch_size ints. Apply once to Q, once to K.
//   HARD CONTRACT: head_dim even, head_dim/2 <= 1024.
void launch_batched_rope(
    float* d_X,
    const int* d_positions,
    int batch_size,
    size_t num_heads,
    size_t head_dim,
    float rope_theta = 500000.0f,
    RopeScaling scaling = {});