#pragma once
#include <cstddef>
#include <cuda_runtime.h>   // cudaStream_t
#include "rope.cuh"   // RopeScaling (shared frequency-rescaling POD)

// ============================================================================
// Qwen3.5 hybrid FULL-attention primitives (head_dim == 256, gated output).
//
// The periodic softmax-attention layers of Qwen3.5 differ from the Qwen2.5/Llama
// decode path in three ways the generic attention.cu kernel cannot express:
//   * head_dim = 256 (> the 128-wide attention_decoding_kernel block);
//   * q_proj emits num_heads * head_dim * 2 channels — per head a [query|gate]
//     pair that must be de-interleaved before q_norm / RoPE;
//   * the attention context is element-wise gated by sigmoid(gate) before o_proj.
// These helpers implement a small, unoptimized ("make it work") path for exactly
// those layers; performance is a later concern.
// ============================================================================

// De-interleave the q_proj output. d_QG is [num_heads, 2*head_dim] laid out per
// head as [query(head_dim) | gate(head_dim)] (HF torch.chunk(..., 2, dim=-1)).
// Writes the query half to d_Q and the gate half to d_gate, each
// [num_heads, head_dim].
void launch_qg_split(const float* d_QG, float* d_Q, float* d_gate,
                     int num_heads, int head_dim);

// rotate_half RoPE applied IN PLACE over only the first `rotary_dim` channels of
// each head (partial rotary; channels [rotary_dim, head_dim) are pass-through),
// for a [num_heads, head_dim] buffer with no cache write. Apply once to Q and
// once to K. At pos == 0 this is the identity (sin == 0).
//   HARD CONTRACT: rotary_dim even, rotary_dim <= head_dim, rotary_dim/2 <= 1024.
void launch_rope_partial_inplace(float* d_X, int pos, int num_heads,
                                 int head_dim, int rotary_dim, float rope_theta,
                                 RopeScaling scaling = {});

// ---------------------------------------------------------------------------
// GLM-4 partial RoPE: INTERLEAVED pairing (HF Glm*/Glm4* apply_rotary_pos_emb).
//
// Pair j is the ADJACENT channel couple (2j, 2j+1) for j < rotary_dim/2, rotated
// by pos * theta^(-2j/rotary_dim) -- the same frequency ladder, in the same
// order, as launch_rope_partial_inplace's half-split pairing above. Channels
// [rotary_dim, head_dim) are left strictly untouched (GLM rotates only
// head_dim * partial_rotary_factor = 64 of its 128 channels).
//
// Q and K are rotated by ONE launch (grid.x == q_heads + kv_heads): both need the
// identical angle for the identical position, and merging them saves a launch per
// layer per token on the decode hot path. At pos == 0 this is the identity.
//   HARD CONTRACT (not diagnosed at runtime):
//     * rotary_dim even, 2 <= rotary_dim <= head_dim, rotary_dim/2 <= 1024;
//     * d_Q is [q_heads, head_dim], d_K is [kv_heads, head_dim], both fp32;
//     * `scaling` must match what every other RoPE call for this model passes --
//       a cached key carries absolute phase, so a mismatched ladder is silent.
void launch_rope_interleaved_partial_inplace(
    float* d_Q, float* d_K, int pos,
    size_t q_heads, size_t kv_heads,
    size_t head_dim, size_t rotary_dim,
    float rope_theta, RopeScaling scaling = {}, cudaStream_t stream = 0);

// Batched sibling of the above over a [num_tokens, num_heads, head_dim] buffer:
// row t is rotated for logical position (start_pos + t). num_tokens == 1 is
// exactly launch_rope_interleaved_partial_inplace on a single buffer. Apply once
// to the Q chunk and once to the K chunk (they have different head counts, so
// unlike the decode launcher this one takes a single buffer).
//   HARD CONTRACT: as above.
void launch_rope_interleaved_partial_inplace_batched(
    float* d_X, int start_pos, size_t num_tokens, size_t num_heads,
    size_t head_dim, size_t rotary_dim,
    float rope_theta, RopeScaling scaling = {}, cudaStream_t stream = 0);

// Append the (already RoPE'd) K and raw V for the current token into a dedicated
// contiguous FP32 cache: float[kv_heads][max_seq_len][head_dim]. No rotation here.
void launch_kv_append(const float* d_K, const float* d_V,
                      float* d_K_cache, float* d_V_cache,
                      int pos, int kv_heads, int head_dim, int max_seq_len);

// Single-token decode attention (online softmax over positions 0..pos), GQA-aware,
// scale = head_dim^-0.5. Supports head_dim up to 256 (blockDim.x == head_dim).
// Cache layout matches launch_kv_append. Result O is [q_heads, head_dim].
void launch_full_attention_decode(const float* d_Q, const float* d_K_cache,
                                  const float* d_V_cache, float* d_O, int pos,
                                  int q_heads, int kv_heads, int head_dim,
                                  int max_seq_len);

// Gate the attention context: d_O[i] *= sigmoid(d_gate[i]) over `n` elements.
void launch_gate_sigmoid_mul(float* d_O, const float* d_gate, int n);
