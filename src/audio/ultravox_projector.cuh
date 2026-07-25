#pragma once
#include <cstddef>
// -----------------------------------------------------------------------------
// Ultravox projector CUDA kernels (audio frontend, preprocessor-only).
//
// FP32 activations throughout: this matches the engine's activation dtype AND
// the FP32 PyTorch golden reference the parity test compares against (cosine
// > 0.999). The dumps produced by scripts/generate_ultravox_audio_dumps.py are
// float32, so there is no half here.
//
// HARD LAUNCH CONTRACTS (not diagnosed at runtime):
//   * These launchers perform NO device allocation. The caller owns d_in/d_out
//     (persistent workspace), per the no-malloc-in-the-forward-pass rule.
//   * d_in and d_out must be distinct (no aliasing) and 16-byte aligned (any
//     cudaMalloc'd / DeviceBuffer base is).
// -----------------------------------------------------------------------------

// StackAudioFrames: [num_frames, hidden_dim] -> [ceil(num_frames/cf), hidden_dim*cf].
// Output row t' is the concatenation of input frames [t'*cf, t'*cf + cf); input
// frames at or past num_frames are ZERO-padded (mirrors Ultravox's
// F.pad(..)-then-reshape when num_frames is not a multiple of cf).
// CONTRACT: hidden_dim % 4 == 0 (float4-vectorized copy).
void launch_stack_audio_frames(const float* d_in,
                               float* d_out,
                               int num_frames,
                               int hidden_dim,
                               int compression_factor);

// SwiGLU: [num_tokens, input_dim] -> [num_tokens, input_dim/2].
//
// Ultravox convention (transformers `SwiGLU`: `x, gate = x.chunk(2, -1);
// return F.silu(gate) * x`): the row splits into two halves; the FIRST half is
// the multiplicand `x`, the SECOND half is the `gate`. Output = SiLU(gate) * x,
// with SiLU(v) = v * sigmoid(v).
//
//   NB: this is the reverse of a naive "gate = first half, up = second half"
//   reading. The golden dump proj_swiglu.bin was produced with the Ultravox
//   assignment above, so the kernel matches it (SiLU is applied to the SECOND
//   half). Getting this backwards fails the cosine-similarity parity check.
//
// CONTRACT: input_dim % 2 == 0 and (input_dim/2) % 4 == 0 (float4).
void launch_swiglu(const float* d_in,
                   float* d_out,
                   int num_tokens,
                   int input_dim);
