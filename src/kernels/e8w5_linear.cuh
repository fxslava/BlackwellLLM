#pragma once
#include <cstddef>

// E8W5 GEMV (5-bit companded-E8 lattice weights, FP32 activations).
//
//   d_out[n] = sum_k  codebook[(coset|u)(n,k)] * scales[n][k/128] * d_in[k]
//
// Expected device-memory layouts (docs/E8W5_FORMAT_SPEC.md; produced by
// scripts/e8w5_convert.py). Output-major, matching the engine's BF16 weight convention
// W[out_features][in_features] rather than AWQ's K-major checkpoint layout:
//   plane_lo : uint32[out_features][in_features/8]    one word per 8-weight E8 block
//   plane_hi : uint8 [out_features][in_features/8]    one byte per block
//   scales   : half  [out_features][in_features/128]
//   codebook : half  [64]                             per tensor
//   d_in     : float[in_features], d_out : float[out_features]
//
// in_features MUST be a multiple of 128 (E8W5_GROUP). That is the format's own group
// alignment, and the converter pads to it, so a violation is a loader bug rather than a
// shape the kernel should silently tolerate -- the launcher rejects it.
// Launch is asynchronous on the default stream.
void launch_e8w5_gemv_kernel(const void* plane_lo,
                             const void* plane_hi,
                             const void* scales,
                             const void* codebook,
                             const float* d_in,
                             float* d_out,
                             size_t out_features,
                             size_t in_features);

// Residual variant: d_residual_accum[n] += GEMV result. Used by o_proj / down_proj, which
// land directly on the residual stream. Mirrors launch_bf16_gemv_residual_kernel.
void launch_e8w5_gemv_residual_kernel(const void* plane_lo,
                                      const void* plane_hi,
                                      const void* scales,
                                      const void* codebook,
                                      const float* d_in,
                                      float* d_residual_accum,
                                      size_t out_features,
                                      size_t in_features);

// Dequantize a whole tensor to FP32 [out_features, in_features]. Not on any hot path: it
// exists so tests can compare the device decode against the reference unpacker directly,
// and so a batched path can be built on top before a fused MMA kernel exists.
void launch_e8w5_dequantize(const void* plane_lo,
                            const void* plane_hi,
                            const void* scales,
                            const void* codebook,
                            float* d_W_out,
                            size_t out_features,
                            size_t in_features);
