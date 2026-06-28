#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// Symmetric INT4 (compressed-tensors "pack-quantized") GEMV — y = W·x
// ============================================================================
// Weight format (verified against the Qwen3.5 checkpoint headers):
//   weight_packed : int32 [out_features, in_features/8]
//                   8 signed int4 per int32, packed along the INPUT dim, with
//                   element t (t=0..7) of a word in bits [4t, 4t+4). The signed
//                   value q in [-8,7] is stored as the UNSIGNED nibble q+8, so it
//                   is recovered as (nibble - 8) -- the compressed-tensors offset
//                   convention, NOT two's-complement. There is no separate
//                   zero-point tensor (symmetric).
//   weight_scale  : bf16  [out_features, in_features/group_size]
//                   one scale per contiguous group of `group_size` input
//                   elements (group_size = 32 for this checkpoint).
//   dequant: W[o, in] = sign_extend(nibble) * scale[o, in / group_size]
//
// x is fp32 (the engine's activation buffers); y is fp32. The _residual variant
// accumulates into an existing fp32 buffer in place (for o_proj / down_proj).
//
// NOTE: the nibble order / sign convention above is the documented compressed-
// tensors pack_to_int32 layout; it is proven self-consistent (pack→unpack→GEMV)
// by test_sym_int4_gemv.cu, and is pinned against the real checkpoint only by the
// golden-dump integration test.

void launch_sym_int4_gemv(
    const int32_t* d_packed,
    const __nv_bfloat16* d_scales,
    const float* d_x,
    float* d_y,
    int out_features,
    int in_features,
    int group_size,
    cudaStream_t stream = 0);

// y_accum[o] += W·x   (in-place residual accumulation)
void launch_sym_int4_gemv_residual(
    const int32_t* d_packed,
    const __nv_bfloat16* d_scales,
    const float* d_x,
    float* d_y_accum,
    int out_features,
    int in_features,
    int group_size,
    cudaStream_t stream = 0);
