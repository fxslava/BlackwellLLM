#pragma once
// E8W5 dequantization primitives -- the engine's copy.
//
// Format: docs/E8W5_FORMAT_SPEC.md. This is a faithful copy of
// research/kernels/e8w5_dequant.cuh; the engine may not include anything from research/
// (research/README.md), so the two live side by side and tests/validation/test_e8w5_linear.cpp
// pins them together by checking this decode against the format's bit definition. If you
// change the algebra here, change it there and re-run `python research/pack_e8w5.py --self-test`.
//
// Storage, per quantized [out_features, in_features] tensor (in_features % 128 == 0):
//   plane_lo : uint32[out_features][in_features/8]   one word per 8-weight E8 block
//   plane_hi : uint8 [out_features][in_features/8]   one byte per block
//   scales   : half  [out_features][in_features/128] one per 128-weight group
//   codebook : half  [64]                            per tensor, indexed by (coset, u)
//
// Deviation from the spec's "A16" name, and it is deliberate: this engine carries
// activations in FP32 end to end (LinearDispatcher::forward takes `const float*`), and the
// golden-dump integration tests pin that precision. So the decode hands back FP32 and the
// dot products use fmaf. The half2/__hfma2 form in the research header is for an FP16-
// activation engine, which this is not.

#include <cuda_fp16.h>
#include <cstdint>

#define E8W5_COORD_BITS 5
#define E8W5_DIM 8                   // coordinates per E8 block
#define E8W5_LUT_N 64                // 2 * 2^b, indexed by (coset, u)
#define E8W5_GROUP 128               // weights per scale group
#define E8W5_BLOCKS_PER_GROUP 16     // GROUP / DIM
#define E8W5_BLOCKS_PER_UINT4 4      // one uint4 of plane L == 4 blocks == 32 weights
#define E8W5_COSET_BIT 28
#define E8W5_PARITY_MASK 0x01111111u // bit 0 of nibbles 0..6
#define E8W5_COSET_CLEAR 0xEFFFFFFFu

// Stage the per-tensor codebook into shared memory. 128 B in FP16, and that precision is a
// memory-system choice as much as a numerical one: 64 halves span 32 four-byte words, one
// per bank, so a 32-lane gather is conflict-free for ANY index pattern. The same table in
// FP32 would be two words per bank and could two-way conflict. __constant__ is worse still:
// it serializes across differing addresses in a warp, and these indices are data-dependent
// and uncorrelated across lanes.
__device__ __forceinline__ void e8w5_stage_codebook(const __half* __restrict__ src,
                                                    __half* __restrict__ smem,
                                                    unsigned tid, unsigned nthreads)
{
    for (unsigned i = tid; i < E8W5_LUT_N; i += nthreads) smem[i] = src[i];
}

// Spread plane H's eight bits so coordinate i's bit 4 lands at bit 4*i -- the same nibble
// grid plane L uses. 3 x (shift, or, and) = 9 ops per block.
__device__ __forceinline__ uint32_t e8w5_spread8(uint32_t h)
{
    h = (h | (h << 12)) & 0x000F000Fu;
    h = (h | (h << 6)) & 0x03030303u;
    h = (h | (h << 3)) & 0x11111111u;
    return h;
}

__device__ __forceinline__ uint32_t e8w5_coset(uint32_t lo4)
{
    return (lo4 >> E8W5_COSET_BIT) & 1u;
}

// Recover u_7's implied low bit from D8's even-sum parity and write it into the slot the
// coset flag occupied. AFTER this, nibble i == u_i & 0xF for ALL eight coordinates, so
// coordinate 7 stops being a special case and the per-coordinate loop stays uniform.
// Three ops buy that uniformity. Read the coset BEFORE calling this.
__device__ __forceinline__ uint32_t e8w5_uniform_nibbles(uint32_t lo4)
{
    const uint32_t parity = static_cast<uint32_t>(__popc(lo4 & E8W5_PARITY_MASK)) & 1u;
    return (lo4 & E8W5_COSET_CLEAR) | (parity << E8W5_COSET_BIT);
}

// One block -> eight FP32 weights in codebook units (the group scale is NOT applied; see
// e8w5_block_dot8 for why folding it once per group is free). The 6-bit index's top bit IS
// the coset flag, so the coset costs no select of its own.
__device__ __forceinline__ void e8w5_decode_block_f32(uint32_t lo4, uint32_t hi8,
                                                      const __half* __restrict__ cb,
                                                      float (&out)[E8W5_DIM])
{
    const uint32_t base = e8w5_coset(lo4) << E8W5_COORD_BITS;
    const uint32_t nib = e8w5_uniform_nibbles(lo4);
    const uint32_t sp = e8w5_spread8(hi8);         // coord i's bit 4 sits at bit 4i

#pragma unroll
    for (int i = 0; i < E8W5_DIM; ++i) {
        const uint32_t s = 4u * static_cast<uint32_t>(i);
        // Extract THEN shift. Pre-shifting sp by 4 would put coordinate 7's MSB at bit 32
        // and drop it off the word -- a bug that only shows up in 1 coordinate of 8, which
        // is why tests/validation/test_e8w5_linear.cpp sweeps all 8 positions explicitly.
        const uint32_t idx = base | ((nib >> s) & 0xFu) | (((sp >> s) & 1u) << 4);
        out[i] = __half2float(cb[idx]);
    }
}

// Eight weights of one block against eight activations, accumulated in FP32.
// The group scale multiplies every weight in the group, so it comes straight out of the
// sum: sum_j (delta * v_j) * x_j == delta * sum_j v_j * x_j. That is ONE multiply per 128
// weights instead of one per weight -- per-group scaling is free in the inner loop, and
// costs only the 0.125 bpw it occupies in the file.
__device__ __forceinline__ float e8w5_block_dot8(uint32_t lo4, uint32_t hi8,
                                                 const __half* __restrict__ cb,
                                                 const float* __restrict__ x)
{
    float w[E8W5_DIM];
    e8w5_decode_block_f32(lo4, hi8, cb, w);
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < E8W5_DIM; ++i) acc = fmaf(w[i], x[i], acc);
    return acc;
}

// One uint4 of plane L + one uint32 of plane H == 4 blocks == 32 weights, all inside one
// scale group (4 | 16), against 32 activations. Returns the UNSCALED partial dot.
__device__ __forceinline__ float e8w5_vec4_dot32(uint4 L, uint32_t H,
                                                 const __half* __restrict__ cb,
                                                 const float* __restrict__ x)
{
    const uint32_t lw[E8W5_BLOCKS_PER_UINT4] = {L.x, L.y, L.z, L.w};
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < E8W5_BLOCKS_PER_UINT4; ++b)
        acc += e8w5_block_dot8(lw[b], (H >> (8 * b)) & 0xFFu, cb,
                               x + b * E8W5_DIM);
    return acc;
}
