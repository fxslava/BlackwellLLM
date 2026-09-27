// E8W5-A16 dequantization primitives: 5-bit companded-E8 weights, 16-bit activations.
//
// Format: docs/E8W5_FORMAT_SPEC.md. Offline packer + verification: research/pack_e8w5.py.
// Every integer operation below is mirrored op-for-op by `cuda_decode_mirror()` in that
// packer and verified bit-exact against the encoder over all 32 coordinate values in both
// cosets -- so if you change the algebra here, change it there and re-run --self-test.
//
// This header is RESEARCH code. It deliberately lives outside src/kernels/ because nothing
// in src/ or tests/ may depend on research/ (research/README.md), and because no kernel has
// been written against this format yet: the format is verified, its end-to-end quality at
// GROUP=128 is not. Read section 9 of the spec before shipping it.
//
// Two decode paths are provided:
//   e8w5_decode_block()        -- one bfe per coordinate; clearest, ~4.4 int ops/weight
//   e8w5_decode_block_bytes()  -- byte-aligned indices via masks; ~3.4 int ops/weight
// Both are static instruction counts, not measured kernels -- the same convention
// research/COMPANDED_E8.md uses for its silicon audit.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// --------------------------------------------------------------------------- //
// format constants -- must match research/pack_e8w5.py
// --------------------------------------------------------------------------- //

#define E8W5_COORD_BITS 5
#define E8W5_DIM 8                  // coordinates per E8 block
#define E8W5_LEVELS 32              // 1 << COORD_BITS
#define E8W5_LUT_N 64               // 2 * LEVELS, indexed by (coset, u)
#define E8W5_GROUP 128              // weights per scale group
#define E8W5_BLOCKS_PER_GROUP 16    // GROUP / DIM
#define E8W5_COSET_BIT 28           // plane-L bit holding the coset flag
#define E8W5_PARITY_MASK 0x01111111u// bit 0 of nibbles 0..6
#define E8W5_COSET_CLEAR 0xEFFFFFFFu

// Per GROUP=128 weights: plane L is 16 x uint32 = 64 B = four aligned uint4,
// plane H is 16 x uint8 = 16 B = one aligned uint4. 80 B total, zero padding.
#define E8W5_PLANE_LO_BYTES_PER_GROUP 64
#define E8W5_PLANE_HI_BYTES_PER_GROUP 16

// --------------------------------------------------------------------------- //
// codebook residency
// --------------------------------------------------------------------------- //
//
// The codebook is 64 conditional means fitted PER TENSOR (and it absorbs lambda, which the
// full-model sweep shows is not always 1.0), so it can be neither a compile-time constant
// nor one __constant__ table shared by the model. It is 128 B in FP16; stage it in shared
// memory once per block launch.
//
// Bank behaviour, which is why FP16 and not FP32: 64 halves span 32 four-byte words, so
// word w sits in bank w and every bank holds exactly one word. Two lanes reading different
// entries either hit different banks or hit the SAME word (an in-word broadcast), so a
// 32-lane gather into this table is conflict-free for ANY index pattern. The same table in
// FP32 would be 64 words over 32 banks -- two words per bank -- and a gather could then
// two-way conflict. The dequant table's precision is a memory-system decision here, not
// only a numerical one.

__device__ __forceinline__ void e8w5_stage_codebook(const __half* __restrict__ src,
                                                    __half* __restrict__ smem,
                                                    unsigned tid, unsigned nthreads)
{
    for (unsigned i = tid; i < E8W5_LUT_N; i += nthreads) smem[i] = src[i];
}

// --------------------------------------------------------------------------- //
// bit primitives
// --------------------------------------------------------------------------- //

// Spread the eight bits of plane H so that coordinate i's bit 4 lands at bit 4*i, i.e. onto
// the same nibble grid plane L already uses. 3 x (shift, or, and) = 9 ops per block.
__device__ __forceinline__ uint32_t e8w5_spread8(uint32_t h)
{
    h = (h | (h << 12)) & 0x000F000Fu;
    h = (h | (h << 6)) & 0x03030303u;
    h = (h | (h << 3)) & 0x11111111u;
    return h;
}

// Recover u_7's implied low bit and write it into the slot the coset flag occupied. After
// this, nibble i == u_i & 0xF for ALL eight coordinates -- coordinate 7 stops being a
// special case, which is what keeps the per-coordinate inner code uniform.
// POPC + AND + LOP3 = 3 ops per block. Read the coset BEFORE calling this.
__device__ __forceinline__ uint32_t e8w5_uniform_nibbles(uint32_t lo4)
{
    const uint32_t parity = static_cast<uint32_t>(__popc(lo4 & E8W5_PARITY_MASK)) & 1u;
    return (lo4 & E8W5_COSET_CLEAR) | (parity << E8W5_COSET_BIT);
}

__device__ __forceinline__ uint32_t e8w5_coset(uint32_t lo4)
{
    return (lo4 >> E8W5_COSET_BIT) & 1u;
}

// --------------------------------------------------------------------------- //
// decode: one block (8 weights) -> four half2, ready for __hfma2
// --------------------------------------------------------------------------- //
//
// The index is 6 bits: [coset | u_i], so the coset flag IS the codebook index's top bit and
// costs no separate select. Values come out in the codebook's own units; the group scale is
// NOT applied here -- see e8w5_group_dot() for why folding it once per group is free.
//
// Static count per block: 9 (spread) + 3 (parity fixup) + 2 (coset) + 1 (base) = 15 ops
// amortized (1.9/weight), plus per coordinate 1 bfe + 1 shift + 1 and + 1 lop3 = 4 ops
// and one LDS.u16. Total ~5.9 int ops + 1 shared load per weight, then 0.5 HFMA2.

__device__ __forceinline__ void e8w5_decode_block(uint32_t lo4, uint32_t hi8,
                                                  const __half* __restrict__ cb,
                                                  __half2 (&out)[4])
{
    const uint32_t base = e8w5_coset(lo4) << E8W5_COORD_BITS;   // index bit 5
    const uint32_t nib = e8w5_uniform_nibbles(lo4);
    const uint32_t sp = e8w5_spread8(hi8);                      // coord i's bit 4 at bit 4i

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t s0 = 8u * static_cast<uint32_t>(i);
        const uint32_t s1 = s0 + 4u;
        // Extract THEN shift: pre-shifting sp by 4 puts coordinate 7's MSB at bit 32 and
        // drops it. The byte-plane variant below is unaffected (its masks never cross 31).
        const uint32_t i0 = base | ((nib >> s0) & 0xFu) | (((sp >> s0) & 1u) << 4);
        const uint32_t i1 = base | ((nib >> s1) & 0xFu) | (((sp >> s1) & 1u) << 4);
        out[i] = __halves2half2(cb[i0], cb[i1]);
    }
}

// Byte-aligned variant: builds all eight 6-bit indices into two uint32 words, four bytes
// each, so the per-coordinate work drops to one byte extract. Even coordinates (0,2,4,6)
// land in `even`, odd ones in `odd`.
//
// Static count per block: 9 (spread) + 3 (parity) + 2 (coset) + 2 (coset byte splat)
// + 3 (even nibbles/msb masks) + 3 (odd) + 4 (two 3-input LOP3 merges) = 26 ops
// amortized (3.25/weight)... which is WORSE than the straightforward path unless the
// compiler folds the masks into the LOP3s it already needs. Measure before choosing:
// the reason to prefer it is register pressure and LDS address arithmetic, not op count.
__device__ __forceinline__ void e8w5_block_indices_bytes(uint32_t lo4, uint32_t hi8,
                                                         uint32_t& even, uint32_t& odd)
{
    const uint32_t csb = (0u - e8w5_coset(lo4)) & 0x20202020u;  // coset -> bit 5 of each byte
    const uint32_t nib = e8w5_uniform_nibbles(lo4);
    const uint32_t sp = e8w5_spread8(hi8);
    even = (nib & 0x0F0F0F0Fu) | ((sp << 4) & 0x10101010u) | csb;
    odd = ((nib >> 4) & 0x0F0F0F0Fu) | (sp & 0x10101010u) | csb;
}

__device__ __forceinline__ void e8w5_decode_block_bytes(uint32_t lo4, uint32_t hi8,
                                                        const __half* __restrict__ cb,
                                                        __half2 (&out)[4])
{
    uint32_t even, odd;
    e8w5_block_indices_bytes(lo4, hi8, even, odd);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t s = 8u * static_cast<uint32_t>(i);
        out[i] = __halves2half2(cb[(even >> s) & 0xFFu], cb[(odd >> s) & 0xFFu]);
    }
}

// --------------------------------------------------------------------------- //
// one scale group: 16 blocks == 128 weights == 5 aligned 16-byte loads
// --------------------------------------------------------------------------- //
//
// The group scale multiplies every weight in the group, so for a dot product it can come
// out of the sum entirely:  sum_j (delta * v_j) * x_j  ==  delta * sum_j v_j * x_j.
// That turns 128 per-weight multiplies into ONE per group. It is the single cheapest thing
// in this header and it is why per-group scales cost nothing in the inner loop -- they cost
// 0.125 bpw in the file, which is the real price (spec section 7).
//
// Accumulation is FP32 on purpose: a K=13696 reduction in FP16 is not safe, and the
// research's dB are all FP32-referenced.

__device__ __forceinline__ float e8w5_group_dot(const uint4* __restrict__ plane_lo,
                                                const uint32_t* __restrict__ plane_hi,
                                                const __half2* __restrict__ x,
                                                const __half* __restrict__ cb,
                                                float scale)
{
    float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int q = 0; q < 4; ++q) {           // four uint4 = 16 blocks = 128 weights
        const uint4 L = plane_lo[q];        // 16 B aligned
        const uint32_t H = plane_hi[q];     // 4 B: one byte per block
        const uint32_t lw[4] = {L.x, L.y, L.z, L.w};
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            __half2 w[4];
            e8w5_decode_block(lw[b], (H >> (8 * b)) & 0xFFu, cb, w);
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float2 p = __half22float2(__hmul2(w[h], x[(q * 4 + b) * 4 + h]));
                acc.x += p.x;
                acc.y += p.y;
            }
        }
    }
    return scale * (acc.x + acc.y);
}

// --------------------------------------------------------------------------- //
// warp tile: 1024 weights of one output row, two coalesced loads per lane
// --------------------------------------------------------------------------- //
//
// Lane l owns blocks 4l..4l+3 -- 32 weights, entirely inside scale group l/4:
//   plane L: lane l reads one uint4 at block 4l  -> 32 x 16 B = 512 B, fully coalesced
//   plane H: lane l reads one uint32 at block 4l ->  32 x 4 B = 128 B, fully coalesced
// Four lanes share each group's scale, so the eight scales per warp tile are broadcast
// reads.
//
// MEASURED register use (`nvcc --ptxas-options=-v`, CUDA 13.2, zero spill stores/loads and
// 128 B smem on all four; the GEMV row kernel is in the compile probe, not this header):
//
//   arch      decode probe   full GEMV row kernel   <=32-register target
//   sm_80     20             32                     met
//   sm_89     21             40                     missed (80% occupancy)
//   sm_90     21             32                     met
//   sm_120    22             40                     missed (80% occupancy)
//
// So the occupancy target is an Ampere/Hopper property, not a property of this format: the
// same source needs 40 registers on Ada and Blackwell. If 100% occupancy matters there,
// force it with __launch_bounds__ and re-read the spill lines -- do not assume.

__device__ __forceinline__ float e8w5_lane_dot32(const uint4 L, uint32_t H,
                                                 const __half2* __restrict__ x,
                                                 const __half* __restrict__ cb)
{
    float2 acc = make_float2(0.0f, 0.0f);
    const uint32_t lw[4] = {L.x, L.y, L.z, L.w};
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        __half2 w[4];
        e8w5_decode_block(lw[b], (H >> (8 * b)) & 0xFFu, cb, w);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float2 p = __half22float2(__hmul2(w[h], x[b * 4 + h]));
            acc.x += p.x;
            acc.y += p.y;
        }
    }
    return acc.x + acc.y;
}
