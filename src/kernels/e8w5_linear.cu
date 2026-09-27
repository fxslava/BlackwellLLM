#include "e8w5_linear.cuh"
#include "e8w5_dequant.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerBlock = 256;              // 8 warps
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

// ============================================================================
// E8W5 GEMV -- one warp per output row, x staged in shared memory.
//
// The shape is bf16_linear.cu's (warp per row, lanes stride K, FP32 accumulate, shuffle
// reduction) because the format forces it: an E8 block is 8 coordinates of ONE output row,
// so there is nothing to coalesce along out_features the way awq_linear.cu does.
//
// What is NOT obvious, and cost this kernel 24x before it was measured: a lane must own a
// WHOLE block, because u_7's low bit is recovered from the parity of the other seven. So a
// lane reads >= 8 consecutive activations, and consecutive lanes are then >= 32 B apart in x.
// Loading plane L as uint4 makes that worse, not better -- 4 blocks per lane means each lane
// wants 128 B of x, so every x load instruction touches 32 DISTINCT cache lines instead of
// one. The 128-bit weight load buys nothing here either, since 32 lanes x 4 B is already a
// full 128 B transaction per instruction.
//
// So: one block per lane, and x goes through shared memory, cooperatively loaded coalesced
// and padded to 9 floats per 8-float group. 9 is coprime with the 32 banks, so lane l
// reading xs[9*l + i] hits a different bank for every l -- conflict-free for all i. Padding
// to 8 would put every lane in the SAME bank (32-way conflict), which is the trap here.
// ============================================================================

constexpr int kTileBlocks = 128;                       // E8 blocks of x staged per tile
constexpr int kTileK = kTileBlocks * E8W5_DIM;         // 1024 activations
constexpr int kXStride = E8W5_DIM + 1;                 // 9: coprime with 32 banks

__global__ void __launch_bounds__(kThreadsPerBlock)
e8w5_gemv_kernel(const uint32_t* __restrict__ plane_lo,
                 const uint32_t* __restrict__ plane_hi,
                 const __half* __restrict__ scales,
                 const __half* __restrict__ codebook,
                 const float* __restrict__ x,
                 float* __restrict__ y,
                 int out_features,
                 int in_features,
                 int accumulate)
{
    __shared__ __half cb[E8W5_LUT_N];
    __shared__ float xs[kTileBlocks * kXStride];

    e8w5_stage_codebook(codebook, cb, threadIdx.x, kThreadsPerBlock);

    const int row = static_cast<int>(blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5));
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const bool live = (row < out_features);

    const int n_blocks = in_features / E8W5_DIM;
    const int n_groups = in_features / E8W5_GROUP;
    const uint32_t* lo_row = plane_lo + static_cast<size_t>(row) * n_blocks;
    const uint32_t* hi_row = plane_hi + static_cast<size_t>(row) * (n_blocks / 4);
    const __half* sc_row = scales + static_cast<size_t>(row) * n_groups;

    float dot = 0.0f;
    for (int k0 = 0; k0 < in_features; k0 += kTileK) {
        const int tile_k = min(kTileK, in_features - k0);
        const int tile_blocks = tile_k / E8W5_DIM;

        // Cooperative, coalesced load of this tile of x into the padded shared layout.
        __syncthreads();
        for (int i = static_cast<int>(threadIdx.x); i < tile_k; i += kThreadsPerBlock)
            xs[(i >> 3) * kXStride + (i & 7)] = x[k0 + i];
        __syncthreads();

        if (!live) continue;

        // Two independent accumulators: e8w5_block_dot8 is an 8-deep dependent FMA chain
        // fed by data-dependent shared loads, so a single accumulator leaves the warp
        // latency-bound rather than bandwidth-bound. Unrolling by two halves that chain.
        const int base_block = k0 / E8W5_DIM;
        float dot0 = 0.0f, dot1 = 0.0f;
        int b = lane;
        for (; b + 32 < tile_blocks; b += 64) {
            const int g0 = base_block + b;
            const int g1 = g0 + 32;
            const uint32_t l0 = lo_row[g0];
            const uint32_t l1 = lo_row[g1];
            const uint32_t w0 = hi_row[g0 >> 2];
            const uint32_t w1 = hi_row[g1 >> 2];
            const float s0 = __half2float(sc_row[g0 >> 4]);
            const float s1 = __half2float(sc_row[g1 >> 4]);
            dot0 = fmaf(s0, e8w5_block_dot8(l0, (w0 >> (8 * (g0 & 3))) & 0xFFu, cb,
                                            xs + b * kXStride), dot0);
            dot1 = fmaf(s1, e8w5_block_dot8(l1, (w1 >> (8 * (g1 & 3))) & 0xFFu, cb,
                                            xs + (b + 32) * kXStride), dot1);
        }
        for (; b < tile_blocks; b += 32) {
            const int gb = base_block + b;
            const uint32_t lo4 = lo_row[gb];                // 32 lanes x 4 B = one line
            const uint32_t hw = hi_row[gb >> 2];            // 4-lane broadcast
            const uint32_t hi8 = (hw >> (8 * (gb & 3))) & 0xFFu;
            const float s = __half2float(sc_row[gb >> 4]);  // 16 blocks per group
            dot0 = fmaf(s, e8w5_block_dot8(lo4, hi8, cb, xs + b * kXStride), dot0);
        }
        dot += dot0 + dot1;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);

    if (live && lane == 0) {
        if (accumulate) y[row] += dot;
        else y[row] = dot;
    }
}

// Reference-shaped full dequantization. One warp per row, same access pattern as the GEMV
// so a mismatch between the two would show up as a layout bug rather than hiding.
__global__ void __launch_bounds__(kThreadsPerBlock)
e8w5_dequantize_kernel(const uint4* __restrict__ plane_lo,
                       const uint32_t* __restrict__ plane_hi,
                       const __half* __restrict__ scales,
                       const __half* __restrict__ codebook,
                       float* __restrict__ w_out,
                       int out_features,
                       int in_features)
{
    __shared__ __half cb[E8W5_LUT_N];
    e8w5_stage_codebook(codebook, cb, threadIdx.x, kThreadsPerBlock);
    __syncthreads();

    const int row = static_cast<int>(blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5));
    const int lane = static_cast<int>(threadIdx.x & 31u);
    if (row >= out_features) return;

    const int n_vec4 = in_features / (E8W5_DIM * E8W5_BLOCKS_PER_UINT4);
    const int n_groups = in_features / E8W5_GROUP;
    const uint4* lo_row = plane_lo + static_cast<size_t>(row) * n_vec4;
    const uint32_t* hi_row = plane_hi + static_cast<size_t>(row) * n_vec4;
    const __half* sc_row = scales + static_cast<size_t>(row) * n_groups;
    float* out_row = w_out + static_cast<size_t>(row) * in_features;

    for (int q = lane; q < n_vec4; q += 32) {
        const uint4 L = lo_row[q];
        const uint32_t H = hi_row[q];
        const float s = __half2float(sc_row[q >> 2]);
        const uint32_t lw[E8W5_BLOCKS_PER_UINT4] = {L.x, L.y, L.z, L.w};
#pragma unroll
        for (int b = 0; b < E8W5_BLOCKS_PER_UINT4; ++b) {
            float w[E8W5_DIM];
            e8w5_decode_block_f32(lw[b], (H >> (8 * b)) & 0xFFu, cb, w);
            float* dst = out_row + static_cast<size_t>(q) * 32 + b * E8W5_DIM;
#pragma unroll
            for (int i = 0; i < E8W5_DIM; ++i) dst[i] = s * w[i];
        }
    }
}

void check_shape(size_t out_features, size_t in_features, const char* who)
{
    if (in_features % E8W5_GROUP != 0)
        throw std::invalid_argument(
            std::string(who) + ": in_features must be a multiple of " +
            std::to_string(E8W5_GROUP) + " (E8W5 scale-group alignment), got " +
            std::to_string(in_features));
    if (out_features == 0 || in_features == 0)
        throw std::invalid_argument(std::string(who) + ": zero-sized projection");
}

int grid_rows(size_t out_features)
{
    return static_cast<int>((out_features + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

} // namespace

void launch_e8w5_gemv_kernel(const void* plane_lo, const void* plane_hi,
                             const void* scales, const void* codebook,
                             const float* d_in, float* d_out,
                             size_t out_features, size_t in_features)
{
    check_shape(out_features, in_features, "launch_e8w5_gemv_kernel");
    e8w5_gemv_kernel<<<grid_rows(out_features), kThreadsPerBlock>>>(
        static_cast<const uint32_t*>(plane_lo), static_cast<const uint32_t*>(plane_hi),
        static_cast<const __half*>(scales), static_cast<const __half*>(codebook),
        d_in, d_out, static_cast<int>(out_features), static_cast<int>(in_features),
        /*accumulate=*/0);
}

void launch_e8w5_gemv_residual_kernel(const void* plane_lo, const void* plane_hi,
                                      const void* scales, const void* codebook,
                                      const float* d_in, float* d_residual_accum,
                                      size_t out_features, size_t in_features)
{
    check_shape(out_features, in_features, "launch_e8w5_gemv_residual_kernel");
    e8w5_gemv_kernel<<<grid_rows(out_features), kThreadsPerBlock>>>(
        static_cast<const uint32_t*>(plane_lo), static_cast<const uint32_t*>(plane_hi),
        static_cast<const __half*>(scales), static_cast<const __half*>(codebook),
        d_in, d_residual_accum, static_cast<int>(out_features),
        static_cast<int>(in_features), /*accumulate=*/1);
}

void launch_e8w5_dequantize(const void* plane_lo, const void* plane_hi,
                            const void* scales, const void* codebook,
                            float* d_W_out, size_t out_features, size_t in_features)
{
    check_shape(out_features, in_features, "launch_e8w5_dequantize");
    e8w5_dequantize_kernel<<<grid_rows(out_features), kThreadsPerBlock>>>(
        static_cast<const uint4*>(plane_lo), static_cast<const uint32_t*>(plane_hi),
        static_cast<const __half*>(scales), static_cast<const __half*>(codebook),
        d_W_out, static_cast<int>(out_features), static_cast<int>(in_features));
}
