#include "awq_linear.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <algorithm>

namespace {

constexpr int kColsPerBlock = 32;  // packed uint32 columns per block (threadIdx.x)
constexpr int kRowsPerBlock = 8;   // row-strided lanes per column (threadIdx.y)
constexpr int kPackFactor   = 8;   // int4 values per uint32

// AWQ packs nibbles in order {0,2,4,6,1,3,5,7}: logical element j sits at
// bit offset 16*(j&1) + 4*(j>>1) inside the packed word.
__device__ __forceinline__ int awq_shift(int j)
{
    return ((j & 1) << 4) | ((j >> 1) << 2);
}

__global__ void __launch_bounds__(kColsPerBlock * kRowsPerBlock)
awq_gemv_kernel(const uint32_t* __restrict__ qweight,
                const half*     __restrict__ scales,
                const uint32_t* __restrict__ qzeros,
                const float*    __restrict__ x,
                float*          __restrict__ y,
                int out_features,
                int in_features,
                int group_size,
                int groups_per_split,
                int accumulate)
{
    const int packed_cols = out_features >> 3;
    const int pc = blockIdx.x * kColsPerBlock + threadIdx.x;
    const int ty = threadIdx.y;

    const int num_groups = (in_features + group_size - 1) / group_size;
    const int g_begin    = blockIdx.y * groups_per_split;
    const int g_end      = min(g_begin + groups_per_split, num_groups);

    float acc[kPackFactor];
#pragma unroll
    for (int j = 0; j < kPackFactor; ++j) acc[j] = 0.0f;

    if (pc < packed_cols) {
        const half* sg = scales + (size_t)pc * kPackFactor;

        for (int g = g_begin; g < g_end; ++g) {
            const uint32_t zq = __ldg(qzeros + (size_t)g * packed_cols + pc);
            const half*    sp = sg + (size_t)g * out_features;

            float s[kPackFactor];
            float z[kPackFactor];
#pragma unroll
            for (int j = 0; j < kPackFactor; ++j) {
                s[j] = __half2float(__ldg(sp + j));
                z[j] = (float)((zq >> awq_shift(j)) & 0xF);
            }

            // Accumulate the group's partial dot in quantized space:
            // sum_k (w - z) * s * x[k] == s * (sum_k w * x[k] - z * sum_k x[k])
            float accq[kPackFactor];
#pragma unroll
            for (int j = 0; j < kPackFactor; ++j) accq[j] = 0.0f;
            float accx = 0.0f;

            const int k_end = min((g + 1) * group_size, in_features);
            for (int k = g * group_size + ty; k < k_end; k += kRowsPerBlock) {
                const uint32_t wq = __ldg(qweight + (size_t)k * packed_cols + pc);
                const float    xk = __ldg(x + k);
                accx += xk;
#pragma unroll
                for (int j = 0; j < kPackFactor; ++j)
                    accq[j] = fmaf((float)((wq >> awq_shift(j)) & 0xF), xk, accq[j]);
            }

#pragma unroll
            for (int j = 0; j < kPackFactor; ++j)
                acc[j] = fmaf(s[j], fmaf(-z[j], accx, accq[j]), acc[j]);
        }
    }

    // Cross-row reduction: each of the 256 threads then owns one of the
    // block's 256 output features.
    __shared__ float smem[kRowsPerBlock][kColsPerBlock][kPackFactor];
#pragma unroll
    for (int j = 0; j < kPackFactor; ++j) smem[ty][threadIdx.x][j] = acc[j];
    __syncthreads();

    const int t   = ty * kColsPerBlock + threadIdx.x;
    const int lx  = t >> 3;
    const int lj  = t & 7;
    const int opc = blockIdx.x * kColsPerBlock + lx;
    if (opc < packed_cols) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < kRowsPerBlock; ++r) sum += smem[r][lx][lj];

        const size_t oi = (size_t)opc * kPackFactor + lj;
        // accumulate=1: y holds the residual stream, so the GEMV result is
        // always added on top (split-K then also rides the same atomics).
        if (accumulate || gridDim.y > 1)
            atomicAdd(y + oi, sum);
        else
            y[oi] = sum;
    }
}

void launch_awq_gemv_impl(const void* qweight,
                          const void* scales,
                          const void* qzeros,
                          const float* d_in,
                          float* d_out,
                          size_t out_features,
                          size_t in_features,
                          int group_size,
                          bool accumulate)
{
    if (out_features == 0 || in_features == 0) return;

    const int gs          = group_size > 0 ? group_size : (int)in_features;
    const int packed_cols = (int)(out_features / kPackFactor);
    const int num_groups  = ((int)in_features + gs - 1) / gs;

    static int num_sms = 0;
    if (num_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
        if (num_sms <= 0) num_sms = 1;
    }

    const int grid_x = (packed_cols + kColsPerBlock - 1) / kColsPerBlock;

    // Narrow layers leave SMs idle; split the reduction dimension across
    // blocks (atomic accumulation) until the grid covers the device.
    int split_k = 1;
    if (grid_x < num_sms)
        split_k = std::min({(num_sms + grid_x - 1) / grid_x, num_groups, 16});
    const int groups_per_split = (num_groups + split_k - 1) / split_k;
    split_k = (num_groups + groups_per_split - 1) / groups_per_split;

    // In accumulate mode d_out is the live residual stream and must NOT be
    // cleared; the kernel adds every partial atomically instead.
    if (split_k > 1 && !accumulate)
        cudaMemsetAsync(d_out, 0, out_features * sizeof(float));

    const dim3 block(kColsPerBlock, kRowsPerBlock);
    const dim3 grid(grid_x, split_k);
    awq_gemv_kernel<<<grid, block>>>(static_cast<const uint32_t*>(qweight),
                                     static_cast<const half*>(scales),
                                     static_cast<const uint32_t*>(qzeros),
                                     d_in,
                                     d_out,
                                     (int)out_features,
                                     (int)in_features,
                                     gs,
                                     groups_per_split,
                                     accumulate ? 1 : 0);
}

}  // namespace

// ============================================================================
// Batched AWQ GEMM (Tensor Cores) — the num_tokens > 1 path.
// ============================================================================
// One warp owns one [WMMA_M tokens x WMMA_N out-features] output tile and marches
// K in WMMA_K steps. Per k-step it stages the FP32 activation tile and the
// dequantized weight tile (int4 unpacked on the fly) into shared memory, then
// runs a TF32 wmma — mirroring batched_bf16_gemm, the only difference being how
// Bs is filled (dequant vs a plain bf16->float load).
namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;

template <bool accumulate>
__global__ void awq_gemm_batched_kernel(const uint32_t* __restrict__ qweight,
                                        const half*     __restrict__ scales,
                                        const uint32_t* __restrict__ qzeros,
                                        const float*    __restrict__ X,
                                        float*          __restrict__ Y,
                                        int M, int K, int T,
                                        int group_size, int packed_cols)
{
    const int tile_n = blockIdx.x * kWmmaN;   // first output feature
    const int tile_m = blockIdx.y * kWmmaM;   // first token row
    const int tid    = threadIdx.x;           // 0..31 (one warp)

    __shared__ float As[kWmmaM * kWmmaK];
    __shared__ float Bs[kWmmaK * kWmmaN];
    __shared__ float Cs[kWmmaM * kWmmaN];

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kWmmaK) {
        // Activation tile As[m,k] = X[(tile_m+m), (k0+k)] (zero past the edges).
        for (int i = tid; i < kWmmaM * kWmmaK; i += 32) {
            const int m = i / kWmmaK, k = i % kWmmaK;
            const int gm = tile_m + m, gk = k0 + k;
            As[i] = (gm < T && gk < K) ? X[(size_t)gm * K + gk] : 0.0f;
        }
        // Dequantized weight tile Bs[k,n] = (w - z) * s for (gk, gn=tile_n+n).
        for (int i = tid; i < kWmmaK * kWmmaN; i += 32) {
            const int k = i / kWmmaN, n = i % kWmmaN;
            const int gk = k0 + k, gn = tile_n + n;
            float bval = 0.0f;
            if (gn < M && gk < K) {
                const int pc  = gn >> 3;          // packed column
                const int j   = gn & 7;           // nibble within the word
                const int sh  = awq_shift(j);
                const int g   = gk / group_size;
                const uint32_t wq = __ldg(qweight + (size_t)gk * packed_cols + pc);
                const uint32_t zq = __ldg(qzeros  + (size_t)g  * packed_cols + pc);
                const float w = (float)((wq >> sh) & 0xF);
                const float z = (float)((zq >> sh) & 0xF);
                const float s = __half2float(__ldg(scales + (size_t)g * M + gn));
                bval = (w - z) * s;
            }
            Bs[i] = bval;
        }
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                               nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                               nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, As, kWmmaK);
        nvcuda::wmma::load_matrix_sync(b_frag, Bs, kWmmaN);
        #pragma unroll
        for (int i = 0; i < a_frag.num_elements; ++i)
            a_frag.x[i] = nvcuda::wmma::__float_to_tf32(a_frag.x[i]);
        #pragma unroll
        for (int i = 0; i < b_frag.num_elements; ++i)
            b_frag.x[i] = nvcuda::wmma::__float_to_tf32(b_frag.x[i]);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(Cs, c_frag, kWmmaN, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < kWmmaM * kWmmaN; i += 32) {
        const int m = i / kWmmaN, n = i % kWmmaN;
        const int gm = tile_m + m, gn = tile_n + n;
        if (gm < T && gn < M) {
            const size_t off = (size_t)gm * M + gn;
            if (accumulate) Y[off] += Cs[i];
            else            Y[off]  = Cs[i];
        }
    }
}

void launch_awq_gemm_batched_impl(const void* qweight, const void* scales,
                                  const void* qzeros, const float* d_X, float* d_Y,
                                  size_t out_features, size_t in_features,
                                  int group_size, size_t num_tokens, bool accumulate)
{
    if (out_features == 0 || in_features == 0 || num_tokens == 0) return;
    const int gs          = group_size > 0 ? group_size : (int)in_features;
    const int packed_cols = (int)(out_features / 8);
    dim3 grid((unsigned)((out_features + kWmmaN - 1) / kWmmaN),
              (unsigned)((num_tokens  + kWmmaM - 1) / kWmmaM));
    const auto* qw = static_cast<const uint32_t*>(qweight);
    const auto* sc = static_cast<const half*>(scales);
    const auto* qz = static_cast<const uint32_t*>(qzeros);
    if (accumulate)
        awq_gemm_batched_kernel<true><<<grid, 32>>>(qw, sc, qz, d_X, d_Y,
            (int)out_features, (int)in_features, (int)num_tokens, gs, packed_cols);
    else
        awq_gemm_batched_kernel<false><<<grid, 32>>>(qw, sc, qz, d_X, d_Y,
            (int)out_features, (int)in_features, (int)num_tokens, gs, packed_cols);
}

}  // namespace

void launch_awq_gemv_kernel(const void* qweight,
                            const void* scales,
                            const void* qzeros,
                            const float* d_in,
                            float* d_out,
                            size_t out_features,
                            size_t in_features,
                            int group_size)
{
    launch_awq_gemv_impl(qweight, scales, qzeros, d_in, d_out,
                         out_features, in_features, group_size, false);
}

void launch_awq_gemv_residual_kernel(const void* qweight,
                                     const void* scales,
                                     const void* qzeros,
                                     const float* d_in,
                                     float* d_residual_accum,
                                     size_t out_features,
                                     size_t in_features,
                                     int group_size)
{
    launch_awq_gemv_impl(qweight, scales, qzeros, d_in, d_residual_accum,
                         out_features, in_features, group_size, true);
}

void launch_batched_awq_gemm(const void* qweight, const void* scales, const void* qzeros,
                             const float* d_X, float* d_Y, size_t out_features,
                             size_t in_features, int group_size, size_t num_tokens)
{
    launch_awq_gemm_batched_impl(qweight, scales, qzeros, d_X, d_Y,
                                 out_features, in_features, group_size, num_tokens, false);
}

void launch_batched_awq_gemm_residual(const void* qweight, const void* scales, const void* qzeros,
                                      const float* d_X, float* d_Y_accum, size_t out_features,
                                      size_t in_features, int group_size, size_t num_tokens)
{
    launch_awq_gemm_batched_impl(qweight, scales, qzeros, d_X, d_Y_accum,
                                 out_features, in_features, group_size, num_tokens, true);
}
