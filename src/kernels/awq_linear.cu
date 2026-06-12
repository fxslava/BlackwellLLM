#include "awq_linear.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
