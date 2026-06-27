#include "sym_int4_linear.cuh"

// One thread per output feature (decode GEMV: M = 1). Each thread walks its
// weight row's packed int32 words, unpacks 8 signed nibbles, and accumulates
// x[in] * dequant(W[o,in]). group_size divides in_features; with group_size=32
// and 8 nibbles/word, a scale spans 4 consecutive words, but we index per input
// element for clarity (the divide is cheap relative to the global loads).
template <bool kResidual>
__global__ void sym_int4_gemv_kernel(const int32_t* __restrict__ packed,
                                     const __nv_bfloat16* __restrict__ scales,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int out_features, int in_features, int group_size) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;

    const int words_per_row  = in_features / 8;
    const int groups_per_row = in_features / group_size;
    const int32_t* __restrict__ wrow       = packed + (size_t)o * words_per_row;
    const __nv_bfloat16* __restrict__ srow = scales + (size_t)o * groups_per_row;

    float acc = 0.0f;
    for (int p = 0; p < words_per_row; ++p) {
        const uint32_t w = static_cast<uint32_t>(wrow[p]);
        const int in_base = p * 8;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int nib = (w >> (4 * t)) & 0xF;
            if (nib > 7) nib -= 16;                       // sign-extend 4-bit -> [-8,7]
            const int in_idx = in_base + t;
            const float scale = __bfloat162float(srow[in_idx / group_size]);
            acc += x[in_idx] * (float)nib * scale;
        }
    }

    if (kResidual) y[o] += acc;
    else           y[o]  = acc;
}

static void launch(const int32_t* d_packed, const __nv_bfloat16* d_scales,
                   const float* d_x, float* d_y, int out_features, int in_features,
                   int group_size, bool residual, cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (out_features + threads - 1) / threads;
    if (residual)
        sym_int4_gemv_kernel<true><<<blocks, threads, 0, stream>>>(
            d_packed, d_scales, d_x, d_y, out_features, in_features, group_size);
    else
        sym_int4_gemv_kernel<false><<<blocks, threads, 0, stream>>>(
            d_packed, d_scales, d_x, d_y, out_features, in_features, group_size);
}

void launch_sym_int4_gemv(const int32_t* d_packed, const __nv_bfloat16* d_scales,
                          const float* d_x, float* d_y, int out_features,
                          int in_features, int group_size, cudaStream_t stream) {
    launch(d_packed, d_scales, d_x, d_y, out_features, in_features, group_size, false, stream);
}

void launch_sym_int4_gemv_residual(const int32_t* d_packed, const __nv_bfloat16* d_scales,
                                   const float* d_x, float* d_y_accum, int out_features,
                                   int in_features, int group_size, cudaStream_t stream) {
    launch(d_packed, d_scales, d_x, d_y_accum, out_features, in_features, group_size, true, stream);
}
