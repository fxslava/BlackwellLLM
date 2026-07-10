#include "batched_bf16_gemm.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// Tile geometry — one warp owns one [WMMA_M x WMMA_N] output tile and marches
// the K dimension in WMMA_K-wide steps. TF32 fragments (m16n16k8) keep the FP32
// activations at ~10 mantissa bits into the Tensor-Core multiply; accumulation
// is FP32, matching the batch=1 GEMV's accumulator.
// ============================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// C[T,M] = X[T,K] @ W^T,  W row-major [M,K], X row-major [T,K], Y row-major [T,M].
// accumulate=true adds into Y in place (residual o_proj / down_proj path).
template <bool accumulate>
__global__ void bf16_gemm_batched_kernel(const __nv_bfloat16* __restrict__ W,
                                         const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         int M, int K, int T)
{
    const int tile_n = blockIdx.x * WMMA_N;   // first output feature of this tile
    const int tile_m = blockIdx.y * WMMA_M;   // first token row of this tile
    const int tid    = threadIdx.x;           // 0..31 (one warp)

    // Staging tiles (zero-padded so arbitrary M/K/T need no launch constraints).
    //   As[m,k] row-major  (ldm = WMMA_K)  -> matrix_a row_major
    //   Bs[k,n] row-major  (ldm = WMMA_N)  -> matrix_b row_major (== W^T tile)
    __shared__ float As[WMMA_M * WMMA_K];
    __shared__ float Bs[WMMA_K * WMMA_N];
    __shared__ float Cs[WMMA_M * WMMA_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // Load the X tile: As[m,k] = X[(tile_m+m), (k0+k)] or 0 past the edges.
        for (int i = tid; i < WMMA_M * WMMA_K; i += 32) {
            const int m = i / WMMA_K, k = i % WMMA_K;
            const int gm = tile_m + m, gk = k0 + k;
            As[i] = (gm < T && gk < K) ? X[(size_t)gm * K + gk] : 0.0f;
        }
        // Load the W^T tile: Bs[k,n] = W[(tile_n+n), (k0+k)] or 0 (bf16 -> fp32).
        for (int i = tid; i < WMMA_K * WMMA_N; i += 32) {
            const int k = i / WMMA_N, n = i % WMMA_N;
            const int gn = tile_n + n, gk = k0 + k;
            Bs[i] = (gn < M && gk < K)
                        ? __bfloat162float(W[(size_t)gn * K + gk])
                        : 0.0f;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, As, WMMA_K);
        wmma::load_matrix_sync(b_frag, Bs, WMMA_N);
        // TF32 requires an explicit round of each fragment element before mma.
        #pragma unroll
        for (int i = 0; i < a_frag.num_elements; ++i)
            a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
        #pragma unroll
        for (int i = 0; i < b_frag.num_elements; ++i)
            b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();   // As/Bs reused next k-step
    }

    wmma::store_matrix_sync(Cs, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < WMMA_M * WMMA_N; i += 32) {
        const int m = i / WMMA_N, n = i % WMMA_N;
        const int gm = tile_m + m, gn = tile_n + n;
        if (gm < T && gn < M) {
            const size_t off = (size_t)gm * M + gn;
            if (accumulate) Y[off] += Cs[i];
            else            Y[off]  = Cs[i];
        }
    }
}

static void launch_impl(const void* d_W_bf16, const float* d_X, float* d_Y,
                        size_t M, size_t K, size_t T, bool accumulate) {
    dim3 grid((unsigned)((M + WMMA_N - 1) / WMMA_N),
              (unsigned)((T + WMMA_M - 1) / WMMA_M));
    const auto* W = reinterpret_cast<const __nv_bfloat16*>(d_W_bf16);
    if (accumulate)
        bf16_gemm_batched_kernel<true><<<grid, 32>>>(W, d_X, d_Y, (int)M, (int)K, (int)T);
    else
        bf16_gemm_batched_kernel<false><<<grid, 32>>>(W, d_X, d_Y, (int)M, (int)K, (int)T);
}

void launch_bf16_gemm_batched(const void* d_W_bf16, const float* d_X, float* d_Y,
                              size_t M, size_t K, size_t num_tokens) {
    launch_impl(d_W_bf16, d_X, d_Y, M, K, num_tokens, /*accumulate=*/false);
}

void launch_bf16_gemm_residual_batched(const void* d_W_bf16, const float* d_X,
                                       float* d_Y_accum, size_t M, size_t K,
                                       size_t num_tokens) {
    launch_impl(d_W_bf16, d_X, d_Y_accum, M, K, num_tokens, /*accumulate=*/true);
}
