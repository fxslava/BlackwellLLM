#include "paged_flash_attention.cuh"
#include <mma.h>
#include <math_constants.h>
#include <cmath>
#include <algorithm>

using namespace nvcuda;
using blackwell::paging::kv_t;
using blackwell::paging::PAGE_SIZE;

// ============================================================================
// Tile geometry
// ============================================================================
//   BLOCK_M : query rows processed per block (one wmma row tile).
//   BLOCK_N : key/value tokens per KV tile == PAGE_SIZE == one wmma col tile.
//             Choosing BLOCK_N = PAGE_SIZE makes each loop iteration consume
//             exactly ONE physical page — the block table walk is trivial and
//             there is never a tile that straddles two pages.
//   head_dim is the wmma K dimension, processed in 16-wide sub-tiles (<= 8).
// ----------------------------------------------------------------------------
#define BLOCK_M       16
#define BLOCK_N       PAGE_SIZE          // 16
#define HEAD_DIM_MAX  128
#define HD_PAD        (HEAD_DIM_MAX + 8) // +8 halfs => skew rows off the same
                                         //   shared-memory bank set (load_matrix
                                         //   reads a column of 16 rows at once).
#define S_PAD         (BLOCK_N + 8)      // same skew for the [M,N] score tile
#define WMMA_T        16

__device__ __forceinline__ float trunc_bf16(float v) {
    // Match the engine's PyTorch-bf16 output parity (see attention.cu).
    return __bfloat162float(__float2bfloat16(v));
}

// ============================================================================
// Paged Flash Attention kernel
// ============================================================================
// Grid : (num_q_heads, num_q_tiles)      Block : 32 threads (one warp)
//   blockIdx.x = query head
//   blockIdx.y = query tile  -> covers logical rows [q_start, q_start+BLOCK_M)
// One warp owns the whole tile: the wmma ops are warp-collective, and the
// online-softmax bookkeeping is done by lanes 0..BLOCK_M-1 (one lane per row).
__global__ void paged_flash_attention_kernel(
    const float*   __restrict__ Q,              // [rows, num_q_heads, head_dim]
    const kv_t*    __restrict__ k_pool_layer,   // [total_pages, num_kv_heads, PAGE_SIZE, head_dim]
    const kv_t*    __restrict__ v_pool_layer,
    float*         __restrict__ O,              // [rows, num_q_heads, head_dim]
    const int32_t* __restrict__ block_table,    // [num_blocks] page ids for this seq
    int   seq_len,
    int   q_start_base,                         // logical pos of tile-0 row-0
    int   num_q_heads,
    int   num_kv_heads,
    int   head_dim,
    float scale)
{
    const int q_head  = blockIdx.x;
    const int q_tile  = blockIdx.y;
    const int tid     = threadIdx.x;            // 0..31

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kvh       = q_head / gqa_ratio;

    const int q_row0  = q_tile * BLOCK_M;       // first Q-tensor row for this tile
    const int q_start = q_start_base + q_row0;  // causal position of local row 0
    const int num_valid = max(0, min(BLOCK_M, seq_len - q_start));
    if (num_valid <= 0) return;

    const int n_dtiles = head_dim / WMMA_T;     // 16-wide K sub-tiles (<= 8)
    const int q_row_stride = num_q_heads * head_dim;

    // ---- shared memory ----------------------------------------------------
    __shared__ kv_t  q_smem[BLOCK_M * HD_PAD];
    __shared__ kv_t  k_smem[BLOCK_N * HD_PAD];
    __shared__ kv_t  v_smem[BLOCK_N * HD_PAD];
    __shared__ kv_t  p_smem[BLOCK_M * S_PAD];   // softmax probabilities (bf16)
    __shared__ float s_smem[BLOCK_M * S_PAD];   // raw scores (fp32)
    __shared__ float o_smem[BLOCK_M * HEAD_DIM_MAX]; // running context, fp32
    __shared__ float m_smem[BLOCK_M];           // running row max
    __shared__ float l_smem[BLOCK_M];           // running row denom
    __shared__ float corr_smem[BLOCK_M];        // per-row rescale this step

    // ---- load Q tile -> bf16 smem (zero-pad invalid rows) -----------------
    for (int i = tid; i < BLOCK_M * head_dim; i += 32) {
        int r = i / head_dim, d = i % head_dim;
        float qv = (r < num_valid)
                 ? Q[(q_row0 + r) * q_row_stride + q_head * head_dim + d]
                 : 0.0f;
        q_smem[r * HD_PAD + d] = (kv_t)qv;
    }
    // init running state
    for (int r = tid; r < BLOCK_M; r += 32) {
        m_smem[r] = -CUDART_INF_F;
        l_smem[r] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * head_dim; i += 32) o_smem[i] = 0.0f;
    __syncthreads();

    const int num_pages = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
    const int per_page   = num_kv_heads * PAGE_SIZE * head_dim;

    for (int pg = 0; pg < num_pages; ++pg) {
        const int page_start = pg * PAGE_SIZE;
        if (page_start > q_start + num_valid - 1) break;   // fully future (causal)

        const int page = block_table[pg];
        const int toks = min(PAGE_SIZE, seq_len - page_start);
        const kv_t* kbase = k_pool_layer + (size_t)page * per_page + (size_t)kvh * PAGE_SIZE * head_dim;
        const kv_t* vbase = v_pool_layer + (size_t)page * per_page + (size_t)kvh * PAGE_SIZE * head_dim;

        // ---- gather one physical page into smem (zero-pad the tail) --------
        for (int i = tid; i < PAGE_SIZE * head_dim; i += 32) {
            int t = i / head_dim, d = i % head_dim;
            kv_t kv = (t < toks) ? kbase[t * head_dim + d] : (kv_t)0.0f;
            kv_t vv = (t < toks) ? vbase[t * head_dim + d] : (kv_t)0.0f;
            k_smem[t * HD_PAD + d] = kv;
            v_smem[t * HD_PAD + d] = vv;
        }
        __syncthreads();

        // ---- S = Q . K^T  (Tensor Cores) ----------------------------------
        // K loaded into a COL-MAJOR b_frag yields K^T directly:
        //   b_frag[kk,n] = k_smem[(dt*16+kk) + n*HD_PAD] = K[n][dt*16+kk].
        wmma::fragment<wmma::accumulator, WMMA_T, WMMA_T, WMMA_T, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
        for (int dt = 0; dt < n_dtiles; ++dt) {
            wmma::fragment<wmma::matrix_a, WMMA_T, WMMA_T, WMMA_T, kv_t, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_T, WMMA_T, WMMA_T, kv_t, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, q_smem + dt * WMMA_T, HD_PAD);
            wmma::load_matrix_sync(b_frag, k_smem + dt * WMMA_T, HD_PAD);
            wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        wmma::store_matrix_sync(s_smem, s_frag, S_PAD, wmma::mem_row_major);
        __syncthreads();

        // ---- online softmax update (one lane per row) ---------------------
        if (tid < num_valid) {
            const int r = tid;
            const int q_pos = q_start + r;
            float rmax = -CUDART_INF_F;
            for (int n = 0; n < PAGE_SIZE; ++n) {
                const int k_pos = page_start + n;
                float s = (n < toks && k_pos <= q_pos)      // causal + tail mask
                        ? s_smem[r * S_PAD + n] * scale
                        : -CUDART_INF_F;
                s_smem[r * S_PAD + n] = s;
                rmax = fmaxf(rmax, s);
            }
            const float m_old = m_smem[r];
            const float m_new = fmaxf(m_old, rmax);
            const float corr  = (m_old == -CUDART_INF_F) ? 0.0f : __expf(m_old - m_new);
            float rowsum = 0.0f;
            for (int n = 0; n < PAGE_SIZE; ++n) {
                float p = (s_smem[r * S_PAD + n] == -CUDART_INF_F)
                        ? 0.0f : __expf(s_smem[r * S_PAD + n] - m_new);
                p_smem[r * S_PAD + n] = (kv_t)p;
                rowsum += p;
            }
            l_smem[r]   = l_smem[r] * corr + rowsum;
            m_smem[r]   = m_new;
            corr_smem[r] = corr;
        }
        __syncthreads();

        // ---- rescale running context by the per-row correction ------------
        for (int i = tid; i < num_valid * head_dim; i += 32) {
            int r = i / head_dim;
            o_smem[i] *= corr_smem[r];
        }
        __syncthreads();

        // ---- O += P . V  (Tensor Cores, accumulate in-place via o_smem) ---
        for (int dt = 0; dt < n_dtiles; ++dt) {
            wmma::fragment<wmma::matrix_a, WMMA_T, WMMA_T, WMMA_T, kv_t, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_T, WMMA_T, WMMA_T, kv_t, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_T, WMMA_T, WMMA_T, float> o_frag;
            wmma::load_matrix_sync(p_frag, p_smem, S_PAD);
            wmma::load_matrix_sync(v_frag, v_smem + dt * WMMA_T, HD_PAD);
            wmma::load_matrix_sync(o_frag, o_smem + dt * WMMA_T, head_dim, wmma::mem_row_major);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            wmma::store_matrix_sync(o_smem + dt * WMMA_T, o_frag, head_dim, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // ---- epilogue: normalize + bf16-parity truncate -> O ------------------
    for (int i = tid; i < num_valid * head_dim; i += 32) {
        int r = i / head_dim, d = i % head_dim;
        float out = o_smem[i] / l_smem[r];
        O[(q_row0 + r) * q_row_stride + q_head * head_dim + d] = trunc_bf16(out);
    }
}

// ============================================================================
// KV append + Copy-on-Write
// ============================================================================
__global__ void paged_kv_append_kernel(
    const float* __restrict__ K, const float* __restrict__ V,
    kv_t* __restrict__ k_pool_layer, kv_t* __restrict__ v_pool_layer,
    int page, int slot, int num_kv_heads, int head_dim)
{
    const int kvh = blockIdx.x;
    const int d   = threadIdx.x;
    if (d >= head_dim) return;
    const size_t off =
        (((size_t)page * num_kv_heads + kvh) * PAGE_SIZE + slot) * head_dim + d;
    k_pool_layer[off] = (kv_t)K[kvh * head_dim + d];
    v_pool_layer[off] = (kv_t)V[kvh * head_dim + d];
}

// Batched append: token blockIdx.y (logical pos start_pos + y) -> its page slot,
// resolved through the shared block table. One thread per head_dim channel.
__global__ void paged_kv_append_batched_kernel(
    const float* __restrict__ K, const float* __restrict__ V,
    kv_t* __restrict__ k_pool_layer, kv_t* __restrict__ v_pool_layer,
    const int32_t* __restrict__ block_table, int start_pos,
    int num_kv_heads, int head_dim)
{
    const int kvh = blockIdx.x;
    const int t   = blockIdx.y;
    const int d   = threadIdx.x;
    if (d >= head_dim) return;

    const int pos  = start_pos + t;
    const int page = block_table[pos / PAGE_SIZE];
    const int slot = pos % PAGE_SIZE;

    const size_t src = ((size_t)t * num_kv_heads + kvh) * head_dim + d;
    const size_t dst =
        (((size_t)page * num_kv_heads + kvh) * PAGE_SIZE + slot) * head_dim + d;
    k_pool_layer[dst] = (kv_t)K[src];
    v_pool_layer[dst] = (kv_t)V[src];
}

__global__ void cow_copy_page_kernel(
    kv_t* __restrict__ k_pool, kv_t* __restrict__ v_pool,
    int src_page, int dst_page, int total_pages, int per_page)
{
    const int layer  = blockIdx.y;
    const size_t base = (size_t)layer * total_pages * per_page;
    const size_t s = base + (size_t)src_page * per_page;
    const size_t d = base + (size_t)dst_page * per_page;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < per_page;
             i += gridDim.x * blockDim.x) {
        k_pool[d + i] = k_pool[s + i];
        v_pool[d + i] = v_pool[s + i];
    }
}

// ============================================================================
// Launchers
// ============================================================================
static void launch_pfa(const float* dQ, const kv_t* dK, const kv_t* dV, float* dO,
                       const int32_t* d_block_table, int seq_len, int q_start_base,
                       int num_q_tiles, int num_q_heads, int num_kv_heads,
                       int head_dim, cudaStream_t stream) {
    const float scale = 1.0f / std::sqrt((float)head_dim);
    dim3 grid(num_q_heads, num_q_tiles);
    paged_flash_attention_kernel<<<grid, 32, 0, stream>>>(
        dQ, dK, dV, dO, d_block_table, seq_len, q_start_base,
        num_q_heads, num_kv_heads, head_dim, scale);
}

void launch_paged_flash_attention_decode(
    const float* d_Q, const kv_t* d_k_pool_layer, const kv_t* d_v_pool_layer,
    float* d_O, const int32_t* d_block_table, int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim, cudaStream_t stream) {
    // One query at pos seq_len-1; q_start_base places local row 0 there.
    launch_pfa(d_Q, d_k_pool_layer, d_v_pool_layer, d_O, d_block_table,
               seq_len, /*q_start_base=*/seq_len - 1, /*num_q_tiles=*/1,
               num_q_heads, num_kv_heads, head_dim, stream);
}

void launch_paged_flash_attention_prefill(
    const float* d_Q, const kv_t* d_k_pool_layer, const kv_t* d_v_pool_layer,
    float* d_O, const int32_t* d_block_table, int seq_len, int num_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim, cudaStream_t stream) {
    const int num_q_tiles = (num_q_tokens + BLOCK_M - 1) / BLOCK_M;
    // The query rows occupy the last num_q_tokens positions of the sequence:
    // local row 0 sits at seq_len - num_q_tokens (0 for a fresh full prefill,
    // start_pos for a chunked/append prefill over an existing prefix).
    launch_pfa(d_Q, d_k_pool_layer, d_v_pool_layer, d_O, d_block_table,
               seq_len, /*q_start_base=*/seq_len - num_q_tokens, num_q_tiles,
               num_q_heads, num_kv_heads, head_dim, stream);
}

void launch_paged_kv_append(
    const float* d_K, const float* d_V, kv_t* d_k_pool_layer, kv_t* d_v_pool_layer,
    int page, int slot, int num_kv_heads, int head_dim, cudaStream_t stream) {
    paged_kv_append_kernel<<<num_kv_heads, head_dim, 0, stream>>>(
        d_K, d_V, d_k_pool_layer, d_v_pool_layer, page, slot, num_kv_heads, head_dim);
}

void launch_paged_kv_append_batched(
    const float* d_K, const float* d_V, kv_t* d_k_pool_layer, kv_t* d_v_pool_layer,
    const int32_t* d_block_table, int start_pos, int num_tokens,
    int num_kv_heads, int head_dim, cudaStream_t stream) {
    dim3 grid(num_kv_heads, num_tokens);
    paged_kv_append_batched_kernel<<<grid, head_dim, 0, stream>>>(
        d_K, d_V, d_k_pool_layer, d_v_pool_layer, d_block_table, start_pos,
        num_kv_heads, head_dim);
}

void launch_cow_copy_page(
    kv_t* d_k_pool, kv_t* d_v_pool, int src_page, int dst_page,
    int num_layers, int total_pages, int num_kv_heads, int head_dim,
    cudaStream_t stream) {
    const int per_page = num_kv_heads * PAGE_SIZE * head_dim;
    dim3 grid(64, num_layers);
    cow_copy_page_kernel<<<grid, 256, 0, stream>>>(
        d_k_pool, d_v_pool, src_page, dst_page, total_pages, per_page);
}
