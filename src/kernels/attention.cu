#include "attention.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>

#define ATTN_BLOCK_SIZE 128

// 🎯 Принудительное усечение мантиссы (Truncation) для паритета с PyTorch Bfloat16
__inline__ __device__ float cast_to_bf16_and_back(float val) {
    return __bfloat162float(__float2bfloat16(val));
    /*unsigned int bits = __float_as_uint(val);
    bits &= 0xFFFF0000;
    return __uint_as_float(bits);*/
}

// Нативная редукция суммы внутри варпа
__inline__ __device__ float attn_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Нативная редукция суммы по всему блоку (128 потоков = 4 варпа)
__inline__ __device__ float attn_block_reduce_sum(float val, float* shared_sums) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    val = attn_warp_reduce_sum(val);

    if (lane_id == 0) {
        shared_sums[warp_id] = val;
    }
    __syncthreads();

    float warp_val = (threadIdx.x < (ATTN_BLOCK_SIZE / 32)) ? shared_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        warp_val = attn_warp_reduce_sum(warp_val);
    }

    __shared__ float s_final_sum;
    if (threadIdx.x == 0) {
        s_final_sum = warp_val;
    }
    __syncthreads();

    return s_final_sum;
}

__global__ void attention_decoding_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    int pos,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    float scale) 
{
    int qh = blockIdx.x;
    int tid = threadIdx.x;

    int gqa_ratio = gridDim.x / kv_heads;
    int kvh = qh / gqa_ratio;

    float q_val = (tid < head_dim) ? Q[qh * head_dim + tid] : 0.0f;

    float m = -CUDART_INF_F; 
    float l = 0.0f;          
    float acc = 0.0f;        

    __shared__ float shared_sums[ATTN_BLOCK_SIZE / 32];

    for (int t = 0; t <= pos; ++t) {
        size_t k_offset = (kvh * max_seq_len + t) * head_dim + tid;
        float k_val = (tid < head_dim) ? K_cache[k_offset] : 0.0f;

        float dot_elem = q_val * k_val;
        float score = attn_block_reduce_sum(dot_elem, shared_sums) * scale;

        float m_prev = m;
        if (score > m) {
            m = score;
        }

        float exp_score = expf(score - m);
        float exp_prev  = expf(m_prev - m);

        l = l * exp_prev + exp_score;

        size_t v_offset = (kvh * max_seq_len + t) * head_dim + tid;
        float v_val = (tid < head_dim) ? V_cache[v_offset] : 0.0f;

        acc = acc * exp_prev + exp_score * v_val;
    }

    if (tid < head_dim) {
        // 🎯 Применяем согласованное усечение к финальному вектору контекста
        float out_val = acc / l;
        O[qh * head_dim + tid] = cast_to_bf16_and_back(out_val);
    }
}

void launch_attention_decoding_kernel(
    const float* d_Q,
    const float* d_K_cache,
    const float* d_V_cache,
    float* d_O,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    cudaStream_t stream)
{
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    dim3 blocks(q_heads);
    dim3 threads(ATTN_BLOCK_SIZE);

    attention_decoding_kernel<<<blocks, threads, 0, stream>>>(
        d_Q, d_K_cache, d_V_cache, d_O, pos, kv_heads, head_dim, max_seq_len, scale
    );
}

// ============================================================================
// Split-K (Flash-Decoding) decode attention — see attention.cuh for the contract
// and for WHY: at batch = 1 the single-block kernel above occupies q_heads SMs
// and there is no other work in flight to cover the rest of the device.
// ============================================================================

namespace {

constexpr int kSplitKWarps = blackwell::attn::kSplitKBlockSize / 32;  // tokens per iteration

// Butterfly max: every lane ends up holding the warp-wide maximum, so phase 2
// needs no separate broadcast of the global softmax max.
__inline__ __device__ float split_k_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

} // namespace

// Phase 1: one block per (query head, context slice).
//
// Two DIFFERENT channel ownerships live in this kernel, which is the whole trick:
//   * the Q*K dot product is warp-local -- warp w owns ONE token and its 32 lanes
//     split that token's channels -- so a score costs warp shuffles only, and the
//     block barrier is paid once per kSplitKWarps tokens instead of once per token
//     (the single-block kernel block-reduces every token: 3 barriers per token);
//   * the softmax state (m, l) and the accumulator are thread-local, thread `tid`
//     owning output channel `tid`. m/l are updated from shared scores that EVERY
//     thread reads, so all 128 threads hold identical statistics and the slice's
//     (m, l) need no reduction at the end.
__global__ void split_k_attention_partial_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ scratch_acc,     // [q_heads][splits][head_dim]
    float* __restrict__ scratch_m,       // [q_heads][splits]
    float* __restrict__ scratch_l,       // [q_heads][splits]
    int context_len,                     // N = pos + 1
    int tokens_per_split,                // L = ceil(N / splits)
    int kv_heads,
    int head_dim,
    int max_seq_len,
    float scale)
{
    const int qh   = blockIdx.x;
    const int sl   = blockIdx.y;
    const int S    = gridDim.y;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int gqa_ratio = gridDim.x / kv_heads;
    const int kvh = qh / gqa_ratio;

    const int t_begin = sl * tokens_per_split;
    const int t_end   = min(t_begin + tokens_per_split, context_len);

    const size_t slice = (size_t)qh * S + sl;

    // Empty slice: write the IDENTITY element of the phase-2 reduction
    // (exp(-inf - M) == 0 kills both l and acc). The launcher re-derives the
    // split count from the slice width so no launched slice is ever empty --
    // this path exists so a hand-rolled launch cannot produce NaNs.
    if (t_begin >= t_end) {
        if (tid < head_dim) scratch_acc[slice * head_dim + tid] = 0.0f;
        if (tid == 0) {
            scratch_m[slice] = -CUDART_INF_F;
            scratch_l[slice] = 0.0f;
        }
        return;
    }

    // Q in the DOT-PRODUCT layout: lane `lane` owns channels lane, lane+32,
    // lane+64, lane+96 of the warp's token. Strided by the warp width so each of
    // the four K reads below is one 32-lane-contiguous transaction for any
    // head_dim <= 128. Hoisted out of the token loop -- Q is read once per block.
    float q_strided[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = lane + 32 * i;
        q_strided[i] = (c < head_dim) ? Q[(size_t)qh * head_dim + c] : 0.0f;
    }

    float m   = -CUDART_INF_F;   // running max over this slice
    float l   = 0.0f;            // running sum of exponentials over this slice
    float acc = 0.0f;            // channel `tid` of this slice's context vector

    __shared__ float s_scores[kSplitKWarps];

    for (int base = t_begin; base < t_end; base += kSplitKWarps) {
        // Identical in every thread, so the consume loop below never diverges.
        const int live = min(kSplitKWarps, t_end - base);

        if (warp < live) {
            const int t = base + warp;
            const size_t k_row = ((size_t)kvh * max_seq_len + t) * head_dim;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = lane + 32 * i;
                dot += q_strided[i] * ((c < head_dim) ? K_cache[k_row + c] : 0.0f);
            }
            dot = attn_warp_reduce_sum(dot);
            if (lane == 0) s_scores[warp] = dot * scale;
        }
        __syncthreads();

        for (int j = 0; j < live; ++j) {
            const float score = s_scores[j];
            const float m_new = fmaxf(m, score);
            // m == -inf only on the first update, where m_new is the (finite)
            // first score, so this is exp(-inf) == 0 and never exp(NaN).
            const float alpha = expf(m - m_new);
            const float w     = expf(score - m_new);

            l = l * alpha + w;

            const size_t v_row = ((size_t)kvh * max_seq_len + (base + j)) * head_dim;
            const float v_val = (tid < head_dim) ? V_cache[v_row + tid] : 0.0f;
            acc = acc * alpha + w * v_val;

            m = m_new;
        }
        __syncthreads();   // s_scores is overwritten by the next iteration
    }

    if (tid < head_dim) scratch_acc[slice * head_dim + tid] = acc;
    if (tid == 0) {
        scratch_m[slice] = m;
        scratch_l[slice] = l;
    }
}

// Phase 2: one block per query head. Folds the S per-slice softmax statistics
// into the global log-sum-exp and rescales the partial context vectors:
//   M = max_s m_s,   L = sum_s l_s * exp(m_s - M),
//   O[h,d] = (1/L) * sum_s acc[h,s,d] * exp(m_s - M).
__global__ void split_k_attention_reduce_kernel(
    const float* __restrict__ scratch_acc,
    const float* __restrict__ scratch_m,
    const float* __restrict__ scratch_l,
    float* __restrict__ O,
    int splits,
    int head_dim)
{
    const int qh   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    __shared__ float s_weight[blackwell::attn::kSplitKMaxSplits];
    __shared__ float s_inv_l;

    const size_t base_slice = (size_t)qh * splits;

    // splits <= kSplitKMaxSplits <= 32, so the whole reduction fits in warp 0 and
    // the butterfly shuffles leave the result in every lane -- no broadcast step.
    // L >= 1 always: the slice owning the global max contributes l_s >= 1 (its own
    // maximal term is exp(0)), so the reciprocal below is finite.
    if (tid < 32) {
        const float m_s = (lane < splits) ? scratch_m[base_slice + lane] : -CUDART_INF_F;
        const float M   = split_k_warp_reduce_max(m_s);
        const float w_s = (lane < splits) ? expf(m_s - M) : 0.0f;
        const float l_s = (lane < splits) ? scratch_l[base_slice + lane] : 0.0f;
        const float L   = attn_warp_reduce_sum(l_s * w_s);
        if (lane < splits) s_weight[lane] = w_s;
        if (lane == 0)     s_inv_l = 1.0f / L;
    }
    __syncthreads();

    if (tid < head_dim) {
        float out_val = 0.0f;
        for (int s = 0; s < splits; ++s)
            out_val += scratch_acc[(base_slice + s) * head_dim + tid] * s_weight[s];
        // The SAME deliberate mantissa truncation the single-block kernel applies
        // (PyTorch bf16 parity), so the two paths differ by at most one bf16 ULP.
        O[(size_t)qh * head_dim + tid] = cast_to_bf16_and_back(out_val * s_inv_l);
    }
}

void launch_attention_decoding_split_k(
    const float* d_Q,
    const float* d_K_cache,
    const float* d_V_cache,
    float* d_O,
    float* d_scratch,
    int splits,
    int pos,
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    cudaStream_t stream)
{
    const int context_len = pos + 1;

    // Clamp the request, then RE-DERIVE it from the slice width the clamp
    // produced: with L = ceil(N/S) the trailing slices can come out empty
    // (N=5, S=4 -> L=2 already covers everything in 3 slices), and an empty slice
    // is a block launched to do nothing. Same normalization as the AWQ GEMV's
    // split-K (src/kernels/awq_linear.cu).
    //
    // The re-derivation is guarded on S > 1, which also keeps the slice width out
    // of the denominator for a context of 0 or 1 tokens: neither is splittable, and
    // context_len == 0 (pos < 0 -- outside the contract, but a crash is a worse
    // answer than the single-block kernel's) would otherwise divide by zero.
    int S = splits;
    if (S > blackwell::attn::kSplitKMaxSplits) S = blackwell::attn::kSplitKMaxSplits;
    if (S > context_len) S = context_len;

    // The SLICE WIDTH is what the grid is then sized from, and it is the width the
    // kernel is launched with -- S slices of L tokens cover [0, N) with none empty
    // exactly because S == ceil(N / L). Deriving L from the normalized S instead
    // would not preserve that.
    int L = 0;
    if (S > 1) {
        L = (context_len + S - 1) / S;          // >= 1, since S <= context_len
        S = (context_len + L - 1) / L;          // slices that width actually needs
    }

    if (S <= 1 || d_scratch == nullptr) {
        launch_attention_decoding_kernel(d_Q, d_K_cache, d_V_cache, d_O, pos,
                                         q_heads, kv_heads, head_dim, max_seq_len, stream);
        return;
    }

    // Scratchpad regions for THIS step's split factor. S never exceeds the
    // max_splits the arena sized the allocation for, so these offsets stay inside
    // it (split_k_scratch_floats documents the layout).
    const size_t slices = q_heads * (size_t)S;
    float* d_acc = d_scratch;
    float* d_m   = d_scratch + slices * head_dim;
    float* d_l   = d_m + slices;

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    split_k_attention_partial_kernel<<<dim3((unsigned)q_heads, (unsigned)S),
                                       blackwell::attn::kSplitKBlockSize, 0, stream>>>(
        d_Q, d_K_cache, d_V_cache, d_acc, d_m, d_l,
        context_len, L, (int)kv_heads, (int)head_dim, (int)max_seq_len, scale);

    split_k_attention_reduce_kernel<<<dim3((unsigned)q_heads),
                                      blackwell::attn::kSplitKBlockSize, 0, stream>>>(
        d_acc, d_m, d_l, d_O, S, (int)head_dim);
}

namespace blackwell {
namespace attn {

SplitKKernelStats query_split_k_kernel_stats() {
    SplitKKernelStats stats;
    cudaFuncAttributes attr{};

    if (cudaFuncGetAttributes(&attr, split_k_attention_partial_kernel) == cudaSuccess) {
        stats.partial_num_regs     = attr.numRegs;
        stats.partial_shared_bytes = attr.sharedSizeBytes;
        stats.partial_local_bytes  = attr.localSizeBytes;
    }
    if (cudaFuncGetAttributes(&attr, split_k_attention_reduce_kernel) == cudaSuccess) {
        stats.reduce_num_regs      = attr.numRegs;
        stats.reduce_shared_bytes  = attr.sharedSizeBytes;
        stats.reduce_local_bytes   = attr.localSizeBytes;
    }
    if (cudaFuncGetAttributes(&attr, attention_decoding_kernel) == cudaSuccess) {
        stats.legacy_num_regs      = attr.numRegs;
        stats.legacy_shared_bytes  = attr.sharedSizeBytes;
        stats.legacy_local_bytes   = attr.localSizeBytes;
    }
    return stats;
}

} // namespace attn
} // namespace blackwell