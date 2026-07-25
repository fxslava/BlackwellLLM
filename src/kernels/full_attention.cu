#include "full_attention.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>

// ============================================================================
// Qwen3.5 hybrid full-attention kernels. See full_attention.cuh for the contract.
// ============================================================================

// bf16 truncation for parity with the PyTorch bf16 reference (mirrors attention.cu).
__inline__ __device__ float fa_cast_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__inline__ __device__ float fa_warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

// Block-wide sum broadcast to every thread. blockDim.x is a multiple of 32 and
// <= 256 (8 warps). s_warp must hold blockDim.x/32 floats.
__inline__ __device__ float fa_block_reduce_sum(float v, float* s_warp, int nwarp) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    v = fa_warp_reduce_sum(v);
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();

    float w = (threadIdx.x < nwarp) ? s_warp[lane] : 0.0f;
    if (wid == 0) w = fa_warp_reduce_sum(w);

    __shared__ float s_final;
    if (threadIdx.x == 0) s_final = w;
    __syncthreads();
    return s_final;
}

// ---------------------------------------------------------------------------
// q_proj de-interleave: [num_heads, 2*head_dim] -> Q + gate (per head)
// ---------------------------------------------------------------------------
__global__ void qg_split_kernel(const float* __restrict__ QG,
                                float* __restrict__ Q,
                                float* __restrict__ gate,
                                int head_dim) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;             // blockDim.x == head_dim
    const int src = h * 2 * head_dim + d;
    const int dst = h * head_dim + d;
    Q[dst]    = QG[src];
    gate[dst] = QG[src + head_dim];
}

void launch_qg_split(const float* d_QG, float* d_Q, float* d_gate,
                     int num_heads, int head_dim) {
    qg_split_kernel<<<num_heads, head_dim>>>(d_QG, d_Q, d_gate, head_dim);
}

// ---------------------------------------------------------------------------
// Partial rotate_half RoPE, in place over the first rotary_dim channels.
// ---------------------------------------------------------------------------
__global__ void rope_partial_kernel(float* __restrict__ X, int pos,
                                    int head_dim, int rotary_dim, float rope_theta,
                                    RopeScaling scaling) {
    const int head = blockIdx.x;
    const int k = threadIdx.x;             // 0 .. rotary_dim/2 - 1
    const int half = rotary_dim / 2;
    if (k >= half) return;

    float* cur = X + head * head_dim;

    // inv_freq[k] = theta^-(2k/rotary_dim); HF builds cos/sin from this base.
    float freq = __fdividef(1.0f,
        powf(rope_theta, __fdividef((float)(2 * k), (float)rotary_dim)));
    freq = apply_rope_scaling(freq, scaling);
    const float angle = pos * freq;
    float s, c;
    sincosf(angle, &s, &c);

    const float x0 = cur[k];
    const float x1 = cur[k + half];
    cur[k]        = x0 * c - x1 * s;
    cur[k + half] = x0 * s + x1 * c;
}

void launch_rope_partial_inplace(float* d_X, int pos, int num_heads,
                                 int head_dim, int rotary_dim, float rope_theta,
                                 RopeScaling scaling) {
    rope_partial_kernel<<<num_heads, rotary_dim / 2>>>(d_X, pos, head_dim, rotary_dim, rope_theta, scaling);
}

// ---------------------------------------------------------------------------
// KV append (no rotation): write full head_dim of K and V into the cache slot.
// ---------------------------------------------------------------------------
__global__ void kv_append_kernel(const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ K_cache,
                                 float* __restrict__ V_cache,
                                 int pos, int head_dim, int max_seq_len) {
    const int kvh = blockIdx.x;
    const int d = threadIdx.x;             // blockDim.x == head_dim
    const size_t slot = ((size_t)kvh * max_seq_len + pos) * head_dim + d;
    K_cache[slot] = K[kvh * head_dim + d];
    V_cache[slot] = V[kvh * head_dim + d];
}

void launch_kv_append(const float* d_K, const float* d_V,
                      float* d_K_cache, float* d_V_cache,
                      int pos, int kv_heads, int head_dim, int max_seq_len) {
    kv_append_kernel<<<kv_heads, head_dim>>>(d_K, d_V, d_K_cache, d_V_cache,
                                             pos, head_dim, max_seq_len);
}

// ---------------------------------------------------------------------------
// Decode attention with online softmax. One block per query head, one thread per
// head channel (blockDim.x == head_dim).
// ---------------------------------------------------------------------------
__global__ void full_attention_decode_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K_cache,
                                             const float* __restrict__ V_cache,
                                             float* __restrict__ O, int pos,
                                             int kv_heads, int head_dim,
                                             int max_seq_len, float scale) {
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nwarp = blockDim.x >> 5;
    const int gqa = gridDim.x / kv_heads;
    const int kvh = qh / gqa;

    const float q_val = Q[qh * head_dim + tid];

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;

    extern __shared__ float s_warp[];      // nwarp floats

    for (int t = 0; t <= pos; ++t) {
        __syncthreads();                   // protect s_warp reuse across iterations
        const size_t koff = ((size_t)kvh * max_seq_len + t) * head_dim + tid;
        const float score = fa_block_reduce_sum(q_val * K_cache[koff], s_warp, nwarp) * scale;

        const float m_prev = m;
        if (score > m) m = score;
        const float exp_score = expf(score - m);
        const float exp_prev = expf(m_prev - m);

        l = l * exp_prev + exp_score;
        acc = acc * exp_prev + exp_score * V_cache[koff];
    }

    O[qh * head_dim + tid] = fa_cast_bf16(acc / l);
}

void launch_full_attention_decode(const float* d_Q, const float* d_K_cache,
                                  const float* d_V_cache, float* d_O, int pos,
                                  int q_heads, int kv_heads, int head_dim,
                                  int max_seq_len) {
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int nwarp = head_dim >> 5;
    full_attention_decode_kernel<<<q_heads, head_dim, nwarp * sizeof(float)>>>(
        d_Q, d_K_cache, d_V_cache, d_O, pos, kv_heads, head_dim, max_seq_len, scale);
}

// ---------------------------------------------------------------------------
// Output gating: O[i] *= sigmoid(gate[i]).
// ---------------------------------------------------------------------------
__global__ void gate_sigmoid_mul_kernel(float* __restrict__ O,
                                        const float* __restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    O[i] = O[i] * (1.0f / (1.0f + expf(-gate[i])));
}

void launch_gate_sigmoid_mul(float* d_O, const float* d_gate, int n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    gate_sigmoid_mul_kernel<<<blocks, threads>>>(d_O, d_gate, n);
}
