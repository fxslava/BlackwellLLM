#include "ssm_kernels.cuh"
#include <cuda_bf16.h>

namespace blackwell { namespace ssm {

// ----------------------------------------------------------------------------
// numerically-stable softplus and silu (device).
// ----------------------------------------------------------------------------
__device__ __forceinline__ float softplus_stable(float x) {
    // log1p(exp(x)) with the large-x linear branch to avoid exp() overflow.
    return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));   // x * sigmoid(x)
}

// ============================================================================
// 1. Causal depthwise conv1d — one thread per channel.
// ============================================================================
__global__ void causal_conv1d_update_kernel(const float* __restrict__ x_new,
                                            float* __restrict__ conv_state,
                                            const float* __restrict__ weight,
                                            const float* __restrict__ bias,
                                            float* __restrict__ out,
                                            int conv_dim, int K, bool apply_silu) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    const float* __restrict__ wc = weight + (size_t)c * K;
    float* __restrict__ sc       = conv_state + (size_t)c * (K - 1);
    const float xc = x_new[c];

    // y = bias + Σ w[j]*win[j], win = [sc[0..K-2], xc]
    float acc = bias ? bias[c] : 0.0f;
    #pragma unroll
    for (int j = 0; j < K - 1; ++j) acc += wc[j] * sc[j];
    acc += wc[K - 1] * xc;

    // Shift the ring buffer left and append the new input.
    #pragma unroll
    for (int j = 0; j < K - 2; ++j) sc[j] = sc[j + 1];
    if (K >= 2) sc[K - 2] = xc;

    out[c] = apply_silu ? silu(acc) : acc;
}

// ============================================================================
// 2. GatedDeltaNet recurrent update.
//    grid.x = head, blockDim.x = value_head_dim (one thread per output channel d).
//    Thread d owns column S[:, d]. Two passes over the Dk contraction:
//      pass 1 computes Sk[d] = Σ_i S[i,d]·k[i]   (value currently keyed by k)
//      pass 2 writes  S[i,d] := a·S[i,d] + β·(v[d] - Sk[d])·k[i]   (delta rule)
//             and accumulates out[d] = Σ_i S_new[i,d]·q[i].
//    S[i,d] is element Sh[i*Dv + d]; threads in a block read the same i with
//    consecutive d, so every state access is coalesced.
// ============================================================================
__global__ void selective_scan_update_kernel(const float* __restrict__ q,
                                             const float* __restrict__ k,
                                             const float* __restrict__ v,
                                             const float* __restrict__ z,
                                             const float* __restrict__ dt_raw,
                                             const float* __restrict__ dt_bias,
                                             const float* __restrict__ A_log,
                                             const float* __restrict__ beta,
                                             float* __restrict__ state,
                                             float* __restrict__ out,
                                             int Dk, int Dv, bool gate_silu) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;          // output channel in [0, Dv)
    if (d >= Dv) return;

    const float dt = softplus_stable(dt_raw[h] + dt_bias[h]);
    const float a  = __expf(dt * (-__expf(A_log[h])));   // gate / decay in (0,1)
    const float b  = beta[h];                            // write strength

    const float* __restrict__ qh = q + (size_t)h * Dk;
    const float* __restrict__ kh = k + (size_t)h * Dk;
    const float vd = v[(size_t)h * Dv + d];
    float* __restrict__ Sh = state + (size_t)h * Dk * Dv;   // [Dk, Dv]

    // pass 1: kv_mem = (decayed state)·k  (HF applies the gate BEFORE the delta).
    float Sk = 0.0f;
    for (int i = 0; i < Dk; ++i) Sk += (a * Sh[(size_t)i * Dv + d]) * kh[i];

    // delta-rule correction for this output channel (scalar in d).
    const float corr = b * (vd - Sk);

    // pass 2: S := a·S + corr·k ; read out with q.
    float acc = 0.0f;
    for (int i = 0; i < Dk; ++i) {
        const float s = a * Sh[(size_t)i * Dv + d] + corr * kh[i];
        Sh[(size_t)i * Dv + d] = s;
        acc += qh[i] * s;
    }

    if (z) {
        float g = z[(size_t)h * Dv + d];
        acc *= gate_silu ? silu(g) : g;
    }
    out[(size_t)h * Dv + d] = acc;
}

// ----------------------------------------------------------------------------
// Launchers
// ----------------------------------------------------------------------------
void launch_causal_conv1d_update(const float* d_x_new, float* d_conv_state,
                                 const float* d_weight, const float* d_bias,
                                 float* d_out, int conv_dim, int kernel_width,
                                 bool apply_silu, cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (conv_dim + threads - 1) / threads;
    causal_conv1d_update_kernel<<<blocks, threads, 0, stream>>>(
        d_x_new, d_conv_state, d_weight, d_bias, d_out,
        conv_dim, kernel_width, apply_silu);
}

void launch_selective_scan_update(const float* d_q, const float* d_k,
                                  const float* d_v, const float* d_z,
                                  const float* d_dt_raw, const float* d_dt_bias,
                                  const float* d_A_log, const float* d_beta,
                                  float* d_state, float* d_out,
                                  int num_heads, int key_head_dim, int value_head_dim,
                                  bool gate_silu, cudaStream_t stream) {
    // One block per head; one thread per output (value) channel.
    dim3 grid(num_heads);
    dim3 block(value_head_dim);
    selective_scan_update_kernel<<<grid, block, 0, stream>>>(
        d_q, d_k, d_v, d_z, d_dt_raw, d_dt_bias, d_A_log, d_beta,
        d_state, d_out, key_head_dim, value_head_dim, gate_silu);
}

// ============================================================================
// 2b. CHUNKED delta rule — parallel prefill over a chunk of tokens.
//     One block per head; the O(C·Dk·Dv) intra-chunk work is spread across the
//     block, chunks (of up to CHUNK tokens) are processed sequentially so the
//     recurrent state S carries forward. This is the direct production port of
//     the verified sandbox (tests/standalone/test_gated_delta_chunk.cu); the math
//     is byte-for-byte that reference, generalized to runtime dims, a token-major
//     layout, and a partial final chunk (cur_C < CHUNK). See the header contract.
// ============================================================================
namespace { constexpr int kDeltaChunk = 64; }   // C: intra-chunk token tile

// token-major index helpers (runtime dims passed in).
__device__ __forceinline__ size_t gd_idxKD(int t, int h, int i, int H, int Dk) {
    return ((size_t)t * H + h) * Dk + i;              // q,k : [T,H,Dk]
}
__device__ __forceinline__ size_t gd_idxVD(int t, int h, int d, int H, int Dv) {
    return ((size_t)t * H + h) * Dv + d;              // v,o : [T,H,Dv]
}
__device__ __forceinline__ size_t gd_idxS(int i, int d, int Dv) {
    return (size_t)i * Dv + d;                        // S(i,d) within a head block
}

// NUMERICALLY STABLE chunked delta rule. The naive form absorbs the cumulative
// decay by dividing v_t / b_t (b_t = Π_{j<=t} a_j); with real GatedDeltaNet decays
// b_t underflows to 0 and v/b_t blows up to inf/NaN (the per-token recurrence
// never divides). We instead carry the cumulative LOG-decay ℓ_t = Σ_{j<=t} log a_j
// (≤ 0) and use ONLY bounded factors: exp(ℓ_t) ∈ (0,1] and the pair ratios
// exp(ℓ_t − ℓ_s) ∈ (0,1] for t ≥ s. Rescaling the corrected writes to ũ_t = b_t u_t
// gives the same outputs and state with every intermediate bounded:
//     Ã[t,s] = β_t exp(ℓ_t−ℓ_s)(k_t·k_s)          (s<t; unit diagonal)
//     ũ_t    = β_t v_t − β_t exp(ℓ_t)(k_t·S_0),   solve (I+Ã)Ũ = RHS
//     o_t    = exp(ℓ_t)(q_t·S_0) + Σ_{s<=t} exp(ℓ_t−ℓ_s)(q_t·k_s) ũ_s
//     S_C    = exp(ℓ_{C-1}) S_0 + Σ_s exp(ℓ_{C-1}−ℓ_s) k_s ũ_s^T
// At C=1 this collapses to the single-token recurrence (β(v−a·k·S_0) etc.). The
// `logdecay` input already IS ℓ per token (= dt·(−exp(A_log)), summed here) so no
// log(exp()) round-trip. See the header contract for layouts.
__global__ void gated_delta_chunked_prefill_kernel(const float* __restrict__ k,
                                                   const float* __restrict__ q,
                                                   const float* __restrict__ v,
                                                   const float* __restrict__ logdecay,
                                                   const float* __restrict__ beta,
                                                   const float* h_prev,   // may alias h_out
                                                   float* __restrict__ o_out,
                                                   float* h_out,          // may alias h_prev
                                                   float* __restrict__ u_scratch,
                                                   int H, int L, int Dk, int Dv) {
    const int h  = blockIdx.x;
    const int tx = threadIdx.x;
    const int NT = blockDim.x;
    const int C  = kDeltaChunk;

    __shared__ float s_ld[kDeltaChunk];       // per-token log-decay ℓ (then cumsum)
    __shared__ float s_beta[kDeltaChunk];
    __shared__ float s_M[kDeltaChunk * kDeltaChunk];   // (I + Ã), unit lower-tri

    // Persistent per-head state lives in h_out (seeded from h_prev, carried, and
    // finally IS the chunk-C state). Seed once (h_prev==h_out => self-copy).
    for (int e = tx; e < Dk * Dv; e += NT)
        h_out[(size_t)h * Dk * Dv + e] = h_prev[(size_t)h * Dk * Dv + e];
    __syncthreads();
    float* __restrict__ S = h_out + (size_t)h * Dk * Dv;
    float* __restrict__ U = u_scratch + (size_t)h * C * Dv;

    for (int base = 0; base < L; base += C) {
        const int cur = min(C, L - base);          // partial final chunk allowed

        // -- per-token log-decay + write strength, then inclusive cumulative sum of
        //    the log-decays into s_ld (== ℓ_t; exp(ℓ_t) is the bounded decay b_t).
        for (int t = tx; t < cur; t += NT) {
            s_ld[t]   = logdecay[(size_t)(base + t) * H + h];
            s_beta[t] = beta    [(size_t)(base + t) * H + h];
        }
        __syncthreads();
        if (tx == 0) {
            float acc = 0.f;
            for (int t = 0; t < cur; ++t) { acc += s_ld[t]; s_ld[t] = acc; }
        }
        __syncthreads();

        // -- unit lower-tri system Ã[t,s] = β_t exp(ℓ_t−ℓ_s)(k_t·k_s) for s<t.
        for (int p = tx; p < cur * cur; p += NT) {
            const int t = p / cur, s = p % cur;
            float m;
            if (s > t)       m = 0.f;
            else if (s == t) m = 1.f;
            else {
                float dot = 0.f;
                for (int i = 0; i < Dk; ++i)
                    dot += k[gd_idxKD(base + t, h, i, H, Dk)] * k[gd_idxKD(base + s, h, i, H, Dk)];
                m = s_beta[t] * __expf(s_ld[t] - s_ld[s]) * dot;
            }
            s_M[t * C + s] = m;
        }
        __syncthreads();

        // -- RHS (rescaled): ũ_t = β_t v_t − β_t exp(ℓ_t)(k_t·S_0)   (no division)
        for (int p = tx; p < cur * Dv; p += NT) {
            const int t = p / Dv, d = p % Dv;
            float kS = 0.f;
            for (int i = 0; i < Dk; ++i)
                kS += k[gd_idxKD(base + t, h, i, H, Dk)] * S[gd_idxS(i, d, Dv)];
            const float vt = v[gd_idxVD(base + t, h, d, H, Dv)];
            U[(size_t)t * Dv + d] = s_beta[t] * (vt - __expf(s_ld[t]) * kS);
        }
        __syncthreads();

        // -- forward substitution: ũ_t -= Σ_{s<t} Ã[t,s] ũ_s  (sequential in t)
        for (int t = 1; t < cur; ++t) {
            for (int d = tx; d < Dv; d += NT) {
                float acc = 0.f;
                for (int s = 0; s < t; ++s) acc += s_M[t * C + s] * U[(size_t)s * Dv + d];
                U[(size_t)t * Dv + d] -= acc;
            }
            __syncthreads();
        }

        // -- outputs: o_t = exp(ℓ_t)(q_t·S_0) + Σ_{s<=t} exp(ℓ_t−ℓ_s)(q_t·k_s) ũ_s
        for (int p = tx; p < cur * Dv; p += NT) {
            const int t = p / Dv, d = p % Dv;
            float qS = 0.f;
            for (int i = 0; i < Dk; ++i)
                qS += q[gd_idxKD(base + t, h, i, H, Dk)] * S[gd_idxS(i, d, Dv)];
            float cross = 0.f;
            for (int s = 0; s <= t; ++s) {
                float qk = 0.f;
                for (int i = 0; i < Dk; ++i)
                    qk += q[gd_idxKD(base + t, h, i, H, Dk)] * k[gd_idxKD(base + s, h, i, H, Dk)];
                cross += __expf(s_ld[t] - s_ld[s]) * qk * U[(size_t)s * Dv + d];
            }
            o_out[gd_idxVD(base + t, h, d, H, Dv)] = __expf(s_ld[t]) * qS + cross;
        }
        __syncthreads();                           // outputs consumed S_0 before overwrite

        // -- state carry: S_C = exp(ℓ_{cur-1}) S_0 + Σ_s exp(ℓ_{cur-1}−ℓ_s) k_s ũ_s^T
        const float bC = __expf(s_ld[cur - 1]);
        for (int p = tx; p < Dk * Dv; p += NT) {
            const int i = p / Dv, d = p % Dv;
            float ku = 0.f;
            for (int s = 0; s < cur; ++s)
                ku += __expf(s_ld[cur - 1] - s_ld[s]) * k[gd_idxKD(base + s, h, i, H, Dk)]
                          * U[(size_t)s * Dv + d];
            S[gd_idxS(i, d, Dv)] = bC * S[gd_idxS(i, d, Dv)] + ku;
        }
        __syncthreads();                           // S_C visible as S_0 for next chunk
    }
}

// Per-head LOG-decay ℓ = dt·(−exp(A_log)) with dt = softplus(a_raw + dt_bias). This
// is exactly log(a) of selective_scan_update's decay a, but kept in log space so
// the chunk kernel's cumulative sum never underflows (feeds `logdecay`).
__global__ void gated_delta_logdecay_kernel(const float* __restrict__ a_raw,
                                           const float* __restrict__ dt_bias,
                                           const float* __restrict__ A_log,
                                           float* __restrict__ logdecay_out, int H) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const float dt = softplus_stable(a_raw[h] + dt_bias[h]);
    logdecay_out[h] = dt * (-__expf(A_log[h]));
}

void launch_gated_delta_chunked_prefill(const float* d_k, const float* d_q, const float* d_v,
                                        const float* d_logdecay, const float* d_beta,
                                        const float* d_h_prev, float* d_o_out, float* d_h_out,
                                        float* d_u_scratch, int num_heads, int num_tokens,
                                        int key_head_dim, int value_head_dim,
                                        cudaStream_t stream) {
    // One block per head; 256 threads spread the intra-chunk work. Shared memory
    // (s_ld/s_beta/s_M) is statically sized to the CHUNK tile.
    if (num_tokens <= 0) return;
    gated_delta_chunked_prefill_kernel<<<num_heads, 256, 0, stream>>>(
        d_k, d_q, d_v, d_logdecay, d_beta, d_h_prev, d_o_out, d_h_out, d_u_scratch,
        num_heads, num_tokens, key_head_dim, value_head_dim);
}

void launch_gated_delta_logdecay(const float* d_a_raw, const float* d_dt_bias,
                                 const float* d_A_log, float* d_logdecay_out,
                                 int num_heads, cudaStream_t stream) {
    const int threads = 256, blocks = (num_heads + threads - 1) / threads;
    gated_delta_logdecay_kernel<<<blocks, threads, 0, stream>>>(
        d_a_raw, d_dt_bias, d_A_log, d_logdecay_out, num_heads);
}

// ============================================================================
// 3. GatedDeltaNet assembly helpers
// ============================================================================
__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ in,
                                    float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void sigmoid_inplace_kernel(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// One block per value head; blockDim.x = head_dim. L2-normalizes the (broadcast)
// query/key head over head_dim and copies the value head through.
__global__ void split_norm_broadcast_kernel(const float* __restrict__ qkv,
                                            float* __restrict__ q_out,
                                            float* __restrict__ k_out,
                                            float* __restrict__ v_out,
                                            int num_key_heads, int num_value_heads,
                                            int hd) {
    const int h = blockIdx.x;           // value head [0, num_value_heads)
    const int t = threadIdx.x;          // channel [0, hd)
    if (h >= num_value_heads || t >= hd) return;

    const int ratio = num_value_heads / num_key_heads;   // GQA group size
    const int kh    = h / ratio;                          // source key/query head
    const int q_off = 0;
    const int k_off = num_key_heads * hd;
    const int v_off = 2 * num_key_heads * hd;

    const float qv = qkv[q_off + kh * hd + t];
    const float kv = qkv[k_off + kh * hd + t];

    // L2 norm over the head's hd channels (block reduction; eps matches HF l2norm).
    __shared__ float red[1024];
    // --- q --- (HF additionally scales the query by 1/sqrt(head_dim))
    red[t] = qv * qv; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    const float q_inv = rsqrtf(red[0] + 1e-6f); __syncthreads();
    // --- k ---
    red[t] = kv * kv; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    const float k_inv = rsqrtf(red[0] + 1e-6f);

    q_out[h * hd + t] = qv * q_inv * rsqrtf((float)hd);   // l2norm(q) * 1/sqrt(Dk)
    k_out[h * hd + t] = kv * k_inv;
    v_out[h * hd + t] = qkv[v_off + h * hd + t];
}

// Per-head gated RMSNorm: out = rmsnorm(x * silu(z)) * gamma over dv channels.
__global__ void gated_rmsnorm_per_head_kernel(const float* __restrict__ x,
                                              const float* __restrict__ z,
                                              const float* __restrict__ gamma,
                                              float* __restrict__ out,
                                              int dv, float eps) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (t >= dv) return;
    // Qwen3.5 RMSNormGated: normalize FIRST, scale by gamma, THEN apply the gate.
    const float xv = x[h * dv + t];
    __shared__ float red[1024];
    red[t] = xv * xv; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / dv + eps);
    const float normed = xv * inv * gamma[t];
    out[h * dv + t] = normed * silu(z[h * dv + t]);
}

void launch_bf16_to_f32(const void* d_in_bf16, float* d_out, int n, cudaStream_t stream) {
    const int threads = 256, blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(d_in_bf16), d_out, n);
}
void launch_sigmoid_inplace(float* d_x, int n, cudaStream_t stream) {
    const int threads = 256, blocks = (n + threads - 1) / threads;
    sigmoid_inplace_kernel<<<blocks, threads, 0, stream>>>(d_x, n);
}
void launch_ssm_split_norm_broadcast(const float* d_qkv, float* d_q_out, float* d_k_out,
                                     float* d_v_out, int num_key_heads, int num_value_heads,
                                     int head_dim, cudaStream_t stream) {
    split_norm_broadcast_kernel<<<num_value_heads, head_dim, 0, stream>>>(
        d_qkv, d_q_out, d_k_out, d_v_out, num_key_heads, num_value_heads, head_dim);
}
void launch_gated_rmsnorm_per_head(const float* d_x, const float* d_z, const float* d_gamma,
                                   float* d_out, int num_heads, int dv, float eps,
                                   cudaStream_t stream) {
    gated_rmsnorm_per_head_kernel<<<num_heads, dv, 0, stream>>>(d_x, d_z, d_gamma, d_out, dv, eps);
}

}} // namespace blackwell::ssm
