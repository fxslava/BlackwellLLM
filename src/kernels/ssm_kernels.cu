#include "ssm_kernels.cuh"

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

    // pass 1: value the current key already retrieves under the PRE-update state.
    float Sk = 0.0f;
    for (int i = 0; i < Dk; ++i) Sk += Sh[(size_t)i * Dv + d] * kh[i];

    // delta-rule correction for this output channel (scalar in d).
    const float corr = b * (vd - Sk);

    // pass 2: write the gated delta update and read out with q.
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

}} // namespace blackwell::ssm
