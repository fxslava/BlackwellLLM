#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// SSM / linear-attention kernels — launcher contracts (single decode step)
// ============================================================================
// These implement the per-token recurrent update for the hybrid model's
// AttnKind::Linear layers. State is fp32 and evolves strictly in place (no CoW),
// matching SsmStatePool's flat per-sequence arenas.
//
// MATH NOTE (GatedDeltaNet recurrence). selective_scan_update implements the
// gated delta rule used by the Qwen3.5 linear layers:
//     dt     = softplus(dt_raw + dt_bias[h])                 (scalar per head)
//     a      = exp(dt * (-exp(A_log[h])))                    (gate / decay in (0,1))
//     β      = beta[h]                                       (write strength, in (0,1))
//     Sk     = S[h]·k        (the value currently associated with key k; ∈ R^Dv)
//     S[h]  := a·S[h] + β·(v - Sk) ⊗ k                       (delta-rule write)
//     o[h]   = S[h]·q                                        (-> Dv)
//     o[h]  := o[h] * silu(z[h])                             (optional output gate)
// This is the dimensionally-consistent form of S := S(aI - β k kᵀ) + β v kᵀ:
// expanding S(aI - βkkᵀ) = a·S - β(S·k)kᵀ, and grouping the write gives
// a·S + β(v - S·k)kᵀ. State S[h] is laid out [Dk][Dv] (element S[i*Dv+d]); the
// kernel reads/writes it in place.
//
// STABILITY PRECONDITION: k must be L2-NORMALIZED per head (‖k‖₂ = 1) before this
// call, exactly as the HF GatedDeltaNet does. The write operator along k is
// (a - β‖k‖²); with ‖k‖²=1 and a,β ∈ (0,1) its eigenvalue stays in (-1,1) so the
// recurrence is a contraction. Un-normalized k makes β‖k‖² > 1 and the state
// diverges. Normalization is the caller's job (kept out of this core op). The standalone test validates THIS recurrence
// against a CPU reference (shape + multi-step stability + parity); exact parity
// with the HF checkpoint is pinned later by the golden-dump integration test.
namespace blackwell { namespace ssm {

// Causal depthwise conv1d, single step. Per channel c in [0, conv_dim):
//   win   = [conv_state[c,0..K-2], x_new[c]]
//   y[c]  = (bias ? bias[c] : 0) + Σ_j weight[c,j] * win[j]
//   conv_state[c] <- win[1..K-1]      (shift in x_new[c])
//   y[c]  = silu(y[c])  when apply_silu
// Layout: conv_state [conv_dim, K-1] row-major; weight [conv_dim, K] row-major.
void launch_causal_conv1d_update(
    const float* d_x_new,
    float*       d_conv_state,
    const float* d_weight,
    const float* d_bias,        // nullable
    float*       d_out,
    int conv_dim,
    int kernel_width,           // K
    bool apply_silu,
    cudaStream_t stream = 0);

// GatedDeltaNet recurrent update, single step.
//   q,k:    [num_heads, key_head_dim]
//   v,z:    [num_heads, value_head_dim]   (z nullable -> no output gate)
//   dt_raw, dt_bias, A_log, beta: [num_heads]   (beta = per-head write strength)
//   state:  [num_heads, key_head_dim, value_head_dim]  (updated in place)
//   out:    [num_heads, value_head_dim]
void launch_selective_scan_update(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    const float* d_z,           // nullable
    const float* d_dt_raw,
    const float* d_dt_bias,
    const float* d_A_log,
    const float* d_beta,
    float*       d_state,
    float*       d_out,
    int num_heads,
    int key_head_dim,
    int value_head_dim,
    bool gate_silu,
    cudaStream_t stream = 0);

// ---- assembly helpers for the GatedDeltaNet decode step --------------------

// bf16 -> fp32 elementwise cast (for per-layer params consumed by fp32 kernels).
void launch_bf16_to_f32(const void* d_in_bf16, float* d_out, int n, cudaStream_t stream = 0);

// sigmoid in place (in_proj_b output -> per-head write strength beta).
void launch_sigmoid_inplace(float* d_x, int n, cudaStream_t stream = 0);

// Split the conv'd qkv [q | k | v] into per-head q,k,v, L2-NORMALIZE q and k per
// head (GatedDeltaNet precondition), and broadcast the num_key_heads q/k onto the
// num_value_heads via repeat-interleave (value head h <- key head h / ratio).
//   qkv : [num_key_heads*hd (q) | num_key_heads*hd (k) | num_value_heads*hd (v)]
//   q_out,k_out,v_out : [num_value_heads, hd]
void launch_ssm_split_norm_broadcast(
    const float* d_qkv, float* d_q_out, float* d_k_out, float* d_v_out,
    int num_key_heads, int num_value_heads, int head_dim, cudaStream_t stream = 0);

// Per-head gated RMSNorm (Mamba2 RMSNormGated): out = rmsnorm(x * silu(z)) * gamma,
// normalized over each head's `dv` channels. x,z,out: [num_heads, dv]; gamma: [dv].
void launch_gated_rmsnorm_per_head(
    const float* d_x, const float* d_z, const float* d_gamma, float* d_out,
    int num_heads, int dv, float eps, cudaStream_t stream = 0);

}} // namespace blackwell::ssm
