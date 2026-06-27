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
// MATH NOTE (reference recurrence). This is the Mamba2 / gated-linear-attention
// ("SSD", diagonal-A) step:
//     dt     = softplus(dt_raw + dt_bias[h])                 (scalar per head)
//     a      = exp(dt * (-exp(A_log[h])))                    (decay in (0,1))
//     S[h]  := a * S[h] + (dt * k) ⊗ v                       (Dk x Dv outer product)
//     o[h]   = qᵀ · S[h]                                     (-> Dv)
//     o[h]  := o[h] * silu(z[h])                             (optional output gate)
// Qwen3.5's GatedDeltaNet additionally carries the delta-rule correction
// S := S(aI - β k kᵀ) + β k vᵀ; that extra term is the documented reconciliation
// point before claiming bit-exact parity with the HF checkpoint. The standalone
// test validates THIS recurrence against a CPU reference (shape + stability +
// parity), proving the kernel/allocator plumbing in isolation.
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

// Gated linear-attention recurrent update, single step.
//   q,k:    [num_heads, key_head_dim]
//   v,z:    [num_heads, value_head_dim]   (z nullable -> no output gate)
//   dt_raw, dt_bias, A_log: [num_heads]
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
    float*       d_state,
    float*       d_out,
    int num_heads,
    int key_head_dim,
    int value_head_dim,
    bool gate_silu,
    cudaStream_t stream = 0);

}} // namespace blackwell::ssm
