#pragma once
#include <cstddef>

// ============================================================================
// Bias epilogue kernels (Qwen2 family: q/k/v projections carry a bias term).
//
// CONTRACT OVERVIEW
// -----------------
// These kernels are backend-agnostic epilogues: they run AFTER any of the
// dispatcher's GEMV backends (bf16 / fp8 / awq) and add a per-output-channel
// bias to the fp32 projection result, in place. They exist so that the
// quantized backends do not need bias-aware variants.
//
// ORDERING CONSTRAINT (engine-level, but binding for any test harness):
// the QKV bias must be applied BEFORE launch_fused_rope_kv_kernel rotates
// Q/K and before V is appended to the KV cache. HF reference computes
// (x @ W^T + b) and only then applies RoPE.
//
// PRECISION
// ---------
// - Activations (d_Y / d_Q / d_K / d_V) are contiguous fp32 device buffers.
// - Bias tensors come straight from the checkpoint via VRAMArena as opaque
//   `const void*`:
//     * BF16 checkpoints (Qwen2.5-Coder base/instruct): __nv_bfloat16[N]
//     * AWQ checkpoints (AutoAWQ keeps biases in half): __half[N]
//   The `dtype` parameter selects the decode path. Conversion to fp32 must
//   use the native intrinsics (__bfloat162float / __half2float); the add is
//   performed once, in fp32 (bf16-parity rule: no intermediate rounding).
// - nullptr bias pointer => kernel launch must be skipped entirely (no-op).
//
// All launches are asynchronous on the default stream, matching the rest of
// src/kernels/.
// ============================================================================

enum class BiasDType {
    BF16, // __nv_bfloat16 elements
    FP16, // __half elements
};

// ----------------------------------------------------------------------------
// launch_bias_add_kernel
//
//   d_Y[i] += to_fp32(d_bias[i])   for i in [0, N)
//
// Memory layout:
//   d_Y    : float[N], contiguous, stride 1 (decode batch == 1 token).
//   d_bias : BF16/FP16[N], contiguous, stride 1.
//
// Grid/block mapping (implementation requirement):
//   block = 256 threads, 1-D
//   grid  = ceil(N / 256), 1-D
//   Thread t handles element (blockIdx.x * blockDim.x + threadIdx.x);
//   guard `if (i >= N) return;`. One global load of bias, one read-modify-
//   write of Y per thread; no shared memory, no reductions.
//
// Edge cases:
//   d_bias == nullptr -> return immediately without launching.
//   N == 0            -> return immediately without launching.
// ----------------------------------------------------------------------------
void launch_bias_add_kernel(float* d_Y,
                            const void* d_bias,
                            size_t N,
                            BiasDType dtype = BiasDType::BF16);

// ----------------------------------------------------------------------------
// launch_fused_qkv_bias_kernel
//
// Single launch covering all three attention projection biases of one layer
// (saves 2 kernel launches per layer; 28 layers x decode loop makes this
// worthwhile on small-batch decode):
//
//   d_Q[i] += to_fp32(d_bias_q[i])   i in [0, q_dim)
//   d_K[i] += to_fp32(d_bias_k[i])   i in [0, kv_dim)
//   d_V[i] += to_fp32(d_bias_v[i])   i in [0, kv_dim)
//
// Memory layout (matches engine buffers allocated in BlackwellEngine::Impl):
//   d_Q      : float[q_dim],  q_dim  = num_attention_heads * head_dim
//              (Qwen2.5-Coder-7B: 28 * 128 = 3584)
//   d_K, d_V : float[kv_dim], kv_dim = num_key_value_heads * head_dim
//              (Qwen2.5-Coder-7B:  4 * 128 = 512)
//   d_bias_* : BF16/FP16 arrays of the matching length, contiguous.
//   Head-major contiguous layout [head][dim], stride 1 — identical to what
//   launch_fused_rope_kv_kernel consumes immediately afterwards.
//
// Grid/block mapping (implementation requirement):
//   N_total = q_dim + 2 * kv_dim          (7B: 3584 + 1024 = 4608)
//   block   = 256 threads, 1-D
//   grid    = ceil(N_total / 256), 1-D
//   Flat thread index t maps:
//     t <  q_dim                      -> Q[t]              += bias_q[t]
//     q_dim <= t < q_dim + kv_dim     -> K[t - q_dim]      += bias_k[...]
//     q_dim + kv_dim <= t < N_total   -> V[t - q_dim - kv_dim] += bias_v[...]
//   Guard `if (t >= N_total) return;`.
//
// Edge cases:
//   All three bias pointers nullptr -> return without launching.
//   Individual nullptr bias is NOT supported (Qwen2 has all three or none);
//   implementations may assert this in debug builds.
// ----------------------------------------------------------------------------
void launch_fused_qkv_bias_kernel(float* d_Q,
                                  float* d_K,
                                  float* d_V,
                                  const void* d_bias_q,
                                  const void* d_bias_k,
                                  const void* d_bias_v,
                                  size_t q_dim,
                                  size_t kv_dim,
                                  BiasDType dtype = BiasDType::BF16);
