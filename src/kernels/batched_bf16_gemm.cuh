#pragma once
#include <cstddef>

// ============================================================================
// Batched BF16 linear projection (Tensor-Core GEMM) — the num_tokens > 1 path.
// ============================================================================
// Companion to the batch=1 GEMV in bf16_linear.cuh: computes, for a tile of
// `num_tokens` activation rows at once,
//
//     Y[t, :] = W @ X[t, :]        (t in [0, num_tokens))
//
// i.e. Y[T, M] = X[T, K] @ W^T with W the row-major [M, K] bf16 weight, exactly
// the GEMV contract broadcast across T rows. The reduction stays in FP32; the
// contraction runs on Hardware Tensor Cores via nvcuda::wmma in TF32 mode, so
// the FP32 activations keep ~10 mantissa bits into the multiply (more faithful
// to the batch=1 FP32 GEMV than a bf16-operand path would be) while still
// paying the Tensor-Core throughput.
//
// LAYOUTS (all row-major, device pointers):
//   d_W_bf16 : [M, K]  __nv_bfloat16   (same weight tensor the GEMV consumes)
//   d_X      : [T, K]  float           (T activation rows)
//   d_Y      : [T, M]  float           (T output rows)
//
// The batch=1 GEMV kernels are latency-critical and deliberately untouched; the
// LinearDispatcher routes here only when num_tokens > 1 (batched prefill / true
// batch mode). No launch-time shape constraints: M, K, T are arbitrary (tail
// tiles are zero-padded in shared memory).
void launch_bf16_gemm_batched(const void* d_W_bf16,
                              const float* d_X,
                              float* d_Y,
                              size_t M,
                              size_t K,
                              size_t num_tokens);

// Residual-accumulating variant (the o_proj / down_proj path): d_Y_accum holds
// [T, M] and the kernel adds W @ X[t] on top of each row in place (+=), never
// overwriting it — mirroring launch_bf16_gemv_residual_kernel for T rows.
void launch_bf16_gemm_residual_batched(const void* d_W_bf16,
                                       const float* d_X,
                                       float* d_Y_accum,
                                       size_t M,
                                       size_t K,
                                       size_t num_tokens);
