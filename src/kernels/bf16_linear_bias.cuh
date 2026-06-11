#pragma once
#include <cstddef>

// ============================================================================
// Fused BF16 GEMV + bias epilogue (fast path for unquantized Qwen2 q/k/v).
//
// CONTRACT OVERVIEW
// -----------------
//   d_Y[row] = sum_{col<K} to_fp32(W[row][col]) * d_X[col] + to_fp32(B[row])
//
// This is the bias-fused sibling of launch_bf16_gemv_kernel (bf16_linear.cuh).
// It is selected by the dispatcher only when quant_strategy == NONE and the
// projection has a bias tensor; quantized backends (fp8/awq) instead use the
// standalone epilogue in bias.cuh. There is deliberately NO residual variant:
// in Qwen2 the only biased projections are q/k/v, and none of them accumulate
// into the residual stream (o_proj / down_proj are bias-free).
//
// NUMERICAL PARITY REQUIREMENT
// ----------------------------
// With d_bias == nullptr this kernel must be bit-exact with
// launch_bf16_gemv_kernel: same warp-per-row mapping, same fp32 accumulation
// order, same __shfl_down_sync cascade (16,8,4,2,1). The bias is decoded with
// __bfloat162float and added ONCE, in fp32, by lane 0 after the warp
// reduction — never folded into the per-lane partial sums and never rounded
// through bf16. Output stays pure fp32.
//
// MEMORY LAYOUT
// -------------
//   d_W_bf16 : __nv_bfloat16[M][K], row-major, row stride = K elements,
//              passed as opaque void* (VRAMArena convention).
//   d_X      : float[K], contiguous (single decode token).
//   d_bias   : __nv_bfloat16[M], contiguous; nullptr => plain GEMV.
//   d_Y      : float[M], contiguous, fully overwritten.
//
// GRID/BLOCK MAPPING (implementation requirement, mirrors bf16_linear.cu)
// -----------------------------------------------------------------------
//   block = 256 threads (8 warps), 1-D
//   grid  = ceil(M * 32 / 256), 1-D
//   row  = global_thread_id / 32   (one warp owns one output row)
//   lane = threadIdx.x % 32        (strided K-loop: col = lane, lane+32, ...)
//   Guard `if (row >= M) return;`. Lane 0 writes the final value.
//
// Typical shapes (Qwen2.5-Coder-7B-Instruct, hidden=3584, head_dim=128):
//   q_proj: M = 3584, K = 3584
//   k_proj: M =  512, K = 3584
//   v_proj: M =  512, K = 3584
//
// Launch is asynchronous on the default stream.
// ============================================================================
void launch_bf16_gemv_bias_kernel(const void* d_W_bf16,
                                  const float* d_X,
                                  const void* d_bias_bf16,
                                  float* d_Y,
                                  size_t M,
                                  size_t K);
