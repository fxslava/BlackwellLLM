#pragma once
#include <cstddef>

// ============================================================================
// BF16 GEMV with residual accumulation (unquantized o_proj / down_proj path).
//
// CONTRACT OVERVIEW
// -----------------
//   d_Y_accum[row] += sum_{col<K} to_fp32(W[row][col]) * d_X[col]
//
// This is the residual sibling of launch_bf16_gemv_kernel (bf16_linear.cuh),
// the unquantized counterpart of launch_fp8_gemv_residual_kernel. It is
// selected by the dispatcher only when quant_strategy == NONE and the caller
// passes a residual accumulator (o_proj and down_proj add their output to the
// residual stream in-place). There is deliberately NO bias variant: in Qwen2
// the residual-accumulating projections are bias-free (the biased q/k/v
// projections use bf16_linear_bias.cuh and never touch the residual stream).
//
// NUMERICAL PARITY REQUIREMENT
// ----------------------------
// The dot product must be bit-exact with launch_bf16_gemv_kernel: same
// warp-per-row mapping, same fp32 accumulation order, same __shfl_down_sync
// cascade (16,8,4,2,1). The accumulation into d_Y_accum happens ONCE, in
// fp32, by lane 0 after the warp reduction — the residual stream is never
// rounded through bf16.
//
// MEMORY LAYOUT
// -------------
//   d_W_bf16  : __nv_bfloat16[M][K], row-major, row stride = K elements,
//               passed as opaque void* (VRAMArena convention).
//   d_X       : float[K], contiguous (single decode token).
//   d_Y_accum : float[M], contiguous, read-modify-write (+=), never zeroed.
//
// GRID/BLOCK MAPPING (implementation requirement, mirrors bf16_linear.cu)
// -----------------------------------------------------------------------
//   block = 256 threads (8 warps), 1-D
//   grid  = ceil(M * 32 / 256), 1-D
//   row  = global_thread_id / 32   (one warp owns one output row)
//   lane = threadIdx.x % 32        (strided K-loop: col = lane, lane+32, ...)
//   Guard `if (row >= M) return;`. Lane 0 performs the += of the final value.
//
// Typical shapes (Qwen2.5-Coder-7B, hidden=3584, intermediate=18944):
//   o_proj:    M = 3584, K =  3584
//   down_proj: M = 3584, K = 18944
//
// Launch is asynchronous on the default stream.
// ============================================================================
void launch_bf16_gemv_residual_kernel(const void* d_W_bf16,
                                      const float* d_X,
                                      float* d_Y_accum,
                                      size_t M,
                                      size_t K);
