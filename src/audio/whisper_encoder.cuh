#pragma once
// -----------------------------------------------------------------------------
// WhisperEncoder CUDA kernels — the element-wise / reduction glue. FP32.
//
// The heavy matmuls (QKV/out projections, FC1/FC2, and the im2col conv lowering)
// run on cuBLAS FP16 tensor cores; the attention is a fused flash kernel — both
// live entirely inside whisper_encoder.cu (file-local launchers). Exposed here:
// the LayerNorm, positional add, transpose and the fused bias / bias+GELU /
// bias+residual epilogues.
//
// HARD LAUNCH CONTRACTS (not diagnosed at runtime):
//   * Launchers perform NO device allocation; the caller owns every pointer.
//   * Every launcher takes the target stream (the whole pass is CUDA-graph-
//     captured on one stream — a launch on the default stream would break it).
//   * Row-major everywhere; [rows, cols] is the logical [T, C] shape.
// -----------------------------------------------------------------------------
#include <cuda_runtime.h>

namespace blackwell::audio {

// Transpose [rows, cols] -> [cols, rows] (permute conv [C,L] -> [L,C]).
void launch_transpose(const float* in, float* out, int rows, int cols,
                      cudaStream_t stream);

// out[t, c] = in[t, c] + pos[t, c].
void launch_add_positional(const float* in, const float* pos, float* out,
                           int rows, int cols, cudaStream_t stream);

// LayerNorm over the last dim (biased variance, nn.LayerNorm semantics):
//   y = (x - mean) / sqrt(var + eps) * gamma + beta.  One block per row.
void launch_layernorm(const float* x, const float* gamma, const float* beta,
                      float* y, int rows, int dim, float eps, cudaStream_t stream);

// --- Fused epilogues (post-cuBLAS-GEMM), all in [rows, cols] = [T, C] --------
// In-place bias add:            x[t, c] += bias[c].              (Q/V projections)
void launch_add_bias(float* x, const float* bias, int rows, int cols,
                     cudaStream_t stream);

// In-place fused bias + GELU:   x[t, c] = gelu(x[t, c] + bias[c]).  (FC1 output)
void launch_bias_gelu(float* x, const float* bias, int rows, int cols,
                      cudaStream_t stream);

// Fused bias + residual:        io[t, c] = io[t, c] + proj[t, c] + bias[c].
// `io` is the residual stream (read + written in place).  (out_proj / FC2 epilogue)
void launch_bias_residual(float* io, const float* proj, const float* bias,
                          int rows, int cols, cudaStream_t stream);

}  // namespace blackwell::audio
