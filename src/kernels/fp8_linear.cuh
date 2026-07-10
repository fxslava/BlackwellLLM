#pragma once
#include <cstddef>
#include <cstdint>

// 🎯 Объявление ядра предварительного расчета динамического скейла токена
// d_token_scale: float[2] device buffer, receives {scale, 1/scale}.
void launch_quantize_per_token_kernel(const float* d_X, float* d_token_scale, size_t K);

// FP8 (E4M3) row-major GEMV.
//
// HARD LAUNCH CONTRACT (violations are NOT diagnosed at runtime):
//   * K must be a multiple of 16. The kernels read weights as uint4 and
//     activations as float4 (16-byte vector loads); the inner loop runs
//     K/16 steps, so a K % 16 tail would be SILENTLY DROPPED, and with
//     K % 16 != 0 odd rows of W (row stride = K bytes) would additionally
//     break the 16-byte alignment required by the uint4 loads.
//   * d_W_fp8 and d_X must be 16-byte aligned (any cudaMalloc'd base is).
//   * d_token_scale must point to a valid device buffer whenever
//     d_input_scale == nullptr: the kernel reads token_scale[0] in that
//     branch even in weight-only mode (where the value is later unused).
//   * weight_scales is indexed as weight_scales[row * scale_stride].
void launch_fp8_gemv_kernel(const void* d_W_fp8,
                            const float* d_X,
                            const void* d_weight_scales,
                            const void* d_input_scale,
                            const float* d_token_scale,
                            float* d_Y,
                            size_t M,
                            size_t K,
                            int scale_stride);

// Residual variant: d_Y_accum[row] += GEMV result. Same hard launch
// contract as launch_fp8_gemv_kernel above (K % 16 == 0, alignment,
// token_scale validity when input_scale == nullptr).
void launch_fp8_gemv_residual_kernel(const void* d_W_fp8,
                                     const float* d_X,
                                     const void* d_weight_scales,
                                     const void* d_input_scale,
                                     const float* d_token_scale,
                                     float* d_Y_accum,
                                     size_t M,
                                     size_t K,
                                     int scale_stride);

// ---------------------------------------------------------------------------
// Batched FP8 GEMM (num_tokens > 1 path): Y[T, M] = X[T, K] @ Wdeq^T, where
// Wdeq[oc, k] = unpack_e4m3(W_fp8[oc][k]) and the row-wise weight scale (plus the
// optional activation scale) are applied in the epilogue -- numerically identical
// to running launch_fp8_gemv per row. E4M3 weights are unpacked to FP32 in shared
// memory and fed to Hardware Tensor Cores (nvcuda::wmma, TF32), like the BF16 /
// AWQ batched kernels.
//
// Activation scaling matches the GEMV exactly and is applied UNIFORMLY across the
// batch: with d_input_scale != nullptr, every token's activations are quantized
// to E4M3 in x/input_scale domain (a per-tensor static scale, so each of the T
// rows uses the same, correct scale) and the epilogue multiplies input_scale
// back; with d_input_scale == nullptr the activations enter the dot product raw
// (weight-only). d_X is [num_tokens, K], d_Y is [num_tokens, M]. K/T/M arbitrary
// (tail tiles zero-padded).
void launch_batched_fp8_gemm(const void* d_W_fp8,
                             const float* d_X,
                             const void* d_weight_scales,
                             const void* d_input_scale,
                             const float* d_token_scale,
                             float* d_Y,
                             size_t M,
                             size_t K,
                             int scale_stride,
                             size_t num_tokens);

// Residual variant: d_Y_accum[T, M] += batched GEMM result (o_proj / down_proj).
void launch_batched_fp8_gemm_residual(const void* d_W_fp8,
                                      const float* d_X,
                                      const void* d_weight_scales,
                                      const void* d_input_scale,
                                      const float* d_token_scale,
                                      float* d_Y_accum,
                                      size_t M,
                                      size_t K,
                                      int scale_stride,
                                      size_t num_tokens);