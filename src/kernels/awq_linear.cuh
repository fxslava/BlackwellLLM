#pragma once
#include <cstddef>

// AWQ int4 GEMV: d_out[oc] = sum_k (unpack(qweight[k][oc]) - unpack(qzeros[k/g][oc])) * scales[k/g][oc] * d_in[k]
//
// Expected device-memory layouts (canonical AutoAWQ checkpoint format):
//   qweight : uint32[in_features][out_features/8], 8 x int4 per word,
//             nibble interleave {0,2,4,6,1,3,5,7} along out_features
//   qzeros  : uint32[ceil(in_features/group_size)][out_features/8], same interleave
//   scales  : half[ceil(in_features/group_size)][out_features]
//   d_in    : float[in_features], d_out : float[out_features]
// group_size <= 0 means one group spanning all of in_features.
// out_features must be a multiple of 8. Launch is asynchronous on the
// default stream.
void launch_awq_gemv_kernel(const void* qweight,
                            const void* scales,
                            const void* qzeros,
                            const float* d_in,
                            float* d_out,
                            size_t out_features,
                            size_t in_features,
                            int group_size);

// Residual variant: d_residual_accum[oc] += GEMV result. Used by o_proj /
// down_proj where the projection lands directly on the residual stream.
// The buffer is never cleared; partials are added atomically.
void launch_awq_gemv_residual_kernel(const void* qweight,
                                     const void* scales,
                                     const void* qzeros,
                                     const float* d_in,
                                     float* d_residual_accum,
                                     size_t out_features,
                                     size_t in_features,
                                     int group_size);

// ---------------------------------------------------------------------------
// Batched AWQ GEMM (num_tokens > 1 path): Y[T, out] = X[T, in] @ Wdeq^T, with
// Wdeq[k, oc] = (unpack(qweight[k][oc]) - unpack(qzeros[k/g][oc])) * scales[k/g][oc].
// Same packing/scale layout as the GEMV above. The int4 weights are unpacked and
// dequantized to FP32 in shared memory on the fly, then fed to Hardware Tensor
// Cores (nvcuda::wmma, TF32) exactly like batched_bf16_gemm -- keeping the FP32
// activation precision the batch=1 GEMV has. d_X is [num_tokens, in_features],
// d_Y is [num_tokens, out_features]. out_features must be a multiple of 8 (the
// packing granularity); K/T are arbitrary (tail tiles zero-padded).
void launch_batched_awq_gemm(const void* qweight,
                             const void* scales,
                             const void* qzeros,
                             const float* d_X,
                             float* d_Y,
                             size_t out_features,
                             size_t in_features,
                             int group_size,
                             size_t num_tokens);

// Residual variant: d_Y_accum[T, out] += batched GEMM result (o_proj / down_proj).
void launch_batched_awq_gemm_residual(const void* qweight,
                                      const void* scales,
                                      const void* qzeros,
                                      const float* d_X,
                                      float* d_Y_accum,
                                      size_t out_features,
                                      size_t in_features,
                                      int group_size,
                                      size_t num_tokens);
