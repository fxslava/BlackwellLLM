#pragma once
#include <cstddef>

void launch_rmsnorm_residual_kernel(float* d_x, 
                                    float* d_residual, 
                                    const void* d_weight, 
                                    size_t seq_len, 
                                    size_t hidden_dim, 
                                    float eps = 1e-5f);

// add_unit_offset: when true, multiply by (1 + weight) in fp32 and latch once
// (Qwen3.5 zero-centered RMSNorm). When false, legacy Llama/Qwen2.5 semantics.
void launch_rmsnorm_kernel(const float* d_input,
                           float* d_output,
                           const void* d_weight,
                           size_t seq_len,
                           size_t hidden_dim,
                           float eps = 1e-5f,
                           bool add_unit_offset = false);

// GLM-4-0414 sandwich norms: normalize the sub-layer output and ACCUMULATE it
// into the residual stream (d_accum += norm(d_input) * w, fp32 add, exactly like
// the GEMV kernels' residual epilogue). Needed because those layers normalize
// AFTER the projection, so the projection's own fused residual-accumulate would
// add the un-normalized output. d_input and d_accum must not alias.
void launch_rmsnorm_accum_kernel(const float* d_input,
                                 float* d_accum,
                                 const void* d_weight,
                                 size_t seq_len,
                                 size_t hidden_dim,
                                 float eps = 1e-5f,
                                 bool add_unit_offset = false);

// FP16-weight sibling of launch_rmsnorm_accum_kernel (AWQ/GPTQ checkpoints).
void launch_rmsnorm_accum_fp16_kernel(const float* d_input,
                                      float* d_accum,
                                      const void* d_weight,
                                      size_t seq_len,
                                      size_t hidden_dim,
                                      float eps = 1e-5f,
                                      bool add_unit_offset = false);

// Same normalization with FP16 layernorm weights (AWQ/GPTQ checkpoints);
// the normalized input and the output are latched to the FP16 grid to mirror
// HF's hidden_states.to(input_dtype) semantics for float16 models.
void launch_rmsnorm_fp16_kernel(const float* d_input,
                                float* d_output,
                                const void* d_weight,
                                size_t seq_len,
                                size_t hidden_dim,
                                float eps = 1e-5f,
                                bool add_unit_offset = false);