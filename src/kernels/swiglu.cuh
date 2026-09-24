#pragma once
#include <cstddef>

void launch_fused_swiglu_kernel(const float* d_gate,
                                const float* d_up,
                                float* d_output,
                                size_t num_elements);

// GLM-4 fused MLP in-projection: d_gate_up is [num_tokens, 2*intermediate_dim]
// row-major, each row laid out [gate(intermediate_dim) | up(intermediate_dim)]
// (HF Glm4MLP: `gate, up = gate_up_proj(x).chunk(2, dim=-1)`). Output is
// [num_tokens, intermediate_dim]. Bit-identical to the contiguous launcher above
// at num_tokens == 1, where gate and up ARE two adjacent contiguous halves.
void launch_fused_swiglu_gate_up_kernel(const float* d_gate_up,
                                        float* d_output,
                                        size_t num_tokens,
                                        size_t intermediate_dim);