#pragma once
#include <cstddef>

void launch_fused_swiglu_kernel(const float* d_gate, 
                                const float* d_up, 
                                float* d_output, 
                                size_t num_elements);