#pragma once
#include <cstddef>

void launch_awq_gemv_kernel(const void* qweight,
                            const void* scales,
                            const void* qzeros,
                            const float* d_in,
                            float* d_out,
                            size_t out_features,
                            size_t in_features,
                            int group_size);
