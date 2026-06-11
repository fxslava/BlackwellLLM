#include "awq_linear.cuh"
#include <cuda_runtime.h>

void launch_awq_gemv_kernel(const void* qweight,
                            const void* scales,
                            const void* qzeros,
                            const float* d_in,
                            float* d_out,
                            size_t out_features,
                            size_t in_features,
                            int group_size)
{
    // Stub: AWQ 4-bit GEMV kernel not yet implemented.
    cudaDeviceSynchronize();
}
