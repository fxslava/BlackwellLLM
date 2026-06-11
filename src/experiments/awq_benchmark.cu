// AWQ int4 GEMV bandwidth/throughput benchmark.
//
// Effective bandwidth counts the bytes the kernel must touch per call
// (packed weights + zeros + fp16 scales + activations + output) and is
// reported against the RTX 5070 theoretical peak: 192-bit GDDR7 @ 28 Gbps
// = 672 GB/s. GEMV at decode time is memory-bound, so %peak is the number
// that matters; FLOP/s is derived from the same timing for reference.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.h"
#include "kernels/awq_linear.cuh"

namespace {

constexpr double kPeakBandwidthGBs = 672.0;  // RTX 5070, GDDR7 192-bit @ 28 Gbps
constexpr int kWarmupIters = 20;
constexpr int kTimedIters  = 200;

void bench_shape(int in_features, int out_features, int group_size) {
    const int gs = group_size > 0 ? group_size : in_features;
    const int num_groups  = (in_features + gs - 1) / gs;
    const int packed_cols = out_features / 8;

    CudaVector<uint32_t> qweight((size_t)in_features * packed_cols);
    CudaVector<uint32_t> qzeros((size_t)num_groups * packed_cols);
    CudaVector<half>     scales((size_t)num_groups * out_features);
    CudaVector<float>    x(in_features);
    CudaVector<float>    y(out_features);

    // Non-zero fill: keeps the run honest against any zero-skip shortcuts.
    CUDA_CHECK(cudaMemset(qweight.d_ptr, 0x55, qweight.num_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(qzeros.d_ptr, 0x44, qzeros.num_elements * sizeof(uint32_t)));

    for (int i = 0; i < kWarmupIters; ++i)
        launch_awq_gemv_kernel(qweight, scales, qzeros, x, y,
                               out_features, in_features, group_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kTimedIters; ++i)
        launch_awq_gemv_kernel(qweight, scales, qzeros, x, y,
                               out_features, in_features, group_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double us = (double)total_ms * 1e3 / kTimedIters;

    const double bytes = (double)in_features * packed_cols * sizeof(uint32_t) +
                         (double)num_groups * packed_cols * sizeof(uint32_t) +
                         (double)num_groups * out_features * sizeof(half) +
                         (double)in_features * sizeof(float) +
                         (double)out_features * sizeof(float);
    const double gbs    = bytes / (us * 1e-6) / 1e9;
    const double gflops = 2.0 * in_features * out_features / (us * 1e-6) / 1e12;

    printf("| %6d | %6d | %5d | %9.2f | %7.1f | %5.1f%% | %7.2f |\n",
           in_features, out_features, group_size, us, gbs,
           100.0 * gbs / kPeakBandwidthGBs, gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

}  // namespace

int main() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("AWQ int4 GEMV benchmark\n");
    printf("Device: %s (%d SMs), reference peak: %.0f GB/s\n\n",
           prop.name, prop.multiProcessorCount, kPeakBandwidthGBs);
    printf("| %6s | %6s | %5s | %9s | %7s | %6s | %7s |\n",
           "IC", "OC", "group", "time(us)", "GB/s", "%peak", "TFLOP/s");
    printf("|--------|--------|-------|-----------|---------|--------|---------|\n");

    // Llama-8B decode shapes: qkv/o projections, MLP up/down, lm_head,
    // plus a narrow layer that exercises the split-K path.
    bench_shape(4096, 4096, 128);
    bench_shape(4096, 6144, 128);
    bench_shape(4096, 14336, 128);
    bench_shape(14336, 4096, 128);
    bench_shape(4096, 128256, 128);
    bench_shape(4096, 1024, 128);

    return 0;
}
