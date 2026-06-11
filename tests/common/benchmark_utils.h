#pragma once
// CUDA-event benchmark harness shared by every performance suite.
//
// Decode-time GEMV kernels are memory-bound, so each benchmark reports
// effective bandwidth (bytes the kernel must touch per call / measured time)
// against the device's theoretical peak; TFLOP/s is derived from the same
// timing for reference. Results print as a markdown-style table and are
// attached to the GTest XML via RecordProperty for CI tracking.
#include <gtest/gtest.h>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>

#include "common.h"  // CudaVector / CUDA_CHECK (src/common.h)

namespace bench_utils {

constexpr int kWarmupIters = 20;
constexpr int kTimedIters  = 200;

// Theoretical peak DRAM bandwidth derived from device attributes (the memory
// clock attribute is in kHz; GDDR transfers twice per clock). Returns 0 when
// the runtime cannot report the clocks, in which case %peak reads as 0.
inline double device_peak_bandwidth_gbs() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0.0;
    int clock_khz = 0, bus_width_bits = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev) != cudaSuccess)
        return 0.0;
    if (clock_khz <= 0 || bus_width_bits <= 0) return 0.0;
    return 2.0 * (double)clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;
}

// Average per-launch time in microseconds: warmup, then one event pair
// around the whole timed loop (per-launch events would serialize the queue).
template <typename LaunchFn>
inline double time_kernel_us(LaunchFn&& launch,
                             int warmup = kWarmupIters,
                             int iters = kTimedIters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (double)total_ms * 1e3 / iters;
}

// Table-formatted reporter. One instance per benchmark test; each row() call
// times a kernel, prints a line and records the numbers as test properties.
class Reporter {
public:
    explicit Reporter(const char* title) : peak_gbs_(device_peak_bandwidth_gbs()) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("\n%s\nDevice: %s (%d SMs), theoretical peak: %.0f GB/s\n\n",
               title, prop.name, prop.multiProcessorCount, peak_gbs_);
        printf("| %-28s | %9s | %7s | %6s | %7s |\n",
               "shape", "time(us)", "GB/s", "%peak", "TFLOP/s");
        printf("|------------------------------|-----------|---------|--------|---------|\n");
    }

    // bytes: data the kernel must touch per call; flops: 2*MAC count (0 to omit).
    template <typename LaunchFn>
    void row(const std::string& name, double bytes, double flops, LaunchFn&& launch) {
        const double us     = time_kernel_us(launch);
        const double gbs    = bytes / (us * 1e-6) / 1e9;
        const double tflops = flops / (us * 1e-6) / 1e12;
        const double pct    = peak_gbs_ > 0.0 ? 100.0 * gbs / peak_gbs_ : 0.0;

        printf("| %-28s | %9.2f | %7.1f | %5.1f%% | %7.2f |\n",
               name.c_str(), us, gbs, pct, tflops);

        ::testing::Test::RecordProperty(name + "_us", std::to_string(us));
        ::testing::Test::RecordProperty(name + "_gbs", std::to_string(gbs));
    }

    ~Reporter() { printf("\n"); }

private:
    double peak_gbs_;
};

}  // namespace bench_utils
