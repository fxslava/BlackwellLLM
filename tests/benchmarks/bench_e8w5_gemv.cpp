// E8W5 (5-bit companded-E8 lattice) GEMV bandwidth benchmark on GLM-4-9B decode shapes,
// with the BF16 GEMV over the same projection as the reference point. Decode GEMV is
// memory-bound, so %peak bandwidth is the number that matters -- and the whole case for a
// 5-bit weight format at batch=1 is that it reads 3.1x fewer bytes than BF16, so the pair of
// rows is the measurement, not either row alone.
//
// Buffers are filled with non-zero byte patterns directly on the device. The pattern does
// not have to be a valid E8 point: the decode is branch-free and its cost is independent of
// the coordinate values, so timing is unaffected while correctness lives in
// validation/test_e8w5_linear.cpp. It does matter that the bytes are non-zero, which keeps
// the run honest against any zero-skip shortcut.
#include <gtest/gtest.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common/benchmark_utils.h"
#include "kernels/bf16_linear.cuh"
#include "kernels/e8w5_linear.cuh"

namespace {

constexpr int kDim = 8;
constexpr int kGroup = 128;

class E8W5Benchmark : public ::testing::Test {};

// Bytes the kernel must touch per call: both planes, the group scales, the codebook, x, y.
double e8w5_bytes(int in_features, int out_features)
{
    const int n_blocks = in_features / kDim;
    const int n_groups = in_features / kGroup;
    return (double)out_features * n_blocks * sizeof(uint32_t)   // plane L
           + (double)out_features * n_blocks * sizeof(uint8_t)  // plane H
           + (double)out_features * n_groups * sizeof(half)     // scales
           + (double)64 * sizeof(half)                          // codebook
           + (double)in_features * sizeof(float)                // x
           + (double)out_features * sizeof(float);              // y
}

void bench_pair(bench_utils::Reporter& reporter, const std::string& name,
                int in_features, int out_features)
{
    const int n_blocks = in_features / kDim;
    const int n_groups = in_features / kGroup;

    CudaVector<uint32_t> plane_lo((size_t)out_features * n_blocks);
    CudaVector<uint8_t> plane_hi((size_t)out_features * n_blocks);
    CudaVector<half> scales((size_t)out_features * n_groups);
    CudaVector<half> codebook(64);
    CudaVector<float> x(in_features);
    CudaVector<float> y(out_features);

    CUDA_CHECK(cudaMemset(plane_lo.d_ptr, 0x55, plane_lo.num_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(plane_hi.d_ptr, 0x27, plane_hi.num_elements * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(scales.d_ptr, 0x35, scales.num_elements * sizeof(half)));
    CUDA_CHECK(cudaMemset(codebook.d_ptr, 0x35, codebook.num_elements * sizeof(half)));
    CUDA_CHECK(cudaMemset(x.d_ptr, 0x3E, x.num_elements * sizeof(float)));

    const double flops = 2.0 * in_features * out_features;

    reporter.row(name + " e8w5", e8w5_bytes(in_features, out_features), flops, [&] {
        launch_e8w5_gemv_kernel(plane_lo, plane_hi, scales, codebook, x, y,
                                out_features, in_features);
    });

    // Same projection in BF16, the format E8W5 replaces. Weight bytes dominate both rows,
    // so the time ratio is essentially the byte ratio if both kernels reach peak.
    CudaVector<__nv_bfloat16> w_bf16((size_t)out_features * in_features);
    CUDA_CHECK(cudaMemset(w_bf16.d_ptr, 0x3E,
                          w_bf16.num_elements * sizeof(__nv_bfloat16)));
    const double bf16_bytes = (double)out_features * in_features * sizeof(__nv_bfloat16)
                              + (double)in_features * sizeof(float)
                              + (double)out_features * sizeof(float);
    reporter.row(name + " bf16", bf16_bytes, flops, [&] {
        launch_bf16_gemv_kernel(w_bf16, x, y, out_features, in_features);
    });
}

}  // namespace

// GLM-4-9B: hidden 4096, GQA KV dim 2 heads * 128 = 256, MLP intermediate 13696 (gate_up
// fused to 2x). Every projection the engine launches per decoded token.
TEST_F(E8W5Benchmark, Glm4DecodeShapes)
{
    bench_utils::Reporter reporter("E8W5 vs BF16 GEMV (GLM-4-9B decode shapes)");
    bench_pair(reporter, "q_proj 4096x4096",       4096, 4096);
    bench_pair(reporter, "kv_proj 4096x256",       4096, 256);
    bench_pair(reporter, "o_proj 4096x4096",       4096, 4096);
    bench_pair(reporter, "gate_up 4096x27392",     4096, 27392);
    bench_pair(reporter, "down 13696x4096",        13696, 4096);
}

// The mission's parity shapes, so the benchmark and the correctness test cover the same K.
TEST_F(E8W5Benchmark, MissionShapes)
{
    bench_utils::Reporter reporter("E8W5 vs BF16 GEMV (K in {4096, 11008, 14336})");
    bench_pair(reporter, "K=4096  x4096",  4096, 4096);
    bench_pair(reporter, "K=11008 x4096",  11008, 4096);
    bench_pair(reporter, "K=14336 x4096",  14336, 4096);
}
