// AWQ int4 GEMV bandwidth benchmark on the Qwen2.5-Coder-7B decode shapes
// (every projection the engine launches per token, plus the lm_head and the
// split-K narrow case). Decode GEMV is memory-bound, so %peak bandwidth is
// the number that matters.
//
// Buffers are filled with non-zero byte patterns directly on the device:
// host-side generation would dominate runtime at these sizes, and non-zero
// data keeps the run honest against any zero-skip shortcuts.
#include <gtest/gtest.h>
#include <cstdint>
#include <cuda_fp16.h>

#include "common/benchmark_utils.h"
#include "kernels/awq_linear.cuh"

namespace {

class AwqBenchmark : public ::testing::Test {};

void bench_shape(bench_utils::Reporter& reporter, const std::string& name,
                 int in_features, int out_features, int group_size) {
    const int gs = group_size > 0 ? group_size : in_features;
    const int num_groups  = (in_features + gs - 1) / gs;
    const int packed_cols = out_features / 8;

    CudaVector<uint32_t> qweight((size_t)in_features * packed_cols);
    CudaVector<uint32_t> qzeros((size_t)num_groups * packed_cols);
    CudaVector<half>     scales((size_t)num_groups * out_features);
    CudaVector<float>    x(in_features);
    CudaVector<float>    y(out_features);

    CUDA_CHECK(cudaMemset(qweight.d_ptr, 0x55, qweight.num_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(qzeros.d_ptr, 0x44, qzeros.num_elements * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(scales.d_ptr, 0x35, scales.num_elements * sizeof(half)));
    CUDA_CHECK(cudaMemset(x.d_ptr, 0x3E, x.num_elements * sizeof(float)));

    const double bytes = (double)in_features * packed_cols * sizeof(uint32_t) +
                         (double)num_groups * packed_cols * sizeof(uint32_t) +
                         (double)num_groups * out_features * sizeof(half) +
                         (double)in_features * sizeof(float) +
                         (double)out_features * sizeof(float);
    const double flops = 2.0 * in_features * out_features;

    reporter.row(name, bytes, flops, [&] {
        launch_awq_gemv_kernel(qweight, scales, qzeros, x, y,
                               out_features, in_features, group_size);
    });
}

}  // namespace

TEST_F(AwqBenchmark, QwenDecodeShapes) {
    bench_utils::Reporter reporter("AWQ int4 GEMV (Qwen2.5-Coder-7B decode shapes)");
    bench_shape(reporter, "q_proj 3584x3584",      3584, 3584,   128);
    bench_shape(reporter, "kv_proj 3584x512",      3584, 512,    128);
    bench_shape(reporter, "gate/up 3584x18944",    3584, 18944,  128);
    bench_shape(reporter, "down 18944x3584",       18944, 3584,  128);
    bench_shape(reporter, "lm_head 3584x152064",   3584, 152064, 128);
    bench_shape(reporter, "split-K narrow 8192x8", 8192, 8,      128);
}
