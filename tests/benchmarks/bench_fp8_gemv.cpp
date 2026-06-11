// FP8 E4M3 GEMV bandwidth benchmark on the Llama 3 8B decode shapes: every
// projection the engine launches per token, the quantized-activation and
// residual-accumulation variants, and the per-token dynamic scale kernel.
// Decode GEMV is memory-bound, so %peak bandwidth is the number that matters.
//
// Buffers are filled with non-zero byte patterns directly on the device:
// host-side generation would dominate runtime at these sizes, and non-zero
// data keeps the run honest against any zero-skip shortcuts.
#include <gtest/gtest.h>
#include <cstdint>
#include <cuda_bf16.h>

#include "common/benchmark_utils.h"
#include "kernels/fp8_linear.cuh"

namespace {

class Fp8Benchmark : public ::testing::Test {};

enum class Variant { kWeightOnly, kQuantizedAct, kResidual };

void bench_shape(bench_utils::Reporter& reporter, const std::string& name,
                 size_t M, size_t K, Variant variant) {
    CudaVector<uint8_t>       weights(M * K);
    CudaVector<__nv_bfloat16> w_scales(M);
    CudaVector<__nv_bfloat16> input_scale(1);
    CudaVector<float>         token_scale(2);
    CudaVector<float>         x(K);
    CudaVector<float>         y(M);

    CUDA_CHECK(cudaMemset(weights.d_ptr, 0x35, weights.num_elements));
    w_scales.upload(std::vector<__nv_bfloat16>(M, __float2bfloat16(0.01f)));
    input_scale.upload({__float2bfloat16(0.02f)});
    token_scale.upload({1.0f, 1.0f});
    CUDA_CHECK(cudaMemset(x.d_ptr, 0x3E, x.num_elements * sizeof(float)));

    const double bytes = (double)M * K * sizeof(uint8_t) +
                         (double)M * sizeof(__nv_bfloat16) +
                         (double)K * sizeof(float) +
                         (double)M * sizeof(float);
    const double flops = 2.0 * M * K;

    const __nv_bfloat16* in_scale =
        variant == Variant::kQuantizedAct ? input_scale.d_ptr : nullptr;

    reporter.row(name, bytes, flops, [&] {
        if (variant == Variant::kResidual) {
            launch_fp8_gemv_residual_kernel(weights, x, w_scales, in_scale,
                                            token_scale, y, M, K, 1);
        } else {
            launch_fp8_gemv_kernel(weights, x, w_scales, in_scale,
                                   token_scale, y, M, K, 1);
        }
    });
}

}  // namespace

TEST_F(Fp8Benchmark, LlamaDecodeShapes) {
    bench_utils::Reporter reporter("FP8 E4M3 GEMV (Llama 3 8B decode shapes, weight-only)");
    bench_shape(reporter, "q/o_proj 4096x4096",  4096, 4096,  Variant::kWeightOnly);
    bench_shape(reporter, "kv_proj 1024x4096",   1024, 4096,  Variant::kWeightOnly);
    bench_shape(reporter, "gate/up 14336x4096",  14336, 4096, Variant::kWeightOnly);
    bench_shape(reporter, "down 4096x14336",     4096, 14336, Variant::kWeightOnly);
}

TEST_F(Fp8Benchmark, QuantizedActivationVariant) {
    bench_utils::Reporter reporter("FP8 E4M3 GEMV (quantized-activation epilogue)");
    bench_shape(reporter, "q/o_proj 4096x4096",  4096, 4096,  Variant::kQuantizedAct);
    bench_shape(reporter, "gate/up 14336x4096",  14336, 4096, Variant::kQuantizedAct);
}

TEST_F(Fp8Benchmark, ResidualVariant) {
    bench_utils::Reporter reporter("FP8 E4M3 GEMV (residual accumulation)");
    bench_shape(reporter, "o_proj 4096x4096",    4096, 4096,  Variant::kResidual);
    bench_shape(reporter, "down 4096x14336",     4096, 14336, Variant::kResidual);
}

TEST_F(Fp8Benchmark, PerTokenScaleKernel) {
    bench_utils::Reporter reporter("FP8 per-token dynamic scale");
    const size_t K = 4096;
    CudaVector<float> x(K);
    CudaVector<float> scale(2);
    CUDA_CHECK(cudaMemset(x.d_ptr, 0x3E, K * sizeof(float)));

    reporter.row("max-reduce 4096", (double)K * sizeof(float), 0.0, [&] {
        launch_quantize_per_token_kernel(x, scale, K);
    });
}
