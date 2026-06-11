// RMSNorm kernels vs the CPU reference: the fused residual-add variant used
// between transformer sublayers and the direct variant used for the final
// pre-logits normalization. The kernels implement HF/BF16 parity semantics:
// weights are read as bf16 and the residual stream / outputs are snapped to
// the bf16 grid (see rmsnorm.cu), so the reference replicates the rounding
// exactly and the comparison can stay far below one bf16 ulp.
#include <gtest/gtest.h>
#include <vector>
#include <cuda_bf16.h>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/rmsnorm.cuh"

namespace {

class RmsnormValidation : public test_utils::CudaTest {};

constexpr float kEps = 1e-5f;

// The only inexact step the reference does not replicate bit-for-bit is the
// order of the double sum-of-squares reduction; 1e-4 sits far below one bf16
// ulp of the O(1) outputs while absorbing that difference.
constexpr float kTolerance = 1e-4f;

std::vector<__nv_bfloat16> bf16_weights(size_t n, unsigned seed) {
    const std::vector<float> w = test_utils::random_uniform(n, seed, 0.9f, 1.1f);
    std::vector<__nv_bfloat16> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __float2bfloat16(w[i]);
    return out;
}

}  // namespace

TEST_F(RmsnormValidation, FusedResidualAddNorm) {
    const size_t seq_len = 4;
    const size_t hidden_dim = 4096;

    const std::vector<float> h_x =
        test_utils::random_uniform(seq_len * hidden_dim, 91, -0.6f, 0.6f);
    const std::vector<float> h_residual =
        test_utils::random_uniform(seq_len * hidden_dim, 92, -0.5f, 1.5f);
    const std::vector<__nv_bfloat16> h_weight = bf16_weights(hidden_dim, 93);

    std::vector<float> ref_x = h_x;
    std::vector<float> ref_residual = h_residual;
    cpu_rmsnorm_residual(ref_x.data(), ref_residual.data(), h_weight.data(),
                         seq_len, hidden_dim, kEps);

    CudaVector<float> d_x(h_x.size());                  d_x.upload(h_x);
    CudaVector<float> d_residual(h_residual.size());    d_residual.upload(h_residual);
    CudaVector<__nv_bfloat16> d_weight(h_weight.size()); d_weight.upload(h_weight);

    launch_rmsnorm_residual_kernel(d_x, d_residual, d_weight, seq_len, hidden_dim, kEps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_x(h_x.size());               d_x.download(gpu_x);
    std::vector<float> gpu_residual(h_residual.size()); d_residual.download(gpu_residual);

    test_utils::expect_allclose(ref_residual, gpu_residual, kTolerance, "residual stream");
    test_utils::expect_allclose(ref_x, gpu_x, kTolerance, "normalized output");
}

TEST_F(RmsnormValidation, DirectNormPreservesInput) {
    const size_t seq_len = 1;
    const size_t hidden_dim = 4096;

    const std::vector<float> h_input =
        test_utils::random_uniform(hidden_dim, 94, -0.5f, 0.5f);
    const std::vector<__nv_bfloat16> h_weight = bf16_weights(hidden_dim, 95);

    std::vector<float> ref_output(hidden_dim, 0.0f);
    cpu_rmsnorm_direct(h_input.data(), ref_output.data(), h_weight.data(),
                       seq_len, hidden_dim, kEps);

    CudaVector<float> d_input(hidden_dim);              d_input.upload(h_input);
    CudaVector<__nv_bfloat16> d_weight(hidden_dim);     d_weight.upload(h_weight);
    CudaVector<float> d_output(hidden_dim);
    test_utils::poison(d_output);

    launch_rmsnorm_kernel(d_input, d_output, d_weight, seq_len, hidden_dim, kEps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_output(hidden_dim);
    d_output.download(gpu_output);
    test_utils::expect_allclose(ref_output, gpu_output, kTolerance, "normalized output");

    // The direct variant reads the input; it must never write it.
    std::vector<float> input_after(hidden_dim);
    d_input.download(input_after);
    ASSERT_EQ(h_input, input_after) << "kernel overwrote its input buffer";
}
