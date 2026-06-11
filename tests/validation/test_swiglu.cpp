// Fused SwiGLU activation vs the CPU reference at both target intermediate
// widths (Llama 14336, Qwen 18944). The kernel implements BF16-parity
// semantics (see swiglu.cu) which the reference replicates; the device's
// fast-math expf/__fdividef can still flip the final bf16 rounding on a few
// elements, so the comparison allows one bf16 ulp (2^-8 relative).
#include <gtest/gtest.h>
#include <vector>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/swiglu.cuh"

namespace {

class SwigluValidation : public test_utils::CudaTest {};

void expect_swiglu_matches_reference(size_t num_elements, unsigned seed) {
    // Wide input range so both saturated sigmoid tails are exercised.
    const std::vector<float> h_gate =
        test_utils::random_uniform(num_elements, seed, -3.0f, 3.0f);
    const std::vector<float> h_up =
        test_utils::random_uniform(num_elements, seed + 1, -2.0f, 2.0f);

    std::vector<float> ref(num_elements, 0.0f);
    cpu_fused_swiglu(h_gate.data(), h_up.data(), ref.data(), num_elements);

    CudaVector<float> d_gate(num_elements); d_gate.upload(h_gate);
    CudaVector<float> d_up(num_elements);   d_up.upload(h_up);
    CudaVector<float> d_output(num_elements);
    test_utils::poison(d_output);

    launch_fused_swiglu_kernel(d_gate, d_up, d_output, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(num_elements);
    d_output.download(gpu);
    const float max_rel = test_utils::max_rel_error(gpu, ref, /*denom_floor=*/1e-2f);
    EXPECT_LE(max_rel, 8e-3f) << "SwiGLU mismatch beyond one bf16 ulp, max_rel=" << max_rel;
}

}  // namespace

TEST_F(SwigluValidation, LlamaIntermediateWidth) {
    expect_swiglu_matches_reference(14336, 101);
}

TEST_F(SwigluValidation, QwenIntermediateWidth) {
    expect_swiglu_matches_reference(18944, 102);
}
