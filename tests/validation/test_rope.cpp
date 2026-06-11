// Fused RoPE + KV-cache append vs the CPU reference (Hugging Face
// rotate_half convention, see rope.cu): Q rotated in place, rotated K and
// untouched V written into the position's cache slots, and the rest of the
// cache left intact.
#include <gtest/gtest.h>
#include <vector>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/rope.cuh"

namespace {

class RopeValidation : public test_utils::CudaTest {};

void expect_rope_matches_reference(size_t q_heads, size_t kv_heads,
                                   size_t head_dim, size_t max_seq_len,
                                   int pos, float theta, unsigned seed) {
    const std::vector<float> h_Q =
        test_utils::random_uniform(q_heads * head_dim, seed, -1.0f, 1.0f);
    const std::vector<float> h_K =
        test_utils::random_uniform(kv_heads * head_dim, seed + 1, -1.0f, 1.0f);
    const std::vector<float> h_V =
        test_utils::random_uniform(kv_heads * head_dim, seed + 2, -1.0f, 1.0f);
    std::vector<float> h_K_cache(kv_heads * max_seq_len * head_dim, 0.0f);
    std::vector<float> h_V_cache(kv_heads * max_seq_len * head_dim, 0.0f);

    std::vector<float> ref_Q = h_Q;
    std::vector<float> ref_K = h_K;
    std::vector<float> ref_K_cache = h_K_cache;
    std::vector<float> ref_V_cache = h_V_cache;
    cpu_fused_rope_kv_append(ref_Q.data(), ref_K.data(), h_V.data(),
                             ref_K_cache.data(), ref_V_cache.data(),
                             pos, q_heads, kv_heads, head_dim, max_seq_len, theta);

    CudaVector<float> d_Q(h_Q.size());             d_Q.upload(h_Q);
    CudaVector<float> d_K(h_K.size());             d_K.upload(h_K);
    CudaVector<float> d_V(h_V.size());             d_V.upload(h_V);
    CudaVector<float> d_K_cache(h_K_cache.size()); d_K_cache.upload(h_K_cache);
    CudaVector<float> d_V_cache(h_V_cache.size()); d_V_cache.upload(h_V_cache);

    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, d_K_cache, d_V_cache,
                                pos, q_heads, kv_heads, head_dim, max_seq_len, theta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_Q(h_Q.size());
    std::vector<float> gpu_K_cache(h_K_cache.size());
    std::vector<float> gpu_V_cache(h_V_cache.size());
    d_Q.download(gpu_Q);
    d_K_cache.download(gpu_K_cache);
    d_V_cache.download(gpu_V_cache);

    // 5e-4: the device computes theta^(-2k/d) with fast-math powf, whose
    // relative error propagates into the rotation angle (up to pos * freq).
    test_utils::expect_allclose(ref_Q, gpu_Q, 5e-4f, "rotated Q");
    // The full-cache comparison also proves the kernel wrote only the slot at
    // `pos`: every other entry must still be the zero it started as.
    test_utils::expect_allclose(ref_K_cache, gpu_K_cache, 5e-4f, "K cache");
    test_utils::expect_allclose(ref_V_cache, gpu_V_cache, 5e-4f, "V cache");
}

}  // namespace

TEST_F(RopeValidation, LlamaHeadsMidPosition) {
    expect_rope_matches_reference(/*q_heads=*/32, /*kv_heads=*/8, /*head_dim=*/128,
                                  /*max_seq_len=*/100, /*pos=*/42,
                                  /*theta=*/500000.0f, /*seed=*/81);
}

TEST_F(RopeValidation, QwenHeadsMidPosition) {
    // Qwen2.5 uses rope_theta=1e6 and a 28/4 GQA head split.
    expect_rope_matches_reference(28, 4, 128, 100, 42, 1000000.0f, 82);
}

// pos=0 means every rotation angle is zero: Q/K must pass through unchanged.
TEST_F(RopeValidation, PositionZeroIsIdentityRotation) {
    expect_rope_matches_reference(32, 8, 128, 16, 0, 500000.0f, 83);
}
