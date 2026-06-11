// Online-softmax decoding attention vs the CPU reference, at the GQA head
// configurations of both target models:
//   Llama 3 8B        : 32 query heads, 8 KV heads (ratio 4)
//   Qwen2.5-Coder-7B  : 28 query heads, 4 KV heads (ratio 7)
#include <gtest/gtest.h>
#include <vector>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/attention.cuh"

namespace {

class AttentionValidation : public test_utils::CudaTest {};

void expect_attention_matches_reference(size_t q_heads, size_t kv_heads,
                                        size_t head_dim, size_t max_seq_len,
                                        int pos, unsigned seed) {
    const std::vector<float> h_Q =
        test_utils::random_uniform(q_heads * head_dim, seed, -0.3f, 0.3f);
    const std::vector<float> h_K_cache =
        test_utils::random_uniform(kv_heads * max_seq_len * head_dim, seed + 1, -0.2f, 0.2f);
    const std::vector<float> h_V_cache =
        test_utils::random_uniform(kv_heads * max_seq_len * head_dim, seed + 2, -0.2f, 0.2f);

    std::vector<float> h_ref(q_heads * head_dim, 0.0f);
    cpu_attention_decoding(h_Q.data(), h_K_cache.data(), h_V_cache.data(),
                           h_ref.data(), pos, q_heads, kv_heads, head_dim,
                           max_seq_len);

    CudaVector<float> d_Q(h_Q.size());             d_Q.upload(h_Q);
    CudaVector<float> d_K_cache(h_K_cache.size()); d_K_cache.upload(h_K_cache);
    CudaVector<float> d_V_cache(h_V_cache.size()); d_V_cache.upload(h_V_cache);
    CudaVector<float> d_O(h_ref.size());
    test_utils::poison(d_O);

    launch_attention_decoding_kernel(d_Q, d_K_cache, d_V_cache, d_O,
                                     pos, q_heads, kv_heads, head_dim, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu(h_ref.size());
    d_O.download(h_gpu);
    test_utils::expect_allclose(h_ref, h_gpu, 1e-3f, "attention output");
}

}  // namespace

TEST_F(AttentionValidation, LlamaGqaMidContext) {
    expect_attention_matches_reference(/*q_heads=*/32, /*kv_heads=*/8,
                                       /*head_dim=*/128, /*max_seq_len=*/200,
                                       /*pos=*/85, /*seed=*/71);
}

TEST_F(AttentionValidation, QwenGqaMidContext) {
    expect_attention_matches_reference(/*q_heads=*/28, /*kv_heads=*/4,
                                       /*head_dim=*/128, /*max_seq_len=*/200,
                                       /*pos=*/85, /*seed=*/72);
}

// pos=0: softmax over a single cached token must reduce to copying V[0].
TEST_F(AttentionValidation, FirstTokenSingleEntryCache) {
    expect_attention_matches_reference(32, 8, 128, 64, /*pos=*/0, 73);
}
