// Stress tests for the non-quantized kernels: shapes that exercise 64-bit
// addressing and capacity limits rather than math (those live in
// validation/).
#include <gtest/gtest.h>
#include <vector>
#include <cuda_bf16.h>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/embedding.cuh"

class CoreOpsStress : public test_utils::CudaTest {};

// Llama-vocab BF16 embedding table (128256 x 4096 = 1.05 GB): row offsets
// overflow 32-bit indexing beyond token ~65535, so lookups near the end of
// the table prove the kernel computes offsets in 64 bits. Only the looked-up
// rows carry a pattern; the rest of the table stays zero to keep host-side
// preparation cheap.
TEST_F(CoreOpsStress, Bf16EmbeddingVocabSizedTable) {
    const size_t vocab_size = 128256;
    const size_t hidden_dim = 4096;
    const std::vector<int> tokens = {0, 42, 1024, 50000, 84042, 100500, 128000, 128255};
    const size_t seq_len = tokens.size();

    std::vector<__nv_bfloat16> h_table(vocab_size * hidden_dim, __float2bfloat16(0.0f));
    std::vector<float> ref(seq_len * hidden_dim, 0.0f);

    for (size_t s = 0; s < seq_len; ++s) {
        const int token_id = tokens[s];
        const size_t row_base = static_cast<size_t>(token_id) * hidden_dim;
        for (size_t h = 0; h < hidden_dim; ++h) {
            const float v = static_cast<float>(token_id % 100) +
                            static_cast<float>(h % 50) * 0.01f;
            const __nv_bfloat16 bf16_v = __float2bfloat16(v);
            h_table[row_base + h] = bf16_v;
            ref[s * hidden_dim + h] = __bfloat162float(bf16_v);
        }
    }

    CudaVector<__nv_bfloat16> d_table(h_table.size()); d_table.upload(h_table);
    CudaVector<int>           d_tokens(seq_len);       d_tokens.upload(tokens);
    CudaVector<float>         d_output(ref.size());
    test_utils::poison(d_output);

    launch_bf16_embedding_kernel(d_tokens, d_table, d_output, seq_len, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(ref.size());
    d_output.download(gpu);
    test_utils::expect_allclose(ref, gpu, 0.0f, "64-bit-offset embedding row");
}
