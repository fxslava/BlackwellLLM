// Embedding-lookup kernels (FP32 and BF16 tables) vs the CPU reference.
// The vocab-sized 64-bit-offset stress variant lives in
// benchmarks/stress_core_ops.cpp.
#include <gtest/gtest.h>
#include <vector>
#include <cuda_bf16.h>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/embedding.cuh"

namespace {

class EmbeddingValidation : public test_utils::CudaTest {};

constexpr size_t kVocabSize = 32000;
constexpr size_t kHiddenDim = 4096;

// Token list deliberately covers both table boundaries.
const std::vector<int> kTokens = {12, 450, 0, 31999, 5, 88, 42, 10};

}  // namespace

TEST_F(EmbeddingValidation, Fp32TableLookup) {
    const size_t seq_len = kTokens.size();
    const std::vector<float> h_table =
        test_utils::random_uniform(kVocabSize * kHiddenDim, 111, -1.0f, 1.0f);

    std::vector<float> ref(seq_len * kHiddenDim, 0.0f);
    cpu_embedding_lookup(kTokens.data(), h_table.data(), ref.data(), seq_len, kHiddenDim);

    CudaVector<float> d_table(h_table.size()); d_table.upload(h_table);
    CudaVector<int>   d_tokens(seq_len);       d_tokens.upload(kTokens);
    CudaVector<float> d_output(ref.size());
    test_utils::poison(d_output);

    launch_embedding_kernel(d_tokens, d_table, d_output, seq_len, kHiddenDim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(ref.size());
    d_output.download(gpu);
    test_utils::expect_allclose(ref, gpu, 0.0f, "embedding row");
}

TEST_F(EmbeddingValidation, Bf16TableLookup) {
    const size_t seq_len = kTokens.size();
    const std::vector<float> h_values =
        test_utils::random_uniform(kVocabSize * kHiddenDim, 112, -1.0f, 1.0f);

    // The reference operates on the bf16-rounded values, so the only thing
    // under test is the kernel's unpack + copy.
    std::vector<__nv_bfloat16> h_bf16_table(h_values.size());
    std::vector<float> h_rounded_table(h_values.size());
    for (size_t i = 0; i < h_values.size(); ++i) {
        h_bf16_table[i] = __float2bfloat16(h_values[i]);
        h_rounded_table[i] = __bfloat162float(h_bf16_table[i]);
    }

    std::vector<float> ref(seq_len * kHiddenDim, 0.0f);
    cpu_embedding_lookup(kTokens.data(), h_rounded_table.data(), ref.data(),
                         seq_len, kHiddenDim);

    CudaVector<__nv_bfloat16> d_table(h_bf16_table.size()); d_table.upload(h_bf16_table);
    CudaVector<int>           d_tokens(seq_len);            d_tokens.upload(kTokens);
    CudaVector<float>         d_output(ref.size());
    test_utils::poison(d_output);

    launch_bf16_embedding_kernel(d_tokens, d_table, d_output, seq_len, kHiddenDim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(ref.size());
    d_output.download(gpu);
    test_utils::expect_allclose(ref, gpu, 0.0f, "bf16 embedding row");
}
