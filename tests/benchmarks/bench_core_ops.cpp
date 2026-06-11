// Bandwidth benchmarks for the non-quantized kernels in the decode hot path:
// BF16 logits projection (lm_head), attention decoding at a long context,
// fused RoPE + KV append, RMSNorm, SwiGLU and argmax sampling. Together with
// the AWQ/FP8 GEMV benchmarks this covers every kernel launched per token.
#include <gtest/gtest.h>
#include <vector>
#include <cuda_bf16.h>

#include "common/benchmark_utils.h"
#include "kernels/attention.cuh"
#include "kernels/bf16_linear.cuh"
#include "kernels/embedding.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/rope.cuh"
#include "kernels/sampling.cuh"
#include "kernels/swiglu.cuh"

namespace {

class CoreOpsBenchmark : public ::testing::Test {};

constexpr size_t kHidden = 4096;        // Llama 3 8B hidden dim
constexpr size_t kIntermediate = 14336; // Llama 3 8B MLP width
constexpr size_t kVocab = 128256;       // Llama 3 vocab
constexpr size_t kQHeads = 32;
constexpr size_t kKvHeads = 8;
constexpr size_t kHeadDim = 128;

}  // namespace

TEST_F(CoreOpsBenchmark, Bf16LogitsProjection) {
    bench_utils::Reporter reporter("BF16 GEMV (Llama lm_head 4096 -> 128256)");
    CudaVector<__nv_bfloat16> w(kVocab * kHidden);
    CudaVector<float> x(kHidden);
    CudaVector<float> y(kVocab);
    CUDA_CHECK(cudaMemset(w.d_ptr, 0x3D, w.num_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(x.d_ptr, 0x3E, x.num_elements * sizeof(float)));

    const double bytes = (double)kVocab * kHidden * sizeof(__nv_bfloat16) +
                         (double)kHidden * sizeof(float) +
                         (double)kVocab * sizeof(float);
    reporter.row("lm_head 128256x4096", bytes, 2.0 * kVocab * kHidden, [&] {
        launch_bf16_gemv_kernel(w, x, y, kVocab, kHidden);
    });
}

TEST_F(CoreOpsBenchmark, AttentionDecodingLongContext) {
    bench_utils::Reporter reporter("Attention decoding (32q/8kv heads, head_dim 128)");
    const size_t max_seq_len = 2048;

    CudaVector<float> q(kQHeads * kHeadDim);
    CudaVector<float> k_cache(kKvHeads * max_seq_len * kHeadDim);
    CudaVector<float> v_cache(kKvHeads * max_seq_len * kHeadDim);
    CudaVector<float> o(kQHeads * kHeadDim);
    CUDA_CHECK(cudaMemset(q.d_ptr, 0x3D, q.num_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(k_cache.d_ptr, 0x3C, k_cache.num_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(v_cache.d_ptr, 0x3C, v_cache.num_elements * sizeof(float)));

    for (int pos : {127, 1023, 2047}) {
        // Per call: Q read, K/V cache rows [0..pos] read, O written.
        const double bytes = (double)kQHeads * kHeadDim * sizeof(float) * 2 +
                             2.0 * kKvHeads * (pos + 1) * kHeadDim * sizeof(float);
        reporter.row("decode @ pos " + std::to_string(pos), bytes, 0.0, [&] {
            launch_attention_decoding_kernel(q, k_cache, v_cache, o, pos,
                                             kQHeads, kKvHeads, kHeadDim, max_seq_len);
        });
    }
}

TEST_F(CoreOpsBenchmark, FusedRopeKvAppend) {
    bench_utils::Reporter reporter("Fused RoPE + KV append (32q/8kv heads)");
    const size_t max_seq_len = 2048;

    CudaVector<float> q(kQHeads * kHeadDim);
    CudaVector<float> k(kKvHeads * kHeadDim);
    CudaVector<float> v(kKvHeads * kHeadDim);
    CudaVector<float> k_cache(kKvHeads * max_seq_len * kHeadDim);
    CudaVector<float> v_cache(kKvHeads * max_seq_len * kHeadDim);

    const double bytes = (double)(kQHeads * 2 + kKvHeads * 4) * kHeadDim * sizeof(float);
    reporter.row("rope+append", bytes, 0.0, [&] {
        launch_fused_rope_kv_kernel(q, k, v, k_cache, v_cache, 1024,
                                    kQHeads, kKvHeads, kHeadDim, max_seq_len);
    });
}

TEST_F(CoreOpsBenchmark, RmsnormAndSwiglu) {
    bench_utils::Reporter reporter("RMSNorm / SwiGLU (Llama widths)");

    CudaVector<float> x(kHidden);
    CudaVector<float> residual(kHidden);
    CudaVector<__nv_bfloat16> weight(kHidden);  // the kernel reads weights as bf16
    // x and residual are read and written, the bf16 weight is read.
    const double rmsnorm_bytes = 4.0 * kHidden * sizeof(float) +
                                 kHidden * sizeof(__nv_bfloat16);
    reporter.row("rmsnorm+residual 4096", rmsnorm_bytes, 0.0, [&] {
        launch_rmsnorm_residual_kernel(x, residual, weight, 1, kHidden);
    });

    CudaVector<float> gate(kIntermediate);
    CudaVector<float> up(kIntermediate);
    CudaVector<float> act(kIntermediate);
    reporter.row("swiglu 14336", 3.0 * kIntermediate * sizeof(float), 0.0, [&] {
        launch_fused_swiglu_kernel(gate, up, act, kIntermediate);
    });
}

TEST_F(CoreOpsBenchmark, ArgmaxSampling) {
    bench_utils::Reporter reporter("Argmax over logits (Llama vocab)");
    CudaVector<float> logits(kVocab);
    CudaVector<int> token(1);
    CUDA_CHECK(cudaMemset(logits.d_ptr, 0x3C, kVocab * sizeof(float)));

    reporter.row("argmax 128256", (double)kVocab * sizeof(float), 0.0, [&] {
        launch_argmax_kernel(logits, token, kVocab);
    });
}
