#pragma once
// BF16 GEMV test support (Llama lm_head / logits projection path): CPU
// problem generator with FP64 accumulation and an RAII device-buffer mirror.
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "common/cuda_test_utils.h"
#include "kernels/bf16_linear.cuh"
#include "kernels/bf16_linear_residual.cuh"

namespace test_utils {

struct Bf16Problem {
    size_t M = 0;
    size_t K = 0;
    std::vector<__nv_bfloat16> weights;  // [M, K]
    std::vector<float> x;                // [K]
    std::vector<float> ref;              // [M]
};

inline Bf16Problem make_bf16_problem(size_t M, size_t K, unsigned seed) {
    Bf16Problem p;
    p.M = M;
    p.K = K;
    p.weights.resize(M * K);
    p.x = random_normal(K, seed ^ 0xC2B2AE35u);
    p.ref.resize(M);

    const std::vector<float> w = random_normal(M * K, seed, 0.05f);
    for (size_t i = 0; i < w.size(); ++i)
        p.weights[i] = __float2bfloat16(w[i]);

    for (size_t i = 0; i < M; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < K; ++j)
            dot += (double)__bfloat162float(p.weights[i * K + j]) * (double)p.x[j];
        p.ref[i] = (float)dot;
    }
    return p;
}

// Residual variant (o_proj / down_proj path): the accumulator starts at a
// non-trivial value and the kernel must add W @ x on top of it (+=), never
// overwrite it.
struct Bf16ResidualProblem {
    Bf16Problem base;                 // base.ref holds the pure W @ x dot
    std::vector<float> accum_init;    // [M] initial residual stream
    std::vector<float> ref;           // [M] accum_init + W @ x
};

inline Bf16ResidualProblem make_bf16_residual_problem(size_t M, size_t K, unsigned seed) {
    Bf16ResidualProblem p;
    p.base = make_bf16_problem(M, K, seed);
    p.accum_init = random_normal(M, seed ^ 0x9E3779B9u);
    p.ref.resize(M);
    for (size_t i = 0; i < M; ++i)
        p.ref[i] = p.accum_init[i] + p.base.ref[i];
    return p;
}

struct Bf16ResidualDevice {
    CudaVector<__nv_bfloat16> weights;
    CudaVector<float> x;
    CudaVector<float> y_accum;

    explicit Bf16ResidualDevice(const Bf16ResidualProblem& p)
        : weights(p.base.weights.size()), x(p.base.x.size()), y_accum(p.base.M) {
        weights.upload(p.base.weights);
        x.upload(p.base.x);
        y_accum.upload(p.accum_init);
    }

    void launch(const Bf16ResidualProblem& p) {
        launch_bf16_gemv_residual_kernel(weights, x, y_accum, p.base.M, p.base.K);
    }

    std::vector<float> run(const Bf16ResidualProblem& p) {
        launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(p.base.M);
        y_accum.download(out);
        return out;
    }
};

struct Bf16Device {
    CudaVector<__nv_bfloat16> weights;
    CudaVector<float> x;
    CudaVector<float> y;

    explicit Bf16Device(const Bf16Problem& p)
        : weights(p.weights.size()), x(p.x.size()), y(p.M) {
        weights.upload(p.weights);
        x.upload(p.x);
        poison(y);
    }

    void launch(const Bf16Problem& p) {
        launch_bf16_gemv_kernel(weights, x, y, p.M, p.K);
    }

    std::vector<float> run(const Bf16Problem& p) {
        launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(p.M);
        y.download(out);
        return out;
    }
};

}  // namespace test_utils
