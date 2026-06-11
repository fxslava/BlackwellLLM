#pragma once
// FP8 E4M3 GEMV test support: CPU problem generators (replicating the
// kernel's quantization exactly), an RAII device-buffer mirror, and the
// assertion helper shared by the validation and stress suites.
//
// Kernel contract notes (see fp8_linear.cu): K must be a multiple of 16
// (uint4/float4 vectorized loads), and token_scale is read unconditionally,
// so every launch here passes a valid 2-float token_scale buffer the way the
// engine does -- never nullptr.
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "common/cuda_test_utils.h"
#include "common/cpu_reference.h"
#include "kernels/fp8_linear.cuh"

namespace test_utils {

// The reference replicates the kernel's quantization exactly (E4M3 RNE,
// bf16-rounded scales), so the only error source is FP32 accumulation order;
// 1.1e-3 was the worst case observed (M=14336), 2e-3 leaves headroom without
// masking real bugs.
constexpr float kFp8RelTolerance = 2e-3f;
constexpr float kFp8DenomFloor = 1e-2f;

// Nearest E4M3 value with round-to-nearest-even tie-breaking, matching the
// hardware float->fp8 conversion for finite, pre-clamped inputs. NaN byte
// patterns (exp=15, mant=7) are excluded from the candidate set.
inline float cpu_quantize_e4m3(float x) {
    x = std::min(std::max(x, -448.0f), 448.0f);
    float best_val = 0.0f;
    float best_diff = INFINITY;
    int best_byte = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b & 0x7F) == 0x7F) continue;
        const float v = cpu_unpack_fp8_e4m3(static_cast<uint8_t>(b));
        const float d = std::fabs(v - x);
        if (d < best_diff ||
            (d == best_diff && (b & 1) == 0 && (best_byte & 1) == 1)) {
            best_diff = d;
            best_val = v;
            best_byte = b;
        }
    }
    return best_val;
}

struct Fp8Problem {
    size_t M = 0;
    size_t K = 0;                         // must be a multiple of 16 (kernel uses uint4/float4 loads)
    std::vector<uint8_t> weights;         // [M, K] raw E4M3 bytes
    std::vector<__nv_bfloat16> w_scales;  // [M] row-wise weight scales
    float act_scale = 0.0f;               // > 0: activation-quantization path (bf16-rounded)
    std::vector<float> x;                 // [K]
    std::vector<float> ref;               // [M]
};

// act_scale <= 0 builds the weight-only path (activations enter the dot
// product untouched). act_scale > 0 builds the quantized-activation path:
// activations are converted to E4M3 in x/act_scale domain and the epilogue
// multiplies the scale back, mirroring fp8_gemv with a non-null input_scale.
inline Fp8Problem make_fp8_problem(size_t M, size_t K, unsigned seed,
                                   float act_scale = 0.0f) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    Fp8Problem p;
    p.M = M;
    p.K = K;
    p.weights.resize(M * K);
    p.w_scales.resize(M);
    p.x = random_normal(K, seed ^ 0x85EBCA6Bu);
    p.ref.resize(M);
    if (act_scale > 0.0f) {
        // Round through bf16 once: the kernel reads the scale as bf16.
        p.act_scale = __bfloat162float(__float2bfloat16(act_scale));
    }

    for (auto& v : p.weights) v = static_cast<uint8_t>(byte(rng));
    for (auto& v : p.w_scales) v = __float2bfloat16(0.01f + 0.005f * std::fabs(nrm(rng)));

    std::vector<float> x_eff = p.x;
    if (p.act_scale > 0.0f)
        for (auto& v : x_eff) v = cpu_quantize_e4m3(v / p.act_scale);

    for (size_t i = 0; i < M; ++i) {
        double dot = 0.0;
        const uint8_t* row = p.weights.data() + i * K;
        for (size_t j = 0; j < K; ++j)
            dot += (double)cpu_unpack_fp8_e4m3(row[j]) * (double)x_eff[j];
        const float w_scale = __bfloat162float(p.w_scales[i]);
        p.ref[i] = (float)dot * w_scale * (p.act_scale > 0.0f ? p.act_scale : 1.0f);
    }
    return p;
}

// Device-side mirror of an Fp8Problem with a poisoned output buffer.
struct Fp8Device {
    CudaVector<uint8_t> weights;
    CudaVector<__nv_bfloat16> w_scales;
    CudaVector<__nv_bfloat16> input_scale;
    CudaVector<float> token_scale;
    CudaVector<float> x;
    CudaVector<float> y;

    explicit Fp8Device(const Fp8Problem& p)
        : weights(p.weights.size()),
          w_scales(p.w_scales.size()),
          input_scale(1),
          token_scale(2),
          x(p.x.size()),
          y(p.M) {
        weights.upload(p.weights);
        w_scales.upload(p.w_scales);
        x.upload(p.x);
        token_scale.upload({1.0f, 1.0f});
        if (p.act_scale > 0.0f)
            input_scale.upload({__float2bfloat16(p.act_scale)});
        poison(y);
    }

    const __nv_bfloat16* input_scale_arg(const Fp8Problem& p) const {
        return p.act_scale > 0.0f ? input_scale.d_ptr : nullptr;
    }

    void launch(const Fp8Problem& p) {
        launch_fp8_gemv_kernel(weights, x, w_scales, input_scale_arg(p),
                               token_scale, y, p.M, p.K, 1);
    }

    std::vector<float> run(const Fp8Problem& p) {
        launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(p.M);
        y.download(out);
        return out;
    }
};

inline void expect_fp8_matches_reference(size_t M, size_t K, unsigned seed,
                                         float act_scale = 0.0f) {
    const Fp8Problem p = make_fp8_problem(M, K, seed, act_scale);
    Fp8Device dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, kFp8DenomFloor);
    EXPECT_LE(max_rel, kFp8RelTolerance)
        << "FP8 GEMV mismatch: M=" << M << " K=" << K
        << " act_scale=" << act_scale << " max_rel=" << max_rel;
}

}  // namespace test_utils
