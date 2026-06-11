#pragma once
// Shared CPU reference math and problem generators for the quantized linear
// kernel test suites (AWQ int4 and FP8 E4M3). Validation suites assert GPU
// results against these references; stress suites reuse the generators for
// edge-case shapes and malformed inputs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cpu_models.h"

namespace test_utils {

inline std::vector<float> random_normal(size_t n, unsigned seed, float stddev = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (auto& e : v) e = dist(rng);
    return v;
}

// Relative error with an absolute floor in the denominator: outputs that land
// near zero by cancellation would otherwise blow up the relative metric.
inline float max_rel_error(const std::vector<float>& gpu,
                           const std::vector<float>& ref,
                           float denom_floor = 1e-3f) {
    float max_rel = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float a = std::fabs(gpu[i] - ref[i]);
        max_rel = std::max(max_rel, a / std::max(denom_floor, std::fabs(ref[i])));
    }
    return max_rel;
}

// ============================================================================
// AWQ int4 (AutoAWQ packing)
// ============================================================================

// AWQ packs 8 int4 values per uint32 in order {0,2,4,6,1,3,5,7}: logical
// element j of a packed word lives at bit offset 16*(j&1) + 4*(j>>1).
constexpr int kAwqOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};

struct AwqProblem {
    int in_features = 0;
    int out_features = 0;
    int group_size = 0;
    std::vector<uint32_t> qweight;  // [in_features][out_features/8]
    std::vector<uint32_t> qzeros;   // [num_groups][out_features/8]
    std::vector<half> scales;       // [num_groups][out_features]
    std::vector<float> x;           // [in_features]
    std::vector<float> ref;         // [out_features], FP64 reference
};

inline AwqProblem make_awq_problem(int in_features, int out_features,
                                   int group_size, unsigned seed) {
    const int gs = group_size > 0 ? group_size : in_features;
    const int num_groups = (in_features + gs - 1) / gs;
    const int packed_cols = out_features / 8;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q4(0, 15);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    std::vector<int> w((size_t)in_features * out_features);
    std::vector<int> z((size_t)num_groups * out_features);

    AwqProblem p;
    p.in_features = in_features;
    p.out_features = out_features;
    p.group_size = group_size;
    p.scales.resize((size_t)num_groups * out_features);
    p.x = random_normal(in_features, seed ^ 0x9E3779B9u);
    p.ref.assign(out_features, 0.0f);

    for (auto& v : w) v = q4(rng);
    for (auto& v : z) v = q4(rng);
    for (auto& v : p.scales) v = __float2half(nrm(rng) * 0.05f);

    p.qweight.assign((size_t)in_features * packed_cols, 0);
    p.qzeros.assign((size_t)num_groups * packed_cols, 0);
    for (int k = 0; k < in_features; ++k)
        for (int pc = 0; pc < packed_cols; ++pc)
            for (int n = 0; n < 8; ++n)
                p.qweight[(size_t)k * packed_cols + pc] |=
                    (uint32_t)w[(size_t)k * out_features + pc * 8 + kAwqOrder[n]] << (4 * n);
    for (int g = 0; g < num_groups; ++g)
        for (int pc = 0; pc < packed_cols; ++pc)
            for (int n = 0; n < 8; ++n)
                p.qzeros[(size_t)g * packed_cols + pc] |=
                    (uint32_t)z[(size_t)g * out_features + pc * 8 + kAwqOrder[n]] << (4 * n);

    for (int oc = 0; oc < out_features; ++oc) {
        double acc = 0.0;
        for (int k = 0; k < in_features; ++k) {
            const int g = k / gs;
            acc += (double)(w[(size_t)k * out_features + oc] - z[(size_t)g * out_features + oc]) *
                   (double)__half2float(p.scales[(size_t)g * out_features + oc]) * (double)p.x[k];
        }
        p.ref[oc] = (float)acc;
    }
    return p;
}

// Constant-pattern problem: every weight nibble = 5, every zero nibble = 4,
// every scale = 0.25, so out[j] == 0.25 * sum(x) for all j. The O(IC)
// reference makes vocab-sized shapes testable without an O(IC*OC) CPU pass.
inline AwqProblem make_awq_constant_problem(int in_features, int out_features,
                                            int group_size, unsigned seed) {
    const int gs = group_size > 0 ? group_size : in_features;
    const int num_groups = (in_features + gs - 1) / gs;
    const int packed_cols = out_features / 8;

    AwqProblem p;
    p.in_features = in_features;
    p.out_features = out_features;
    p.group_size = group_size;
    p.qweight.assign((size_t)in_features * packed_cols, 0x55555555u);
    p.qzeros.assign((size_t)num_groups * packed_cols, 0x44444444u);
    p.scales.assign((size_t)num_groups * out_features, __float2half(0.25f));
    p.x = random_normal(in_features, seed);

    double sum_x = 0.0;
    for (float v : p.x) sum_x += v;
    p.ref.assign(out_features, (float)(0.25 * sum_x));
    return p;
}

// ============================================================================
// FP8 E4M3
// ============================================================================

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

}  // namespace test_utils
