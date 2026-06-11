#pragma once
// AWQ int4 GEMV test support: CPU problem generators in canonical AutoAWQ
// packing, an RAII device-buffer mirror, and the assertion helper shared by
// the validation and stress suites.
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/cuda_test_utils.h"
#include "kernels/awq_linear.cuh"

namespace test_utils {

// FP16 scales bound the achievable precision; 4.2e-3 was the worst case
// observed across shapes, 5e-3 leaves headroom without masking real bugs.
constexpr float kAwqRelTolerance = 5e-3f;

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

// Device-side mirror of an AwqProblem with a poisoned output buffer.
struct AwqDevice {
    CudaVector<uint32_t> qweight;
    CudaVector<uint32_t> qzeros;
    CudaVector<half> scales;
    CudaVector<float> x;
    CudaVector<float> y;

    explicit AwqDevice(const AwqProblem& p)
        : qweight(p.qweight.size()),
          qzeros(p.qzeros.size()),
          scales(p.scales.size()),
          x(p.x.size()),
          y(p.out_features) {
        qweight.upload(p.qweight);
        qzeros.upload(p.qzeros);
        scales.upload(p.scales);
        x.upload(p.x);
        poison(y);
    }

    void launch(const AwqProblem& p) {
        launch_awq_gemv_kernel(qweight, scales, qzeros, x, y,
                               p.out_features, p.in_features, p.group_size);
    }

    std::vector<float> run(const AwqProblem& p) {
        launch(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(p.out_features);
        y.download(out);
        return out;
    }
};

inline void expect_awq_matches_reference(int in_features, int out_features,
                                         int group_size, unsigned seed) {
    const AwqProblem p = make_awq_problem(in_features, out_features, group_size, seed);
    AwqDevice dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref);
    EXPECT_LE(max_rel, kAwqRelTolerance)
        << "AWQ GEMV mismatch: IC=" << in_features << " OC=" << out_features
        << " group_size=" << group_size << " max_rel=" << max_rel;
}

}  // namespace test_utils
