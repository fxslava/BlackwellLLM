#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.h"
#include "kernels/awq_linear.cuh"

namespace {

// AWQ packs 8 int4 values per uint32 in order {0,2,4,6,1,3,5,7}:
// logical element j of a packed word lives at bit offset 16*(j&1) + 4*(j>>1).
constexpr int kAwqOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};

struct AwqTestCase {
    std::vector<uint32_t> qweight;  // [in_features][out_features/8]
    std::vector<uint32_t> qzeros;   // [num_groups][out_features/8]
    std::vector<half> scales;       // [num_groups][out_features]
    std::vector<float> x;           // [in_features]
    std::vector<float> ref;         // [out_features], FP64 reference
};

AwqTestCase make_case(int in_features, int out_features, int group_size, unsigned seed) {
    const int gs = group_size > 0 ? group_size : in_features;
    const int num_groups = (in_features + gs - 1) / gs;
    const int packed_cols = out_features / 8;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q4(0, 15);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    std::vector<int> w((size_t)in_features * out_features);
    std::vector<int> z((size_t)num_groups * out_features);

    AwqTestCase tc;
    tc.scales.resize((size_t)num_groups * out_features);
    tc.x.resize(in_features);
    tc.ref.assign(out_features, 0.0f);

    for (auto& v : w) v = q4(rng);
    for (auto& v : z) v = q4(rng);
    for (auto& v : tc.scales) v = __float2half(nrm(rng) * 0.05f);
    for (auto& v : tc.x) v = nrm(rng);

    tc.qweight.assign((size_t)in_features * packed_cols, 0);
    tc.qzeros.assign((size_t)num_groups * packed_cols, 0);
    for (int k = 0; k < in_features; ++k)
        for (int pc = 0; pc < packed_cols; ++pc)
            for (int n = 0; n < 8; ++n)
                tc.qweight[(size_t)k * packed_cols + pc] |=
                    (uint32_t)w[(size_t)k * out_features + pc * 8 + kAwqOrder[n]] << (4 * n);
    for (int g = 0; g < num_groups; ++g)
        for (int pc = 0; pc < packed_cols; ++pc)
            for (int n = 0; n < 8; ++n)
                tc.qzeros[(size_t)g * packed_cols + pc] |=
                    (uint32_t)z[(size_t)g * out_features + pc * 8 + kAwqOrder[n]] << (4 * n);

    for (int oc = 0; oc < out_features; ++oc) {
        double acc = 0.0;
        for (int k = 0; k < in_features; ++k) {
            const int g = k / gs;
            acc += (double)(w[(size_t)k * out_features + oc] - z[(size_t)g * out_features + oc]) *
                   (double)__half2float(tc.scales[(size_t)g * out_features + oc]) * (double)tc.x[k];
        }
        tc.ref[oc] = (float)acc;
    }
    return tc;
}

// Relative error with a small absolute floor in the denominator: outputs that
// land near zero by cancellation would otherwise blow up the relative metric.
float max_rel_error(const std::vector<float>& gpu, const std::vector<float>& ref) {
    float max_rel = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float a = std::fabs(gpu[i] - ref[i]);
        max_rel = std::max(max_rel, a / std::max(1e-3f, std::fabs(ref[i])));
    }
    return max_rel;
}

// FP16 scales bound the achievable precision; 4.2e-3 was the worst case
// observed across shapes, 5e-3 leaves headroom without masking real bugs.
constexpr float kRelTolerance = 5e-3f;

void run_and_check(int in_features, int out_features, int group_size, unsigned seed) {
    const AwqTestCase tc = make_case(in_features, out_features, group_size, seed);

    CudaVector<uint32_t> d_qw(tc.qweight.size()); d_qw.upload(tc.qweight);
    CudaVector<uint32_t> d_qz(tc.qzeros.size());  d_qz.upload(tc.qzeros);
    CudaVector<half>     d_s(tc.scales.size());   d_s.upload(tc.scales);
    CudaVector<float>    d_x(tc.x.size());        d_x.upload(tc.x);
    CudaVector<float>    d_y(out_features);
    // Poison the output so a kernel that silently writes nothing fails loudly.
    CUDA_CHECK(cudaMemset(d_y.d_ptr, 0xCC, out_features * sizeof(float)));

    launch_awq_gemv_kernel(d_qw, d_s, d_qz, d_x, d_y,
                           out_features, in_features, group_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(out_features);
    d_y.download(gpu);

    const float max_rel = max_rel_error(gpu, tc.ref);
    EXPECT_LE(max_rel, kRelTolerance)
        << "AWQ GEMV mismatch: IC=" << in_features << " OC=" << out_features
        << " group_size=" << group_size << " max_rel=" << max_rel;
}

}  // namespace

class AWQKernel : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[CUDA ERROR]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(AWQKernel, NumericalCorrectness) {
    run_and_check(4096, 4096, 128, 1);
}

TEST_F(AWQKernel, LlamaMlpProjection) {
    run_and_check(11008, 4096, 128, 2);
}

// OC=8 -> a single packed column; the launcher must split the reduction
// dimension across blocks (atomicAdd path) to keep the GPU busy.
TEST_F(AWQKernel, ExtremeSplitK) {
    run_and_check(8192, 8, 128, 3);
}

// group_size <= 0 means per-channel quantization (one group over all of IC).
TEST_F(AWQKernel, PerChannelQuantization) {
    run_and_check(4096, 256, -1, 4);
}

// in_features not divisible by group_size: the last group is ragged.
TEST_F(AWQKernel, RaggedTailGroup) {
    run_and_check(200, 64, 64, 5);
}

TEST_F(AWQKernel, SmallGroupSize) {
    run_and_check(512, 512, 64, 6);
}
