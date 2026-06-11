#pragma once
// Shared GTest scaffolding for CUDA kernel tests: base fixture, seeded random
// input generators and error metrics. Device memory management (CudaVector)
// and CUDA_CHECK come from the engine's src/common.h.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "common.h"  // CudaVector / CUDA_CHECK (src/common.h)

namespace test_utils {

// Base fixture for every kernel test: the CUDA context must be error-free
// when the test ends, otherwise a failed asynchronous launch in this test
// would surface as a confusing failure in the next one.
class CudaTest : public ::testing::Test {
protected:
    void TearDown() override {
        const cudaError_t err = cudaGetLastError();
        EXPECT_EQ(err, cudaSuccess)
            << "CUDA error left pending after test: " << cudaGetErrorString(err);
    }
};

// ----------------------------------------------------------------------------
// Random input generation (seeded, reproducible across runs and platforms)
// ----------------------------------------------------------------------------

inline std::vector<float> random_normal(size_t n, unsigned seed, float stddev = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (auto& e : v) e = dist(rng);
    return v;
}

inline std::vector<float> random_uniform(size_t n, unsigned seed,
                                         float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& e : v) e = dist(rng);
    return v;
}

inline std::vector<int> random_token_ids(size_t n, unsigned seed, int vocab_size) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, vocab_size - 1);
    std::vector<int> v(n);
    for (auto& e : v) e = dist(rng);
    return v;
}

// ----------------------------------------------------------------------------
// Error metrics
// ----------------------------------------------------------------------------

inline float max_abs_error(const std::vector<float>& gpu, const std::vector<float>& ref) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(gpu[i] - ref[i]));
    return max_abs;
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

// Element-wise comparison with per-index failure messages; used by the
// non-quantized kernel suites where outputs must match almost bitwise.
inline void expect_allclose(const std::vector<float>& ref,
                            const std::vector<float>& gpu,
                            float abs_tolerance,
                            const char* what) {
    ASSERT_EQ(ref.size(), gpu.size()) << what << ": size mismatch";
    for (size_t i = 0; i < ref.size(); ++i)
        ASSERT_NEAR(ref[i], gpu[i], abs_tolerance) << what << " mismatch at index " << i;
}

// Poison a device buffer so a kernel that silently writes nothing fails
// loudly instead of comparing leftover zeros against the reference.
template <typename T>
inline void poison(CudaVector<T>& v) {
    CUDA_CHECK(cudaMemset(v.d_ptr, 0xCC, v.num_elements * sizeof(T)));
}

}  // namespace test_utils
