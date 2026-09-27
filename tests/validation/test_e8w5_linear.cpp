// E8W5 (5-bit companded-E8 lattice) device decode and GEMV correctness.
//
// Format: docs/E8W5_FORMAT_SPEC.md. The reference packer/unpacker below is an independent
// C++ restatement of the bit layout -- deliberately NOT a call into the CUDA primitives --
// so the two can only agree if src/kernels/e8w5_dequant.cuh implements the format the spec
// describes. research/pack_e8w5.py checks the same layout on the Python side against the
// real quantizer, so the three stay pinned together.
//
// The dequantization claim is BIT-EXACT: it is elementwise (codebook lookup times a group
// scale), so there is no reduction order to diverge. The GEMV claim is a tolerance, because
// a warp-shuffle tree and a sequential CPU sum do not associate the same way; it is checked
// against an FP64 accumulation of the SAME decoded weights, which isolates the reduction
// from the decode.
//
// Shapes are the mission's K in {4096, 11008, 14336} plus GLM-4-9B's 13696 (the tensor the
// research measured), all multiples of the format's 128-weight scale group.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/cuda_test_utils.h"
#include "kernels/e8w5_linear.cuh"

namespace {

constexpr int kDim = 8;        // E8 coordinates per block
constexpr int kLevels = 32;    // 2^5 per coordinate
constexpr int kLutN = 64;      // 2 * kLevels, indexed by (coset, u)
constexpr int kGroup = 128;    // weights per scale group
constexpr int kLo = -16;       // coordinate box [-16, +15]
constexpr int kHi = 15;
constexpr int kCosetBit = 28;

// ---------------------------------------------------------------------------
// reference pack / unpack -- the spec's §3 bit layout, written out longhand
// ---------------------------------------------------------------------------

// Plane L: nibble i = u_i & 0xF for i in 0..6; nibble 7 = (u_7 & 0xE) | coset, because
// u_7's low bit is implied by D8's even-sum parity. Plane H: bit i = u_i >> 4.
void ref_pack_block(const int (&k)[kDim], int coset, uint32_t& lo4, uint8_t& hi)
{
    int sum = 0;
    for (int i = 0; i < kDim; ++i) sum += k[i];
    if ((sum & 1) != 0)
        throw std::logic_error("reference packer: odd coordinate sum is not a D8 point");

    lo4 = 0;
    hi = 0;
    for (int i = 0; i < kDim; ++i) {
        const uint32_t u = static_cast<uint32_t>(k[i] - kLo);
        if (u >= kLevels) throw std::logic_error("reference packer: coordinate out of box");
        if (i < kDim - 1) lo4 |= (u & 0xFu) << (4 * i);
        hi |= static_cast<uint8_t>(((u >> 4) & 1u) << i);
    }
    const uint32_t u7 = static_cast<uint32_t>(k[kDim - 1] - kLo);
    lo4 |= ((u7 & 0xEu) | static_cast<uint32_t>(coset)) << kCosetBit;
}

// Returns the 6-bit codebook indices the decode must produce.
void ref_unpack_indices(uint32_t lo4, uint8_t hi, int (&idx)[kDim])
{
    const uint32_t coset = (lo4 >> kCosetBit) & 1u;
    uint32_t u[kDim];
    uint32_t parity = 0;
    for (int i = 0; i < kDim - 1; ++i) {
        u[i] = ((lo4 >> (4 * i)) & 0xFu) | (((static_cast<uint32_t>(hi) >> i) & 1u) << 4);
        parity ^= (u[i] & 1u);
    }
    u[kDim - 1] = (((lo4 >> kCosetBit) & 0xEu) | parity)
                  | (((static_cast<uint32_t>(hi) >> (kDim - 1)) & 1u) << 4);
    for (int i = 0; i < kDim; ++i) idx[i] = static_cast<int>((coset << 5) | u[i]);
}

// ---------------------------------------------------------------------------
// a packed problem
// ---------------------------------------------------------------------------

struct E8W5Problem {
    int out_features = 0, in_features = 0;
    std::vector<uint32_t> plane_lo;   // [out][in/8]
    std::vector<uint8_t> plane_hi;    // [out][in/8]
    std::vector<half> scales;         // [out][in/128]
    std::vector<half> codebook;       // [64]
    std::vector<float> w_ref;         // [out][in] expected dequantization, bit-exact
    std::vector<float> x;             // [in]
};

E8W5Problem make_problem(int out_features, int in_features, unsigned seed)
{
    E8W5Problem p;
    p.out_features = out_features;
    p.in_features = in_features;
    const int n_blocks = in_features / kDim;
    const int n_groups = in_features / kGroup;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coord(kLo, kHi);
    std::uniform_int_distribution<int> flip(0, 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // A codebook shaped like a real one: monotone, roughly Gaussian-spaced, so the test
    // exercises realistic magnitudes rather than an index ramp. Entry (c,u) ordering is the
    // format's: index = (coset << 5) | u.
    p.codebook.resize(kLutN);
    for (int c = 0; c < 2; ++c)
        for (int u = 0; u < kLevels; ++u) {
            const float t = (static_cast<float>(u) + 0.5f * static_cast<float>(c)) / kLevels;
            p.codebook[(c << 5) | u] = __float2half(2.0f * t - 1.0f);
        }

    p.scales.resize(static_cast<size_t>(out_features) * n_groups);
    for (auto& s : p.scales)
        s = __float2half(0.01f + 0.02f * std::fabs(normal(rng)));

    p.plane_lo.resize(static_cast<size_t>(out_features) * n_blocks);
    p.plane_hi.resize(static_cast<size_t>(out_features) * n_blocks);
    p.w_ref.assign(static_cast<size_t>(out_features) * in_features, 0.0f);

    for (int n = 0; n < out_features; ++n) {
        for (int b = 0; b < n_blocks; ++b) {
            int k[kDim];
            int sum = 0;
            for (int i = 0; i < kDim; ++i) { k[i] = coord(rng); sum += k[i]; }
            if (sum & 1) {                      // repair parity like the real encoder does
                if (k[0] < kHi) k[0] += 1; else k[0] -= 1;
            }
            const int coset = flip(rng);
            const size_t blk = static_cast<size_t>(n) * n_blocks + b;
            ref_pack_block(k, coset, p.plane_lo[blk], p.plane_hi[blk]);

            int idx[kDim];
            ref_unpack_indices(p.plane_lo[blk], p.plane_hi[blk], idx);
            const int g = (b * kDim) / kGroup;
            const float s = __half2float(p.scales[static_cast<size_t>(n) * n_groups + g]);
            for (int i = 0; i < kDim; ++i) {
                // Same operand order the kernel uses (s * codebook_value), so a bit-exact
                // comparison is meaningful rather than luck.
                p.w_ref[static_cast<size_t>(n) * in_features + b * kDim + i] =
                    s * __half2float(p.codebook[idx[i]]);
            }
        }
    }

    p.x.resize(in_features);
    for (auto& v : p.x) v = normal(rng);
    return p;
}

// Device-side mirror of a problem, freed on destruction.
struct DeviceProblem {
    void* lo = nullptr;
    void* hi = nullptr;
    void* sc = nullptr;
    void* cb = nullptr;
    float* x = nullptr;
    float* out = nullptr;

    // gtest's ASSERT_* expands to `return;`, which a constructor may not do, so allocation
    // failures throw here and surface as a test error rather than a silent null pointer.
    explicit DeviceProblem(const E8W5Problem& p, size_t out_elems)
    {
        auto ck = [](cudaError_t e, const char* what) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("DeviceProblem: ") + what + ": " +
                                         cudaGetErrorString(e));
        };
        auto up = [&](void** d, const void* h, size_t bytes) {
            ck(cudaMalloc(d, bytes), "cudaMalloc");
            ck(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        };
        up(&lo, p.plane_lo.data(), p.plane_lo.size() * sizeof(uint32_t));
        up(&hi, p.plane_hi.data(), p.plane_hi.size() * sizeof(uint8_t));
        up(&sc, p.scales.data(), p.scales.size() * sizeof(half));
        up(&cb, p.codebook.data(), p.codebook.size() * sizeof(half));
        up(reinterpret_cast<void**>(&x), p.x.data(), p.x.size() * sizeof(float));
        ck(cudaMalloc(&out, out_elems * sizeof(float)), "cudaMalloc out");
        ck(cudaMemset(out, 0, out_elems * sizeof(float)), "cudaMemset out");
    }
    ~DeviceProblem()
    {
        cudaFree(lo); cudaFree(hi); cudaFree(sc); cudaFree(cb); cudaFree(x); cudaFree(out);
    }
    DeviceProblem(const DeviceProblem&) = delete;
    DeviceProblem& operator=(const DeviceProblem&) = delete;
};

class E8W5Validation : public test_utils::CudaTest {};

// The shapes the mission names, plus GLM-4-9B's down_proj K. out_features kept modest so
// the suite stays in the fast `validation` label.
class E8W5Shapes : public E8W5Validation,
                   public ::testing::WithParamInterface<int> {};

INSTANTIATE_TEST_SUITE_P(DecodeShapes, E8W5Shapes,
                         ::testing::Values(4096, 11008, 13696, 14336));

} // namespace

// ---------------------------------------------------------------------------
// 1. bit-exact dequantization
// ---------------------------------------------------------------------------
TEST_P(E8W5Shapes, DequantizeIsBitExactVsCpuReference)
{
    const int K = GetParam();
    const int M = 64;
    const E8W5Problem p = make_problem(M, K, /*seed=*/1234u + K);

    DeviceProblem d(p, static_cast<size_t>(M) * K);
    launch_e8w5_dequantize(d.lo, d.hi, d.sc, d.cb, d.out, M, K);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> got(static_cast<size_t>(M) * K);
    ASSERT_EQ(cudaMemcpy(got.data(), d.out, got.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    size_t mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != p.w_ref[i]) {          // exact: elementwise, no reduction involved
            if (mismatches == 0) first = i;
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u)
        << "K=" << K << ": " << mismatches << " of " << got.size()
        << " weights differ; first at flat index " << first
        << " (row " << first / K << ", col " << first % K << ") got " << got[first]
        << " expected " << p.w_ref[first];
}

// ---------------------------------------------------------------------------
// 2. GEMV against an FP64 sum of the SAME decoded weights
// ---------------------------------------------------------------------------
TEST_P(E8W5Shapes, GemvMatchesFp64ReferenceSum)
{
    const int K = GetParam();
    const int M = 256;
    const E8W5Problem p = make_problem(M, K, /*seed=*/777u + K);

    DeviceProblem d(p, static_cast<size_t>(M));
    launch_e8w5_gemv_kernel(d.lo, d.hi, d.sc, d.cb, d.x, d.out, M, K);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> got(M);
    ASSERT_EQ(cudaMemcpy(got.data(), d.out, got.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    double num = 0.0, den = 0.0;
    for (int n = 0; n < M; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += static_cast<double>(p.w_ref[static_cast<size_t>(n) * K + k])
                   * static_cast<double>(p.x[k]);
        const double diff = static_cast<double>(got[n]) - acc;
        num += diff * diff;
        den += acc * acc;
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    EXPECT_LT(rel, 1e-6) << "K=" << K << " relative Frobenius error " << rel;
}

// ---------------------------------------------------------------------------
// 3. the residual variant accumulates rather than overwrites
// ---------------------------------------------------------------------------
TEST_F(E8W5Validation, ResidualVariantAccumulatesIntoTheStream)
{
    const int K = 4096, M = 128;
    const E8W5Problem p = make_problem(M, K, /*seed=*/99u);

    std::vector<float> seed(M);
    for (int n = 0; n < M; ++n) seed[n] = 0.5f * static_cast<float>(n) - 10.0f;

    DeviceProblem d(p, static_cast<size_t>(M));
    ASSERT_EQ(cudaMemcpy(d.out, seed.data(), seed.size() * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);
    launch_e8w5_gemv_residual_kernel(d.lo, d.hi, d.sc, d.cb, d.x, d.out, M, K);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> with_residual(M);
    ASSERT_EQ(cudaMemcpy(with_residual.data(), d.out, M * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaMemset(d.out, 0, M * sizeof(float)), cudaSuccess);
    launch_e8w5_gemv_kernel(d.lo, d.hi, d.sc, d.cb, d.x, d.out, M, K);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> plain(M);
    ASSERT_EQ(cudaMemcpy(plain.data(), d.out, M * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    for (int n = 0; n < M; ++n)
        EXPECT_FLOAT_EQ(with_residual[n], seed[n] + plain[n]) << "row " << n;
}

// ---------------------------------------------------------------------------
// 4. every coordinate value and both cosets are actually reachable
// ---------------------------------------------------------------------------
// The parity trick makes u_7 the one coordinate whose low bit is not stored, and the coset
// flag shares nibble 7 with it. A decode bug there would be invisible in random data if it
// only affected rare values, so sweep all 32 values at all 8 positions in both cosets.
TEST_F(E8W5Validation, EveryCoordinateValueAndBothCosetsDecodeExactly)
{
    const int K = 4096;                      // 512 blocks per row
    const int M = 1;
    E8W5Problem p = make_problem(M, K, /*seed=*/5u);

    // Overwrite the first 2*8*32 = 512 blocks with the exhaustive sweep.
    const int n_blocks = K / kDim;
    const int n_groups = K / kGroup;
    ASSERT_GE(n_blocks, 2 * kDim * kLevels);
    int b = 0;
    for (int c = 0; c < 2; ++c)
        for (int pos = 0; pos < kDim; ++pos)
            for (int v = kLo; v <= kHi; ++v, ++b) {
                int k[kDim] = {0, 0, 0, 0, 0, 0, 0, 0};
                k[pos] = v;
                const int fix = (pos != 1) ? 1 : 2;   // repair parity elsewhere
                int sum = 0;
                for (int i = 0; i < kDim; ++i) sum += k[i];
                k[fix] += (sum & 1);
                ref_pack_block(k, c, p.plane_lo[b], p.plane_hi[b]);
                int idx[kDim];
                ref_unpack_indices(p.plane_lo[b], p.plane_hi[b], idx);
                const float s = __half2float(p.scales[(b * kDim) / kGroup]);
                for (int i = 0; i < kDim; ++i)
                    p.w_ref[static_cast<size_t>(b) * kDim + i] =
                        s * __half2float(p.codebook[idx[i]]);
            }
    ASSERT_EQ(b, 2 * kDim * kLevels);
    (void)n_groups;

    DeviceProblem d(p, static_cast<size_t>(M) * K);
    launch_e8w5_dequantize(d.lo, d.hi, d.sc, d.cb, d.out, M, K);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(static_cast<size_t>(M) * K);
    ASSERT_EQ(cudaMemcpy(got.data(), d.out, got.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < 2 * kDim * kLevels * kDim; ++i)
        ASSERT_EQ(got[i], p.w_ref[i]) << "exhaustive sweep diverged at coordinate " << i;
}

// ---------------------------------------------------------------------------
// 5. the group-alignment contract is enforced, not silently tolerated
// ---------------------------------------------------------------------------
TEST_F(E8W5Validation, RejectsInFeaturesNotAMultipleOfTheScaleGroup)
{
    float* dummy = nullptr;
    ASSERT_EQ(cudaMalloc(&dummy, 16 * sizeof(float)), cudaSuccess);
    EXPECT_THROW(launch_e8w5_gemv_kernel(dummy, dummy, dummy, dummy, dummy, dummy,
                                         8, /*in_features=*/130),
                 std::invalid_argument);
    EXPECT_THROW(launch_e8w5_dequantize(dummy, dummy, dummy, dummy, dummy,
                                        8, /*in_features=*/64),
                 std::invalid_argument);
    cudaFree(dummy);
}
