// ============================================================================
// Standalone math/memory oracle for the Paged Flash Attention kernel.
// ============================================================================
// Validates, against an unoptimized FP32 CPU reference:
//   * the Tensor-Core (wmma) Q.K^T and P.V tiling,
//   * the shared-memory +8 skew loads (a fault here surfaces as a CUDA error),
//   * the block-table gather (physical pages are deliberately SCATTERED, so a
//     wrong page id reads wrong/zero KV and blows past the epsilon),
//   * causal + tail masking on a non-page-aligned sequence length,
//   * the FP32 -> BF16 storage/operand precision drift (absorbed by epsilon).
//
// Self-contained: depends only on the CUDA runtime, cuda_bf16, and the kernel
// launchers. No GoogleTest, no engine, no test fixtures.
//
// Build (Windows: from an "x64 Native Tools" prompt with CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 ^
//       tests/standalone/test_paged_attention.cpp ^
//       src/kernels/paged_flash_attention.cu ^
//       -o test_paged_attention.exe
//   ./test_paged_attention.exe
// (Use the architecture of your card; RTX 5070 == sm_120.)
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../../src/kernels/paged_flash_attention.cuh"

using blackwell::paging::kv_t;
using blackwell::paging::PAGE_SIZE;

#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "[CUDA] %s\n  at %s:%d\n",                    \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

// ----------------------------------------------------------------------------
// FP32 ground truth. K_log / V_log are LOGICAL [seq_len][kv_heads][head_dim];
// Q is [num_q][q_heads][head_dim] with row r living at logical position
// q_start + r. Causal: query position p attends to keys [0, p]. FP64 accum.
// ----------------------------------------------------------------------------
static void cpu_paged_attention_reference(
    const std::vector<float>& Q, const std::vector<float>& K_log,
    const std::vector<float>& V_log, std::vector<float>& O,
    int num_q, int q_start, int seq_len,
    int q_heads, int kv_heads, int head_dim)
{
    const double scale = 1.0 / std::sqrt((double)head_dim);
    const int gqa = q_heads / kv_heads;
    O.assign((size_t)num_q * q_heads * head_dim, 0.0f);

    for (int r = 0; r < num_q; ++r) {
        const int q_pos = q_start + r;
        for (int qh = 0; qh < q_heads; ++qh) {
            const int kvh = qh / gqa;
            const float* q = &Q[((size_t)r * q_heads + qh) * head_dim];

            std::vector<double> score(q_pos + 1);
            double mx = -1e300;
            for (int t = 0; t <= q_pos; ++t) {       // causal bound
                const float* k = &K_log[((size_t)t * kv_heads + kvh) * head_dim];
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d) dot += (double)q[d] * (double)k[d];
                score[t] = dot * scale;
                mx = std::max(mx, score[t]);
            }
            double sum = 0.0;
            for (int t = 0; t <= q_pos; ++t) { score[t] = std::exp(score[t] - mx); sum += score[t]; }

            float* o = &O[((size_t)r * q_heads + qh) * head_dim];
            for (int t = 0; t <= q_pos; ++t) {
                const double p = score[t] / sum;
                const float* v = &V_log[((size_t)t * kv_heads + kvh) * head_dim];
                for (int d = 0; d < head_dim; ++d) o[d] += (float)(p * (double)v[d]);
            }
        }
    }
}

// Scatter logical token KV into a BF16 physical-page arena via the block table.
//   pool layout: kv_t[total_pages][kv_heads][PAGE_SIZE][head_dim]
static void scatter_into_pages(
    const std::vector<float>& K_log, const std::vector<float>& V_log,
    const std::vector<int32_t>& block_table, int seq_len, int kv_heads, int head_dim,
    std::vector<kv_t>& k_pool, std::vector<kv_t>& v_pool)
{
    for (int t = 0; t < seq_len; ++t) {
        const int page = block_table[t / PAGE_SIZE];
        const int slot = t % PAGE_SIZE;
        for (int h = 0; h < kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t dst = (((size_t)page * kv_heads + h) * PAGE_SIZE + slot) * head_dim + d;
                const size_t src = ((size_t)t * kv_heads + h) * head_dim + d;
                k_pool[dst] = __float2bfloat16(K_log[src]);   // the actual FP32->BF16 quantization
                v_pool[dst] = __float2bfloat16(V_log[src]);
            }
        }
    }
}

struct CaseResult { double max_abs_err; bool ok; };

static CaseResult run_case(const char* name, bool decode,
                           int seq_len, int q_heads, int kv_heads, int head_dim,
                           float epsilon, unsigned seed)
{
    const int num_q   = decode ? 1 : seq_len;                 // decode: single query token
    const int q_start = decode ? (seq_len - 1) : 0;
    const int num_blocks = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;

    // Scattered physical pages -> exercise the block-table indirection.
    const int32_t scatter[] = {5, 2, 7, 0, 4, 1, 6, 3};
    const int total_pages = (int)(sizeof(scatter) / sizeof(scatter[0]));
    if (num_blocks > total_pages) { std::fprintf(stderr, "seq too long for arena\n"); std::exit(2); }
    std::vector<int32_t> block_table(scatter, scatter + num_blocks);

    // Deterministic inputs. Ranges chosen so softmax is selective enough that a
    // mis-gathered page changes the result beyond BF16 noise.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dq(-0.8f, 0.8f), dk(-0.8f, 0.8f), dv(-1.0f, 1.0f);
    std::vector<float> Q((size_t)num_q * q_heads * head_dim);
    std::vector<float> K_log((size_t)seq_len * kv_heads * head_dim);
    std::vector<float> V_log((size_t)seq_len * kv_heads * head_dim);
    for (auto& x : Q)     x = dq(rng);
    for (auto& x : K_log) x = dk(rng);
    for (auto& x : V_log) x = dv(rng);

    // CPU oracle.
    std::vector<float> O_ref;
    cpu_paged_attention_reference(Q, K_log, V_log, O_ref, num_q, q_start, seq_len,
                                  q_heads, kv_heads, head_dim);

    // BF16 page arena (single layer is sufficient for the attention kernel).
    const size_t per_page = (size_t)kv_heads * PAGE_SIZE * head_dim;
    std::vector<kv_t> hk((size_t)total_pages * per_page, __float2bfloat16(0.0f));
    std::vector<kv_t> hv((size_t)total_pages * per_page, __float2bfloat16(0.0f));
    scatter_into_pages(K_log, V_log, block_table, seq_len, kv_heads, head_dim, hk, hv);

    // Device buffers.
    float *dQ = nullptr, *dO = nullptr; kv_t *dK = nullptr, *dV = nullptr; int32_t* dBT = nullptr;
    CK(cudaMalloc(&dQ, Q.size() * sizeof(float)));
    CK(cudaMalloc(&dO, O_ref.size() * sizeof(float)));
    CK(cudaMalloc(&dK, hk.size() * sizeof(kv_t)));
    CK(cudaMalloc(&dV, hv.size() * sizeof(kv_t)));
    CK(cudaMalloc(&dBT, block_table.size() * sizeof(int32_t)));
    CK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hk.data(), hk.size() * sizeof(kv_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hv.data(), hv.size() * sizeof(kv_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBT, block_table.data(), block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemset(dO, 0xFF, O_ref.size() * sizeof(float)));   // poison: catch unwritten outputs

    if (decode)
        launch_paged_flash_attention_decode(dQ, dK, dV, dO, dBT, seq_len,
                                            q_heads, kv_heads, head_dim);
    else
        launch_paged_flash_attention_prefill(dQ, dK, dV, dO, dBT, seq_len, num_q,
                                             q_heads, kv_heads, head_dim);

    CK(cudaGetLastError());        // launch config / illegal launch
    CK(cudaDeviceSynchronize());   // execution faults (bad smem access, etc.)

    std::vector<float> O_gpu(O_ref.size());
    CK(cudaMemcpy(O_gpu.data(), dO, O_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0; size_t arg = 0;
    for (size_t i = 0; i < O_ref.size(); ++i) {
        double e = std::fabs((double)O_gpu[i] - (double)O_ref[i]);
        if (e > max_err) { max_err = e; arg = i; }
    }

    cudaFree(dQ); cudaFree(dO); cudaFree(dK); cudaFree(dV); cudaFree(dBT);

    const bool ok = (max_err < epsilon) && std::isfinite(max_err);
    std::printf("[%-8s] seq_len=%-3d q_heads=%d kv_heads=%d head_dim=%d  "
                "max|err|=%.3e (eps=%.1e)  @elem %zu  -> %s\n",
                name, seq_len, q_heads, kv_heads, head_dim, max_err, epsilon, arg,
                ok ? "PASS" : "FAIL");
    return { max_err, ok };
}

int main() {
    int dev = 0;
    if (cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return 2;
    }
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Epsilon is generous: BF16 K/V/Q quantization (~2^-8 relative) compounded
    // over a head_dim=128 reduction plus Tensor-Core accumulation order, then a
    // final BF16 truncation of the output. The printed max|err| is the real signal.
    const float eps = 5e-2f;
    bool all_ok = true;

    // Decode: single query at a non-page-aligned length (last page tail-masked).
    all_ok &= run_case("decode",  /*decode=*/true,  /*seq_len=*/37,
                       /*q_heads=*/4, /*kv_heads=*/2, /*head_dim=*/128, eps, 1234).ok;
    // GQA stress on the decode path (ratio 4).
    all_ok &= run_case("decode",  true,  /*seq_len=*/45,
                       /*q_heads=*/8, /*kv_heads=*/2, /*head_dim=*/128, eps, 5678).ok;
    // Prefill: full Tensor-Core tiles, multi-tile, causal across scattered pages.
    all_ok &= run_case("prefill", /*decode=*/false, /*seq_len=*/40,
                       /*q_heads=*/4, /*kv_heads=*/2, /*head_dim=*/128, eps, 9012).ok;
    // Prefill at a smaller head_dim (exercises n_dtiles != 8).
    all_ok &= run_case("prefill", false, /*seq_len=*/33,
                       /*q_heads=*/4, /*kv_heads=*/4, /*head_dim=*/64,  eps, 3456).ok;

    std::printf("\n%s\n", all_ok ? "ALL CASES PASSED" : "FAILURES DETECTED");
    return all_ok ? 0 : 1;
}
