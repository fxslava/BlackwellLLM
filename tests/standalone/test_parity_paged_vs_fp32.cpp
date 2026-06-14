// ============================================================================
// Dual-Oracle parity test: Paged Flash Attention (BF16, GPU) vs the legacy
// continuous-cache FP32 math, at the real Qwen-2.5-14B / Llama-3-8B GQA shapes.
// ============================================================================
// We do NOT overwrite the FP32 goldens; they stay the absolute source of truth.
// Two references bracket the GPU kernel:
//
//   Oracle 1 (FP32 GOLDEN)  : pure fp64 continuous attention, no bf16 anywhere.
//                             The absolute truth -> measures total quant DRIFT.
//   Oracle 2 (BF16 CPU REF) : mirrors EVERY quantization point the kernel hits
//                             (Q,K,V -> bf16, P -> bf16, output -> bf16) with
//                             otherwise-exact fp64 arithmetic. "Mathematically
//                             perfect bf16 execution" -> isolates memory routing
//                             / GQA / block-table correctness from accumulation.
//
//   STRICT : GPU vs Oracle 2, tight epsilon. Residual is ONLY tensor-core fp32
//            accumulation order + the final bf16-output-grid straddle.
//   DRIFT  : GPU vs Oracle 1, loose epsilon, logged. Tracks quantization budget.
//
// THE 1e-4 QUESTION: the kernel's epilogue applies trunc_bf16 to the output
// (parity with the old engine). Comparing any two bf16-grid values floors the
// achievable strict error at ~1 bf16 ULP (~1e-3 here), so 1e-4 is unreachable
// by construction, not by bug. What makes the strict bound meaningful is the
// SEPARATION: a routing bug reads wrong operands and errs by ~full magnitude
// (~250 ULP). The NEGATIVE CONTROL below corrupts the block table and asserts
// the error explodes past the strict bound, proving it has teeth.
//
// Build (links the real kernel .cu):
//   nvcc -std=c++17 -arch=sm_120 ^
//       tests/standalone/test_parity_paged_vs_fp32.cpp ^
//       src/kernels/paged_flash_attention.cu -o test_parity.exe
//   ./test_parity.exe
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
    do { cudaError_t _e = (call); if (_e != cudaSuccess) {                     \
        std::fprintf(stderr, "[CUDA] %s\n  at %s:%d\n",                        \
                     cudaGetErrorString(_e), __FILE__, __LINE__);              \
        std::exit(2); } } while (0)

// Round-trip a float through bf16 (the exact storage/operand quantization).
static inline float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }

// Tolerances (see header note for why STRICT is ~1 bf16 ULP, not 1e-4).
static constexpr float STRICT_EPS = 4e-3f;   // memory routing / GQA correctness
static constexpr float DRIFT_EPS  = 5e-2f;   // FP32->BF16 quantization budget

// ---------------------------------------------------------------------------
// Oracle 1: pure FP32/FP64 continuous attention (the absolute golden truth).
// Decode: a single query at logical position pos = seq_len-1.
// ---------------------------------------------------------------------------
static void oracle_fp32(const std::vector<float>& Q, const std::vector<float>& K,
                        const std::vector<float>& V, std::vector<float>& O,
                        int seq_len, int q_heads, int kv_heads, int head_dim) {
    const double scale = 1.0 / std::sqrt((double)head_dim);
    const int gqa = q_heads / kv_heads;
    O.assign((size_t)q_heads * head_dim, 0.0f);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / gqa;
        const float* q = &Q[(size_t)qh * head_dim];
        std::vector<double> s(seq_len);
        double mx = -1e300;
        for (int t = 0; t < seq_len; ++t) {
            const float* k = &K[((size_t)t * kv_heads + kvh) * head_dim];
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) dot += (double)q[d] * (double)k[d];
            s[t] = dot * scale; mx = std::max(mx, s[t]);
        }
        double l = 0.0;
        for (int t = 0; t < seq_len; ++t) { s[t] = std::exp(s[t] - mx); l += s[t]; }
        float* o = &O[(size_t)qh * head_dim];
        for (int t = 0; t < seq_len; ++t) {
            const double p = s[t] / l;
            const float* v = &V[((size_t)t * kv_heads + kvh) * head_dim];
            for (int d = 0; d < head_dim; ++d) o[d] += (float)(p * (double)v[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Oracle 2: mathematically perfect BF16 execution, mirroring the kernel's
// quantization points exactly (Q,K,V -> bf16, P -> bf16, output -> bf16; l from
// the fp p, fp64 accumulation everywhere else).
// ---------------------------------------------------------------------------
static void oracle_bf16(const std::vector<float>& Q, const std::vector<float>& K,
                        const std::vector<float>& V, std::vector<float>& O,
                        int seq_len, int q_heads, int kv_heads, int head_dim) {
    const double scale = 1.0 / std::sqrt((double)head_dim);
    const int gqa = q_heads / kv_heads;
    O.assign((size_t)q_heads * head_dim, 0.0f);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / gqa;
        const float* q = &Q[(size_t)qh * head_dim];
        std::vector<double> s(seq_len);
        double mx = -1e300;
        for (int t = 0; t < seq_len; ++t) {
            const float* k = &K[((size_t)t * kv_heads + kvh) * head_dim];
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d)
                dot += (double)bf16r(q[d]) * (double)bf16r(k[d]);   // bf16 Q,K
            s[t] = dot * scale; mx = std::max(mx, s[t]);
        }
        std::vector<double> p(seq_len); double l = 0.0;
        for (int t = 0; t < seq_len; ++t) { p[t] = std::exp(s[t] - mx); l += p[t]; }
        float* o = &O[(size_t)qh * head_dim];
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (int t = 0; t < seq_len; ++t) {
                const float* v = &V[((size_t)t * kv_heads + kvh) * head_dim];
                acc += (double)bf16r((float)p[t]) * (double)bf16r(v[d]);   // bf16 P, bf16 V
            }
            o[d] = bf16r((float)(acc / l));   // bf16 output (kernel's trunc_bf16)
        }
    }
}

// Scatter logical token KV into the bf16 physical-page arena via a block table.
static void scatter_into_pages(const std::vector<float>& K, const std::vector<float>& V,
                               const std::vector<int32_t>& block_table, int seq_len,
                               int kv_heads, int head_dim,
                               std::vector<kv_t>& kp, std::vector<kv_t>& vp) {
    for (int t = 0; t < seq_len; ++t) {
        const int page = block_table[t / PAGE_SIZE], slot = t % PAGE_SIZE;
        for (int h = 0; h < kv_heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
                const size_t dst = (((size_t)page * kv_heads + h) * PAGE_SIZE + slot) * head_dim + d;
                const size_t src = ((size_t)t * kv_heads + h) * head_dim + d;
                kp[dst] = __float2bfloat16(K[src]);
                vp[dst] = __float2bfloat16(V[src]);
            }
    }
}

static double max_abs(const std::vector<float>& a, const std::vector<float>& b, double* mean = nullptr) {
    double mx = 0.0, sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double e = std::fabs((double)a[i] - (double)b[i]); mx = std::max(mx, e); sum += e; }
    if (mean) *mean = sum / a.size();
    return mx;
}

static bool run_model(const char* name, int q_heads, int kv_heads, int head_dim,
                      int seq_len, unsigned seed) {
    std::printf("== %s : q_heads=%d kv_heads=%d (gqa=%d) head_dim=%d seq_len=%d ==\n",
                name, q_heads, kv_heads, q_heads / kv_heads, head_dim, seq_len);

    // Deterministic inputs.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> du(-1.0f, 1.0f);
    std::vector<float> Q((size_t)q_heads * head_dim);
    std::vector<float> K((size_t)seq_len * kv_heads * head_dim);
    std::vector<float> V((size_t)seq_len * kv_heads * head_dim);
    for (auto& x : Q) x = du(rng);
    for (auto& x : K) x = du(rng);
    for (auto& x : V) x = du(rng);

    // Both oracles.
    std::vector<float> O_fp32, O_bf16;
    oracle_fp32(Q, K, V, O_fp32, seq_len, q_heads, kv_heads, head_dim);
    oracle_bf16(Q, K, V, O_bf16, seq_len, q_heads, kv_heads, head_dim);

    // Scattered physical pages -> exercise the block-table indirection.
    const int32_t scatter[] = {7,3,11,1,9,5,13,0,14,2,10,4,12,6,15,8};
    const int total_pages = (int)(sizeof(scatter)/sizeof(scatter[0]));
    const int num_blocks = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
    if (num_blocks > total_pages) { std::fprintf(stderr, "seq too long\n"); std::exit(2); }
    std::vector<int32_t> bt(scatter, scatter + num_blocks);

    const size_t per_page = (size_t)kv_heads * PAGE_SIZE * head_dim;
    std::vector<kv_t> hk((size_t)total_pages * per_page, __float2bfloat16(0.0f));
    std::vector<kv_t> hv((size_t)total_pages * per_page, __float2bfloat16(0.0f));
    scatter_into_pages(K, V, bt, seq_len, kv_heads, head_dim, hk, hv);

    // Device buffers.
    float *dQ = nullptr, *dO = nullptr; kv_t *dK = nullptr, *dV = nullptr; int32_t* dBT = nullptr;
    CK(cudaMalloc(&dQ, Q.size() * sizeof(float)));
    CK(cudaMalloc(&dO, O_fp32.size() * sizeof(float)));
    CK(cudaMalloc(&dK, hk.size() * sizeof(kv_t)));
    CK(cudaMalloc(&dV, hv.size() * sizeof(kv_t)));
    CK(cudaMalloc(&dBT, bt.size() * sizeof(int32_t)));
    CK(cudaMemcpy(dQ, Q.data(), Q.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hk.data(), hk.size()*sizeof(kv_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hv.data(), hv.size()*sizeof(kv_t), cudaMemcpyHostToDevice));

    auto launch_and_fetch = [&](const std::vector<int32_t>& block_table, std::vector<float>& out) {
        CK(cudaMemcpy(dBT, block_table.data(), block_table.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemset(dO, 0xFF, O_fp32.size()*sizeof(float)));   // poison
        launch_paged_flash_attention_decode(dQ, dK, dV, dO, dBT, seq_len, q_heads, kv_heads, head_dim);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        out.resize(O_fp32.size());
        CK(cudaMemcpy(out.data(), dO, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
    };

    // --- correct routing ---
    std::vector<float> O_gpu;
    launch_and_fetch(bt, O_gpu);

    double strict_mean = 0.0;
    const double strict = max_abs(O_gpu, O_bf16, &strict_mean);   // GPU vs Oracle 2
    const double drift  = max_abs(O_gpu, O_fp32);                 // GPU vs Oracle 1

    const bool strict_ok = (strict < STRICT_EPS) && std::isfinite(strict);
    const bool drift_ok  = (drift  < DRIFT_EPS)  && std::isfinite(drift);
    std::printf("  STRICT  GPU vs bf16-ref : max|err|=%.3e mean=%.3e  (eps=%.1e)  -> %s\n",
                strict, strict_mean, (double)STRICT_EPS, strict_ok ? "PASS" : "FAIL");
    std::printf("  DRIFT   GPU vs fp32-gold: max|err|=%.3e             (eps=%.1e)  -> %s\n",
                drift, (double)DRIFT_EPS, drift_ok ? "PASS" : "FAIL");

    // --- negative control: a stale/wrong page id ---
    // Proves the STRICT bound catches a memory-routing bug despite being ~1 ULP,
    // not 1e-4. NOTE: simply SWAPPING two block-table entries is a no-op for
    // decode -- the query attends to every position, so the output is a sum over
    // the whole (K,V) set and is permutation-invariant. A real bug must change
    // the MULTISET of attended data, so we repoint logical block 0 at an unused
    // zero-filled page (an uninitialized/stale block-table entry).
    bool teeth_ok = true;
    if (num_blocks >= 1 && num_blocks < total_pages) {
        std::vector<int32_t> bad = bt;
        bad[0] = scatter[num_blocks];        // an unused, zero-filled physical page
        std::vector<float> O_bad; launch_and_fetch(bad, O_bad);
        const double bad_err = max_abs(O_bad, O_bf16);
        teeth_ok = bad_err > STRICT_EPS;
        std::printf("  CONTROL stale page id (block 0 -> unused page): max|err|=%.3e  (> eps -> bug caught? %s)\n",
                    bad_err, teeth_ok ? "YES" : "NO");
    }

    cudaFree(dQ); cudaFree(dO); cudaFree(dK); cudaFree(dV); cudaFree(dBT);
    std::printf("\n");
    return strict_ok && drift_ok && teeth_ok;
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) { std::fprintf(stderr, "No CUDA device.\n"); return 2; }
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    bool ok = true;
    ok &= run_model("Qwen-2.5-14B", /*q=*/40, /*kv=*/8, /*hd=*/128, /*seq=*/70, 0xC0FFEE);
    ok &= run_model("Llama-3-8B",   /*q=*/32, /*kv=*/8, /*hd=*/128, /*seq=*/66, 0xBADF00D);

    std::printf("%s\n", ok ? "ALL PARITY CHECKS PASSED" : "PARITY FAILURES DETECTED");
    return ok ? 0 : 1;
}
