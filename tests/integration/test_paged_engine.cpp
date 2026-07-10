// Engine-level smoke test for the Paged KV-cache strategy: initializes
// BlackwellEngine in Paged mode, drives a real multi-token decode through the
// paged-flash-attention path, then exercises the branch API (fork / rewind /
// CoW-on-next-write) -- asserting the whole flow reports Success statuses.
//
// This is a liveness / integration check (the kernel MATH is validated
// separately by tests/standalone/test_parity_paged_vs_fp32.cpp); the bar here is
// "EngineStatus::Success everywhere, no CUDA faults" through the engine seam
// (decode is status-tier per the Hybrid error doctrine; the fork/rewind admin
// API stays exception-tier).
//
// SKIPs if the model checkpoint is absent (default Qwen2.5-Coder-7B-AWQ; override
// with BLACKWELL_QWEN_INDEX).
#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "blackwell/engine.h"
#include "engine_impl.h"                     // white-box: read d_logits / vocab
#include "engine_prefill_coordinator.h"      // prefill_driver().prefill_prompt

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

std::string qwen_index_path() {
    return env_or("BLACKWELL_QWEN_INDEX",
                  "F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ/model.safetensors.index.json");
}

bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

constexpr int kQwenBos = 151643;   // <|endoftext|>, a safe seed token

// One status-tier decode step; any non-Success is a test failure. Returns the
// sampled token (-1 on failure, keeping the driver loops finite).
int fwd(BlackwellEngine& e, int tok, int pos, float temp = 0.6f, float top_p = 0.9f,
        int seq = 0) {
    int next = -1;
    const auto st = e.forward_status(tok, pos, temp, top_p, seq, &next);
    EXPECT_EQ(st, blackwell::EngineStatus::Success)
        << "forward_status(pos=" << pos << ", seq=" << seq
        << "): " << blackwell::to_string(st);
    return next;
}

// Copy the engine's current logits (whatever the last forward / prefill left in
// d_logits) to host. White-box: reaches through get_impl() into the OBJECT lib.
std::vector<float> snapshot_logits(BlackwellEngine& e) {
    auto* impl = e.get_impl();
    const size_t V = impl->m_config.vocab_size;
    std::vector<float> h(V);
    EXPECT_EQ(cudaMemcpy(h.data(), static_cast<const float*>(impl->d_logits),
                         V * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return h;
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

// Relative Frobenius distance between two logit vectors.
float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return (float)std::sqrt(num / std::max(den, 1e-30));
}

// A deterministic, reproducible pseudo-prompt of valid token ids (no tokenizer
// dependency): correctness of the batched vs single-token math is token-agnostic.
std::vector<int> make_prompt(int n, int vocab) {
    std::vector<int> t(n);
    // Seed off n so different-length prompts share no common prefix (else the
    // second run's coordinator would prefix-hit the first run's committed pages).
    uint32_t s = 0x9E3779B9u ^ (uint32_t)(n * 2654435761u);
    t[0] = kQwenBos;
    for (int i = 1; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        t[i] = 1 + (int)(s % (uint32_t)(vocab - 1));
    }
    return t;
}

}  // namespace

// The heart of Step 4: batched prefill must reproduce the old single-token
// prefill loop's logits. Path A drives the prompt token-by-token through
// forward_status (run_token); Path B runs the whole prompt through the prefix-
// cache coordinator, which now sweeps it in batched Tensor-Core chunks
// (run_chunk). The final-token logit distributions must agree.
TEST(PagedEngineIntegration, BatchedPrefillMatchesSingleTokenLoop) {
    if (!file_exists(qwen_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << qwen_index_path();
    }
    // max_seq_len 256 with token_capacity 64 => a >64-token prompt exercises the
    // multi-tile path in run_delta as well as a single-chunk one.
    BlackwellEngine engine(qwen_index_path(), /*max_seq_len=*/256,
                           /*num_gpu_layers=*/static_cast<size_t>(-1),
                           BlackwellEngine::KVCacheMode::Paged);
    ASSERT_TRUE(engine.has_prefix_cache())
        << "dense paged model must expose the prefill coordinator";

    const int vocab = (int)engine.get_impl()->m_config.vocab_size;

    for (int n : {20, 80}) {   // one chunk, then two tiles (64 + 16)
        const std::vector<int> prompt = make_prompt(n, vocab);

        // --- Path A: single-token decode loop over the whole prompt (seq 0). ---
        int next = -1;
        for (int pos = 0; pos < n; ++pos) {
            const auto st = engine.forward_status(prompt[pos], pos, 0.6f, 0.9f, 0, &next);
            ASSERT_EQ(st, blackwell::EngineStatus::Success) << "single-token pos " << pos;
        }
        const std::vector<float> single = snapshot_logits(engine);

        // --- Path B: batched prefill through the coordinator (run_chunk tiles). ---
        auto& driver = engine.prefill_driver();
        auto r = driver.prefill_prompt(prompt.data(), n);
        ASSERT_EQ(r.status, blackwell::EngineStatus::Success);
        ASSERT_NE(r.engine_seq, -1);
        ASSERT_EQ(r.cached_tokens, 0) << "empty tree: nothing should be prefix-cached";
        ASSERT_EQ(r.computed_tokens, n);
        const std::vector<float> batched = snapshot_logits(engine);
        driver.finish(r.engine_seq);

        // The two paths share the AWQ per-row GEMV projections and the same paged
        // KV / RoPE kernels, so the last-token logits agree to reduced-precision
        // rounding: same greedy prediction, tiny relative L2.
        const float rl2 = rel_l2(batched, single);
        EXPECT_LE(rl2, 2e-2f) << "n=" << n << " batched-vs-single logits rel_l2=" << rl2;
        EXPECT_EQ(argmax(batched), argmax(single))
            << "n=" << n << " greedy next-token disagrees (rel_l2=" << rl2 << ")";
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    SUCCEED() << "Batched prefill logits match the single-token loop.";
}

TEST(PagedEngineIntegration, ForwardForkRewindNoThrow) {
    if (!file_exists(qwen_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << qwen_index_path()
                     << " (set BLACKWELL_QWEN_INDEX to override)";
    }

    // Small context keeps both the (still-allocated) FP32 cache and the paged
    // pool tiny; weights dominate VRAM.
    BlackwellEngine engine(qwen_index_path(), /*max_seq_len=*/128,
                           /*num_gpu_layers=*/static_cast<size_t>(-1),
                           BlackwellEngine::KVCacheMode::Paged);

    // --- dummy forward pass: a few sequential decode steps on sequence 0 ---
    int tok = kQwenBos;
    for (int pos = 0; pos < 4; ++pos) tok = fwd(engine, tok, pos);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during paged decode";

    // --- branch: fork sequence 0 -> 1 (CoW share, no data copy yet) ---
    EXPECT_NO_THROW(engine.fork(/*parent=*/0, /*child=*/1));

    // --- rewind sequence 0 back to 2 tokens (drops the trailing page if owned) ---
    EXPECT_NO_THROW(engine.rewind(/*seq=*/0, /*pos=*/2));

    // --- continue decoding seq 0 from pos 2: the next write hits a fork-shared
    //     partial page, triggering lazy Copy-on-Write inside the engine. ---
    (void)fwd(engine, tok, /*pos=*/2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during post-fork CoW decode";

    SUCCEED() << "Paged decode + fork + rewind + CoW completed with Success statuses.";
}

TEST(PagedEngineIntegration, ForkedBranchIsDecodable) {
    if (!file_exists(qwen_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << qwen_index_path();
    }
    BlackwellEngine engine(qwen_index_path(), /*max_seq_len=*/128,
                           /*num_gpu_layers=*/static_cast<size_t>(-1),
                           BlackwellEngine::KVCacheMode::Paged);

    // Decode 4 tokens on sequence 0.
    int t0 = kQwenBos, pos0 = 0;
    for (; pos0 < 4; ++pos0) t0 = fwd(engine, t0, pos0);

    // Branch at length 4; branch 1 shares seq 0's KV pages (CoW).
    ASSERT_NO_THROW(engine.fork(/*parent=*/0, /*child=*/1));

    // Interleave decode of BOTH branches via seq_id -- the new plumbing. Branch 1
    // must be decodable independently; its first write CoWs the shared page.
    int t1 = t0, pos1 = 4;
    for (int s = 0; s < 3; ++s) {
        t1 = fwd(engine, t1, pos1++, 0.6f, 0.9f, /*seq_id=*/1);
        t0 = fwd(engine, t0, pos0++, 0.6f, 0.9f, /*seq_id=*/0);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during multi-branch decode";

    // An unknown sequence id must fail cleanly (an error status, not a CUDA
    // fault and not an unwind -- forward_status is noexcept).
    int nx = -1;
    EXPECT_NE(engine.forward_status(t0, pos0, 0.6f, 0.9f, /*seq_id=*/42, &nx),
              blackwell::EngineStatus::Success);

    SUCCEED() << "Forked branch decoded independently via seq_id.";
}

// Regression: forward(token, pos) is position-addressed, so re-decoding an
// existing position (pos < current length) must reconcile (rewind+append), NOT
// throw "pos out of sync". This is the pattern the playground adapter hits via
// incremental KV reuse / eos re-feed across two requests.
TEST(PagedEngineIntegration, RedecodeExistingPositionReconciles) {
    if (!file_exists(qwen_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << qwen_index_path();
    }
    BlackwellEngine engine(qwen_index_path(), /*max_seq_len=*/128,
                           /*num_gpu_layers=*/static_cast<size_t>(-1),
                           BlackwellEngine::KVCacheMode::Paged);

    int t = kQwenBos;
    for (int pos = 0; pos < 5; ++pos) t = fwd(engine, t, pos);  // length -> 5

    // Re-decode position 4 (pos < length 5): the eos-refeed / replay pattern.
    t = fwd(engine, t, /*pos=*/4);
    // Reset-style replay from position 0, then continue.
    t = fwd(engine, kQwenBos, /*pos=*/0);
    t = fwd(engine, t, /*pos=*/1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during position-addressed redecode";

    // A forward gap (pos beyond the end) is still a clean error status, not a fault.
    int nx = -1;
    EXPECT_NE(engine.forward_status(t, /*pos=*/50, 0.6f, 0.9f, 0, &nx),
              blackwell::EngineStatus::Success);
    SUCCEED();
}

TEST(PagedEngineIntegration, ContinuousModeRejectsBranching) {
    if (!file_exists(qwen_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << qwen_index_path();
    }
    // Default (Continuous) strategy must reject branching, not silently no-op.
    BlackwellEngine engine(qwen_index_path(), /*max_seq_len=*/128);
    EXPECT_THROW(engine.fork(0, 1), std::runtime_error);
    EXPECT_THROW(engine.rewind(0, 0), std::runtime_error);
}
