// Engine-level smoke test for the Paged KV-cache strategy: initializes
// BlackwellEngine in Paged mode, drives a real multi-token decode through the
// paged-flash-attention path, then exercises the branch API (fork / rewind /
// CoW-on-next-write) -- asserting the whole flow runs without throwing.
//
// This is a liveness / integration check (the kernel MATH is validated
// separately by tests/standalone/test_parity_paged_vs_fp32.cpp); the bar here is
// "no exceptions, no CUDA faults" through the engine seam.
//
// SKIPs if the model checkpoint is absent (default Qwen2.5-Coder-7B-AWQ; override
// with BLACKWELL_QWEN_INDEX).
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#include "blackwell/engine.h"

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

}  // namespace

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
    ASSERT_NO_THROW({
        for (int pos = 0; pos < 4; ++pos) {
            tok = engine.forward(tok, pos);
        }
    });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during paged decode";

    // --- branch: fork sequence 0 -> 1 (CoW share, no data copy yet) ---
    EXPECT_NO_THROW(engine.fork(/*parent=*/0, /*child=*/1));

    // --- rewind sequence 0 back to 2 tokens (drops the trailing page if owned) ---
    EXPECT_NO_THROW(engine.rewind(/*seq=*/0, /*pos=*/2));

    // --- continue decoding seq 0 from pos 2: the next write hits a fork-shared
    //     partial page, triggering lazy Copy-on-Write inside the engine. ---
    ASSERT_NO_THROW({ engine.forward(tok, /*pos=*/2); });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during post-fork CoW decode";

    SUCCEED() << "Paged decode + fork + rewind + CoW completed without exceptions.";
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
    ASSERT_NO_THROW({ for (; pos0 < 4; ++pos0) t0 = engine.forward(t0, pos0); });

    // Branch at length 4; branch 1 shares seq 0's KV pages (CoW).
    ASSERT_NO_THROW(engine.fork(/*parent=*/0, /*child=*/1));

    // Interleave decode of BOTH branches via seq_id -- the new plumbing. Branch 1
    // must be decodable independently; its first write CoWs the shared page.
    int t1 = t0, pos1 = 4;
    ASSERT_NO_THROW({
        for (int s = 0; s < 3; ++s) {
            t1 = engine.forward(t1, pos1++, 0.6f, 0.9f, /*seq_id=*/1);
            t0 = engine.forward(t0, pos0++, 0.6f, 0.9f, /*seq_id=*/0);
        }
    });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during multi-branch decode";

    // An unknown sequence id must fail cleanly (clear throw, not a CUDA fault).
    EXPECT_THROW(engine.forward(t0, pos0, 0.6f, 0.9f, /*seq_id=*/42), std::runtime_error);

    SUCCEED() << "Forked branch decoded independently via seq_id.";
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
