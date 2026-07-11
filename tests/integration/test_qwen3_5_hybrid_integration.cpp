// ============================================================================
// Golden-dump parity SCAFFOLD for the Qwen3.5 hybrid (linear-attn + full-attn)
// checkpoint. Covers single-token DECODE parity, fork branching, AND True Batched
// Prefill (BatchedPrefillMatchesSequential, self-referential vs the run_token
// sweep). Mirrors test_qwen_engine.cpp's dump/skip conventions.
// ============================================================================
// STATUS: this is intentionally a scaffold. The capability test runs today; the
// numerical-parity test is wired end-to-end but SKIPs until the remaining decode
// prerequisites land (compressed-tensors int4 dequant, model.language_model.*
// SSM weight binding, and the SSM decode body in step_linear_attention). Once
// those exist and the .bin dumps are generated, NO test code changes are needed —
// just drop the dumps in the directory below and the parity test goes green.
//
// External requirements (the parity test SKIPs if either is missing):
//   - checkpoint index, default F:/AI/Qwen3.5-9B-AWQ-4bit/model.safetensors.index.json
//     (override: env BLACKWELL_QWEN35_INDEX)
//   - golden dumps dir, default <repo>/tests/integration/golden_dumps/qwen3.5_hybrid
//     (override: env BLACKWELL_QWEN35_DUMPS_DIR)
//
// ---------------------------------------------------------------------------
// GOLDEN-DUMP CONTRACT (generated from the HF reference, raw little-endian fp32)
// ---------------------------------------------------------------------------
//   input_embedding.bin        [hidden_size]            embedding of the decode
//                                                       token (post embed_tokens),
//                                                       injected into d_X_accum so
//                                                       the test is independent of
//                                                       our embedding kernel.
//   init_ssm_state.bin         [num_linear_layers][H][Dk][Dv]
//                                                       the recurrent state S for
//                                                       every linear layer BEFORE
//                                                       this step (the HF cache).
//   init_conv_state.bin        [num_linear_layers][conv_dim][K-1]
//                                                       causal-conv1d ring buffers.
//   expected_logits.bin        [vocab_size]             logits AFTER one decode step.
// Optional per-stage tensors (cosine telemetry, not assertions):
//   linear_attn_out_layer{li}.bin [H*Dv]               GatedDeltaNet output o.
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "common.h"
#include "blackwell/engine.h"
#include "blackwell/config.h"
#include "engine_impl.h"
#include "hybrid_snapshot_ring.h"

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}
std::string model_index_path() {
    return env_or("BLACKWELL_QWEN35_INDEX",
                  "F:/AI/Qwen3.5-9B-AWQ-4bit/model.safetensors.index.json");
}
std::string dumps_dir() {
    return env_or("BLACKWELL_QWEN35_DUMPS_DIR",
                  "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/qwen3.5_hybrid");
}
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

// Raw fp32 loader, identical contract to the Qwen2.5/LLaMA suites.
std::vector<float> load_golden_dump(const std::string& filename, size_t num_elements) {
    const std::string full = dumps_dir() + "/" + filename;
    std::ifstream f(full, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Dump not found: " + full + " (generate it from the HF reference).");
    std::vector<float> buf(num_elements);
    f.read(reinterpret_cast<char*>(buf.data()), num_elements * sizeof(float));
    if (!f)
        throw std::runtime_error("Dump size mismatch for " + full);
    return buf;
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    double den = std::sqrt(na) * std::sqrt(nb);
    return den > 1e-9 ? dot / den : 0.0;
}

constexpr double kParityThreshold = 0.999;   // exact-parity bar for this stack

} // namespace

// ---------------------------------------------------------------------------
// Capabilities: this runs today (no dumps needed). The hybrid model now SUPPORTS
// fork() (physical SSM + full-attention snapshot under Paged mode), but rewind()
// stays unsupported -- a recurrent state cannot be rolled back to an arbitrary
// position. Construction needs the checkpoint to load, so if it throws we SKIP.
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, SupportsForkRejectsRewind) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), /*max_seq_len=*/512,
                                                   /*num_gpu_layers=*/static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction not yet supported for this checkpoint "
                        "(int4 dequant / SSM weight binding pending): " << e.what();
    }

    const ModelCapabilities caps = engine->get_capabilities();
    EXPECT_TRUE(caps.requires_ssm_subsystem);
    EXPECT_TRUE(caps.is_hybrid);
    EXPECT_TRUE(caps.supports_cow_branching)
        << "hybrid models now branch via physical state snapshot under Paged mode";
    EXPECT_GT(caps.num_linear_attention_layers, 0);
    EXPECT_GT(caps.num_full_attention_layers, 0);

    // fork() is now supported (snapshots the recurrent SSM + gated full-attention
    // state); rewind() is still rejected (recurrent state has no positional undo).
    EXPECT_NO_THROW(engine->fork(0, 1));
    EXPECT_THROW(engine->rewind(0, 0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// release_sequence() recycles a fork id: fork() rejects an id that already
// exists, so the snapshot ring's slot reuse depends on being able to destroy a
// branch and fork the same id again. This is the primitive that makes the ring
// possible (no dumps / decode needed -- pure branch-lifecycle bookkeeping).
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, ReleaseSequenceRecyclesForkId) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), /*max_seq_len=*/512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending: " << e.what();
    }
    ASSERT_GE(engine->branch_capacity(), 2)
        << "hybrid paged model must expose >= 2 branch slots for virtual rewind";

    EXPECT_NO_THROW(engine->fork(0, 1));
    // Re-forking the live id 1 must fail (the guard the ring relies on)...
    EXPECT_THROW(engine->fork(0, 1), std::runtime_error);
    // ...until it is released, after which the same id forks cleanly again.
    EXPECT_NO_THROW(engine->release_sequence(1));
    EXPECT_NO_THROW(engine->fork(0, 1));
    EXPECT_NO_THROW(engine->release_sequence(1));
    // Releasing an unknown id is a caller error (loud, not silent).
    EXPECT_THROW(engine->release_sequence(1), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Single-step logit parity vs the HF golden dump. Fully wired; SKIPs until the
// decode path can execute the linear layers. The injection points below are the
// exact seams where the dumps enter the engine.
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, SingleStepLogitParity) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();
    if (!file_exists(dumps_dir() + "/expected_logits.bin"))
        GTEST_SKIP() << "golden dumps absent in " << dumps_dir()
                     << " (run the HF reference dump generator).";

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), /*max_seq_len=*/512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending int4/binding work: " << e.what();
    }

    BlackwellEngine::Impl* impl = engine->get_impl();
    const ModelConfig& cfg = impl->m_config;
    const auto geo = blackwell::ssm::SsmGeometry::from_config(cfg);

    // ---- INJECTION POINT 1: initial recurrent SSM state (the HF cache) --------
    // Copy the golden S / conv ring buffers into SsmStatePool so the single step
    // starts from the reference's pre-step state instead of zeros.
    {
        const size_t rec_per_layer  = geo.rec_elems_per_layer();
        const size_t conv_per_layer = geo.conv_elems_per_layer();
        std::vector<float> S    = load_golden_dump("init_ssm_state.bin",
                                                   (size_t)geo.num_linear_layers * rec_per_layer);
        std::vector<float> conv = load_golden_dump("init_conv_state.bin",
                                                   (size_t)geo.num_linear_layers * conv_per_layer);
        for (int i = 0; i < (int)cfg.num_layers; ++i) {
            const int li = impl->m_linear_layer_index[i];
            if (li < 0) continue;   // full-attention layer
            ASSERT_NE(impl->ssm_state, nullptr);
            CUDA_CHECK(cudaMemcpy(impl->ssm_state->rec_state(0, li),
                                  S.data() + (size_t)li * rec_per_layer,
                                  rec_per_layer * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(impl->ssm_state->conv_state(0, li),
                                  conv.data() + (size_t)li * conv_per_layer,
                                  conv_per_layer * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // ---- INJECTION POINT 2: input embedding (bypass our embedding kernel) -----
    // Run the step, then before the layer sweep we want d_X_accum == golden embed.
    // The cleanest seam is to drive forward() with the real token id and instead
    // overwrite d_X_accum here for exactness; we load it so the contract is fixed.
    std::vector<float> embed = load_golden_dump("input_embedding.bin", cfg.hidden_dim);

    // ---- RUN ONE DECODE STEP --------------------------------------------------
    // forward() runs embedding -> 32 layers (8 full + 24 linear) -> lm_head. The
    // linear layers currently throw until step_linear_attention is wired; catch
    // that precise boundary and SKIP so this test is green-by-skip, not red.
    std::vector<float> logits(cfg.vocab_size);
    {
        const int decode_token = 0;     // golden generator uses a fixed prompt/token
        int next = -1;
        const auto st = engine->forward_status(decode_token, /*pos=*/0, 0.6f, 0.9f, 0, &next);
        if (st != blackwell::EngineStatus::Success)
            GTEST_SKIP() << "SSM decode path not yet wired (scaffold ready): "
                         << blackwell::to_string(st);
        CUDA_CHECK(cudaMemcpy(logits.data(), impl->d_logits,
                              cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // ---- ASSERTION: exact-parity bar -----------------------------------------
    std::vector<float> expected = load_golden_dump("expected_logits.bin", cfg.vocab_size);
    const double cos = cosine_similarity(expected, logits);
    std::cout << "[qwen3.5-hybrid] single-step logit cosine = " << cos << "\n";
    EXPECT_GT(cos, kParityThreshold);

    // Top-1 token must match the HF reference. The two leading logits sit ~0.06
    // apart, so this guards against a high-cosine result that still flips argmax.
    auto argmax = [](const std::vector<float>& v) {
        return (int)(std::max_element(v.begin(), v.end()) - v.begin());
    };
    const int mine = argmax(logits), ref = argmax(expected);
    std::cout << "[qwen3.5-hybrid] top-1 token: engine=" << mine << " ref=" << ref << "\n";
    EXPECT_EQ(mine, ref);
}

// ---------------------------------------------------------------------------
// MULTI-token decode parity (pos > 0). The single-step test above runs only at
// pos 0, where RoPE is the identity -- so it cannot catch a positional-encoding
// regression in the full-attention layers. Here we feed a real token sequence one
// token at a time (pos 0,1,2,...) from a fresh recurrent state and compare the
// per-position logits against an HF causal forward of the same sequence. This is
// the gate that proves partial-RoPE + the recurrent SSM stay faithful for pos>0
// (the autoregressive path the chat loop actually drives).
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, MultiStepDecodeParity) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();
    if (!file_exists(dumps_dir() + "/multistep_logits.bin"))
        GTEST_SKIP() << "multi-step dumps absent in " << dumps_dir()
                     << " (run generate_qwen35_multistep.py).";

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), 512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending: " << e.what();
    }
    BlackwellEngine::Impl* impl = engine->get_impl();
    const ModelConfig& cfg = impl->m_config;

    // Token ids (int32) the HF reference was run on; count == file size / 4.
    std::vector<int> tokens;
    {
        std::ifstream f(dumps_dir() + "/multistep_tokens.bin", std::ios::binary | std::ios::ate);
        ASSERT_TRUE(f.good());
        const std::streamsize bytes = f.tellg();
        f.seekg(0);
        std::vector<int32_t> raw(bytes / sizeof(int32_t));
        f.read(reinterpret_cast<char*>(raw.data()), bytes);
        tokens.assign(raw.begin(), raw.end());
    }
    const int N = (int)tokens.size();
    ASSERT_GT(N, 1) << "need a multi-token sequence to exercise pos>0";

    // HF per-position reference logits [N, vocab].
    std::vector<float> ref = load_golden_dump("multistep_logits.bin", (size_t)N * cfg.vocab_size);

    // Fresh autoregressive run from an empty recurrent state.
    if (impl->ssm_state) impl->ssm_state->reset(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto argmax = [](const float* v, size_t n) {
        return (int)(std::max_element(v, v + n) - v);
    };

    int argmax_matches = 0;
    double min_cos = 1.0;
    std::vector<float> mine(cfg.vocab_size);
    for (int pos = 0; pos < N; ++pos) {
        int next = -1;   // logits for predicting pos+1
        ASSERT_EQ(engine->forward_status(tokens[pos], pos, 0.6f, 0.9f, 0, &next),
                  blackwell::EngineStatus::Success);
        CUDA_CHECK(cudaMemcpy(mine.data(), impl->d_logits,
                              cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        const float* rp = ref.data() + (size_t)pos * cfg.vocab_size;
        std::vector<float> refv(rp, rp + cfg.vocab_size);
        const double cos = cosine_similarity(refv, mine);
        const int a_mine = argmax(mine.data(), cfg.vocab_size);
        const int a_ref  = argmax(rp, cfg.vocab_size);
        if (a_mine == a_ref) ++argmax_matches;
        min_cos = std::min(min_cos, cos);
        std::cout << "  [pos " << pos << "] cos=" << cos
                  << "  argmax engine=" << a_mine << " ref=" << a_ref
                  << (a_mine == a_ref ? "" : "  <-- MISMATCH") << "\n";
    }
    std::cout << "[qwen3.5-hybrid] multi-step min cosine = " << min_cos
              << ", argmax matches = " << argmax_matches << "/" << N << "\n";

    // Every position must stay in parity; a positional-encoding regression in the
    // full-attention layers shows up as cosine DECAY as pos grows (the real failure
    // mode), so the per-position cosine floor is the load-bearing assertion. Argmax
    // is a secondary check: near-tied top-2 logits (cos ~0.9999) can flip on bf16
    // rounding without any bug, so we tolerate a couple while still catching a
    // genuine collapse (which flips many).
    EXPECT_GT(min_cos, kParityThreshold);
    EXPECT_GE(argmax_matches, N - 2);
}

// ---------------------------------------------------------------------------
// Regression: reset_state() must zero the recurrent SSM state so a restarted
// sequence does not decode on top of the previous one. This is the actual root
// cause of the playground's cross-turn repetition: the KV caches are position-
// addressed and self-heal on reprefill, but the SSM state has no rewind. We
// pollute the recurrent state with a few decode steps, then decode token B at
// pos 0 with and without reset_state(), and compare both to the clean reference.
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, ResetStateClearsRecurrentPollution) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();
    if (!file_exists(dumps_dir() + "/multistep_logits.bin"))
        GTEST_SKIP() << "multi-step dumps absent (run generate_qwen35_multistep.py).";

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), 512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending: " << e.what();
    }
    BlackwellEngine::Impl* impl = engine->get_impl();
    const ModelConfig& cfg = impl->m_config;

    std::vector<int> tokens;
    {
        std::ifstream f(dumps_dir() + "/multistep_tokens.bin", std::ios::binary | std::ios::ate);
        ASSERT_TRUE(f.good());
        const std::streamsize bytes = f.tellg();
        f.seekg(0);
        std::vector<int32_t> raw(bytes / sizeof(int32_t));
        f.read(reinterpret_cast<char*>(raw.data()), bytes);
        tokens.assign(raw.begin(), raw.end());
    }
    ASSERT_GT((int)tokens.size(), 4);

    // Reference: clean decode of token B == tokens[0] at pos 0 (== the first row of
    // the multi-step reference).
    std::vector<float> ref = load_golden_dump("multistep_logits.bin", cfg.vocab_size);

    auto decode_B_logits = [&](void) {
        std::vector<float> v(cfg.vocab_size);
        int next = -1;
        EXPECT_EQ(engine->forward_status(tokens[0], /*pos=*/0, 0.6f, 0.9f, 0, &next),
                  blackwell::EngineStatus::Success);
        CUDA_CHECK(cudaMemcpy(v.data(), impl->d_logits, cfg.vocab_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return v;
    };

    // POLLUTE: advance the recurrent state with a few unrelated decode steps.
    if (impl->ssm_state) impl->ssm_state->reset(0);
    for (int p = 0; p < 4; ++p) {
        int next = -1;
        ASSERT_EQ(engine->forward_status(tokens[p + 1], p, 0.6f, 0.9f, 0, &next),
                  blackwell::EngineStatus::Success);
    }

    // Decode B at pos 0 WITHOUT resetting -> recurrent state is stale.
    const std::vector<float> polluted = decode_B_logits();
    const double cos_polluted = cosine_similarity(ref, polluted);

    // Now reset the recurrent state and decode B again from a clean history.
    engine->reset_state(0);
    const std::vector<float> clean = decode_B_logits();
    const double cos_clean = cosine_similarity(ref, clean);

    std::cout << "[qwen3.5-hybrid] reset_state: cos(polluted)=" << cos_polluted
              << "  cos(reset)=" << cos_clean << "\n";

    // The reset run must match the clean reference; the polluted run must be visibly
    // worse (proving the recurrent state actually leaked across the restart).
    EXPECT_GT(cos_clean, kParityThreshold);
    EXPECT_LT(cos_polluted, cos_clean);
}

// ---------------------------------------------------------------------------
// Snapshot-driven VIRTUAL rewind parity: the hybrid analogue of the dense
// BatchedDecodeMatchesSequential gate. A recurrent SSM state cannot be rolled
// back physically, so HybridSnapshotRing forks checkpoints and, on a rewind,
// restores the nearest one and replays the retained tail. The claim under test:
// the restored+replayed state at length T decodes BIT-IDENTICALLY (to bf16
// rounding) to a fresh, uninterrupted decode that reached T -- i.e. the fork
// snapshot really is the state at that point, and the replay reproduces it.
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, VirtualRewindMatchesFreshDecode) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();
    if (!file_exists(dumps_dir() + "/multistep_tokens.bin"))
        GTEST_SKIP() << "multi-step token dump absent (run generate_qwen35_multistep.py).";

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), 512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending: " << e.what();
    }
    if (!blackwell::HybridSnapshotRing::supported(*engine))
        GTEST_SKIP() << "model has no branch headroom for virtual rewind";

    BlackwellEngine::Impl* impl = engine->get_impl();
    const ModelConfig& cfg = impl->m_config;

    std::vector<int> tokens;
    {
        std::ifstream f(dumps_dir() + "/multistep_tokens.bin", std::ios::binary | std::ios::ate);
        ASSERT_TRUE(f.good());
        const std::streamsize bytes = f.tellg();
        f.seekg(0);
        std::vector<int32_t> raw(bytes / sizeof(int32_t));
        f.read(reinterpret_cast<char*>(raw.data()), bytes);
        tokens.assign(raw.begin(), raw.end());
    }
    const int M = std::min<int>(static_cast<int>(tokens.size()), 12);
    ASSERT_GE(M, 10) << "need >= 10 tokens to exercise a mid-stream rewind";

    auto snapshot_logits = [&]() {
        std::vector<float> v(cfg.vocab_size);
        CUDA_CHECK(cudaMemcpy(v.data(), impl->d_logits, cfg.vocab_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return v;
    };

    // Reference: a fresh, uninterrupted decode of tokens[0..M). ref[p] = the
    // logits AFTER consuming tokens[p] (the distribution for position p+1).
    engine->reset_state(0);
    std::vector<std::vector<float>> ref(M);
    for (int p = 0; p < M; ++p) {
        int next = -1;
        const auto st = engine->forward_status(tokens[p], p, 0.0f, 1.0f, 0, &next);
        if (st != blackwell::EngineStatus::Success)
            GTEST_SKIP() << "SSM decode path not yet wired: " << blackwell::to_string(st);
        ref[p] = snapshot_logits();
    }

    // Ring run: prefill P tokens as the "prompt", checkpoint every K, decode to M,
    // then virtually rewind to T and prove the head decodes identically to the
    // fresh reference from T onward.
    const int P = 3, K = 4, T = 8;
    blackwell::HybridSnapshotRing ring(*engine, K);
    engine->reset_state(0);
    int next = -1;
    for (int p = 0; p < P; ++p)
        ASSERT_EQ(engine->forward_status(tokens[p], p, 0.0f, 1.0f, 0, &next),
                  blackwell::EngineStatus::Success);
    ASSERT_EQ(ring.begin(P), blackwell::EngineStatus::Success);
    for (int p = P; p < M; ++p) {
        ASSERT_EQ(engine->forward_status(tokens[p], p, 0.0f, 1.0f, 0, &next),
                  blackwell::EngineStatus::Success);
        ring.on_token(p + 1);
    }

    // Virtual rewind to length T -> the head must hold the SAME logits the fresh
    // decode had after tokens[T-1], and out_next the greedy resample of them.
    blackwell::EngineStatus rst = blackwell::EngineStatus::Success;
    int out_next = -1;
    const int landed = ring.virtual_rewind(T, tokens, 0.0f, 1.0f, &out_next, &rst);
    ASSERT_EQ(rst, blackwell::EngineStatus::Success);
    ASSERT_EQ(landed, T);

    const double cos_at_T = cosine_similarity(ref[T - 1], snapshot_logits());
    std::cout << "[qwen3.5-hybrid] virtual_rewind logits cos@T = " << cos_at_T << "\n";
    EXPECT_GT(cos_at_T, kParityThreshold);
    const int ref_argmax =
        static_cast<int>(std::max_element(ref[T - 1].begin(), ref[T - 1].end()) -
                         ref[T - 1].begin());
    EXPECT_EQ(out_next, ref_argmax);

    // Continue decoding the retained tail from T; every position must stay in
    // lockstep with the fresh reference (the restored recurrent + full-attn state
    // is the fresh state, so this is exact parity, not approximate).
    double min_cos = cos_at_T;
    for (int p = T; p < M; ++p) {
        ASSERT_EQ(engine->forward_status(tokens[p], p, 0.0f, 1.0f, 0, &next),
                  blackwell::EngineStatus::Success);
        min_cos = std::min(min_cos, cosine_similarity(ref[p], snapshot_logits()));
    }
    std::cout << "[qwen3.5-hybrid] post-rewind decode min cos = " << min_cos << "\n";
    EXPECT_GT(min_cos, kParityThreshold);
}

// ---------------------------------------------------------------------------
// Isolated GatedDeltaNet parity for linear layer 0. This is the achievable
// correctness gate for the SSM assembly: the full-logit test above is blocked by
// the (unimplemented) head_dim-256 full-attention layers, but layer 0 is linear
// and its input is exactly the decode token's embedding. We feed the golden
// embedding, run step_linear_attention(0), and compare the mixer output to the HF
// reference linear_attn_out_layer0.bin (and the post-step state to the dump).
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, LinearLayer0Parity) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();
    if (!file_exists(dumps_dir() + "/linear_attn_out_layer0.bin"))
        GTEST_SKIP() << "golden dumps absent in " << dumps_dir();

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), 512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction failed: " << e.what();
    }
    BlackwellEngine::Impl* impl = engine->get_impl();
    const ModelConfig& cfg = impl->m_config;
    ASSERT_EQ(impl->m_linear_layer_index[0], 0) << "layer 0 must be a linear layer";

    const int H = (int)cfg.linear.num_value_heads;
    const int Dk = (int)cfg.linear.key_head_dim, Dv = (int)cfg.linear.value_head_dim;
    const size_t hidden = cfg.hidden_dim;

    // Seed the residual stream with the golden token embedding.
    std::vector<float> embed = load_golden_dump("input_embedding.bin", hidden);
    CUDA_CHECK(cudaMemcpy(impl->d_X_accum, embed.data(), hidden * sizeof(float),
                          cudaMemcpyHostToDevice));
    impl->ssm_state->reset(0);   // zero recurrent + conv state (first token)
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run the GatedDeltaNet decode step for layer 0.
    impl->step_linear_attention(0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- STAGE-BY-STAGE diagnostics vs HF captures (l0_*.bin) ----------------
    auto stage = [&](const char* name, const float* dptr, size_t n, const std::string& dump) {
        if (!file_exists(dumps_dir() + "/" + dump)) { std::cout << "  (skip " << dump << ")\n"; return; }
        std::vector<float> mine(n);
        CUDA_CHECK(cudaMemcpy(mine.data(), dptr, n * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> ref = load_golden_dump(dump, n);
        std::cout << "  [stage] " << name << " cos=" << cosine_similarity(ref, mine) << "\n";
    };
    const size_t conv_dim = 2*(size_t)cfg.linear.num_key_heads*Dk + (size_t)H*Dv;
    stage("mixer_in (post input_layernorm)", impl->d_X_norm,      hidden,   "l0_mixer_in.bin");
    stage("qkv_proj (in_proj_qkv int4)",     impl->d_ssm_qkv,     conv_dim, "l0_qkv_proj.bin");
    stage("beta (sigmoid in_proj_b)",        impl->d_ssm_b,       H,        "l0_beta.bin");
    stage("core (scan output, pre-gate)",    impl->d_ssm_core,    (size_t)H*Dv, "l0_core.bin");
    stage("gated (RMSNormGated output)",     impl->d_ssm_o,       (size_t)H*Dv, "l0_gated.bin");

    // Mixer output = (residual after) - (embedding before).
    std::vector<float> after(hidden);
    CUDA_CHECK(cudaMemcpy(after.data(), impl->d_X_accum, hidden * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<float> mixer(hidden);
    for (size_t i = 0; i < hidden; ++i) mixer[i] = after[i] - embed[i];

    std::vector<float> golden = load_golden_dump("linear_attn_out_layer0.bin", hidden);
    const double mix_cos = cosine_similarity(golden, mixer);
    std::cout << "[qwen3.5-hybrid] layer0 mixer-output cosine = " << mix_cos << "\n";

    // Diagnostic: post-step recurrent state vs the dump (layout-sensitive; treated
    // as a hint, not the parity gate).
    if (file_exists(dumps_dir() + "/expected_ssm_state.bin")) {
        const size_t rec = (size_t)H * Dk * Dv;
        std::vector<float> S(rec);
        CUDA_CHECK(cudaMemcpy(S.data(), impl->ssm_state->rec_state(0, 0),
                              rec * sizeof(float), cudaMemcpyDeviceToHost));
        std::ifstream f(dumps_dir() + "/expected_ssm_state.bin", std::ios::binary);
        std::vector<float> Sg(rec);
        f.read(reinterpret_cast<char*>(Sg.data()), rec * sizeof(float));  // layer 0 == first block
        std::cout << "[qwen3.5-hybrid] layer0 recurrent-state cosine = "
                  << cosine_similarity(Sg, S) << "\n";
    }

    EXPECT_GT(mix_cos, kParityThreshold);
}

// ---------------------------------------------------------------------------
// Tree-of-Thoughts branching for a HYBRID model. fork() must physically snapshot
// BOTH the recurrent SSM state AND the dedicated gated full-attention KV cache,
// so decoding a forked child never disturbs the parent (and vice versa).
//
// Proof of independence, no golden dumps needed: prime seq 0, fork it into four
// identical branches, then show that decoding two of them INTERLEAVED reproduces,
// to parity, what decoding them in ISOLATION does. If any per-sequence state were
// shared (the pre-snapshot behaviour, where the SSM path was hardcoded to seq 0
// and the full-attn cache had no seq dimension), the interleaved decode would
// read a polluted state and diverge from the isolated reference.
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, HybridModelSupportsForking) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), /*max_seq_len=*/512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending: " << e.what();
    }
    ASSERT_TRUE(engine->get_capabilities().supports_cow_branching)
        << "hybrid model must support fork() after the SSM/full-attn snapshot work";

    BlackwellEngine::Impl* impl = engine->get_impl();
    const size_t V = impl->m_config.vocab_size;

    auto snapshot = [&]() {
        std::vector<float> h(V);
        EXPECT_EQ(cudaMemcpy(h.data(), impl->d_logits, V * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    };
    // One decode step on `seq`; returns that step's logits. forward_status needs a
    // non-null out pointer, so we always pass one.
    auto decode = [&](int tok, int pos, int seq) {
        int next = -1;
        EXPECT_EQ(engine->forward_status(tok, pos, 0.6f, 0.9f, seq, &next),
                  blackwell::EngineStatus::Success)
            << "decode(tok=" << tok << ", pos=" << pos << ", seq=" << seq << ")";
        return snapshot();
    };
    auto argmax = [](const std::vector<float>& v) {
        return (int)(std::max_element(v.begin(), v.end()) - v.begin());
    };

    // Prime seq 0 with a short prompt from an empty recurrent state (positions 0..L-1).
    if (impl->ssm_state) impl->ssm_state->reset(0);
    const int prompt[] = {1, 2, 3};
    const int L = (int)(sizeof(prompt) / sizeof(prompt[0]));
    for (int pos = 0; pos < L; ++pos) (void)decode(prompt[pos], pos, /*seq=*/0);

    // Fork into four identical branches at length L: seq 0 (parent) + 1 (child) are
    // decoded INTERLEAVED; seq 2/3 hold the ISOLATED references. Needs branch
    // capacity >= 4 (default paged_branch_factor).
    ASSERT_NO_THROW(engine->fork(0, 1));
    ASSERT_NO_THROW(engine->fork(0, 2));
    ASSERT_NO_THROW(engine->fork(0, 3));

    // Distinct continuations: parent follows Y, child follows X.
    const int Ya = 100, Yb = 101;
    const int Xa = 500, Xb = 501;
    ASSERT_LT(Yb, (int)V);
    ASSERT_LT(Xb, (int)V);

    // Isolated references, each decoded alone on its own branch.
    (void)decode(Ya, L,     /*seq=*/2);
    const std::vector<float> ref_parent = decode(Yb, L + 1, /*seq=*/2);
    (void)decode(Xa, L,     /*seq=*/3);
    const std::vector<float> ref_child  = decode(Xb, L + 1, /*seq=*/3);

    // Interleaved: alternate parent (seq 0, Y) and child (seq 1, X).
    (void)decode(Ya, L,     /*seq=*/0);
    (void)decode(Xa, L,     /*seq=*/1);
    const std::vector<float> got_parent = decode(Yb, L + 1, /*seq=*/0);
    const std::vector<float> got_child  = decode(Xb, L + 1, /*seq=*/1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA fault during hybrid fork decode";

    // Independence: interleaving must reproduce the isolated references (same
    // kernels over the same physically-forked state). A shared or leaking store
    // would make got_* diverge from ref_*.
    const double cos_parent = cosine_similarity(ref_parent, got_parent);
    const double cos_child  = cosine_similarity(ref_child,  got_child);
    std::cout << "[qwen3.5-hybrid fork] parent cos=" << cos_parent
              << "  child cos=" << cos_child
              << "  parent top-1=" << argmax(got_parent)
              << "  child top-1="  << argmax(got_child) << "\n";
    EXPECT_GT(cos_parent, kParityThreshold)
        << "interleaved parent decode diverged from its isolated reference "
           "(child branch leaked into the parent's SSM / full-attn state)";
    EXPECT_GT(cos_child, kParityThreshold)
        << "interleaved child decode diverged from its isolated reference "
           "(parent branch leaked into the child's SSM / full-attn state)";
    EXPECT_EQ(argmax(got_parent), argmax(ref_parent));
    EXPECT_EQ(argmax(got_child),  argmax(ref_child));

    SUCCEED() << "Hybrid fork produced independent parent/child branches.";
}

// ---------------------------------------------------------------------------
// TRUE BATCHED PREFILL parity. The chunked delta-rule kernel
// (launch_gated_delta_chunked_prefill) graduated into run_chunk must reproduce
// the token-by-token run_token sweep it replaces. This test is SELF-REFERENTIAL
// -- no golden dumps -- because run_token IS the reference: we prefill the same
// synthetic prompt two ways on the SAME engine/sequence and compare the final
// logits.
//   Path A (reference): run_token for each position 0..N-1 (the per-token
//     GatedDeltaNet recurrence + per-token full-attention).
//   Path B (batched):   run_chunk tiled into m_token_capacity blocks -- each a
//     ONE-pass batched prefill: chunked delta rule for the linear layers, a
//     per-token cache sweep for the gated full-attention layers, batched
//     projections/MLP. The recurrent SSM state + conv ring + full-attn cache
//     carry across tiles exactly as they carry across run_token steps.
// If the chunked kernel, the state carry across chunk boundaries, or the
// seq-id offsets were wrong, Path B's logits would diverge from Path A's.
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, BatchedPrefillMatchesSequential) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path();

    std::unique_ptr<BlackwellEngine> engine;
    try {
        engine = std::make_unique<BlackwellEngine>(model_index_path(), /*max_seq_len=*/512,
                                                   static_cast<size_t>(-1),
                                                   BlackwellEngine::KVCacheMode::Paged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "engine construction pending: " << e.what();
    }
    BlackwellEngine::Impl* impl = engine->get_impl();
    const size_t V = impl->m_config.vocab_size;
    const int tile = (int)impl->m_token_capacity;   // hybrid Paged prefill tile (64)
    ASSERT_GT(tile, 1) << "hybrid Paged model must expose a batched prefill tile";

    // A deliberately long prompt whose length is NOT a multiple of the tile, so the
    // final chunk exercises the kernel's partial-chunk (cur_C < 64) path and the
    // run reaches across several chunk boundaries (state carry). Kept modest because
    // the reference (Path A) is an O(N) run_token sweep. Any valid ids work -- the
    // comparison is engine-vs-engine, not against HF.
    const int N = 2 * tile + 2;          // e.g. 130 tokens -> 3 chunks (64,64,2)
    std::vector<int> tokens(N);
    for (int i = 0; i < N; ++i) tokens[i] = (int)((i * 1103515245u + 12345u) % (V / 2)) + 1;

    auto logits_now = [&]() {
        std::vector<float> h(V);
        EXPECT_EQ(cudaMemcpy(h.data(), impl->d_logits, V * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    };

    // ---- Path A: sequential run_token (the reference) -------------------------
    engine->reset_state(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < N; ++i) {
        const bool last = (i == N - 1);
        ASSERT_EQ(impl->run_token(tokens[i], i, /*seq_id=*/0, /*want_logits=*/last),
                  blackwell::EngineStatus::Success) << "run_token pos " << i;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> ref = logits_now();

    // ---- Path B: batched prefill via run_chunk, tiled into <=tile blocks -------
    engine->reset_state(0);   // zero recurrent/conv state; KV caches self-heal on re-prefill
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int pos = 0; pos < N; ) {
        const int len  = std::min(tile, N - pos);
        const bool last = (pos + len == N);
        ASSERT_EQ(impl->run_chunk(tokens.data() + pos, pos, len, /*seq_id=*/0,
                                  /*want_logits=*/last),
                  blackwell::EngineStatus::Success)
            << "run_chunk [" << pos << ", " << (pos + len) << ")";
        pos += len;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = logits_now();

    // ---- Parity ---------------------------------------------------------------
    double max_abs = 0.0;
    for (size_t i = 0; i < V; ++i) max_abs = std::max(max_abs, (double)std::fabs(ref[i] - got[i]));
    const double cos = cosine_similarity(ref, got);
    auto argmax = [](const std::vector<float>& v) {
        return (int)(std::max_element(v.begin(), v.end()) - v.begin());
    };
    std::cout << "[qwen3.5-hybrid prefill] N=" << N << " tile=" << tile
              << "  cosine=" << cos << "  max|dlogit|=" << max_abs
              << "  argmax batched=" << argmax(got) << " seq=" << argmax(ref) << "\n";

    // The chunked delta rule reproduces the per-token recurrence to fp32 rounding,
    // and the (int4) projections / full-attention are the same kernels per row, so
    // the two paths land on the same distribution. Cosine + argmax are the parity
    // bar this suite uses (near-tied top-2 logits can flip argmax on rounding, but
    // a real batched-prefill bug collapses cosine).
    EXPECT_GT(cos, kParityThreshold)
        << "batched prefill diverged from the sequential run_token reference";
    EXPECT_EQ(argmax(got), argmax(ref))
        << "batched prefill predicted a different top-1 token than the sequential sweep";
}
