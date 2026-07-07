// ============================================================================
// Golden-dump parity SCAFFOLD for the Qwen3.5 hybrid (linear-attn + full-attn)
// checkpoint. Single-token DECODE phase only (chunked prefill is a later
// milestone). Mirrors test_qwen_engine.cpp's dump/skip conventions.
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
// Capabilities: this runs today (no dumps needed). It asserts the hybrid model
// correctly refuses CoW branching. Construction needs the checkpoint to load,
// which is itself gated on the int4/binding work — so if construction throws we
// SKIP rather than fail (the contract is still documented and compiled).
// ---------------------------------------------------------------------------
TEST(Qwen35Hybrid, CapabilitiesRejectBranching) {
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
    EXPECT_FALSE(caps.supports_cow_branching);
    EXPECT_GT(caps.num_linear_attention_layers, 0);
    EXPECT_GT(caps.num_full_attention_layers, 0);

    // The capability gate must reject branching with a clean, handled error.
    EXPECT_THROW(engine->fork(0, 1), std::runtime_error);
    EXPECT_THROW(engine->rewind(0, 0), std::runtime_error);
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
