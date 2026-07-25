// =============================================================================
// Llama-3.1-8B-Instruct AWQ-INT4 backbone LOAD smoke test.
//
// Purpose-built to answer one question: does the real downloaded 8B AWQ
// checkpoint LOAD correctly through BlackwellEngine on the GPU -- geometry
// resolved from its own config.json, all weight shards mapped, quant path
// selected -- and run at least one real forward step through the AWQ INT4
// GEMV kernels, with NO cross-quant golden-dump parity involved?
//
// This is the AWQ sibling of test_llama8b_fp8_integration.cpp. That test asserts
// logit PARITY against a PyTorch reference; this one deliberately does NOT --
// there is no AWQ reference dump, and the audio pipeline's uv8b_* references were
// generated with the FP8 backbone (different quantization), so a parity check
// here would be meaningless. We assert the achievable "the weights load without
// shape mismatch / assertion and the forward pass returns Success + an in-range
// token" bar, and REPORT the sampled continuation as telemetry.
//
// Checkpoint geometry (from F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/config.json):
//   hidden 4096, 32 layers, 32 heads / 8 KV heads, vocab 128256,
//   rope_theta 500000, quant_method "awq" (bits 4, group 128, gemm).
//   NB: this export has "rope_scaling": null -- i.e. VANILLA RoPE, not the
//   llama3 wavelength scaling. The engine honors the checkpoint's own config,
//   so we assert rope_scaling_type == 0 to lock that observed fact.
//
// External requirement (SKIPs if missing):
//   - checkpoint index  F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json
//                       (override: env BLACKWELL_AWQ_INDEX)
// =============================================================================
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "blackwell/engine.h"
#include "blackwell/engine_status.h"
#include "engine_impl.h"

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}
std::string awq_index_path() {
    return env_or("BLACKWELL_AWQ_INDEX",
                  "F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json");
}
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

// Llama-3.1-8B geometry the AWQ checkpoint must resolve to (config.json above).
constexpr int kHidden   = 4096;
constexpr int kLayers   = 32;
constexpr int kHeads    = 32;
constexpr int kKvHeads  = 8;
constexpr int kVocab    = 128256;
constexpr int kBosToken = 128000;   // config.json bos_token_id

}  // namespace

// -----------------------------------------------------------------------------
// Load the AWQ 8B backbone and run a short greedy decode; no golden-dump parity.
// -----------------------------------------------------------------------------
TEST(Llama8bAwqBackboneLoad, LoadsAndRunsForwardStep) {
    const std::string index = awq_index_path();
    if (!file_exists(index))
        GTEST_SKIP() << "AWQ checkpoint absent: " << index << " (set BLACKWELL_AWQ_INDEX)";

    // --- Load: construction parses config.json + maps every weight shard -------
    std::cout << "[awq] constructing engine from " << index << " ...\n";
    BlackwellEngine engine(index, /*max_seq_len=*/32);
    auto* core = engine.get_impl();
    ASSERT_NE(core, nullptr);

    // --- Geometry resolved from the checkpoint's own config (no shape mismatch) -
    EXPECT_EQ(static_cast<int>(core->m_config.hidden_dim),          kHidden);
    EXPECT_EQ(static_cast<int>(core->m_config.num_layers),          kLayers);
    EXPECT_EQ(static_cast<int>(core->m_config.num_attention_heads), kHeads);
    EXPECT_EQ(static_cast<int>(core->m_config.num_key_value_heads), kKvHeads);
    EXPECT_EQ(static_cast<int>(core->m_config.vocab_size),          kVocab);
    EXPECT_EQ(core->m_config.quant_method, std::string("awq"));
    EXPECT_EQ(core->m_config.quant_strategy, QuantStrategy::WEIGHT_ONLY_PACKED);
    EXPECT_NEAR(core->m_config.rope_theta, 500000.0f, 1.0f);
    // This AWQ export declares "rope_scaling": null -> vanilla RoPE (type 0).
    EXPECT_EQ(core->m_config.rope_scaling_type, 0)
        << "AWQ config.json has rope_scaling null; engine should use vanilla RoPE";

    std::cout << "[awq] loaded: hidden=" << core->m_config.hidden_dim
              << " layers=" << core->m_config.num_layers
              << " heads=" << core->m_config.num_attention_heads
              << " kv_heads=" << core->m_config.num_key_value_heads
              << " vocab=" << core->m_config.vocab_size
              << " quant=" << core->m_config.quant_method << "\n";

    // --- One real forward step per position through the AWQ INT4 kernels --------
    // Greedy (temperature 0 -> argmax in sample_top_p); feed each token back so
    // the KV cache advances. Success + in-range token id is the whole contract.
    const size_t vocab = core->m_config.vocab_size;
    int token = kBosToken;
    std::cout << "[awq] greedy decode from BOS: " << token;
    for (int pos = 0; pos < 4; ++pos) {
        int next = -1;
        const blackwell::EngineStatus st =
            engine.forward_status(token, pos, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                  /*seq_id=*/0, &next);
        ASSERT_EQ(st, blackwell::EngineStatus::Success)
            << "forward_status failed at pos " << pos;
        ASSERT_GE(next, 0);
        ASSERT_LT(static_cast<size_t>(next), vocab)
            << "sampled token " << next << " out of vocab range at pos " << pos;
        std::cout << " -> " << next;
        token = next;
    }
    std::cout << "\n[awq] backbone load + forward smoke test OK\n";
}
