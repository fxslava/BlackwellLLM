// End-to-end engine verification for Qwen2.5-Coder-7B-Instruct-AWQ: runs one
// BOS-token decode step through the real BlackwellEngine (28 layers of int4
// AWQ weights) and compares per-layer telemetry plus the final logits against
// golden PyTorch dumps produced by generate_qwen_dumps.py.
//
// Convergence metric (quantization drift is expected on this stack):
//   cosine_similarity(final logits) >= 0.95
//   OR the PyTorch golden top-1 token appears in the engine's top-5.
//
// The loading / metric / table / verdict plumbing lives in
// tests/common/engine_test_harness.h, shared with the Llama and GLM-4 suites.
//
// External requirements (the test SKIPs if either is missing):
//   - model checkpoint index, default
//     F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ/model.safetensors.index.json
//     (override with env var BLACKWELL_QWEN_INDEX)
//   - golden dumps directory produced by generate_qwen_dumps.py, default
//     <repo>/tests/integration/golden_dumps/qwen2.5_awq
//     (override with env var BLACKWELL_QWEN_DUMPS_DIR)
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "common.h"
#include "common/engine_test_harness.h"
#include "blackwell/engine.h"
#include "engine_impl.h"

namespace {

std::string model_index_path() {
    return engine_test::env_or("BLACKWELL_QWEN_INDEX",
                               "F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ/model.safetensors.index.json");
}

std::string dumps_dir() {
    return engine_test::env_or("BLACKWELL_QWEN_DUMPS_DIR",
                               "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/qwen2.5_awq");
}

}  // namespace

TEST(QwenEngineIntegration, AwqBosDecodeConvergence) {
    using namespace engine_test;

    if (!file_exists(model_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << model_index_path()
                     << " (set BLACKWELL_QWEN_INDEX to override)";
    }
    if (!file_exists(dumps_dir() + "/logits_out.bin")) {
        GTEST_SKIP() << "Golden dumps not found in " << dumps_dir()
                     << " (run generate_qwen_dumps.py, or set BLACKWELL_QWEN_DUMPS_DIR)";
    }

    // Qwen2.5-Coder-7B geometry (config.json at the checkpoint root).
    const size_t hidden_dim       = 3584;   // 28 heads x 128
    const size_t kv_dim           = 512;    // 4 KV heads x 128
    const size_t intermediate_dim = 18944;
    const size_t vocab_size       = 152064;
    const int    num_layers       = 28;
    const int    start_token      = 151643; // <|endoftext|> (BOS)
    const int    pos              = 0;

    std::cout << "\n[Integration] Initializing BlackwellEngine (Qwen2.5 AWQ)...\n";
    BlackwellEngine engine(model_index_path(), 2048);
    auto* core = engine.get_impl();

    const std::string dd = dumps_dir();
    TelemetryTablePrinter table;
    table.header("executing 28 layers with telemetry");

    // ========================================================================
    // STAGE 1: embedding lookup (telemetry only; the binding verdict is the
    // final-logits convergence metric below).
    // ========================================================================
    core->step_embedding(start_token);
    table.probe(dd, "embed_out.bin", core->d_X_accum, hidden_dim, -1, "Embedding");

    // ========================================================================
    // STAGE 2: layer-by-layer execution with telemetry
    // ========================================================================
    for (int l = 0; l < num_layers; ++l) {
        const std::string l_str = std::to_string(l);

        core->step_attention_norm(l);
        if (l == 0)
            table.probe(dd, "layer_0_input_norm.bin", core->d_X_norm, hidden_dim, l, "InputNorm");

        core->step_attention_qkv_projections(l);
        if (l == 0) {
            table.probe(dd, "layer_0_q_proj.bin", core->d_Q, hidden_dim, l, "Q_proj");
            table.probe(dd, "layer_0_k_proj.bin", core->d_K, kv_dim,     l, "K_proj");
            table.probe(dd, "layer_0_v_proj.bin", core->d_V, kv_dim,     l, "V_proj");
        }

        core->step_attention_math(l, pos);
        if (l == 0)
            table.probe(dd, "layer_0_attn_math.bin", core->d_Attn_out, hidden_dim, l, "AttnMath");

        core->step_attention_out(l);
        core->step_mlp_norm(l);

        core->step_mlp_projections(l);
        if (l == 0) {
            table.probe(dd, "layer_0_gate_proj.bin", core->d_Gate, intermediate_dim, l, "Gate_proj");
            table.probe(dd, "layer_0_up_proj.bin",   core->d_Up,   intermediate_dim, l, "Up_proj");
        }

        core->step_mlp_out(l);
        table.probe(dd, "layer_" + l_str + "_accum_out.bin", core->d_X_accum, hidden_dim,
                    l, "Accum_out");
    }
    table.footer();

    // ========================================================================
    // STAGE 3: final norm, vocab projection and the convergence verdict
    // ========================================================================
    std::cout << "\n[Integration] Stage 3: vocab projection and convergence metric...\n";
    core->step_final_ops();

    const std::vector<float> golden_logits =
        load_golden_bin(dd, "logits_out.bin", vocab_size);

    LogitsVerdict verdict;
    ASSERT_TRUE(assert_logits_convergence(core->d_logits, golden_logits, vocab_size,
                                          /*min_cosine=*/0.95, /*top_k=*/5, &verdict));

    std::cout << "  [SUCCESS] Convergence metric met (cosine " << verdict.cosine
              << ", golden top-1 in engine top-5: "
              << (verdict.top1_in_engine_topk ? "yes" : "no") << ").\n";
}
