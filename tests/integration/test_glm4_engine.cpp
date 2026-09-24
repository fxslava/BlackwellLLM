// End-to-end parity for GLM-4-9B-Chat-1M: replays a short prompt through the real
// BlackwellEngine and compares per-stage telemetry, attention entropy and the final
// logits against PyTorch dumps taken from the checkpoint's own modeling code.
//
// WHAT THIS SUITE IS POSITIONED TO CATCH, that nothing else can:
//   * the INTERLEAVED partial RoPE (GLM rotates channels (2j, 2j+1) over the first
//     64 of 128 per head). q_rope / k_rope are dumped precisely so the pairing is
//     witnessed against the model's own apply_rotary_pos_emb rather than inferred
//     from downstream damage;
//   * the FUSED mlp.gate_up_proj ordering -- gate is the FIRST half, and getting it
//     backwards passes every shape check while producing fluent nonsense;
//   * attention ENTROPY over a real multi-token prefix. A pos=0 probe cannot see
//     this: one key means p == 1 and entropy identically 0, so a single-token test
//     would pass with the attention distribution completely wrong.
//
// The prompt is 6 tokens and every dump holds the LAST token's row -- the row whose
// attention spans the whole prefix and whose logits pick the next token. The engine
// reaches that row by replaying tokens 0..4 through the ordinary decode path first,
// which is also a test that its KV cache accumulates correctly.
//
// PRECISION FLOOR. The reference runs bf16 activations (and GLM latches its RoPE
// cos/sin table to bf16 before rotating); the engine runs fp32 activations over bf16
// weights. Elementwise equality is therefore impossible by construction -- cosine
// similarity is the metric, exactly as in the Llama and Qwen suites.
//
// External requirements (the test SKIPs if either is missing):
//   - HF-NATIVE GLM-4 checkpoint index, default
//     F:/AI/models/GLM-4-9B-Chat-1M-hf/model.safetensors.index.json
//     (override with BLACKWELL_GLM4_INDEX). The upstream THUDM tree is NOT
//     loadable -- convert it with scripts/convert_glm4_thudm_to_hf.py.
//   - golden dumps from scripts/generate_glm4_dumps.py, default
//     <repo>/tests/integration/golden_dumps/glm4_9b
//     (override with BLACKWELL_GLM4_DUMPS_DIR)
//
// VRAM: 17.7 GiB of bf16 weights against a 11.9 GiB card, so the engine is
// constructed with layer offloading (BLACKWELL_GLM4_GPU_LAYERS, default 16).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "common/engine_test_harness.h"
#include "blackwell/engine.h"
#include "engine_impl.h"

namespace {

std::string model_index_path() {
    return engine_test::env_or(
        "BLACKWELL_GLM4_INDEX",
        "F:/AI/models/GLM-4-9B-Chat-1M-hf/model.safetensors.index.json");
}

std::string dumps_dir() {
    return engine_test::env_or(
        "BLACKWELL_GLM4_DUMPS_DIR",
        "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/glm4_9b");
}

size_t gpu_layers() {
    const std::string v = engine_test::env_or("BLACKWELL_GLM4_GPU_LAYERS", "16");
    return static_cast<size_t>(std::strtoul(v.c_str(), nullptr, 10));
}

// GLM-4-9B-Chat-1M geometry (converted config.json; mirrored in the dumps' meta.json).
constexpr size_t kHidden     = 4096;
constexpr size_t kInter      = 13696;
constexpr size_t kVocab      = 151552;
constexpr size_t kHeads      = 32;
constexpr size_t kKvHeads    = 4;
constexpr size_t kHeadDim    = 128;
constexpr size_t kRotaryDim  = 64;    // partial_rotary_factor 0.5
constexpr size_t kQDim       = kHeads * kHeadDim;     // 4096
constexpr size_t kKvDim      = kKvHeads * kHeadDim;   // 512
constexpr int    kNumLayers  = 40;
constexpr size_t kMaxSeqLen  = 64;
const int kProbeLayers[] = {0, 19, 39};

bool is_probe_layer(int l) {
    for (int p : kProbeLayers) if (p == l) return true;
    return false;
}

// Reconstructs the attention distribution the engine actually produced, from the
// engine's OWN tensors: the rotated query it holds after step_attention_math and the
// rotated keys it wrote into its KV cache. The decode attention kernel fuses the
// softmax, so the probabilities are never materialized in memory -- but they are
// fully determined by these two, and recomputing them here is what makes the
// entropy comparison a measurement of the engine rather than of the dump.
//
// Continuous KV layout: float[kv_heads][max_seq_len][head_dim]. Scale is
// head_dim^-0.5, matching launch_attention_decoding_kernel.
std::vector<double> engine_attention_entropy_per_head(const float* d_q,
                                                      const float* d_k_cache,
                                                      int seq_len) {
    const std::vector<float> q = engine_test::download(d_q, kHeads * kHeadDim);
    const std::vector<float> k =
        engine_test::download(d_k_cache, kKvHeads * kMaxSeqLen * kHeadDim);

    const size_t group = kHeads / kKvHeads;   // 8
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    std::vector<double> entropy(kHeads, 0.0);
    for (size_t h = 0; h < kHeads; ++h) {
        const float* qh = q.data() + h * kHeadDim;
        const size_t kvh = h / group;
        std::vector<float> scores(static_cast<size_t>(seq_len));
        for (int s = 0; s < seq_len; ++s) {
            const float* ks = k.data() + (kvh * kMaxSeqLen + static_cast<size_t>(s)) * kHeadDim;
            double dot = 0.0;
            for (size_t d = 0; d < kHeadDim; ++d) dot += double(qh[d]) * double(ks[d]);
            scores[static_cast<size_t>(s)] = static_cast<float>(dot * scale);
        }
        entropy[h] = engine_test::compute_entropy(engine_test::softmax(scores));
    }
    return entropy;
}

double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

}  // namespace

TEST(Glm4EngineIntegration, LayerByLayerParityAndAttentionEntropy) {
    using namespace engine_test;

    if (!file_exists(model_index_path())) {
        GTEST_SKIP() << "GLM-4 checkpoint not found at " << model_index_path()
                     << ". The upstream THUDM tree is not loadable directly -- run "
                        "scripts/convert_glm4_thudm_to_hf.py, or set BLACKWELL_GLM4_INDEX.";
    }
    if (!file_exists(dumps_dir() + "/logits_out.bin") ||
        !file_exists(dumps_dir() + "/tokens.bin")) {
        GTEST_SKIP() << "Golden dumps not found in " << dumps_dir()
                     << " (run scripts/generate_glm4_dumps.py, or set "
                        "BLACKWELL_GLM4_DUMPS_DIR)";
    }

    const std::string dd = dumps_dir();
    const std::vector<int> tokens = load_golden_i32(dd + "/tokens.bin");
    ASSERT_GE(tokens.size(), 2u)
        << "the dumps were generated with a single-token prompt; attention entropy "
           "is identically zero there and this suite cannot measure it";
    const int seq_len = static_cast<int>(tokens.size());
    const int last_pos = seq_len - 1;
    ASSERT_LT(static_cast<size_t>(seq_len), kMaxSeqLen);

    std::cout << "\n[Integration] GLM-4-9B-Chat-1M: " << seq_len << "-token prompt (";
    for (int t : tokens) std::cout << " " << t;
    std::cout << " ), probing layers 0/19/39 on the last token.\n";
    std::cout << "[Integration] Initializing BlackwellEngine ("
              << gpu_layers() << " of " << kNumLayers << " layers VRAM-resident; "
              << "17.7 GiB of weights does not fit an 11.9 GiB card)...\n";

    BlackwellEngine engine(model_index_path(), kMaxSeqLen, gpu_layers(),
                           BlackwellEngine::KVCacheMode::Continuous);
    auto* core = engine.get_impl();

    // The parse the whole suite rests on: if the loader silently treated this as a
    // half-split full-rotary model, every RoPE probe below would be meaningless.
    ASSERT_EQ(core->m_config.rope_pairing, RopePairing::Interleaved);
    ASSERT_EQ(core->m_config.rotary_dim, kRotaryDim);
    ASSERT_TRUE(core->m_config.mlp_fused_gate_up);
    ASSERT_FALSE(core->m_config.has_sandwich_norms);   // "glm", not the 0414 "glm4"
    ASSERT_TRUE(core->m_config.has_qkv_bias);
    ASSERT_EQ(core->m_config.num_key_value_heads, kKvHeads);
    ASSERT_FLOAT_EQ(core->m_config.rope_theta, 1.0e8f);   // 10000 * rope_ratio(10000)

    // ========================================================================
    // Replay the prefix through the ordinary decode path, so the KV cache holds
    // positions 0..last_pos-1 exactly as a real generation would leave them.
    // ========================================================================
    for (int t = 0; t < last_pos; ++t) {
        const auto st = core->run_token(tokens[static_cast<size_t>(t)], t, /*seq_id=*/0,
                                        /*want_logits=*/false);
        ASSERT_EQ(st, blackwell::EngineStatus::Success)
            << "prefix replay failed at position " << t << ": "
            << blackwell::to_string(st);
    }

    // ========================================================================
    // The last token, stage by stage.
    // ========================================================================
    TelemetryTablePrinter table;
    table.header("last-token stage parity (cosine < 0.99 flags DIVERGING)");

    core->step_embedding(tokens[static_cast<size_t>(last_pos)]);
    auto embed = table.probe(dd, "embed_out.bin", core->d_X_accum, kHidden, -1, "Embedding");
    ASSERT_TRUE(embed.has_value());
    EXPECT_GE(embed->cosine_sim, 0.9999) << "the embedding row is a pure table lookup: "
                                            "anything below bf16 rounding here is a "
                                            "tokenizer or vocab-layout mismatch";

    core->kv_mgr->prepare_decode_step(/*seq=*/0, last_pos);

    struct LayerProbe {
        std::optional<Telemetry> q_rope, k_rope, input_norm, attn_math, gate_up, accum;
        std::vector<double> engine_entropy;
        std::vector<float>  attn_accum_host;   // residual after o_proj, for mlp_out
    };
    std::vector<LayerProbe> probes(static_cast<size_t>(kNumLayers));

    for (int l = 0; l < kNumLayers; ++l) {
        const bool probe = is_probe_layer(l);
        const std::string tag = "layer_" + std::to_string(l) + "_";
        LayerProbe& P = probes[static_cast<size_t>(l)];

        core->arena.prefetch_layer(l + 1, last_pos);

        core->step_attention_norm(l);
        if (probe)
            P.input_norm = table.probe(dd, tag + "input_norm.bin", core->d_X_norm,
                                       kHidden, l, "InputNorm");

        core->step_attention_qkv_projections(l);
        if (probe) {
            // Pre-RoPE, post-bias: the fused query_key_value split lands here.
            table.probe(dd, tag + "q_proj.bin", core->d_Q, kQDim,  l, "Q_proj");
            table.probe(dd, tag + "k_proj.bin", core->d_K, kKvDim, l, "K_proj");
            table.probe(dd, tag + "v_proj.bin", core->d_V, kKvDim, l, "V_proj");
        }

        // RoPE happens INSIDE attention_math (fused with the KV append), rotating
        // d_Q and d_K in place -- so after this call they hold the post-RoPE values.
        core->step_attention_math(l, last_pos);
        if (probe) {
            P.q_rope = table.probe(dd, tag + "q_rope.bin", core->d_Q, kQDim,  l, "Q_rope");
            P.k_rope = table.probe(dd, tag + "k_rope.bin", core->d_K, kKvDim, l, "K_rope");
            P.attn_math = table.probe(dd, tag + "attn_math.bin", core->d_Attn_out,
                                      kQDim, l, "AttnMath");
            P.engine_entropy = engine_attention_entropy_per_head(
                core->d_Q, static_cast<const float*>(core->kv_mgr->get_layer_k_ptr(l)),
                seq_len);
        }

        core->step_attention_out(l);
        if (probe) {
            table.probe(dd, tag + "attn_accum.bin", core->d_X_accum, kHidden, l, "AttnAccum");
            P.attn_accum_host = download(core->d_X_accum, kHidden);
        }

        core->step_mlp_norm(l);
        if (probe)
            table.probe(dd, tag + "post_attn_norm.bin", core->d_X_norm, kHidden,
                        l, "PostAttnNorm");

        core->step_mlp_projections(l);
        if (probe)
            P.gate_up = table.probe(dd, tag + "gate_up.bin", core->d_GateUp,
                                    2 * kInter, l, "GateUp");

        core->step_mlp_out(l);
        if (probe) {
            P.accum = table.probe(dd, tag + "accum_out.bin", core->d_X_accum, kHidden,
                                  l, "Accum_out");
            // down_proj's output is fused into the residual, so the pre-residual MLP
            // output is never a tensor in the engine. It is exactly the delta the
            // step added, which IS comparable against the dump.
            const std::vector<float> after = download(core->d_X_accum, kHidden);
            std::vector<float> implied(kHidden);
            for (size_t i = 0; i < kHidden; ++i)
                implied[i] = after[i] - P.attn_accum_host[i];
            try {
                const auto golden_mlp = load_golden_bin(dd, tag + "mlp_out.bin", kHidden);
                const Telemetry t = compute_telemetry(golden_mlp, implied);
                table.row(l, "MlpOut(implied)", t);
                EXPECT_GE(t.cosine_sim, 0.999)
                    << "layer " << l << " implied mlp_out (accum_out - attn_accum) "
                    << "diverged: cosine " << t.cosine_sim;
            } catch (const std::exception&) {
                table.skipped(l, "MlpOut(implied)", "[mlp_out dump absent]");
            }
        }
    }
    table.footer();

    // ========================================================================
    // Assertions on the probed stages.
    // ========================================================================
    // RoPE is the whole point of the GLM-4 bring-up: a wrong pairing still yields
    // plausible magnitudes, so the floor here is deliberately the tightest one.
    for (int l : kProbeLayers) {
        const LayerProbe& P = probes[static_cast<size_t>(l)];
        ASSERT_TRUE(P.q_rope.has_value() && P.k_rope.has_value())
            << "layer " << l << ": RoPE dumps missing -- regenerate with "
               "scripts/generate_glm4_dumps.py";
        EXPECT_GE(P.q_rope->cosine_sim, 0.999)
            << "layer " << l << " q_rope cosine " << P.q_rope->cosine_sim
            << ": the interleaved partial rotation disagrees with the checkpoint's "
               "own apply_rotary_pos_emb";
        EXPECT_GE(P.k_rope->cosine_sim, 0.999)
            << "layer " << l << " k_rope cosine " << P.k_rope->cosine_sim;
        ASSERT_TRUE(P.gate_up.has_value());
        EXPECT_GE(P.gate_up->cosine_sim, 0.999)
            << "layer " << l << " gate_up cosine " << P.gate_up->cosine_sim
            << ": check the fused-projection row order (gate is the FIRST half)";
    }

    // The early layers carry no accumulated drift, so they hold the tight floor.
    {
        const LayerProbe& P0 = probes[0];
        ASSERT_TRUE(P0.input_norm.has_value() && P0.attn_math.has_value() &&
                    P0.accum.has_value());
        EXPECT_GE(P0.input_norm->cosine_sim, 0.999) << "layer 0 input_norm";
        EXPECT_GE(P0.attn_math->cosine_sim, 0.999) << "layer 0 attn_math";
        EXPECT_GE(P0.accum->cosine_sim, 0.999)     << "layer 0 accum_out";
    }

    // ========================================================================
    // Attention entropy: the engine's own distribution vs PyTorch's, per probe
    // layer. Tolerance 5% of the reference value.
    // ========================================================================
    std::cout << "\n[Integration] attention entropy (nats), last query row over "
              << seq_len << " keys; max possible " << std::log(double(seq_len)) << ":\n";
    for (int l : kProbeLayers) {
        const LayerProbe& P = probes[static_cast<size_t>(l)];
        ASSERT_EQ(P.engine_entropy.size(), kHeads);

        // Golden probs are dumped as [num_heads, seq_len]; entropy per head.
        std::vector<float> golden_probs;
        try {
            golden_probs = load_golden_bin(dd, "layer_" + std::to_string(l) +
                                               "_attn_probs.bin",
                                           kHeads * static_cast<size_t>(seq_len));
        } catch (const std::exception& e) {
            GTEST_FAIL() << "attention-probability dump missing for layer " << l
                         << ": " << e.what();
        }

        std::vector<double> golden_entropy(kHeads);
        for (size_t h = 0; h < kHeads; ++h) {
            const float* row = golden_probs.data() + h * static_cast<size_t>(seq_len);
            golden_entropy[h] = compute_entropy(row, static_cast<size_t>(seq_len));
        }

        const double g = mean(golden_entropy);
        const double e = mean(P.engine_entropy);
        const double rel = (g > 1e-9) ? std::abs(e - g) / g : std::abs(e - g);
        std::cout << "  layer " << l << ": reference " << g << "  engine " << e
                  << "  (rel. diff " << rel * 100.0 << "%)\n";

        EXPECT_GT(g, 1e-6) << "layer " << l << " reference entropy is ~0: the dumps "
                              "cannot have come from a multi-token prompt";
        EXPECT_LE(rel, 0.05)
            << "layer " << l << " attention entropy off by " << rel * 100.0
            << "% (reference " << g << " nats, engine " << e << " nats): the engine's "
               "attention distribution has a different shape, not just a different "
               "magnitude";
    }

    // ========================================================================
    // Final logits.
    // ========================================================================
    std::cout << "\n[Integration] final norm, vocab projection and the verdict...\n";
    core->step_final_ops();
    table.header("final stages");
    table.probe(dd, "final_norm_out.bin", core->d_X_norm, kHidden, kNumLayers, "FinalNorm");
    table.footer();

    const std::vector<float> golden_logits = load_golden_bin(dd, "logits_out.bin", kVocab);
    LogitsVerdict verdict;
    ASSERT_TRUE(assert_logits_convergence(core->d_logits, golden_logits, kVocab,
                                          /*min_cosine=*/0.95, /*top_k=*/5, &verdict));
    EXPECT_TRUE(verdict.top1_exact)
        << "greedy next token differs: reference " << verdict.golden_top1
        << " vs engine " << verdict.engine_top1;

    std::cout << "  [SUCCESS] GLM-4 parity: logits cosine " << verdict.cosine
              << ", top-1 " << (verdict.top1_exact ? "exact" : "DIFFERENT")
              << " (reference token " << verdict.golden_top1 << ").\n";
}
