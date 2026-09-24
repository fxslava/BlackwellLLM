// ============================================================================
// Pure-logic tests for the tier-3 config builder (build_and_validate_runtime).
// No GPU, no checkpoint -- exercises the preset-expansion + validation rules in
// isolation, so the branching veto / head_dim cap / context bound stay pinned.
// ============================================================================
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "blackwell/config.h"
#include "blackwell/runtime_config.h"

using blackwell::InferenceConfig;
using blackwell::RuntimeConfig;
using blackwell::RuntimeOverrides;
using blackwell::build_and_validate_runtime;
using blackwell::derive_capabilities;
using KVCacheMode = BlackwellEngine::KVCacheMode;

namespace {

// Minimal dense (all full-attention) topology, à la Qwen2.5/Llama.
ModelConfig dense_model(size_t head_dim = 128) {
    ModelConfig m{};
    m.hidden_dim = 4096;
    m.num_layers = 32;
    m.num_attention_heads = 32;
    m.num_key_value_heads = 8;
    m.head_dim = head_dim;
    m.max_position_embeddings = 4096;
    m.attn_output_gate = false;       // generic attention path
    // layer_types empty == legacy uniform full-attention.
    return m;
}

// Hybrid topology: a couple of linear-attention (SSM) layers among full ones,
// gated head_dim-256 attention (Qwen3.5 shape).
ModelConfig hybrid_model() {
    ModelConfig m = dense_model(/*head_dim=*/256);
    m.attn_output_gate = true;        // dedicated full-attention path
    m.num_layers = 4;
    m.layer_types = {AttnKind::Linear, AttnKind::Linear, AttnKind::Linear, AttnKind::Full};
    return m;
}

} // namespace

// Capability derivation mirrors the topology.
TEST(RuntimeConfig, DeriveCapabilitiesCountsLayers) {
    const auto dense = derive_capabilities(dense_model());
    EXPECT_EQ(dense.num_full_attention_layers, 32);
    EXPECT_EQ(dense.num_linear_attention_layers, 0);
    EXPECT_FALSE(dense.is_hybrid);
    EXPECT_FALSE(dense.requires_ssm_subsystem);

    const auto hy = derive_capabilities(hybrid_model());
    EXPECT_EQ(hy.num_linear_attention_layers, 3);
    EXPECT_EQ(hy.num_full_attention_layers, 1);
    EXPECT_TRUE(hy.is_hybrid);
    EXPECT_TRUE(hy.requires_ssm_subsystem);
}

// Default request on a dense model: Continuous, all layers resident.
TEST(RuntimeConfig, DenseDefaultPlanIsContinuous) {
    const auto m = dense_model();
    InferenceConfig req;
    req.max_context_length = 2048;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_EQ(rt.kv_mode, KVCacheMode::Continuous);
    EXPECT_EQ(rt.max_seq_len, 2048u);
    EXPECT_FALSE(rt.uses_dedicated_full_attention);
}

// require_branching on a dense model -> Paged, allowed.
TEST(RuntimeConfig, DenseBranchingForcesPaged) {
    const auto m = dense_model();
    InferenceConfig req;
    req.require_branching = true;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_EQ(rt.kv_mode, KVCacheMode::Paged);
}

// require_branching on a hybrid SSM model -> rejected (un-snapshot-able state).
TEST(RuntimeConfig, HybridBranchingRejected) {
    const auto m = hybrid_model();
    InferenceConfig req;
    req.require_branching = true;
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::runtime_error);
}

// Batched requests (max_batch_size > 1) on a hybrid SSM model -> rejected: the
// recurrent linear-attention state cannot advance a tile of tokens in parallel.
TEST(RuntimeConfig, HybridBatchingRejected) {
    const auto m = hybrid_model();
    InferenceConfig req;
    req.max_batch_size = 4;                       // require_branching stays false
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::runtime_error);
}

// The same batched request on a dense model is accepted (no SSM state to serialize).
TEST(RuntimeConfig, DenseBatchingAllowed) {
    const auto m = dense_model();
    InferenceConfig req;
    req.max_batch_size = 4;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_EQ(rt.max_sequences, 4u);
}

// Hybrid model in (explicitly overridden) Paged mode is fine: its head_dim-256
// gated layers use the dedicated path, so the paged-flash 128-cap is waived.
TEST(RuntimeConfig, HybridPagedAllowedViaDedicatedPath) {
    const auto m = hybrid_model();
    InferenceConfig req;                         // require_branching stays false
    RuntimeOverrides ov;
    ov.kv_mode = KVCacheMode::Paged;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req, ov);
    EXPECT_EQ(rt.kv_mode, KVCacheMode::Paged);
    EXPECT_TRUE(rt.uses_dedicated_full_attention);
}

// A dense model with head_dim > 128 cannot use the paged-flash kernel.
TEST(RuntimeConfig, PagedRejectsOversizedHeadDimWithoutDedicatedPath) {
    const auto m = dense_model(/*head_dim=*/256);
    InferenceConfig req;
    req.require_branching = true;                 // forces Paged
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::runtime_error);
}

// Context length beyond the model's trained positional range is rejected.
TEST(RuntimeConfig, ContextBeyondPositionalRangeRejected) {
    const auto m = dense_model();                // max_position_embeddings = 4096
    InferenceConfig req;
    req.max_context_length = 8192;
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
}

// Degenerate sizing requests are rejected.
TEST(RuntimeConfig, ZeroContextRejected) {
    const auto m = dense_model();
    InferenceConfig req;
    req.max_context_length = 0;
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
}

// Low-level override: explicit residency split survives into the plan.
TEST(RuntimeConfig, NumGpuLayersOverrideApplied) {
    const auto m = dense_model();
    InferenceConfig req;
    RuntimeOverrides ov;
    ov.num_gpu_layers = 8;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req, ov);
    EXPECT_EQ(rt.num_gpu_layers, 8u);
}

// GEMV<->batched-GEMM crossover: default, tier-2 request value, override
// precedence, and the >= 1 guard.
TEST(RuntimeConfig, BatchedGemmThresholdResolution) {
    const auto m = dense_model();
    const auto caps = derive_capabilities(m);

    // Default (nothing set) is 16.
    EXPECT_EQ(build_and_validate_runtime(m, caps, InferenceConfig{}).batched_gemm_threshold, 16);

    // Tier-2 request value flows through when no override is present.
    InferenceConfig req;
    req.batched_gemm_threshold = 8;
    EXPECT_EQ(build_and_validate_runtime(m, caps, req).batched_gemm_threshold, 8);

    // Override wins over the request.
    RuntimeOverrides ov;
    ov.batched_gemm_threshold = 32;
    EXPECT_EQ(build_and_validate_runtime(m, caps, req, ov).batched_gemm_threshold, 32);

    // A threshold of 1 is legal ("always batch when num_tokens > 1").
    RuntimeOverrides ov1;
    ov1.batched_gemm_threshold = 1;
    EXPECT_EQ(build_and_validate_runtime(m, caps, InferenceConfig{}, ov1).batched_gemm_threshold, 1);
}

TEST(RuntimeConfig, BatchedGemmThresholdBelowOneRejected) {
    const auto m = dense_model();
    RuntimeOverrides ov;
    ov.batched_gemm_threshold = 0;
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), InferenceConfig{}, ov),
                 std::invalid_argument);
}

// ── audio streaming plan (dynamic overlap reconciliation) ───────────────────

// Default request leaves the streaming plan inert (frontend has not opted in).
TEST(RuntimeConfig, AudioStreamingInertByDefault) {
    const auto m = dense_model();
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), InferenceConfig{});
    EXPECT_FALSE(rt.audio_streaming.enabled);
    EXPECT_EQ(rt.audio_streaming.window_tokens, 0);
    EXPECT_EQ(rt.audio_streaming.hop_tokens, 0);
}

// Opt-in resolves ms intent to whole soft-token counts (160 ms/token) and derives
// the overlap span; on a dense model reconciliation is enabled.
TEST(RuntimeConfig, AudioStreamingResolvesMsToTokens) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;            // defaults: 2240 ms / 320 ms
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_TRUE(rt.audio_streaming.enabled);
    EXPECT_EQ(rt.audio_streaming.window_tokens, 14);   // 2240 / 160
    EXPECT_EQ(rt.audio_streaming.hop_tokens, 2);        // 320 / 160
    EXPECT_EQ(rt.audio_streaming.overlap_tokens, 12);   // 14 - 2
    EXPECT_FLOAT_EQ(rt.audio_streaming.reconciliation_threshold, 0.999f);
    EXPECT_EQ(rt.audio_streaming.max_rewind_tokens, 8);  // min(8, overlap 12)
}

// A window/hop not on a whole soft-token boundary is rejected (would misalign the
// overlap against the injected tokens).
TEST(RuntimeConfig, AudioStreamingRejectsUnalignedMs) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.hop_size_ms = 300;        // not a multiple of 160
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
}

// window_tokens must be >= hop_tokens.
TEST(RuntimeConfig, AudioStreamingRejectsHopExceedingWindow) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.window_size_ms = 320;     // 2 tokens
    req.audio_streaming.hop_size_ms    = 640;     // 4 tokens > window
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
}

// Threshold must be in (0, 1]; cap must be >= 0.
TEST(RuntimeConfig, AudioStreamingRejectsBadThresholdAndCap) {
    const auto m = dense_model();
    const auto caps = derive_capabilities(m);
    {   InferenceConfig req; req.audio_streaming.enable = true;
        req.audio_streaming.overlap_reconciliation_threshold = 1.5f;
        EXPECT_THROW(build_and_validate_runtime(m, caps, req), std::invalid_argument); }
    {   InferenceConfig req; req.audio_streaming.enable = true;
        req.audio_streaming.overlap_reconciliation_threshold = 0.0f;
        EXPECT_THROW(build_and_validate_runtime(m, caps, req), std::invalid_argument); }
    {   InferenceConfig req; req.audio_streaming.enable = true;
        req.audio_streaming.max_reconciliation_rewind_tokens = -1;
        EXPECT_THROW(build_and_validate_runtime(m, caps, req), std::invalid_argument); }
}

// The rewind cap is clamped to the overlap span (can never rewind past it).
TEST(RuntimeConfig, AudioStreamingClampsRewindCapToOverlap) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.max_reconciliation_rewind_tokens = 99;   // > overlap 12
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_EQ(rt.audio_streaming.max_rewind_tokens, rt.audio_streaming.overlap_tokens);  // 12
}

// SSM/hybrid capability gate: the plan still resolves the token geometry, but
// reconciliation is DISABLED (recurrent state can't be rewound+re-injected).
TEST(RuntimeConfig, AudioStreamingReconciliationGatedOffForHybrid) {
    const auto m = hybrid_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_FALSE(rt.audio_streaming.enabled);        // gated off
    EXPECT_EQ(rt.audio_streaming.window_tokens, 14); // geometry still resolved
    EXPECT_EQ(rt.audio_streaming.hop_tokens, 2);
}

// A streaming OVERRIDE activates plan resolution even when .enable is false, and
// wins over the request defaults.
TEST(RuntimeConfig, AudioStreamingOverrideActivatesAndWins) {
    const auto m = dense_model();
    InferenceConfig req;                              // enable stays false
    RuntimeOverrides ov;
    ov.audio_window_size_ms = 1600;                  // 10 tokens
    ov.audio_hop_size_ms    = 160;                   // 1 token
    ov.audio_reconciliation_threshold = 0.995f;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req, ov);
    EXPECT_TRUE(rt.audio_streaming.enabled);
    EXPECT_EQ(rt.audio_streaming.window_tokens, 10);
    EXPECT_EQ(rt.audio_streaming.hop_tokens, 1);
    EXPECT_EQ(rt.audio_streaming.overlap_tokens, 9);
    EXPECT_FLOAT_EQ(rt.audio_streaming.reconciliation_threshold, 0.995f);
}

// The reconciliation on/off override is honored (off => plan disabled, geometry kept).
TEST(RuntimeConfig, AudioStreamingReconciliationOverrideOff) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    RuntimeOverrides ov;
    ov.audio_overlap_reconciliation = false;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req, ov);
    EXPECT_FALSE(rt.audio_streaming.enabled);
    EXPECT_EQ(rt.audio_streaming.window_tokens, 14);
}

// ── CenterSlice mode (append-only center-token streaming) ────────────────────

// The default mode stays Reconcile, but the edge margins resolve to tokens either
// way (they are inert outside CenterSlice).
TEST(RuntimeConfig, AudioStreamingDefaultModeIsReconcileWithResolvedEdges) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_EQ(rt.audio_streaming.mode, blackwell::AudioStreamingMode::Reconcile);
    EXPECT_EQ(rt.audio_streaming.left_edge_tokens, 2);    // 320 / 160
    EXPECT_EQ(rt.audio_streaming.right_edge_tokens, 3);   // 480 / 160
}

// CenterSlice resolves edge ms -> tokens and is enabled on a dense model even
// with overlap_reconciliation opted out (the cosine knob belongs to the other mode).
TEST(RuntimeConfig, CenterSliceResolvesEdgesAndIgnoresReconcileOptOut) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.mode = blackwell::AudioStreamingMode::CenterSlice;
    req.audio_streaming.overlap_reconciliation = false;   // must not matter
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_TRUE(rt.audio_streaming.enabled);
    EXPECT_EQ(rt.audio_streaming.mode, blackwell::AudioStreamingMode::CenterSlice);
    EXPECT_EQ(rt.audio_streaming.window_tokens, 14);
    EXPECT_EQ(rt.audio_streaming.hop_tokens, 2);
    EXPECT_EQ(rt.audio_streaming.left_edge_tokens, 2);
    EXPECT_EQ(rt.audio_streaming.right_edge_tokens, 3);
}

// Edge margins must land on whole soft-token boundaries, like window/hop.
TEST(RuntimeConfig, CenterSliceRejectsUnalignedEdgeMs) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.mode = blackwell::AudioStreamingMode::CenterSlice;
    req.audio_streaming.left_edge_ms = 300;               // not a multiple of 160
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
}

// Negative edges are rejected in every mode.
TEST(RuntimeConfig, AudioStreamingRejectsNegativeEdgeMs) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.right_edge_ms = -160;
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
}

// K >= 1: a zero pause-commit margin leaves nothing to complete the phrase with.
TEST(RuntimeConfig, CenterSliceRejectsZeroRightEdge) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.mode = blackwell::AudioStreamingMode::CenterSlice;
    req.audio_streaming.right_edge_ms = 0;
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
    // ...but right_edge_ms = 0 is fine in Reconcile mode (the knob is inert there).
    req.audio_streaming.mode = blackwell::AudioStreamingMode::Reconcile;
    EXPECT_NO_THROW(build_and_validate_runtime(m, derive_capabilities(m), req));
}

// Center-span continuity: left + right + hop must fit in the window, else audio
// between consecutive center spans would never be committed.
TEST(RuntimeConfig, CenterSliceRejectsGapGeometry) {
    const auto m = dense_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.mode = blackwell::AudioStreamingMode::CenterSlice;
    req.audio_streaming.window_size_ms = 640;   // 4 tokens
    req.audio_streaming.hop_size_ms    = 320;   // 2 tokens; defaults L=2, K=3 -> 2+3+2 > 4
    EXPECT_THROW(build_and_validate_runtime(m, derive_capabilities(m), req),
                 std::invalid_argument);
    // The boundary case (L + K + h == W) is legal: centers tile exactly.
    req.audio_streaming.window_size_ms = 1120;  // 7 tokens == 2 + 3 + 2
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_EQ(rt.audio_streaming.window_tokens, 7);
}

// Mode + edge overrides activate resolution (even with .enable false) and win
// over the request values.
TEST(RuntimeConfig, CenterSliceModeAndEdgeOverridesWin) {
    const auto m = dense_model();
    InferenceConfig req;                                  // enable stays false
    RuntimeOverrides ov;
    ov.audio_streaming_mode = blackwell::AudioStreamingMode::CenterSlice;
    ov.audio_left_edge_ms  = 160;                        // 1 token
    ov.audio_right_edge_ms = 160;                        // K = 1 token
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req, ov);
    EXPECT_TRUE(rt.audio_streaming.enabled);
    EXPECT_EQ(rt.audio_streaming.mode, blackwell::AudioStreamingMode::CenterSlice);
    EXPECT_EQ(rt.audio_streaming.left_edge_tokens, 1);
    EXPECT_EQ(rt.audio_streaming.right_edge_tokens, 1);
}

// The SSM capability gate applies to CenterSlice too: the pause/resume pointer
// rollback rewinds the KV, which a recurrent state cannot follow.
TEST(RuntimeConfig, CenterSliceGatedOffForHybrid) {
    const auto m = hybrid_model();
    InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.mode = blackwell::AudioStreamingMode::CenterSlice;
    const auto rt = build_and_validate_runtime(m, derive_capabilities(m), req);
    EXPECT_FALSE(rt.audio_streaming.enabled);             // gated off
    EXPECT_EQ(rt.audio_streaming.mode, blackwell::AudioStreamingMode::CenterSlice);
    EXPECT_EQ(rt.audio_streaming.left_edge_tokens, 2);    // geometry still resolved
}

// ============================================================================
// ConfigLoader: the GLM-4 parse branches (tier-1 facts, not the tier-3 plan).
// These write a throwaway config.json and read it back through the real loader —
// the mapping from checkpoint JSON to ModelConfig is exactly where a GLM-4 bring-up
// goes silently wrong (a defaulted eps, an unrotated pairing, a missed alias).
// ============================================================================
namespace {

// Writes `body` as a config.json in a uniquely named temp dir; returns its path.
// The dir is left behind (a few hundred bytes in TEMP) rather than risking a
// recursive remove in a test process.
std::string write_temp_config(const char* tag, const std::string& body) {
    static int counter = 0;
    const auto dir = std::filesystem::temp_directory_path() /
                     ("blackwell_cfg_" + std::string(tag) + "_" + std::to_string(++counter));
    std::filesystem::create_directories(dir);
    const auto path = dir / "config.json";
    std::ofstream out(path);
    out << body;
    out.close();
    return path.string();
}

// GLM-4-9B-Chat as transformers >= 4.46 exports it (model_type "glm"): HF-native
// tensor names, partial rotary at the document root, no sandwich norms.
const char* kGlm4HfConfig = R"({
  "model_type": "glm",
  "architectures": ["GlmForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 13696,
  "num_hidden_layers": 40,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "head_dim": 128,
  "vocab_size": 151552,
  "rms_norm_eps": 1.5625e-07,
  "rope_theta": 10000.0,
  "partial_rotary_factor": 0.5,
  "attention_bias": true,
  "tie_word_embeddings": false,
  "max_position_embeddings": 131072
})";

} // namespace

TEST(ConfigLoaderGlm4, HfNativeGlmTopology) {
    const auto path = write_temp_config("glm_hf", kGlm4HfConfig);
    const ModelConfig c = ConfigLoader::load_from_json(path);

    EXPECT_EQ(c.hidden_dim, 4096u);
    EXPECT_EQ(c.intermediate_dim, 13696u);
    EXPECT_EQ(c.num_layers, 40u);
    EXPECT_EQ(c.num_attention_heads, 32u);
    EXPECT_EQ(c.num_key_value_heads, 2u);
    EXPECT_EQ(c.head_dim, 128u);
    EXPECT_EQ(c.vocab_size, 151552u);
    EXPECT_EQ(c.max_position_embeddings, 131072u);

    // The three GLM-4 deltas.
    EXPECT_EQ(c.rope_pairing, RopePairing::Interleaved);
    EXPECT_EQ(c.rotary_dim, 64u);                  // head_dim * 0.5
    EXPECT_TRUE(c.mlp_fused_gate_up);
    EXPECT_FALSE(c.has_sandwich_norms);            // "glm", not "glm4"

    // Parsed, not defaulted: 1.5625e-07 is ~6x below the loader's 1e-6 fallback.
    EXPECT_FLOAT_EQ(c.rms_norm_eps, 1.5625e-07f);
    EXPECT_TRUE(c.has_qkv_bias);
    EXPECT_FALSE(c.tie_word_embeddings);
    EXPECT_FALSE(c.norm_add_unit_offset);          // GLM uses plain-weight RMSNorm
    EXPECT_FALSE(c.has_qk_norm);
    EXPECT_EQ(c.weight_prefix, "model.");
    EXPECT_EQ(c.rope_scaling_type, 0);             // plain theta stretch
}

// The 0414 family ("glm4") additionally normalizes each sub-layer output.
TEST(ConfigLoaderGlm4, Glm4FamilyEnablesSandwichNorms) {
    std::string body = kGlm4HfConfig;
    body.replace(body.find("\"glm\""), 5, "\"glm4\"");
    const auto path = write_temp_config("glm4_0414", body);
    const ModelConfig c = ConfigLoader::load_from_json(path);

    EXPECT_TRUE(c.has_sandwich_norms);
    EXPECT_TRUE(c.mlp_fused_gate_up);
    EXPECT_EQ(c.rope_pairing, RopePairing::Interleaved);
}

// THUDM parameter spellings resolve to the same topology as the HF ones. The
// canonical key always wins when both are present (asserted by the HF case above,
// which carries only canonical keys).
TEST(ConfigLoaderGlm4, ThudmParameterAliases) {
    const auto path = write_temp_config("glm_alias", R"({
      "model_type": "glm",
      "architectures": ["GlmForCausalLM"],
      "hidden_size": 4096,
      "ffn_hidden_size": 13696,
      "num_layers": 40,
      "num_attention_heads": 32,
      "multi_query_group_num": 2,
      "kv_channels": 128,
      "padded_vocab_size": 151552,
      "layernorm_epsilon": 1.5625e-07,
      "rope_ratio": 500,
      "partial_rotary_factor": 0.5,
      "add_qkv_bias": true,
      "seq_length": 131072
    })");
    const ModelConfig c = ConfigLoader::load_from_json(path);

    EXPECT_EQ(c.intermediate_dim, 13696u);         // ffn_hidden_size
    EXPECT_EQ(c.num_layers, 40u);                  // num_layers
    EXPECT_EQ(c.num_key_value_heads, 2u);          // multi_query_group_num
    EXPECT_EQ(c.head_dim, 128u);                   // kv_channels
    EXPECT_EQ(c.vocab_size, 151552u);              // padded_vocab_size
    EXPECT_EQ(c.max_position_embeddings, 131072u); // seq_length
    EXPECT_FLOAT_EQ(c.rms_norm_eps, 1.5625e-07f);  // layernorm_epsilon
    EXPECT_FLOAT_EQ(c.rope_theta, 5000000.0f);     // 10000 * rope_ratio
    EXPECT_TRUE(c.has_qkv_bias);                   // add_qkv_bias
    EXPECT_EQ(c.rotary_dim, 64u);
}

// A THUDM-native checkpoint must be refused with an actionable message, not
// mis-loaded: its weights live in a different namespace entirely
// (transformer.encoder.* with a fused query_key_value).
TEST(ConfigLoaderGlm4, ThudmNativeCheckpointRejected) {
    const auto path = write_temp_config("chatglm", R"({
      "model_type": "chatglm",
      "architectures": ["ChatGLMForConditionalGeneration"],
      "hidden_size": 4096,
      "ffn_hidden_size": 13696,
      "num_layers": 40,
      "num_attention_heads": 32,
      "multi_query_group_num": 2,
      "kv_channels": 128,
      "padded_vocab_size": 151552,
      "layernorm_epsilon": 1.5625e-07,
      "add_qkv_bias": true,
      "seq_length": 131072
    })");
    try {
        ConfigLoader::load_from_json(path);
        FAIL() << "expected a THUDM-native rejection";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("transformer.encoder"), std::string::npos) << msg;
        EXPECT_NE(msg.find("gate_up_proj"), std::string::npos) << msg;
    }
}

// Non-GLM checkpoints are untouched by every branch above: full rotary span,
// half-split pairing, separate gate/up, no sandwich norms.
TEST(ConfigLoaderGlm4, NonGlmCheckpointKeepsHalfSplitFullRotary) {
    const auto path = write_temp_config("qwen2", R"({
      "model_type": "qwen2",
      "architectures": ["Qwen2ForCausalLM"],
      "hidden_size": 3584,
      "intermediate_size": 18944,
      "num_hidden_layers": 28,
      "num_attention_heads": 28,
      "num_key_value_heads": 4,
      "vocab_size": 152064,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "max_position_embeddings": 32768
    })");
    const ModelConfig c = ConfigLoader::load_from_json(path);

    EXPECT_EQ(c.rope_pairing, RopePairing::HalfSplit);
    EXPECT_EQ(c.rotary_dim, c.head_dim);           // 128, full rotary
    EXPECT_FALSE(c.mlp_fused_gate_up);
    EXPECT_FALSE(c.has_sandwich_norms);
    EXPECT_TRUE(c.has_qkv_bias);                   // qwen2 hardcodes it
}
