// ============================================================================
// Pure-logic tests for the tier-3 config builder (build_and_validate_runtime).
// No GPU, no checkpoint -- exercises the preset-expansion + validation rules in
// isolation, so the branching veto / head_dim cap / context bound stay pinned.
// ============================================================================
#include <gtest/gtest.h>
#include <stdexcept>

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
