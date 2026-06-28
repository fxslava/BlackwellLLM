#include "blackwell/runtime_config.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace blackwell {

using KVCacheMode = BlackwellEngine::KVCacheMode;

// SIZE_MAX ("all resident") may be overridden by BLACKWELL_GPU_LAYERS so the VRAM
// split is tunable without touching any call site. Centralized here (was an
// engine.cpp static) so the env contract lives with the rest of plan resolution.
static size_t resolve_num_gpu_layers(const std::optional<size_t>& override_val) {
    if (override_val && *override_val != RuntimeConfig::kAllLayersResident)
        return *override_val;
    if (const char* env = std::getenv("BLACKWELL_GPU_LAYERS")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env && *end == '\0') return static_cast<size_t>(v);
        throw std::invalid_argument(
            "BLACKWELL_GPU_LAYERS is not a valid non-negative integer: " + std::string(env));
    }
    return RuntimeConfig::kAllLayersResident;
}

ModelCapabilities derive_capabilities(const ModelConfig& model) {
    ModelCapabilities caps;
    int linear = 0, full = 0;
    for (AttnKind k : model.layer_types)
        (k == AttnKind::Linear ? linear : full)++;
    if (model.layer_types.empty())
        full = static_cast<int>(model.num_layers);   // legacy uniform full-attention

    caps.num_linear_attention_layers = linear;
    caps.num_full_attention_layers   = full;
    caps.requires_ssm_subsystem      = linear > 0;
    caps.is_hybrid                   = linear > 0 && full > 0;
    // Finalized by the engine once kv_mode is known: branching additionally
    // requires the paged cache. Left false here (topology says only whether the
    // model COULD branch, i.e. has no recurrent SSM state -- see the builder).
    caps.supports_cow_branching      = false;
    return caps;
}

RuntimeConfig build_and_validate_runtime(const ModelConfig& model,
                                         const ModelCapabilities& caps,
                                         const InferenceConfig& request,
                                         const RuntimeOverrides& overrides) {
    RuntimeConfig rt;

    // --- sizing ---------------------------------------------------------------
    if (request.max_context_length == 0)
        throw std::invalid_argument(
            "InferenceConfig: max_context_length must be > 0");
    if (request.max_batch_size == 0)
        throw std::invalid_argument(
            "InferenceConfig: max_batch_size must be >= 1");
    if (model.max_position_embeddings != 0 &&
        request.max_context_length > model.max_position_embeddings)
        throw std::invalid_argument(
            "InferenceConfig: max_context_length (" +
            std::to_string(request.max_context_length) +
            ") exceeds the model's trained positional range max_position_embeddings (" +
            std::to_string(model.max_position_embeddings) + ")");
    rt.max_seq_len   = request.max_context_length;
    rt.max_sequences = request.max_batch_size;

    // --- branching capability gate -------------------------------------------
    // The model can branch at all only if it carries no recurrent linear-attention
    // (SSM) state -- that state is deliberately not snapshot-able. A request to
    // branch on such a model is rejected here, before any VRAM is touched, instead
    // of failing later at the first fork().
    const bool model_can_branch = caps.num_linear_attention_layers == 0;
    if (request.require_branching && !model_can_branch)
        throw std::runtime_error(
            "InferenceConfig: require_branching=true is unsupported for this model -- it "
            "has " + std::to_string(caps.num_linear_attention_layers) +
            " linear-attention (SSM) layer(s) whose recurrent state cannot be "
            "snapshot for Copy-on-Write fork/rewind. Run a single linear sequence.");

    // --- KV-cache strategy ----------------------------------------------------
    // Branching forces Paged; otherwise default Continuous. An explicit override
    // wins either way (e.g. exercising the paged cache without forking).
    rt.kv_mode = overrides.kv_mode.value_or(
        request.require_branching ? KVCacheMode::Paged : KVCacheMode::Continuous);

    rt.paged_branch_factor = overrides.paged_branch_factor.value_or(4);
    if (rt.paged_branch_factor < 1)
        throw std::invalid_argument("RuntimeConfig: paged_branch_factor must be >= 1");

    // --- attention dispatch / head_dim validation -----------------------------
    // Qwen3.5 hybrid gated attention (head_dim 256) uses a dedicated naive path,
    // so the paged-flash 128-cap does not apply to it. For every other model the
    // paged kernel must actually be able to serve the head dimension.
    rt.uses_dedicated_full_attention = model.attn_output_gate;
    if (rt.kv_mode == KVCacheMode::Paged && !rt.uses_dedicated_full_attention &&
        model.head_dim > static_cast<size_t>(KernelLimits::kPagedFlashHeadDimMax))
        throw std::runtime_error(
            "RuntimeConfig: Paged KV cache requires head_dim <= " +
            std::to_string(KernelLimits::kPagedFlashHeadDimMax) +
            " (paged-flash kernel limit), but the model's head_dim is " +
            std::to_string(model.head_dim) +
            ". Use KVCacheMode::Continuous for this model.");

    // --- weight residency -----------------------------------------------------
    rt.num_gpu_layers = resolve_num_gpu_layers(overrides.num_gpu_layers);

    return rt;
}

} // namespace blackwell
