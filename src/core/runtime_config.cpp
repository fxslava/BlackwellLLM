#include "blackwell/runtime_config.h"
#include <algorithm>
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

    // --- batching capability gate --------------------------------------------
    // True batch mode (max_batch_size > 1) and batched prefill drive many token
    // positions through one forward pass. Hybrid linear-attention (SSM) models
    // carry a per-layer RECURRENT state that advances strictly one position at a
    // time and cannot be evaluated across a tile of tokens in parallel -- so they
    // are excluded from batching here, before any VRAM is sized against a width
    // the SSM path can never honour.
    if (request.max_batch_size > 1 && caps.requires_ssm_subsystem)
        throw std::runtime_error(
            "InferenceConfig: max_batch_size > 1 (requested " +
            std::to_string(request.max_batch_size) + ") is unsupported for this model -- it "
            "has " + std::to_string(caps.num_linear_attention_layers) +
            " linear-attention (SSM) layer(s) whose recurrent state advances one "
            "position at a time and cannot be batched. Run batch=1 (single sequence).");

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

    // Linear-projection GEMV<->batched-GEMM crossover (override wins over the
    // tier-2 request default). A threshold of 1 means "always batch when
    // num_tokens > 1"; larger values keep small deltas on the low-latency GEMV.
    rt.batched_gemm_threshold =
        overrides.batched_gemm_threshold.value_or(request.batched_gemm_threshold);
    if (rt.batched_gemm_threshold < 1)
        throw std::invalid_argument("RuntimeConfig: batched_gemm_threshold must be >= 1");

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

    // --- tiered KV prefix-cache sizing -----------------------------------------
    rt.kv_vram_cache_pages = overrides.kv_vram_cache_pages.value_or(0);
    rt.kv_ram_slots        = overrides.kv_ram_slots.value_or(RuntimeConfig::kMirrorDevicePool);
    rt.kv_disk_slots       = overrides.kv_disk_slots.value_or(0);
    rt.kv_spill_path       = overrides.kv_spill_path.value_or(std::string());

    if (rt.kv_vram_cache_pages < 0)
        throw std::invalid_argument("RuntimeConfig: kv_vram_cache_pages must be >= 0");
    if (rt.kv_ram_slots < 0 && rt.kv_ram_slots != RuntimeConfig::kMirrorDevicePool)
        throw std::invalid_argument(
            "RuntimeConfig: kv_ram_slots must be >= 0 (or kMirrorDevicePool)");
    if (rt.kv_disk_slots < 0)
        throw std::invalid_argument("RuntimeConfig: kv_disk_slots must be >= 0");
    if (rt.kv_disk_slots > 0 && rt.kv_spill_path.empty())
        throw std::invalid_argument(
            "RuntimeConfig: kv_disk_slots > 0 requires a non-empty kv_spill_path "
            "(the NVMe spill tier needs a backing file)");

    // --- forced translation languages -------------------------------------------
    // Free-form pass-through (the frontend owns the language list and prompt
    // construction); an empty request normalizes to the "auto" sentinel so tier-3
    // consumers never observe an unset value.
    rt.source_language =
        request.source_language.empty() ? std::string("auto") : request.source_language;
    rt.target_language =
        request.target_language.empty() ? std::string("auto") : request.target_language;

    // --- audio streaming plan (ms intent -> validated soft-token counts) --------
    // Resolved when the frontend opts in (.enable) OR any streaming override is
    // present. Requests are in ms; one soft-token == kAudioSoftTokenMs, so the ms
    // values must land on whole soft-token boundaries (else the sliding window's
    // overlap would not align to injected tokens and reconciliation would compare
    // misaligned spans). Left all-zero / disabled otherwise.
    {
        const auto& a = request.audio_streaming;
        const bool any_override =
            overrides.audio_streaming_mode ||
            overrides.audio_window_size_ms || overrides.audio_hop_size_ms ||
            overrides.audio_overlap_reconciliation || overrides.audio_reconciliation_threshold ||
            overrides.audio_max_reconciliation_rewind_tokens ||
            overrides.audio_left_edge_ms || overrides.audio_right_edge_ms;

        if (a.enable || any_override) {
            const AudioStreamingMode mode = overrides.audio_streaming_mode.value_or(a.mode);
            const int win_ms = overrides.audio_window_size_ms.value_or(a.window_size_ms);
            const int hop_ms = overrides.audio_hop_size_ms.value_or(a.hop_size_ms);
            const float thr =
                overrides.audio_reconciliation_threshold.value_or(a.overlap_reconciliation_threshold);
            const int cap =
                overrides.audio_max_reconciliation_rewind_tokens.value_or(a.max_reconciliation_rewind_tokens);
            const bool want_recon =
                overrides.audio_overlap_reconciliation.value_or(a.overlap_reconciliation);
            const int left_ms  = overrides.audio_left_edge_ms.value_or(a.left_edge_ms);
            const int right_ms = overrides.audio_right_edge_ms.value_or(a.right_edge_ms);

            if (win_ms <= 0 || hop_ms <= 0)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming window/hop ms must be > 0");
            if (win_ms % kAudioSoftTokenMs != 0 || hop_ms % kAudioSoftTokenMs != 0)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming window_size_ms (" + std::to_string(win_ms) +
                    ") and hop_size_ms (" + std::to_string(hop_ms) +
                    ") must each be a whole multiple of one soft-token (" +
                    std::to_string(kAudioSoftTokenMs) + " ms) so the sliding-window overlap "
                    "aligns to injected tokens");
            const int wt = win_ms / kAudioSoftTokenMs;
            const int ht = hop_ms / kAudioSoftTokenMs;
            if (ht < 1 || wt < ht)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming requires window_tokens (" + std::to_string(wt) +
                    ") >= hop_tokens (" + std::to_string(ht) + ") >= 1");
            if (thr <= 0.0f || thr > 1.0f)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming overlap_reconciliation_threshold must be "
                    "in (0, 1]");
            if (cap < 0)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming max_reconciliation_rewind_tokens must be >= 0");

            // CenterSlice edge margins: same whole-soft-token alignment rule as
            // window/hop (a fractional edge would split a soft-token).
            if (left_ms < 0 || right_ms < 0)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming left/right edge ms must be >= 0");
            if (left_ms % kAudioSoftTokenMs != 0 || right_ms % kAudioSoftTokenMs != 0)
                throw std::invalid_argument(
                    "InferenceConfig: audio_streaming left_edge_ms (" + std::to_string(left_ms) +
                    ") and right_edge_ms (" + std::to_string(right_ms) +
                    ") must each be a whole multiple of one soft-token (" +
                    std::to_string(kAudioSoftTokenMs) + " ms)");
            const int lt = left_ms / kAudioSoftTokenMs;
            const int kt = right_ms / kAudioSoftTokenMs;
            if (mode == AudioStreamingMode::CenterSlice) {
                // K >= 1: the pause-commit needs at least one tentative edge token
                // to complete the phrase (and the resume rollback to un-commit).
                if (kt < 1)
                    throw std::invalid_argument(
                        "InferenceConfig: CenterSlice requires right_edge_ms >= one soft-token (" +
                        std::to_string(kAudioSoftTokenMs) + " ms): the VAD-pause commit margin K "
                        "must be at least 1 token");
                // Center-span continuity: window W sliding by hop h keeps consecutive
                // center spans [w0+L, w0+W-K) contiguous iff L + K + h <= W; a larger
                // hop would leave audio no window's center ever covers.
                if (lt + kt + ht > wt)
                    throw std::invalid_argument(
                        "InferenceConfig: CenterSlice requires left_edge + right_edge + hop <= "
                        "window (" + std::to_string(lt) + " + " + std::to_string(kt) + " + " +
                        std::to_string(ht) + " > " + std::to_string(wt) + " tokens): consecutive "
                        "windows' center spans would leave an uncovered gap — widen the window or "
                        "shrink the edges/hop");
            }

            rt.audio_streaming.mode                   = mode;
            rt.audio_streaming.window_tokens          = wt;
            rt.audio_streaming.hop_tokens             = ht;
            rt.audio_streaming.overlap_tokens         = wt - ht;
            rt.audio_streaming.reconciliation_threshold = thr;
            rt.audio_streaming.max_rewind_tokens      = std::min(cap, wt - ht);
            rt.audio_streaming.left_edge_tokens       = lt;
            rt.audio_streaming.right_edge_tokens      = kt;

            // Capability gate: BOTH modes rewind the (position-addressed) KV --
            // Reconcile on tail divergence, CenterSlice on pause/resume -- but a
            // hybrid model's recurrent linear-attention (SSM) state cannot be
            // rewound, so streaming is disabled there and the frontend falls back
            // to Phase-1 tail-slicing. In Reconcile mode the explicit
            // overlap_reconciliation opt-out also disables the plan; CenterSlice
            // ignores that knob (it never runs the cosine reconcile).
            const bool mode_active =
                (mode == AudioStreamingMode::CenterSlice) || want_recon;
            rt.audio_streaming.enabled = mode_active && !caps.requires_ssm_subsystem;
        }
    }

    return rt;
}

} // namespace blackwell
