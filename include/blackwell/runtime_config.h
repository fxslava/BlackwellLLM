#pragma once
#include "blackwell/config.h"   // ModelConfig (tier 1)
#include "blackwell/engine.h"   // BlackwellEngine::KVCacheMode, ModelCapabilities
#include <cstddef>
#include <functional>
#include <optional>
#include <string>

// ============================================================================
// Three-tier configuration pipeline ("video-codec" model).
//
// Like an industrial codec (x264/FFmpeg), settings flow through three layers of
// decreasing abstraction and increasing concreteness:
//
//   ModelConfig     (tier 1)  immutable FACTS parsed from the checkpoint's
//                             config.json -- topology you cannot change
//                             (head_dim, num_layers, norm_add_unit_offset, ...).
//                             Lives in blackwell/config.h.
//
//   InferenceConfig (tier 2)  what the USER/API asks for -- a hardware-agnostic
//                             "preset + target": context length, batch width,
//                             whether tree-search branching is needed.
//
//   RuntimeConfig   (tier 3)  the fully-resolved EXECUTION PLAN the engine and
//                             memory managers actually consume. No env lookups,
//                             no "auto" sentinels, no defaults survive here --
//                             every field is decided and validated.
//
// build_and_validate_runtime() is the codec's preset-expansion + param-validation
// pass: it reconciles tiers 1+2 (plus optional low-level overrides) into a tier-3
// plan, or throws an actionable error when the request is unsatisfiable.
// ============================================================================

namespace blackwell {

// One Ultravox audio soft-token spans conv_stride(2) * stack_factor(8) mel frames
// at the Whisper 100-frame/s log-mel rate = 160 ms. Streaming requests are given in
// hardware/stack-agnostic MILLISECONDS and resolved to whole soft-token counts here.
// MUST match audio::kMelFramesPerSoftToken (src/audio/sliding_audio_window.h): 16
// mel frames * 10 ms = 160 ms.
inline constexpr int kAudioSoftTokenMs = 160;

// How the streaming frontend maintains the LLM KV across sliding-window hops.
//   Reconcile   — cosine-compare the window's overlap tail against the injected
//                 history; on divergence, rollback + re-inject the corrected tail
//                 (dynamic overlap reconciliation).
//   CenterSlice — append-only during active speech: the Whisper CNN's receptive
//                 field distorts a window's edge tokens, so both edges are
//                 discarded and only the stable CENTER tokens are committed (no
//                 cosine checks, no mid-speech rollbacks; pos only moves forward).
//                 The right-edge tokens are committed only at a VAD pause to
//                 complete the phrase, and un-committed by a pointer-only rollback
//                 if speech resumes.
enum class AudioStreamingMode { Reconcile, CenterSlice };

// ── Tier 2: high-level streaming request (audio sliding-window + reconciliation) ─
// All fields are USER INTENT in ms / ratios; build_and_validate_runtime resolves
// them into a validated AudioStreamingPlan (tier 3). Nothing here is consumed
// directly — the pipeline reads only the resolved plan.
struct AudioStreamingConfig {
    bool  enable                           = false;   // audio frontends opt in
    AudioStreamingMode mode                = AudioStreamingMode::Reconcile;
    int   window_size_ms                   = 2240;    // acoustic context window (14 tokens)
    int   hop_size_ms                      = 320;     // new audio committed per hop (2 tokens)
    bool  overlap_reconciliation           = true;    // (Reconcile) rewrite tail when history diverges
    float overlap_reconciliation_threshold = 0.999f;  // (Reconcile) cosine below this ⇒ divergence
    int   max_reconciliation_rewind_tokens = 8;       // (Reconcile) safety cap on re-prefill depth
    // (CenterSlice) edge margins discarded from every window. The left (past) edge
    // is dropped outright; the right (unstable future) edge K is withheld during
    // speech and committed only at a VAD pause.
    int   left_edge_ms                     = 320;     // 2 tokens of distorted past edge
    int   right_edge_ms                    = 480;     // K = 3 tokens of tentative future edge
};

// ── Tier 2: high-level request ──────────────────────────────────────────────
struct InferenceConfig {
    // Max tokens (prompt + generated) for a single sequence. Drives KV-cache and
    // activation sizing. Analogous to a target resolution.
    size_t max_context_length = 2048;

    // Peak number of concurrently-live sequences (batch width / ReAct tree
    // branches). > 1 reserves multi-sequence capacity; the current decode loop
    // still drives them one at a time.
    size_t max_batch_size = 1;

    // The caller intends to fork()/rewind() sequences (agent tree-search). Forces
    // a Copy-on-Write paged KV cache, and is REJECTED for models whose recurrent
    // linear-attention (SSM) state cannot be snapshot (hybrid Qwen3.5).
    bool require_branching = false;

    // Crossover width at which a batched linear projection switches from the
    // per-row GEMV sweep to the Tensor-Core batched GEMM. The GEMM wins on
    // throughput for wide prefill chunks but carries higher fixed launch/setup
    // latency (shared memory, wmma pipelining), so a small delta (e.g. 2-10
    // tokens of live typing) is faster looping the GEMV. num_tokens < threshold
    // uses the GEMV sweep; >= threshold uses the batched GEMM. Must be >= 1.
    int batched_gemm_threshold = 16;

    // Default sampling knobs. Per-call forward() arguments still override these.
    float temperature = 0.6f;
    float top_p       = 0.9f;

    // Audio streaming (sliding-window + dynamic overlap reconciliation). Inert
    // unless .enable is set (or a streaming override is present); resolved into
    // RuntimeConfig::audio_streaming.
    AudioStreamingConfig audio_streaming;

    // Forced translation languages for speech frontends (startup defaults the
    // app's runtime language selectors may later override). Free-form English
    // language names ("Russian", "German", ...); "auto" = no forcing. The core
    // only carries the intent — prompt construction from it is the frontend's
    // job, so no closed language list is validated here.
    std::string source_language = "auto";
    std::string target_language = "auto";
};

// Resolved streaming plan (tier 3): ms intent expanded to whole soft-token counts,
// validated, and gated on model capability. The audio pipeline consumes ONLY this.
struct AudioStreamingPlan {
    bool  enabled                  = false;   // streaming active AFTER the capability gate
    AudioStreamingMode mode        = AudioStreamingMode::Reconcile;
    int   window_tokens            = 0;       // window_size_ms / kAudioSoftTokenMs
    int   hop_tokens               = 0;       // hop_size_ms    / kAudioSoftTokenMs
    int   overlap_tokens           = 0;       // window_tokens - hop_tokens (the compare span)
    float reconciliation_threshold = 0.999f;  // (Reconcile) cosine below ⇒ divergence
    int   max_rewind_tokens        = 0;       // (Reconcile) clamped to [0, overlap_tokens]
    // (CenterSlice) validated edge margins, in soft-tokens. Invariant:
    // left + right + hop <= window (consecutive centers stay contiguous — no
    // uncovered gap between one window's center span and the next's).
    int   left_edge_tokens         = 0;       // dropped past edge per window
    int   right_edge_tokens        = 0;       // K: pause-commit margin (>= 1 in CenterSlice)
};

// ── Tier 3: resolved execution plan ─────────────────────────────────────────
struct RuntimeConfig {
    static constexpr size_t kAllLayersResident = static_cast<size_t>(-1);

    // --- sizing (validated against ModelConfig) ---
    size_t max_seq_len   = 2048;   // == InferenceConfig::max_context_length
    size_t max_sequences = 1;      // == InferenceConfig::max_batch_size

    // --- KV-cache execution strategy ---
    BlackwellEngine::KVCacheMode kv_mode = BlackwellEngine::KVCacheMode::Continuous;
    int paged_branch_factor = 4;   // host-mirror page headroom for CoW fork branches

    // --- weight residency / offloading ---
    // Leading layers kept resident in VRAM; the remainder stream from pinned host
    // RAM. kAllLayersResident == every layer resident.
    size_t num_gpu_layers = kAllLayersResident;

    // --- attention dispatch ---
    // Qwen3.5 hybrid routes its head_dim-256 gated layers through a dedicated naive
    // path instead of the 128-capped paged/continuous kernels. Resolved here so the
    // engine and the validator agree on the dispatch (and so the head_dim cap is
    // correctly waived for these models).
    bool uses_dedicated_full_attention = false;

    // --- linear-projection dispatch ---
    // GEMV-sweep vs batched-GEMM crossover width (== InferenceConfig::
    // batched_gemm_threshold, validated >= 1). The LinearDispatcher consults this
    // to pick the lowest-latency path per projection: num_tokens below it loops
    // the fast batch=1 GEMV, at/above it launches the Tensor-Core batched GEMM.
    int batched_gemm_threshold = 16;

    // --- tiered KV prefix-cache substrate (Paged mode only; docs/TIERED_KV_AND_AOT.md §5.1) ---
    // One page = paging::PAGE_SIZE (16) tokens of KV across all layers.
    //
    // Floor on the paged device pool, in pages. The pool is always at least the
    // sequence budget (ceil(max_seq_len/PAGE_SIZE) * paged_branch_factor); raising
    // this floor keeps that many cached prefix branches VRAM-resident beyond the
    // live sequences' own needs. 0 = sequence budget only.
    int kv_vram_cache_pages = 0;
    // Pinned host-RAM demotion tier capacity, in pages. kMirrorDevicePool (the
    // one deliberate late-bound value in this struct: the device-pool page count
    // is only known once the KV manager exists) sizes it 1:1 with the device
    // pool, i.e. a full VRAM's worth of cold branches can wait in RAM.
    static constexpr int kMirrorDevicePool = -1;
    int kv_ram_slots = kMirrorDevicePool;
    // NVMe spill tier capacity, in pages. 0 = disk tier off (demotions stop at
    // RAM; RAM pressure falls back to LRU tree eviction).
    int kv_disk_slots = 0;
    // Backing file for the spill tier. Must be non-empty iff kv_disk_slots > 0.
    std::string kv_spill_path;

    // --- audio streaming plan (resolved from InferenceConfig::audio_streaming) ---
    AudioStreamingPlan audio_streaming;

    // --- forced translation languages (from InferenceConfig; never empty here:
    //     an unset request normalizes to the "auto" sentinel) ---
    std::string source_language = "auto";
    std::string target_language = "auto";
};

// Optional low-level overrides applied AFTER the automatic plan is derived but
// BEFORE validation. std::nullopt means "let the builder decide". This is the
// `-x264-params` seam: poke an individual knob without restating the whole plan.
struct RuntimeOverrides {
    std::optional<BlackwellEngine::KVCacheMode> kv_mode;
    std::optional<size_t> num_gpu_layers;     // kAllLayersResident forces all-resident
    std::optional<int>    paged_branch_factor;
    std::optional<int>    batched_gemm_threshold;  // GEMV<->batched-GEMM crossover (>= 1)

    // Tiered KV prefix-cache sizing (see the RuntimeConfig fields for semantics).
    std::optional<int>         kv_vram_cache_pages;
    std::optional<int>         kv_ram_slots;       // kMirrorDevicePool = mirror device pool
    std::optional<int>         kv_disk_slots;      // 0 = disk tier off
    std::optional<std::string> kv_spill_path;

    // Audio streaming (see AudioStreamingConfig for semantics). Any of these being
    // present also activates the streaming-plan resolution even if .enable is false.
    std::optional<AudioStreamingMode> audio_streaming_mode;
    std::optional<int>   audio_window_size_ms;
    std::optional<int>   audio_hop_size_ms;
    std::optional<bool>  audio_overlap_reconciliation;
    std::optional<float> audio_reconciliation_threshold;
    std::optional<int>   audio_max_reconciliation_rewind_tokens;
    std::optional<int>   audio_left_edge_ms;    // (CenterSlice) past-edge margin
    std::optional<int>   audio_right_edge_ms;   // (CenterSlice) pause-commit margin K

    // Observability seam, not an execution-plan knob (deliberately absent from
    // RuntimeConfig): invoked from the engine-constructing thread while model
    // weights stream from disk, with cumulative (bytes_done, bytes_total) --
    // drive a load-progress UI from it. Called per tensor; keep it O(1) and
    // never let it throw. Empty = no reporting.
    std::function<void(size_t bytes_done, size_t bytes_total)> load_progress;
};

// Hard kernel/hardware limits the validator asserts against. These mirror
// compile-time kernel constants; they are NOT tunable, only checkable.
struct KernelLimits {
    static constexpr int kPagedFlashHeadDimMax = 128; // paged_flash_attention HEAD_DIM_MAX
};

// Derive the static model capabilities from the parsed topology alone
// (counts / hybrid flags). supports_cow_branching is left false here: it depends
// on the chosen kv_mode and is finalized by the engine once the runtime plan is
// known. An empty layer_types means the legacy uniform full-attention layout.
ModelCapabilities derive_capabilities(const ModelConfig& model);

// Reconcile tiers 1+2 (+ overrides) into the tier-3 execution plan. Throws
// std::invalid_argument / std::runtime_error with an actionable message when the
// request cannot be satisfied:
//   * require_branching on a hybrid SSM model (un-snapshot-able recurrent state);
//   * max_context_length beyond the model's trained positional range;
//   * paged KV on a head_dim the paged-flash kernel cannot serve (and which has
//     no dedicated fallback);
//   * a degenerate sizing request (zero context / batch).
RuntimeConfig build_and_validate_runtime(const ModelConfig& model,
                                         const ModelCapabilities& caps,
                                         const InferenceConfig& request,
                                         const RuntimeOverrides& overrides = {});

} // namespace blackwell
