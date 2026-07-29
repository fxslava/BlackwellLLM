#pragma once
// -----------------------------------------------------------------------------
// ContinuousStreamingConfig — every tunable of the re-translation pipeline
// (docs/CONTINUOUS_STREAMING.md) in ONE place, so no magic number is spelled out
// at a call site, in a CLI parser, and in an ImGui slider independently.
//
// The struct owns its own BOUNDS as well as its defaults. That is the load-
// bearing part: the panel's sliders, the CLI's validation and clamp() all read
// the same constants, so a value the UI can produce is by construction a value
// the pipeline accepts, and the two cannot drift apart as the ranges are tuned.
//
// EVICTION WATERMARKS, stated explicitly because the names admit two readings.
// These are GC-style watermarks, not an amount-per-eviction:
//   eviction_high_water_mark  TRIGGER. Once the committed zone [S, C) reaches
//                             this many tokens, an eviction is planned.
//   eviction_target_tokens    LOW WATER. Eviction reclaims down TO this size, so
//                             one pass frees (committed - target) tokens and the
//                             gap between the two IS the hysteresis: a bigger gap
//                             means rarer, larger compactions.
// Hence target <= high_water is a hard invariant, not a preference (a low water
// above the high water would evict on every single commit).
//
// The DRAFT zone is deliberately absent from these numbers: eviction operates
// strictly behind C, so the draft's worst case (max_utterance_ms of audio plus
// the decode cap) is headroom the CALLER must leave between
// eviction_high_water_mark and the model's context length. draft_headroom_tokens()
// computes that worst case so the check has one implementation.
//
// DEPENDENCIES: STL only, plus src/vad/speech_segmenter.hpp for the derivation
// below (header-only, STL-only, no ONNXRuntime — including it does NOT pull the
// ORT link dependency, which stays PRIVATE to blackwell_vad).
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstdint>

#include "speech_segmenter.hpp"   // blackwell::vad::SegmenterConfig

namespace blackwell::bridge {

struct ContinuousStreamingConfig {
    // ---- audio gating (consumed by SpeechSegmenter) -------------------------
    int pre_roll_ms        = 250;    // audio prepended before the onset block
    int hangover_ms        = 400;    // sub-release audio that closes an utterance
    int max_utterance_ms   = 15000;  // forced commit for a speaker who never pauses

    // ---- redraft cadence ----------------------------------------------------
    // Every Partial costs a rewind + re-prefill + re-decode of the utterance so
    // far, so this trades translation latency against GPU budget. It is not a
    // detection knob: it cannot change WHERE a boundary falls.
    int partial_cadence_ms = 500;

    // ---- KV eviction watermarks (see the header block) ----------------------
    int eviction_high_water_mark = 3000;
    int eviction_target_tokens   = 1000;

    // ---- bounds (shared by clamp(), the CLI and the panel sliders) ----------
    static constexpr int kMinPreRollMs        = 0,   kMaxPreRollMs        = 1000;
    static constexpr int kMinHangoverMs       = 100, kMaxHangoverMs       = 2000;
    static constexpr int kMinMaxUtteranceMs   = 1000, kMaxMaxUtteranceMs  = 30000;
    static constexpr int kMinPartialCadenceMs = 100, kMaxPartialCadenceMs = 5000;
    static constexpr int kMinHighWaterTokens  = 128, kMaxHighWaterTokens  = 131072;
    static constexpr int kMinTargetTokens     = 0;

    // Total, in-place, and idempotent: ANY input becomes a legal configuration.
    // Ordering matters — the pairwise invariant is enforced after the per-field
    // ranges, so clamping high_water can never leave target stranded above it.
    void clamp() noexcept {
        pre_roll_ms        = clamp_to(pre_roll_ms,        kMinPreRollMs,        kMaxPreRollMs);
        hangover_ms        = clamp_to(hangover_ms,        kMinHangoverMs,       kMaxHangoverMs);
        max_utterance_ms   = clamp_to(max_utterance_ms,   kMinMaxUtteranceMs,   kMaxMaxUtteranceMs);
        partial_cadence_ms = clamp_to(partial_cadence_ms, kMinPartialCadenceMs, kMaxPartialCadenceMs);
        eviction_high_water_mark =
            clamp_to(eviction_high_water_mark, kMinHighWaterTokens, kMaxHighWaterTokens);
        eviction_target_tokens =
            clamp_to(eviction_target_tokens, kMinTargetTokens, eviction_high_water_mark);
    }

    bool valid() const noexcept {
        ContinuousStreamingConfig copy = *this;
        copy.clamp();
        return copy == *this;
    }

    bool operator==(const ContinuousStreamingConfig&) const noexcept = default;

    // ---- derivations --------------------------------------------------------
    // The segmenter's own defaults (thresholds, tail pad, min utterance) are left
    // alone: they are detector policy, not user-facing streaming knobs.
    vad::SegmenterConfig to_segmenter_config(int sample_rate = 16000,
                                             int block_samples = 160) const noexcept {
        vad::SegmenterConfig s{};
        s.sample_rate       = sample_rate;
        s.block_samples     = block_samples;
        s.preroll_ms        = pre_roll_ms;
        s.hangover_ms       = hangover_ms;
        s.max_utterance_ms  = max_utterance_ms;
        s.partial_period_ms = partial_cadence_ms;
        return s;
    }

    bool should_evict(std::uint32_t committed_tokens) const noexcept {
        return committed_tokens >= static_cast<std::uint32_t>(eviction_high_water_mark);
    }

    // Tokens to reclaim to bring the committed zone back to the low watermark —
    // i.e. the min_delta argument for KvLedger::plan_evict_head. Zero when the
    // committed zone is already at or below target.
    std::uint32_t eviction_min_delta(std::uint32_t committed_tokens) const noexcept {
        const auto target = static_cast<std::uint32_t>(eviction_target_tokens);
        return committed_tokens > target ? committed_tokens - target : 0u;
    }

    // Worst-case DRAFT size: one maximal utterance's audio soft tokens plus the
    // decode cap. The caller must keep eviction_high_water_mark + this below the
    // model's context length, or a redraft can run off the end of the cache
    // between two evictions.
    std::uint32_t draft_headroom_tokens(int max_new_tokens,
                                        int audio_ms_per_token = 160) const noexcept {
        const int per = audio_ms_per_token < 1 ? 1 : audio_ms_per_token;
        const int audio = (max_utterance_ms + per - 1) / per;   // ceil
        const int text = max_new_tokens < 0 ? 0 : max_new_tokens;
        return static_cast<std::uint32_t>(audio + text);
    }

    bool fits_context(int context_length, int max_new_tokens) const noexcept {
        const std::int64_t need = static_cast<std::int64_t>(eviction_high_water_mark) +
                                  static_cast<std::int64_t>(draft_headroom_tokens(max_new_tokens));
        return need <= static_cast<std::int64_t>(context_length);
    }

private:
    static int clamp_to(int v, int lo, int hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

// Atomics-backed publisher for the same settings. The panel (UI thread) stores;
// the DSP and engine threads load. Field-wise atomic rather than a mutex, in
// keeping with the rest of the control plane — a reader may observe a torn PAIR
// of fields mid-drag, which is harmless because every field is independently
// valid and the next frame's store settles it. The one invariant that must never
// tear (target <= high_water) is preserved by storing high_water FIRST when it
// grows and target first when it shrinks.
class LiveStreamingConfig {
public:
    LiveStreamingConfig() noexcept { store(ContinuousStreamingConfig{}); }
    explicit LiveStreamingConfig(const ContinuousStreamingConfig& cfg) noexcept { store(cfg); }

    void store(const ContinuousStreamingConfig& in) noexcept {
        ContinuousStreamingConfig cfg = in;
        cfg.clamp();
        pre_roll_ms_.store(cfg.pre_roll_ms, std::memory_order_relaxed);
        hangover_ms_.store(cfg.hangover_ms, std::memory_order_relaxed);
        max_utterance_ms_.store(cfg.max_utterance_ms, std::memory_order_relaxed);
        partial_cadence_ms_.store(cfg.partial_cadence_ms, std::memory_order_relaxed);
        // Order the watermark pair so a reader between the two stores still sees
        // target <= high_water, whichever direction the change went.
        if (cfg.eviction_high_water_mark >= high_water_.load(std::memory_order_relaxed)) {
            high_water_.store(cfg.eviction_high_water_mark, std::memory_order_relaxed);
            target_.store(cfg.eviction_target_tokens, std::memory_order_release);
        } else {
            target_.store(cfg.eviction_target_tokens, std::memory_order_relaxed);
            high_water_.store(cfg.eviction_high_water_mark, std::memory_order_release);
        }
    }

    ContinuousStreamingConfig load() const noexcept {
        ContinuousStreamingConfig cfg{};
        cfg.pre_roll_ms              = pre_roll_ms_.load(std::memory_order_relaxed);
        cfg.hangover_ms              = hangover_ms_.load(std::memory_order_relaxed);
        cfg.max_utterance_ms         = max_utterance_ms_.load(std::memory_order_relaxed);
        cfg.partial_cadence_ms       = partial_cadence_ms_.load(std::memory_order_relaxed);
        cfg.eviction_high_water_mark = high_water_.load(std::memory_order_acquire);
        cfg.eviction_target_tokens   = target_.load(std::memory_order_acquire);
        cfg.clamp();   // a torn pair cannot escape this accessor
        return cfg;
    }

private:
    std::atomic<int> pre_roll_ms_{0};
    std::atomic<int> hangover_ms_{0};
    std::atomic<int> max_utterance_ms_{0};
    std::atomic<int> partial_cadence_ms_{0};
    std::atomic<int> high_water_{0};
    std::atomic<int> target_{0};
};

}  // namespace blackwell::bridge
