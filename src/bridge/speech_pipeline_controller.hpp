#pragma once
// =============================================================================
// bridge/speech_pipeline_controller.hpp — the state machine behind the opaque
// SpeechPipelineHandle. See include/bridge/speculative_bridge_api.h.
//
// CONCURRENCY
//   * push_pcm() runs on the single audio/VAD thread (its VAD accumulators are
//     that thread's private state).
//   * on_speech_start()/on_silence_timeout() may be called from that thread (via
//     the internal VAD) OR from an external VAD/ASR thread — they only touch
//     atomics and marshal engine work, so they are safe from any thread.
//   * Token/state callbacks fire from whichever thread performs the transition;
//     the engine-thread bookkeeping hooks (set_verified_prompt_tokens / add_
//     speculative_tokens) are called by the IEngineControl implementor.
//   No OS mutex sits on any of these paths (lock-free, allocation-free at steady
//   state) — the controller never touches CUDA; it flags epochs and marshals.
// =============================================================================
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "bridge/speculative_bridge_api.h"  // C types: state / config / event / callbacks / BridgeStatus
#include "bridge_internal.hpp"              // IEngineControl, TokenSink, opaque handles

namespace blackwell::bridge {

class SpeechPipelineController {
public:
    // Factory: validates the config + handles, resolves the engine seam, and
    // allocates the controller. Returns a BridgeStatus (no throw on the happy
    // path). *out receives the controller on BRIDGE_OK.
    static BridgeStatus create(const SpeechPipelineConfig& cfg, SpeechPipelineController** out);
    ~SpeechPipelineController() = default;

    SpeechPipelineController(const SpeechPipelineController&) = delete;
    SpeechPipelineController& operator=(const SpeechPipelineController&) = delete;

    // ---- C-ABI-facing operations (all noexcept) -----------------------------
    void register_callbacks(SpeechTokenCallback token_cb, SpeechStateCallback state_cb,
                            void* user) noexcept;
    void push_pcm(const float* samples, size_t count) noexcept;
    void on_speech_start() noexcept;     // barge-in: bump gen, cancel, rewind, -> PREFILL
    void on_silence_timeout() noexcept;  // commit soft-tokens + decode -> DECODE_TRANSLATING
    // Runtime retune of the auto-commit hangover (UI slider seam). Any thread;
    // one atomic store, effective on the next VAD block. 0 = auto-commit off.
    void set_silence_hangover_ms(uint32_t ms) noexcept;
    // MANUAL (push-to-talk) mode: while enabled, the internal threshold VAD may
    // never drive a state transition — no auto onset, no auto barge-in, no
    // silence auto-commit. Boundaries come EXCLUSIVELY from the explicit
    // on_speech_start()/on_silence_timeout() calls (hotkey press/release), so a
    // background VAD trigger can never race a key event mid-utterance. PCM keeps
    // flowing to the ring and speculative warming stays active. Any thread.
    void set_manual_mode(bool enabled) noexcept;
    bool manual_mode() const noexcept { return manual_mode_.load(std::memory_order_acquire); }

    // EXTERNAL VAD SCORER (see SpeechVadScoreFn). Installing one replaces the
    // built-in RMS threshold as the answer to "is this block speech?" and
    // NOTHING else: hangover, warm-prefill throttle, manual mode and barge-in
    // routing all keep running on top of that answer.
    // With no scorer installed the RMS path runs, unchanged and by default.
    // Any thread; effective on the next VAD block.
    void set_vad_scorer(SpeechVadScoreFn fn, void* user) noexcept;
    void set_vad_threshold(float threshold) noexcept;
    bool has_vad_scorer() const noexcept {
        return vad_scorer_.load(std::memory_order_acquire) != nullptr;
    }
    float vad_threshold() const noexcept {
        return vad_threshold_.load(std::memory_order_relaxed);
    }

    // ---- Engine-thread bookkeeping hooks (called by the IEngineControl impl) -
    // The engine owns the exact token accounting; it publishes it here so the
    // barge-in micro-rewind knows the verified prefix and the speculative depth.
    void set_verified_prompt_tokens(uint32_t n) noexcept {
        verified_prompt_tokens_.store(n, std::memory_order_release);
    }
    void add_speculative_tokens(uint32_t n) noexcept {
        speculative_tokens_count_.fetch_add(n, std::memory_order_acq_rel);
    }

    SpeechPipelineState state() const noexcept { return state_.load(std::memory_order_acquire); }
    uint64_t generation_id() const noexcept { return current_gen_id_.load(std::memory_order_acquire); }

private:
    SpeechPipelineController(const SpeechPipelineConfig& cfg, IEngineControl* control,
                             AudioStreamHandle stream) noexcept;

    void set_state(SpeechPipelineState next, uint64_t gen) noexcept;

    // The per-block speech verdict, split into the two questions the state
    // machine asks. `onset` uses the higher threshold (start / barge in on
    // this?), `sustain` the lower one (is the speaker still going?) — that
    // difference IS the hysteresis, and it is identical in shape whether the
    // numbers came from dBFS or from a neural probability.
    struct VadDecision {
        bool onset   = false;
        bool sustain = false;
    };
    // Picks the scorer path or the RMS path and applies the matching thresholds.
    VadDecision evaluate_block(const float* block, uint32_t count, float db) noexcept;
    void vad_on_block(VadDecision decision) noexcept;  // internal auto-VAD per 10 ms block
    TokenSink make_token_sink() noexcept;
    void on_decode_final(uint64_t gen) noexcept;
    // Trampoline: adapt the engine's raw TokenSink callback into a SpeechTokenEvent.
    static void token_trampoline(void* user, const char* utf8, int32_t index,
                                 int32_t is_final, BridgeStatus status);

    // Resolved dependencies (non-owning; the app owns engine + stream).
    SpeechPipelineConfig cfg_;
    IEngineControl* control_ = nullptr;
    AudioStreamHandle stream_ = nullptr;

    // Derived VAD geometry (sample counts). hangover_samples_ is atomic: it is
    // the ONE knob retunable at runtime (set_silence_hangover_ms from the UI
    // thread) while the audio thread reads it per 10 ms block.
    uint32_t block_size_ = 160;          // 10 ms @ sample_rate
    std::atomic<uint32_t> hangover_samples_{0};  // silence_hangover_ms -> samples (0 = off)
    std::atomic<bool> manual_mode_{false};       // true = auto-VAD transitions muted (PTT)
    uint32_t warm_interval_samples_ = 0; // warm_prefill_interval_ms -> samples (0 = disabled)
    float release_db_ = 0.0f;            // resolved hysteresis release threshold

    // ---- Shared epoch/state (atomic; read by engine thread + callbacks) -----
    std::atomic<SpeechPipelineState> state_{SPEECH_STATE_IDLE};
    std::atomic<uint64_t> current_gen_id_{0};        // monotone; fetch-add on barge-in
    std::atomic<uint32_t> verified_prompt_tokens_{0};// safe KV rewind point
    std::atomic<uint32_t> speculative_tokens_count_{0};

    // ---- Callbacks (atomic pointers; lock-free reads from the engine thread) -
    std::atomic<SpeechTokenCallback> token_cb_{nullptr};
    std::atomic<SpeechStateCallback> state_cb_{nullptr};
    std::atomic<void*> cb_user_{nullptr};

    // ---- External VAD scorer (optional; null => built-in RMS threshold) -------
    // Read once per block on the audio thread, written from the UI/setup thread.
    std::atomic<SpeechVadScoreFn> vad_scorer_{nullptr};
    std::atomic<void*> vad_scorer_user_{nullptr};
    std::atomic<float> vad_threshold_{0.5f};   // probability onset threshold
    // Hysteresis for the probability path, mirroring vad_release_db on the RMS
    // path: a block sustains speech while it stays above (threshold - kDrop),
    // never below kFloor. Without it a probability hovering at the threshold
    // would chatter the state machine every block.
    static constexpr float kProbReleaseDrop  = 0.15f;
    static constexpr float kProbReleaseFloor = 0.05f;

    // ---- Internal VAD accumulators --------------------------------------------
    // Pure block RMS is audio-thread-private; the two boundary counters are also
    // written by on_speech_start() (any thread) so they are atomic.
    double vad_sumsq_ = 0.0;             // audio thread only
    uint32_t vad_count_ = 0;             // audio thread only
    // The block's samples, retained so an installed scorer can be handed the
    // audio itself rather than a summary statistic. Audio thread only; sized to
    // the fixed 10 ms block (the RMS path leaves it untouched and unread).
    static constexpr uint32_t kVadBlockCapacity = 160;
    float vad_block_[kVadBlockCapacity] = {};
    // Last verdict from an installed scorer, held when it returns "no opinion"
    // (a negative score). Audio thread only.
    VadDecision last_decision_{};
    std::atomic<uint32_t> silence_samples_{0};
    std::atomic<uint32_t> samples_since_warm_{0};
};

}  // namespace blackwell::bridge
