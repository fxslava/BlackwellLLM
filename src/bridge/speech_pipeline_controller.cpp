// =============================================================================
// bridge/speech_pipeline_controller.cpp — state machine + C-ABI entry points
// for include/bridge/speculative_bridge_api.h.
// =============================================================================
#include "speech_pipeline_controller.hpp"

#include <cmath>
#include <exception>
#include <new>

#include "bridge_internal.hpp"
#include "common.h"  // blackwell::cuda_error

using blackwell::EngineStatus;

namespace blackwell::bridge {
namespace {

constexpr uint32_t kVadBlock = 160;  // 10 ms @ 16 kHz

// RMS(block) in dBFS: 20*log10(rms + 1e-6). Matches the audio_recorder VAD.
float rms_db(double sumsq, uint32_t n) noexcept {
    if (n == 0) return -120.0f;
    const double rms = std::sqrt(sumsq / static_cast<double>(n));
    return static_cast<float>(20.0 * std::log10(rms + 1e-6));
}

}  // namespace

// ---- construction -----------------------------------------------------------
SpeechPipelineController::SpeechPipelineController(const SpeechPipelineConfig& cfg,
                                                  IEngineControl* control,
                                                  AudioStreamHandle stream) noexcept
    : cfg_(cfg), control_(control), stream_(stream) {
    const uint32_t sr = (cfg_.sample_rate != 0) ? cfg_.sample_rate : 16000u;
    block_size_ = kVadBlock;
    hangover_samples_.store(static_cast<uint32_t>(
        static_cast<uint64_t>(cfg_.silence_hangover_ms) * sr / 1000u),
        std::memory_order_relaxed);
    warm_interval_samples_ = static_cast<uint32_t>(
        static_cast<uint64_t>(cfg_.warm_prefill_interval_ms) * sr / 1000u);
    // Hysteresis: release <= onset. 0 (unset) => no hysteresis (release == onset).
    release_db_ = (cfg_.vad_release_db != 0.0f) ? cfg_.vad_release_db : cfg_.vad_threshold_db;
}

BridgeStatus SpeechPipelineController::create(const SpeechPipelineConfig& cfg,
                                             SpeechPipelineController** out) {
    if (out == nullptr) return BRIDGE_ERR_INVALID_ARG;
    *out = nullptr;
    if (cfg.engine == nullptr || cfg.audio_stream == nullptr) return BRIDGE_ERR_INVALID_HANDLE;

    IEngineControl* control = cfg.engine->control;
    if (control == nullptr) return BRIDGE_ERR_INVALID_HANDLE;

    *out = new SpeechPipelineController(cfg, control, cfg.audio_stream);
    return BRIDGE_OK;
}

// ---- callbacks / state ------------------------------------------------------
void SpeechPipelineController::register_callbacks(SpeechTokenCallback token_cb,
                                                  SpeechStateCallback state_cb,
                                                  void* user) noexcept {
    // Publish user before the callbacks so a concurrent invoke never reads a
    // callback with a not-yet-visible user pointer.
    cb_user_.store(user, std::memory_order_release);
    token_cb_.store(token_cb, std::memory_order_release);
    state_cb_.store(state_cb, std::memory_order_release);
}

void SpeechPipelineController::set_state(SpeechPipelineState next, uint64_t gen) noexcept {
    const SpeechPipelineState prev = state_.exchange(next, std::memory_order_acq_rel);
    if (prev == next) return;
    if (SpeechStateCallback cb = state_cb_.load(std::memory_order_acquire)) {
        cb(cb_user_.load(std::memory_order_acquire), prev, next, gen);
    }
}

// ---- barge-in: monotone gen bump + KV micro-rewind --------------------------
void SpeechPipelineController::on_speech_start() noexcept {
    // Already warming this utterance -> nothing to interrupt.
    if (state_.load(std::memory_order_acquire) == SPEECH_STATE_PREFILL_SPEAKING) return;

    // 1. Fetch-and-add the epoch: any in-flight decode task tagged with an older
    //    gen is now stale (the engine drops it at its next checkpoint).
    const uint64_t gen = current_gen_id_.fetch_add(1, std::memory_order_acq_rel) + 1;

    // 2. Signal cancel (immediate, latest-wins) and 3. marshal the KV micro-rewind
    //    to the verified prefix onto the engine thread. The audio thread issues
    //    both but executes neither — no CUDA is touched here.
    control_->cancel_generation(gen);
    set_state(SPEECH_STATE_INTERRUPTION_REWIND, gen);
    (void)control_->rewind_kv(stream_, verified_prompt_tokens_.load(std::memory_order_acquire), gen);

    // 4. Reset speculative accounting, 5. enter PREFILL_SPEAKING.
    speculative_tokens_count_.store(0, std::memory_order_release);
    silence_samples_.store(0, std::memory_order_relaxed);
    samples_since_warm_.store(0, std::memory_order_relaxed);
    set_state(SPEECH_STATE_PREFILL_SPEAKING, gen);
}

// ---- stable boundary: commit soft-tokens + begin decode ---------------------
void SpeechPipelineController::on_silence_timeout() noexcept {
    // Only meaningful while actively warming a spoken utterance.
    if (state_.load(std::memory_order_acquire) != SPEECH_STATE_PREFILL_SPEAKING) return;

    // Commit under the CURRENT epoch (this is not a new generation — it finalizes
    // the speech we have been warming).
    const uint64_t gen = current_gen_id_.load(std::memory_order_acquire);
    set_state(SPEECH_STATE_DECODE_TRANSLATING, gen);
    (void)control_->commit_and_decode(stream_, make_token_sink(), gen);
}

// ---- streaming edge + internal VAD ------------------------------------------
void SpeechPipelineController::push_pcm(const float* samples, size_t count) noexcept {
    // The buffer an external scorer is handed must hold a whole block. Both are
    // compile-time constants, which is what makes the clamp below an identity
    // rather than a silent truncation.
    static_assert(kVadBlock == kVadBlockCapacity,
                  "vad_block_ must be sized to exactly one VAD block");
    if (samples == nullptr || count == 0) return;

    // Forward to the SPSC ring so the engine thread can (speculatively) prefill.
    // Backpressure (ring full) is intentionally swallowed here — the VAD/decode
    // logic must not stall the producer; dropped samples degrade quality, not
    // correctness.
    //
    // TRUE PUSH-TO-TALK: in manual mode the ring receives audio ONLY while an
    // utterance is being captured (between the explicit press and release, i.e.
    // PREFILL_SPEAKING). Idle/decoding background noise is discarded at the
    // door, so a press can never feed pre-press audio into the encoder. The VAD
    // block accounting below keeps running either way (it feeds the level meter
    // and re-arms instantly when manual mode is turned off).
    const bool capture =
        !manual_mode_.load(std::memory_order_acquire) ||
        state_.load(std::memory_order_acquire) == SPEECH_STATE_PREFILL_SPEAKING;
    if (capture) (void)stream_->ring.push_samples(samples, count);

    // Internal VAD in fixed 10 ms blocks (sample-accurate, no wall clock). The
    // block's samples are retained alongside the running sum of squares so an
    // installed scorer can be handed the audio; the RMS path ignores them.
    for (size_t i = 0; i < count; ++i) {
        const double s = static_cast<double>(samples[i]);
        vad_sumsq_ += s * s;
        if (vad_count_ < kVadBlockCapacity) vad_block_[vad_count_] = samples[i];
        if (++vad_count_ >= block_size_) {
            // The scorer is third-party code: hand it only what was actually
            // written. Today block_size_ == kVadBlockCapacity (static_assert at
            // the top of this function), so this clamp is an identity — it
            // exists so a future config-driven block size cannot become an
            // overread inside someone else's callback.
            const uint32_t scored = (vad_count_ < kVadBlockCapacity) ? vad_count_
                                                                     : kVadBlockCapacity;
            vad_on_block(evaluate_block(vad_block_, scored, rms_db(vad_sumsq_, vad_count_)));
            vad_sumsq_ = 0.0;
            vad_count_ = 0;
        }
    }
}

// The ONE place the two detectors diverge. Everything downstream consumes the
// VadDecision and cannot tell which produced it.
SpeechPipelineController::VadDecision SpeechPipelineController::evaluate_block(
    const float* block, uint32_t count, float db) noexcept {
    const SpeechVadScoreFn fn = vad_scorer_.load(std::memory_order_acquire);
    if (fn != nullptr) {
        void* user = vad_scorer_user_.load(std::memory_order_acquire);
        const float p = fn(user, block, static_cast<size_t>(count));
        if (p >= 0.0f) {
            const float onset   = vad_threshold_.load(std::memory_order_relaxed);
            const float release = (onset - kProbReleaseDrop > kProbReleaseFloor)
                                      ? onset - kProbReleaseDrop
                                      : kProbReleaseFloor;
            last_decision_ = VadDecision{p > onset, p > release};
        }
        // A negative score is "no opinion": hold the previous verdict rather
        // than silently falling back to RMS, which would interleave two
        // detectors' hysteresis and could flap the state machine.
        return last_decision_;
    }
    // Built-in RMS threshold — the default, byte-for-byte the original policy.
    return VadDecision{db > cfg_.vad_threshold_db, db > release_db_};
}

void SpeechPipelineController::vad_on_block(VadDecision decision) noexcept {
    const SpeechPipelineState s = state_.load(std::memory_order_acquire);
    // MANUAL/PTT gate: while set, the threshold VAD may not drive ANY state
    // transition (onset, auto-commit, barge-in) — those come exclusively from
    // the explicit on_speech_start/on_silence_timeout key events, so a
    // background VAD trigger can never race a hotkey mid-utterance. Speculative
    // warming below is NOT a transition and stays active during a PTT hold.
    const bool manual = manual_mode_.load(std::memory_order_acquire);

    switch (s) {
        case SPEECH_STATE_IDLE:
            if (!manual && decision.onset) on_speech_start();  // onset
            break;

        case SPEECH_STATE_PREFILL_SPEAKING: {
            // The hangover is runtime-retunable (UI slider): one relaxed load per
            // 10 ms block. 0 = auto-commit disabled. In manual mode the silence
            // clock does not even accumulate (a later mode flip must not fire an
            // instant commit off stale silence).
            const uint32_t hangover = hangover_samples_.load(std::memory_order_relaxed);
            if (!manual) {
                if (decision.sustain) {
                    silence_samples_.store(0, std::memory_order_relaxed);  // still speaking
                } else {
                    const uint32_t sil = silence_samples_.fetch_add(block_size_,
                                                                    std::memory_order_relaxed) +
                                         block_size_;
                    if (hangover != 0 && sil >= hangover) {  // stable boundary reached
                        on_silence_timeout();
                        break;
                    }
                }
            }
            // TrackUpdate analogue: throttle speculative warming prefills.
            if (warm_interval_samples_ != 0) {
                const uint32_t acc = samples_since_warm_.fetch_add(block_size_,
                                                                   std::memory_order_relaxed) +
                                     block_size_;
                if (acc >= warm_interval_samples_) {
                    samples_since_warm_.store(0, std::memory_order_relaxed);
                    (void)control_->warm_prefill(stream_,
                                                 current_gen_id_.load(std::memory_order_acquire));
                }
            }
            break;
        }

        case SPEECH_STATE_DECODE_TRANSLATING:
            if (!manual && decision.onset) on_speech_start();  // barge-in mid-translation
            break;

        case SPEECH_STATE_INTERRUPTION_REWIND:
            // Transient; the next block re-evaluates once PREFILL_SPEAKING is set.
            break;
    }
}

// ---- runtime VAD retune -----------------------------------------------------
void SpeechPipelineController::set_silence_hangover_ms(uint32_t ms) noexcept {
    const uint32_t sr = (cfg_.sample_rate != 0) ? cfg_.sample_rate : 16000u;
    hangover_samples_.store(
        static_cast<uint32_t>(static_cast<uint64_t>(ms) * sr / 1000u),
        std::memory_order_relaxed);
}

void SpeechPipelineController::set_manual_mode(bool enabled) noexcept {
    manual_mode_.store(enabled, std::memory_order_release);
    // A mode flip mid-utterance must not inherit the other mode's silence
    // accounting: the auto-commit clock restarts from the flip (belt-and-braces
    // with the !manual accumulation gate in vad_on_block).
    silence_samples_.store(0, std::memory_order_relaxed);
}

void SpeechPipelineController::set_vad_scorer(SpeechVadScoreFn fn, void* user) noexcept {
    // User pointer first, so the audio thread can never observe a live scorer
    // paired with a stale user pointer. (Clearing goes the other way: drop the
    // function first, then the pointer it would have been called with.)
    if (fn != nullptr) {
        vad_scorer_user_.store(user, std::memory_order_release);
        vad_scorer_.store(fn, std::memory_order_release);
    } else {
        vad_scorer_.store(nullptr, std::memory_order_release);
        vad_scorer_user_.store(nullptr, std::memory_order_release);
    }
    // A detector swap must not inherit the other one's verdict or its silence
    // accounting — the next block starts the decision fresh.
    last_decision_ = VadDecision{};
    silence_samples_.store(0, std::memory_order_relaxed);
}

void SpeechPipelineController::set_vad_threshold(float threshold) noexcept {
    const float clamped = (threshold < 0.0f) ? 0.0f : (threshold > 1.0f ? 1.0f : threshold);
    vad_threshold_.store(clamped, std::memory_order_relaxed);
}

// ---- token adaptation -------------------------------------------------------
TokenSink SpeechPipelineController::make_token_sink() noexcept {
    return TokenSink{&SpeechPipelineController::token_trampoline, this};
}

void SpeechPipelineController::token_trampoline(void* user, const char* utf8, int32_t index,
                                                int32_t is_final, BridgeStatus /*status*/) {
    auto* self = static_cast<SpeechPipelineController*>(user);
    const uint64_t gen = self->current_gen_id_.load(std::memory_order_acquire);

    SpeechTokenEvent ev;
    ev.text = (utf8 != nullptr) ? utf8 : "";
    ev.token_id = 0;  // surfaced by the engine through a richer sink later
    ev.position = static_cast<uint32_t>(index < 0 ? 0 : index);
    ev.is_translation = true;  // decode-loop output (ASR partials would set false)

    if (SpeechTokenCallback cb = self->token_cb_.load(std::memory_order_acquire)) {
        cb(self->cb_user_.load(std::memory_order_acquire), &ev, gen);
    }
    if (is_final != 0) self->on_decode_final(gen);
}

void SpeechPipelineController::on_decode_final(uint64_t gen) noexcept {
    // Return to IDLE only if no newer epoch (barge-in) has superseded this one —
    // otherwise the barge-in already owns the state.
    if (gen == current_gen_id_.load(std::memory_order_acquire)) {
        set_state(SPEECH_STATE_IDLE, gen);
    }
}

}  // namespace blackwell::bridge

// -----------------------------------------------------------------------------
// C-ABI entry points.
// -----------------------------------------------------------------------------
namespace {

using blackwell::bridge::SpeechPipelineController;

SpeechPipelineController* to_ctrl(SpeechPipelineHandle h) noexcept {
    return reinterpret_cast<SpeechPipelineController*>(h);
}

// Exception firewall for the allocating entry points (create). The streaming +
// event entries call only noexcept controller methods, so they need no guard.
template <class Fn>
BridgeStatus guarded(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const blackwell::cuda_error& e) {
        return (e.code() == cudaErrorMemoryAllocation) ? BRIDGE_ERR_OUT_OF_MEMORY
                                                       : BRIDGE_ERR_CUDA;
    } catch (const std::bad_alloc&) {
        return BRIDGE_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return BRIDGE_ERR_INTERNAL;
    }
}

}  // namespace

extern "C" {

BRIDGE_API BridgeStatus speech_pipeline_create(const SpeechPipelineConfig* config,
                                               SpeechPipelineHandle* out_handle) {
    if (config == nullptr || out_handle == nullptr) return BRIDGE_ERR_INVALID_ARG;
    *out_handle = nullptr;
    return guarded([&]() -> BridgeStatus {
        SpeechPipelineController* ctrl = nullptr;
        const BridgeStatus st = SpeechPipelineController::create(*config, &ctrl);
        if (st != BRIDGE_OK) return st;
        *out_handle = reinterpret_cast<SpeechPipelineHandle>(ctrl);
        return BRIDGE_OK;
    });
}

BRIDGE_API BridgeStatus speech_pipeline_destroy(SpeechPipelineHandle handle) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    delete to_ctrl(handle);
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_register_callbacks(SpeechPipelineHandle handle,
                                                           SpeechTokenCallback token_cb,
                                                           SpeechStateCallback state_cb,
                                                           void* user) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->register_callbacks(token_cb, state_cb, user);
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_push_pcm(SpeechPipelineHandle handle,
                                                 const float* samples, size_t count) {
    // HOT PATH: no lock, no try/catch, no allocation — push_pcm is noexcept.
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    if (samples == nullptr && count != 0) return BRIDGE_ERR_INVALID_ARG;
    if (count == 0) return BRIDGE_OK;
    to_ctrl(handle)->push_pcm(samples, count);
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_on_speech_start(SpeechPipelineHandle handle) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->on_speech_start();
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_on_silence_timeout(SpeechPipelineHandle handle) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->on_silence_timeout();
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_set_silence_hangover_ms(SpeechPipelineHandle handle,
                                                                uint32_t silence_hangover_ms) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->set_silence_hangover_ms(silence_hangover_ms);
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_set_manual_mode(SpeechPipelineHandle handle,
                                                        bool enabled) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->set_manual_mode(enabled);
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_set_vad_scorer(SpeechPipelineHandle handle,
                                                       SpeechVadScoreFn fn, void* user) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->set_vad_scorer(fn, user);
    return BRIDGE_OK;
}

BRIDGE_API BridgeStatus speech_pipeline_set_vad_threshold(SpeechPipelineHandle handle,
                                                          float threshold) {
    if (handle == nullptr) return BRIDGE_ERR_INVALID_HANDLE;
    to_ctrl(handle)->set_vad_threshold(threshold);
    return BRIDGE_OK;
}

}  // extern "C"
