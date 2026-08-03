#pragma once
// -----------------------------------------------------------------------------
// ConversationalMode — Mode A behind ISpeechMode: the CLASSIC path, unchanged.
// Listen -> silence -> generate -> barge-in, driven by the speech_pipeline_*
// C-ABI over EngineControlBridge's SPSC command ring.
//
// THIS FILE MOVES CODE, IT DOES NOT CHANGE IT. Everything here was previously
// inline in translator/main.cpp; the construction order, the config values, the
// callback registration, the VAD-scorer install and the teardown order are
// byte-for-byte what main() did. That is deliberate and it is the whole claim of
// the Mode A extraction: if this file contains a behavioural decision, the
// extraction was done wrong.
//
// OWNERSHIP, and why it is split the way it is:
//   OWNED   the bridge plumbing (EngineHandle / AudioStreamHandle /
//           SpeechPipelineHandle). Its lifetime is exactly one activation of this
//           mode, so the destructor is the natural place to unwind it — in
//           reverse construction order, as main() did.
//   BORROWED  the control plane and the VAD. Both outlive any single mode: the
//           settings panel holds the control directly (its atomics ARE the UI
//           seam), and the Silero session is a startup-cost resource that must
//           survive a mode switch rather than be reloaded per flip.
//
// THE PREFILL SEAM. prefill_system_prompt() lives on the concrete control
// (RealEngineControl / SimulatedEngineControl), not on EngineControlBridge, so it
// arrives as a callable rather than a virtual. That is not a testability
// contortion — how a frozen prefix gets laid down genuinely differs per control
// implementation, and taking it explicitly is what lets a CPU test drive this
// exact class against SimulatedEngineControl with no GPU.
//
// THREADING: see ISpeechMode. Nothing here relaxes the single-engine-thread
// doctrine — pump_engine() is still the only caller into the engine.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include "bridge/engine_api.h"              // EngineHandle / AudioStreamHandle / status
#include "bridge/speculative_bridge_api.h"  // speech_pipeline_* C ABI, SpeechVadScoreFn
#include "bridge_internal.hpp"              // bridge_wrap_engine / bridge_release_engine
#include "engine_control_bridge.hpp"        // EngineControlBridge

#include "speech_mode.hpp"

namespace rt {

class ConversationalMode final : public ISpeechMode {
public:
    // Every field here was a local in main(); the defaults are the ones that were
    // spelled out at the speech_pipeline_create call site.
    struct Config {
        std::uint32_t sample_rate            = 16000;
        float         vad_threshold_db       = -40.0f;   // onset
        float         vad_release_db         = -45.0f;   // hysteresis release (<= onset)
        std::uint32_t silence_hangover_ms    = 800;      // stable boundary -> commit decode
        std::uint32_t warm_prefill_interval_ms = 320;    // speculative warm-prefill throttle
        float         vad_probability_threshold = 0.5f;  // neural scorer decision point
        // Acoustic head kept when a speech-start flush clears the audio ring, in
        // ms. The other half of "where does an utterance begin?" — the hangover
        // above decides where it ENDS. Default mirrors
        // ContinuousStreamingConfig::pre_roll_ms; see set_pre_roll_ms.
        int           pre_roll_ms            = 250;
        std::string   system_prompt;
    };

    using PrefillSystemPromptFn = std::function<std::uint32_t(const std::string&)>;

    // `control` is BORROWED and must outlive this object.
    //
    // The VAD arrives as the C ABI's OWN seam — a SpeechVadScoreFn plus its user
    // pointer — not as a SileroVAD*. That is deliberate: SpeechVadScoreFn exists
    // precisely so nothing downstream has to know what is scoring, and naming a
    // concrete detector here would drag ONNXRuntime into every consumer of this
    // header (including a CPU-only test that never scores anything). `vad_fn`
    // null leaves the pipeline on its built-in RMS threshold detector, exactly as
    // main() did when the model failed to load.
    //
    // Throws on a plumbing failure — INIT tier, unwinding whatever it already built.
    ConversationalMode(blackwell::bridge::EngineControlBridge* control,
                       PrefillSystemPromptFn prefill,
                       SpeechVadScoreFn vad_fn, void* vad_user,
                       const Config& cfg,
                       SpeechTokenCallback token_cb,
                       SpeechStateCallback state_cb,
                       void* callback_user)
        : control_(control), prefill_(std::move(prefill)),
          vad_fn_(vad_fn), vad_user_(vad_user), cfg_(cfg) {
        if (control_ == nullptr) throw std::runtime_error("ConversationalMode: null control");

        eng_ = blackwell::bridge::bridge_wrap_engine(control_);
        if (eng_ == nullptr) throw std::runtime_error("bridge_wrap_engine failed");

        if (BridgeStatus s = engine_create_audio_stream(eng_, &stream_); s != BRIDGE_OK) {
            unwind();
            throw std::runtime_error(std::string("engine_create_audio_stream: ") +
                                     bridge_status_to_string(s));
        }

        SpeechPipelineConfig scfg{};
        scfg.engine                  = eng_;
        scfg.audio_stream            = stream_;
        scfg.sample_rate             = cfg_.sample_rate;
        scfg.vad_threshold_db        = cfg_.vad_threshold_db;
        scfg.vad_release_db          = cfg_.vad_release_db;
        scfg.silence_hangover_ms     = cfg_.silence_hangover_ms;
        scfg.warm_prefill_interval_ms = cfg_.warm_prefill_interval_ms;
        if (BridgeStatus s = speech_pipeline_create(&scfg, &pipe_); s != BRIDGE_OK) {
            unwind();
            throw std::runtime_error(std::string("speech_pipeline_create: ") +
                                     bridge_status_to_string(s));
        }

        speech_pipeline_register_callbacks(pipe_, token_cb, state_cb, callback_user);

        // The pre-roll is the one audio-gating knob that does NOT go to the
        // pipeline: the pipeline never moves the ring's read cursor (SPSC — only
        // the engine thread may), so the flush it triggers, and therefore the
        // decision of how much acoustic head to spare, both live on the control.
        // Set here so a mode is fully configured the moment it is constructed,
        // rather than only after the first live-settings push.
        control_->set_pre_roll_ms(cfg_.pre_roll_ms);

        // The scorer replaces the RMS threshold as the answer to "is this block
        // speech?" — and ONLY that. The hangover, the warm-prefill throttle and
        // push-to-talk muting all keep running on top of its verdict.
        if (vad_fn_ != nullptr) {
            (void)speech_pipeline_set_vad_threshold(pipe_, cfg_.vad_probability_threshold);
            (void)speech_pipeline_set_vad_scorer(pipe_, vad_fn_, vad_user_);
        }
    }

    ~ConversationalMode() override { unwind(); }

    ConversationalMode(const ConversationalMode&) = delete;
    ConversationalMode& operator=(const ConversationalMode&) = delete;

    const char* name() const noexcept override { return "Conversational"; }

    void start_on_engine_thread() override {
        if (prefill_) frozen_prefix_ = prefill_(cfg_.system_prompt);
    }

    void on_pcm_block(const float* samples, std::size_t count) noexcept override {
        if (pipe_ != nullptr) (void)speech_pipeline_push_pcm(pipe_, samples, count);
    }

    // Parks at 0% CPU until a boundary event arrives, then drains the ring.
    void pump_engine() override { (void)control_->wait_and_pump(); }

    // Abort any in-flight decode, then wake the pump so the thread can exit —
    // the same two-step main() performed at shutdown, in the same order.
    void stop() noexcept override {
        control_->cancel_generation(control_->active_generation() + 1);
        control_->stop();
    }

    ModeTelemetry telemetry() const noexcept override {
        ModeTelemetry t;
        t.name = name();
        return t;
    }

    // Handles the settings panel still needs (it talks to the pipeline directly
    // for hangover / push-to-talk / VAD threshold).
    SpeechPipelineHandle pipeline() const noexcept { return pipe_; }
    std::uint32_t frozen_prefix_tokens() const noexcept { return frozen_prefix_; }

private:
    // Reverse construction order, exactly as main()'s shutdown block did. Safe to
    // run on a partially-built object: every handle is null-checked and nulled.
    void unwind() noexcept {
        if (pipe_ != nullptr) { speech_pipeline_destroy(pipe_); pipe_ = nullptr; }
        if (stream_ != nullptr) { engine_destroy_audio_stream(stream_); stream_ = nullptr; }
        if (eng_ != nullptr) { blackwell::bridge::bridge_release_engine(eng_); eng_ = nullptr; }
    }

    blackwell::bridge::EngineControlBridge* control_ = nullptr;   // borrowed
    PrefillSystemPromptFn                   prefill_;
    SpeechVadScoreFn                        vad_fn_ = nullptr;    // null == built-in RMS
    void*                                   vad_user_ = nullptr;  // borrowed
    Config                                  cfg_;

    EngineHandle         eng_    = nullptr;   // owned
    AudioStreamHandle    stream_ = nullptr;   // owned
    SpeechPipelineHandle pipe_   = nullptr;   // owned
    std::uint32_t        frozen_prefix_ = 0;
};

}  // namespace rt
