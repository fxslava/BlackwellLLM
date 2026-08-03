// =============================================================================
// tests/bridge/conversational_mode_test.cpp
//
// The Mode A EXTRACTION PROOF. ConversationalMode claims to be a pure relocation
// of the wiring that used to sit inline in translator/main.cpp — same handles,
// same config, same callback registration, same teardown order. A claim like that
// is only worth anything if it is checked, and "it compiles" does not check it.
//
// So this suite builds BOTH: the old wiring, spelled out by hand exactly as
// main() did it, and a ConversationalMode — each over its own SimulatedEngineControl —
// feeds them THE SAME audio, and asserts the two are indistinguishable in what
// they marshal onto the engine thread and what text comes back out.
//
// If someone later "tidies" a config value or reorders construction inside
// ConversationalMode, the hand-written control path here will not follow, and the
// A/B diverges. That is the entire point.
//
// CPU-only: SimulatedEngineControl drives the REAL EngineControlBridge (SPSC ring,
// barge-in epoch, decode loop) with a canned payload, and the VAD stays null so
// the pipeline uses its built-in RMS detector — no ONNXRuntime, no model file, no
// GPU. Silero's own behaviour is pinned in tests/vad/silero_vad_test.cpp; what is
// under test here is the wiring, not the detector.
// =============================================================================
#include "conversational_mode.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "simulated_engine_control.hpp"

namespace {

// Accumulates everything the pipeline reports, so two runs can be compared as
// whole observations rather than field by field.
//
// The timeline records WHEN each transition fired (in 10 ms blocks), not just the
// order. That is load-bearing: an earlier version of this test recorded only the
// state SEQUENCE, and a mutation of silence_hangover_ms (800 -> 400) sailed
// straight through it — the drive script feeds 1200 ms of silence, which clears
// either hangover, so the same states arrive in the same order, just sooner.
// Stamping the block index makes every timing-affecting config value in
// ConversationalMode::Config observable, which is what this suite claims to pin.
struct Sink {
    int now = 0;                                   // current block index, set by the driver
    std::string text;
    std::vector<std::pair<int, int>> timeline;     // (block index, state)
    std::uint64_t finals = 0;

    bool operator==(const Sink& o) const {
        return text == o.text && timeline == o.timeline && finals == o.finals;
    }
};

void on_token(void* user, const SpeechTokenEvent* event, std::uint64_t) {
    auto* s = static_cast<Sink*>(user);
    if (s != nullptr && event != nullptr && event->text != nullptr) s->text += event->text;
}

void on_state(void* user, SpeechPipelineState, SpeechPipelineState next, std::uint64_t) {
    auto* s = static_cast<Sink*>(user);
    if (s == nullptr) return;
    s->timeline.emplace_back(s->now, static_cast<int>(next));
    if (next == SPEECH_STATE_IDLE) ++s->finals;
}

constexpr std::uint32_t kSampleRate = 16000;
constexpr std::size_t   kBlock      = 160;    // 10 ms
constexpr const char*   kSystemPrompt =
    "You are a real-time speech transcriber and translator.";

// One utterance: loud enough to clear the -40 dBFS onset, then silence past the
// 800 ms hangover so the controller commits. Identical for both paths.
template <typename PushFn, typename PumpFn>
void drive_one_utterance(Sink* sink, PushFn push, PumpFn pump) {
    const std::vector<float> loud(kBlock, 0.3f);     // ~-10 dBFS
    const std::vector<float> quiet(kBlock, 0.0f);
    for (int i = 0; i < 60; ++i) {                   // 600 ms speech
        sink->now = i;
        push(loud.data(), kBlock);
        pump();
    }
    for (int i = 0; i < 120; ++i) {                  // 1200 ms silence
        sink->now = 60 + i;
        push(quiet.data(), kBlock);
        pump();
    }
    sink->now = 180;
    for (int i = 0; i < 200; ++i) pump();            // drain the decode
}

// ---- Path A: the wiring as main.cpp used to spell it out ---------------------
// Deliberately hand-written and deliberately NOT calling into ConversationalMode.
class LegacyWiring {
public:
    explicit LegacyWiring(rt::SimulatedEngineControl* control, Sink* sink) {
        eng_ = blackwell::bridge::bridge_wrap_engine(control);
        EXPECT_NE(eng_, nullptr);
        EXPECT_EQ(engine_create_audio_stream(eng_, &stream_), BRIDGE_OK);

        SpeechPipelineConfig scfg{};
        scfg.engine                   = eng_;
        scfg.audio_stream             = stream_;
        scfg.sample_rate              = kSampleRate;
        scfg.vad_threshold_db         = -40.0f;
        scfg.vad_release_db           = -45.0f;
        scfg.silence_hangover_ms      = 800;
        scfg.warm_prefill_interval_ms = 320;
        EXPECT_EQ(speech_pipeline_create(&scfg, &pipe_), BRIDGE_OK);
        speech_pipeline_register_callbacks(pipe_, &on_token, &on_state, sink);
    }
    ~LegacyWiring() {
        speech_pipeline_destroy(pipe_);
        engine_destroy_audio_stream(stream_);
        blackwell::bridge::bridge_release_engine(eng_);
    }
    void push(const float* s, std::size_t n) { (void)speech_pipeline_push_pcm(pipe_, s, n); }

private:
    EngineHandle         eng_    = nullptr;
    AudioStreamHandle    stream_ = nullptr;
    SpeechPipelineHandle pipe_   = nullptr;
};

// THE PROOF. Same audio, same mock, two wirings — one observation.
TEST(ConversationalModeExtraction, IsIndistinguishableFromTheOldInlineWiring) {
    Sink legacy_sink;
    std::uint32_t legacy_prefix = 0;
    {
        rt::SimulatedEngineControl control;
        LegacyWiring w(&control, &legacy_sink);
        legacy_prefix = control.prefill_system_prompt(kSystemPrompt);
        drive_one_utterance(&legacy_sink,
                            [&](const float* s, std::size_t n) { w.push(s, n); },
                            [&] { (void)control.pump(); });
    }

    Sink mode_sink;
    std::uint32_t mode_prefix = 0;
    {
        rt::SimulatedEngineControl control;
        rt::ConversationalMode::Config cfg;
        cfg.sample_rate   = kSampleRate;
        cfg.system_prompt = kSystemPrompt;
        rt::ConversationalMode mode(
            &control,
            [&control](const std::string& p) { return control.prefill_system_prompt(p); },
            /*vad_fn=*/nullptr, /*vad_user=*/nullptr, cfg, &on_token, &on_state,
            &mode_sink);
        mode.start_on_engine_thread();
        mode_prefix = mode.frozen_prefix_tokens();
        drive_one_utterance(&mode_sink,
            [&](const float* s, std::size_t n) { mode.on_pcm_block(s, n); },
            [&] { (void)control.pump(); });
    }

    // The utterance actually happened — otherwise "identical" would be the
    // trivially-true comparison of two empty runs.
    EXPECT_FALSE(legacy_sink.text.empty());
    EXPECT_GT(legacy_sink.finals, 0u);

    EXPECT_EQ(legacy_prefix, mode_prefix);
    EXPECT_EQ(legacy_sink.text, mode_sink.text);
    EXPECT_EQ(legacy_sink.timeline, mode_sink.timeline);
    EXPECT_EQ(legacy_sink.finals, mode_sink.finals);
    EXPECT_TRUE(legacy_sink == mode_sink);
}

// ---- the ISpeechMode contract itself ----------------------------------------

// start_on_engine_thread() is where the mode's frozen prefix is laid down, and it
// is per-mode BY DESIGN (a conversational assistant and a verbatim transcriber
// want different system prompts) — which is what makes a mode switch a session
// boundary rather than a free toggle.
TEST(ConversationalModeExtraction, StartOnEngineThreadFreezesTheSystemPrefix) {
    rt::SimulatedEngineControl control;
    rt::ConversationalMode::Config cfg;
    cfg.system_prompt = kSystemPrompt;
    Sink sink;
    rt::ConversationalMode mode(
        &control,
        [&control](const std::string& p) { return control.prefill_system_prompt(p); },
        nullptr, nullptr, cfg, &on_token, &on_state, &sink);

    EXPECT_EQ(mode.frozen_prefix_tokens(), 0u) << "nothing prefilled before the hook runs";
    mode.start_on_engine_thread();
    EXPECT_GT(mode.frozen_prefix_tokens(), 0u);
    EXPECT_EQ(control.system_prefix_tokens(), mode.frozen_prefix_tokens());
}

// The runner joins by calling stop() and then falling out of pump_engine(). If
// stop() did not unblock the pump, the engine thread would hang at shutdown —
// which is exactly the bug the two-step (cancel then stop) exists to avoid.
TEST(ConversationalModeExtraction, StopUnblocksThePumpSoTheRunnerCanJoin) {
    rt::SimulatedEngineControl control;
    rt::ConversationalMode::Config cfg;
    Sink sink;
    rt::ConversationalMode mode(
        &control,
        [&control](const std::string& p) { return control.prefill_system_prompt(p); },
        nullptr, nullptr, cfg, &on_token, &on_state, &sink);

    std::atomic<bool> running{true};
    std::atomic<bool> exited{false};
    std::thread runner([&] {
        while (running.load(std::memory_order_acquire)) mode.pump_engine();
        exited.store(true, std::memory_order_release);
    });

    running.store(false, std::memory_order_release);
    mode.stop();
    runner.join();     // hangs here if stop() failed to wake the pump
    EXPECT_TRUE(exited.load(std::memory_order_acquire));
}

// ---- the VAD pre-roll, and why it is pinned HERE -----------------------------
// pre_roll_ms spent its first life as dead config: the struct held it, the panel
// had a slider for it, and the only consumer was the OFFLINE re-translation
// driver — so the live engine hard-flushed the ring at speech onset and clipped
// the first syllable off every utterance. The flush itself lives in a CUDA-only
// header, but the failure was never in the arithmetic: it was that the value the
// user set never reached the control that flushes. That IS testable without a
// GPU, and it is what these two pin.
TEST(ConversationalModeExtraction, PreRollReachesTheControlAtConstruction) {
    rt::SimulatedEngineControl control;
    rt::ConversationalMode::Config cfg;
    cfg.pre_roll_ms = 180;                   // deliberately not the default
    Sink sink;
    rt::ConversationalMode mode(
        &control,
        [&control](const std::string& p) { return control.prefill_system_prompt(p); },
        nullptr, nullptr, cfg, &on_token, &on_state, &sink);

    // Published by the ctor, not by a later live-settings push: a mode is fully
    // configured the moment it exists, so the FIRST onset already honours it.
    EXPECT_EQ(control.pre_roll_ms(), 180);

    // And it stays live — the settings modal writes this while audio is flowing.
    control.set_pre_roll_ms(400);
    EXPECT_EQ(control.pre_roll_ms(), 400);
    control.set_pre_roll_ms(-1);             // clamped at the setter, never negative
    EXPECT_EQ(control.pre_roll_ms(), 0);
}

// The ms -> samples conversion the flush subtracts from available_samples().
// 0 must mean "keep nothing" exactly (the pre-fix hard flush stays reachable).
TEST(ConversationalModeExtraction, PreRollSamplesConvertsAtTheStreamRate) {
    rt::SimulatedEngineControl control;

    control.set_pre_roll_ms(250);
    EXPECT_EQ(control.pre_roll_samples(16000), 4000u);   // the shipping default
    control.set_pre_roll_ms(0);
    EXPECT_EQ(control.pre_roll_samples(16000), 0u);
    control.set_pre_roll_ms(1000);                        // the UI's ceiling
    EXPECT_EQ(control.pre_roll_samples(16000), 16000u);
}

// The null-VAD fallback (a missing 2.3 MB model must not take the app down) is
// covered by the A/B above rather than by a dedicated assertion: both paths there
// run with vad == nullptr and still commit a full utterance, which is only
// possible if the pipeline's built-in RMS detector stayed in place.

}  // namespace
