// =============================================================================
// tests/bridge/speech_pipeline_manual_mode_test.cpp
//
// Tier-1 regression for the MANUAL / push-to-talk VAD mute: while manual mode is
// set, the internal threshold VAD must never drive a state transition (no auto
// onset, no silence auto-commit, no auto barge-in) — boundaries come EXCLUSIVELY
// from the explicit on_speech_start / on_silence_timeout events. This pins the
// fix for the race where a background VAD trigger fired in parallel with hotkey
// events and broke the state machine mid-switch.
//
// Drives the REAL SpeechPipelineController (real state machine, real 10 ms block
// VAD, real epoch bumps) against a recording IEngineControl mock, so every
// marshal the controller issues (cancel / rewind / warm / commit) is counted.
// The only CUDA touched is the stream ring's pinned allocation (AudioRingBuffer
// uses cudaHostAlloc); a machine without a CUDA device SKIPs rather than fails.
// =============================================================================
#include "speech_pipeline_controller.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "bridge_internal.hpp"  // IEngineControl, BridgeAudioStreamOpaque, bridge_wrap_engine

namespace {

using blackwell::EngineStatus;
using namespace blackwell::bridge;

// Records every marshal the controller issues; never touches an engine.
class RecordingControl : public IEngineControl {
public:
    size_t audio_ring_capacity_samples() const override { return 16000; }
    bool generation_in_flight() const override { return false; }
    EngineStatus submit_multimodal(AudioRingBuffer&, const char*, TokenSink) override {
        return EngineStatus::Success;
    }
    void cancel_generation(uint64_t gen) noexcept override { ++cancels; last_gen = gen; }
    EngineStatus rewind_kv(AudioStreamHandle, uint32_t keep, uint64_t) noexcept override {
        ++rewinds;
        last_keep = keep;
        return EngineStatus::Success;
    }
    EngineStatus warm_prefill(AudioStreamHandle, uint64_t) noexcept override {
        ++warms;
        return EngineStatus::Success;
    }
    EngineStatus commit_and_decode(AudioStreamHandle, TokenSink, uint64_t) noexcept override {
        ++commits;
        return EngineStatus::Success;
    }

    int cancels = 0, rewinds = 0, warms = 0, commits = 0;
    uint32_t last_keep = 0;
    uint64_t last_gen = 0;
};

class SpeechPipelineManualMode : public ::testing::Test {
protected:
    static constexpr uint32_t kSampleRate = 16000;
    static constexpr uint32_t kHangoverMs = 300;

    void SetUp() override {
        try {
            stream_ = std::make_unique<BridgeAudioStreamOpaque>(/*capacity=*/1u << 15);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "pinned ring unavailable (no CUDA device?): " << e.what();
        }
        engine_ = bridge_wrap_engine(&control_);
        ASSERT_NE(engine_, nullptr);

        SpeechPipelineConfig cfg{};
        cfg.engine = engine_;
        cfg.audio_stream = stream_.get();
        cfg.sample_rate = kSampleRate;
        cfg.vad_threshold_db = -40.0f;
        cfg.vad_release_db = 0.0f;            // 0 => release == onset (no hysteresis)
        cfg.silence_hangover_ms = kHangoverMs;
        cfg.warm_prefill_interval_ms = 0;     // warming off: counts reflect boundaries only
        ASSERT_EQ(SpeechPipelineController::create(cfg, &pipe_), BRIDGE_OK);
    }

    void TearDown() override {
        delete pipe_;
        if (engine_ != nullptr) bridge_release_engine(engine_);
    }

    // Feed `ms` of constant-amplitude PCM (whole 10 ms VAD blocks). 0.5 ≈ -6 dBFS
    // (loud, above the -40 onset); 0.0 ≈ -120 dBFS (silence, below release).
    void push_ms(float amplitude, int ms) {
        std::vector<float> pcm(static_cast<size_t>(ms) * (kSampleRate / 1000), amplitude);
        pipe_->push_pcm(pcm.data(), pcm.size());
    }

    RecordingControl control_;
    std::unique_ptr<BridgeAudioStreamOpaque> stream_;
    EngineHandle engine_ = nullptr;
    SpeechPipelineController* pipe_ = nullptr;
};

// Sanity baseline: with manual mode OFF the auto-VAD drives the full cycle
// (onset -> PREFILL, hangover silence -> commit + DECODE).
TEST_F(SpeechPipelineManualMode, AutoVadDrivesStateWithoutManualMode) {
    push_ms(0.5f, 50);                                       // loud onset
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
    EXPECT_EQ(control_.cancels, 1);
    EXPECT_EQ(control_.rewinds, 1);

    push_ms(0.0f, 400);                                      // silence past the hangover
    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

// The fix under test: in manual mode NO auto trigger may fire — not the onset,
// not the silence auto-commit, not the mid-decode barge-in. Explicit press /
// release events remain the exclusive drivers and still work.
TEST_F(SpeechPipelineManualMode, ManualModeMutesEveryAutoTrigger) {
    pipe_->set_manual_mode(true);

    push_ms(0.5f, 100);                                      // would auto-onset
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_IDLE);
    EXPECT_EQ(control_.cancels, 0);
    EXPECT_EQ(control_.rewinds, 0);

    pipe_->on_speech_start();                                // explicit key press
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
    EXPECT_EQ(control_.rewinds, 1);

    push_ms(0.5f, 100);                                      // speaking while held
    push_ms(0.0f, 600);                                      // silence >> 300 ms hangover
    EXPECT_EQ(control_.commits, 0);                          // auto-commit muted
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    pipe_->on_silence_timeout();                             // explicit key release
    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);

    push_ms(0.5f, 100);                                      // would auto barge-in
    EXPECT_EQ(control_.cancels, 1);                          // only the press's epoch bump
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

// TRUE PUSH-TO-TALK: in manual mode the ring receives PCM only while capturing
// (PREFILL_SPEAKING, i.e. between press and release). Idle and mid-decode
// background audio is dropped at the producer edge, so a press can never feed
// pre-press noise into the engine; auto mode keeps the always-feed behaviour.
TEST_F(SpeechPipelineManualMode, ManualModeGatesRingToCaptureWindow) {
    const auto samples_of_ms = [](int ms) {
        return static_cast<size_t>(ms) * (kSampleRate / 1000);
    };
    pipe_->set_manual_mode(true);

    push_ms(0.5f, 100);                                      // idle background noise
    EXPECT_EQ(stream_->ring.available_samples(), 0u);        // dropped at the door

    pipe_->on_speech_start();                                // press: capture opens
    push_ms(0.5f, 100);
    EXPECT_EQ(stream_->ring.available_samples(), samples_of_ms(100));

    pipe_->on_silence_timeout();                             // release: decode begins
    push_ms(0.5f, 100);                                      // noise during decode
    EXPECT_EQ(stream_->ring.available_samples(), samples_of_ms(100));  // still dropped

    pipe_->set_manual_mode(false);                           // auto mode: always feeds
    push_ms(0.0f, 50);                                       // silent (no auto barge-in)
    EXPECT_EQ(stream_->ring.available_samples(), samples_of_ms(150));
}

// A manual -> auto flip must not fire an instant commit off silence accumulated
// while muted: the auto-commit clock restarts from the flip.
TEST_F(SpeechPipelineManualMode, DisablingManualModeRestartsSilenceClock) {
    pipe_->set_manual_mode(true);
    pipe_->on_speech_start();
    push_ms(0.0f, 600);                                      // silent hold, muted clock
    EXPECT_EQ(control_.commits, 0);

    pipe_->set_manual_mode(false);                           // back to auto
    push_ms(0.0f, 290);                                      // 290 < 300 ms from the flip
    EXPECT_EQ(control_.commits, 0);
    push_ms(0.0f, 20);                                       // crosses the hangover
    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

}  // namespace
