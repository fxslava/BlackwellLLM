// =============================================================================
// tests/bridge/speech_pipeline_vad_scorer_test.cpp
//
// Tier-1 regression for the EXTERNAL VAD SCORER seam (SpeechVadScoreFn): the
// hook that lets a neural detector replace the built-in RMS threshold as the
// answer to "is this block speech?", without touching anything built on top of
// that answer.
//
// Deliberately uses a SCRIPTED FAKE scorer, not Silero: this suite must stay
// CPU-only, deterministic and free of ONNXRuntime and the model file. The real
// model's behaviour is pinned separately in tests/vad/silero_vad_test.cpp; what
// is under test HERE is the contract between a scorer and the state machine.
//
// The two directions that matter, and neither is expressible with an RMS
// threshold alone:
//   * LOUD audio the scorer rejects  -> no onset  (the false-trigger fix: a
//     slammed door or keyboard clatter no longer starts an utterance)
//   * QUIET audio the scorer accepts -> onset     (soft speech that sits under
//     the dBFS floor is no longer missed)
//
// Drives the REAL SpeechPipelineController against a recording IEngineControl,
// exactly like the manual-mode and background-listening suites next door.
// =============================================================================
#include "speech_pipeline_controller.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "bridge_internal.hpp"

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

// A scorer whose verdict the test sets directly, standing in for the neural
// model. Counts its invocations so "was it consulted at all?" is observable.
struct ScriptedScorer {
    float probability = 0.0f;   // what the next block scores
    int calls = 0;              // how many blocks it has been asked about
    size_t last_count = 0;      // samples handed over on the last call
    bool saw_nonzero_audio = false;

    static float score(void* user, const float* block, size_t count) {
        auto* self = static_cast<ScriptedScorer*>(user);
        ++self->calls;
        self->last_count = count;
        for (size_t i = 0; i < count; ++i) {
            if (block[i] != 0.0f) {
                self->saw_nonzero_audio = true;
                break;
            }
        }
        return self->probability;
    }
};

class SpeechPipelineVadScorer : public ::testing::Test {
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
        cfg.vad_release_db = 0.0f;         // 0 => release == onset (no hysteresis)
        cfg.silence_hangover_ms = kHangoverMs;
        cfg.warm_prefill_interval_ms = 0;  // warming off: counts reflect boundaries only
        ASSERT_EQ(SpeechPipelineController::create(cfg, &pipe_), BRIDGE_OK);
    }

    void TearDown() override {
        delete pipe_;
        if (engine_ != nullptr) bridge_release_engine(engine_);
    }

    void install_scorer() { pipe_->set_vad_scorer(&ScriptedScorer::score, &scorer_); }

    // 0.5 ≈ -6 dBFS (far above the -40 dBFS onset); 0.001 ≈ -60 dBFS (far below
    // it); 0.0 is digital silence.
    void push_ms(float amplitude, int ms) {
        std::vector<float> pcm(static_cast<size_t>(ms) * (kSampleRate / 1000), amplitude);
        pipe_->push_pcm(pcm.data(), pcm.size());
    }

    RecordingControl control_;
    ScriptedScorer scorer_;
    std::unique_ptr<BridgeAudioStreamOpaque> stream_;
    EngineHandle engine_ = nullptr;
    SpeechPipelineController* pipe_ = nullptr;
};

// Default state: no scorer installed, so the RMS threshold runs and behaves
// exactly as it always has. This is the "nothing changed for existing callers"
// guarantee the additive ABI rests on.
TEST_F(SpeechPipelineVadScorer, WithoutScorerRmsPathIsUnchanged) {
    EXPECT_FALSE(pipe_->has_vad_scorer());

    push_ms(0.5f, 50);  // loud
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    push_ms(0.0f, 400);  // silence past the hangover
    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
    EXPECT_EQ(scorer_.calls, 0) << "an uninstalled scorer must never be consulted";
}

// The scorer is handed the block's ACTUAL SAMPLES, one call per 10 ms block --
// a neural model cannot work from a summary statistic.
TEST_F(SpeechPipelineVadScorer, ScorerReceivesEveryBlocksSamples) {
    install_scorer();
    EXPECT_TRUE(pipe_->has_vad_scorer());

    scorer_.probability = 0.0f;
    push_ms(0.25f, 100);  // 10 blocks of 10 ms

    EXPECT_EQ(scorer_.calls, 10);
    EXPECT_EQ(scorer_.last_count, 160u) << "one call per 10 ms block at 16 kHz";
    EXPECT_TRUE(scorer_.saw_nonzero_audio) << "the scorer must see the audio, not zeros";
}

// DIRECTION 1 — the false-trigger fix. Audio loud enough to clear the dBFS onset
// by ~34 dB, which the scorer rejects, must NOT start an utterance.
TEST_F(SpeechPipelineVadScorer, LoudNonSpeechRejectedByScorerDoesNotTrigger) {
    install_scorer();
    scorer_.probability = 0.02f;  // "not speech"

    push_ms(0.5f, 500);  // -6 dBFS: the RMS path would have fired immediately

    EXPECT_GT(scorer_.calls, 0);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_IDLE);
    EXPECT_EQ(control_.cancels, 0);
    EXPECT_EQ(control_.rewinds, 0);
    EXPECT_EQ(control_.commits, 0);
}

// DIRECTION 2 — the missed-speech fix. Audio well below the dBFS floor that the
// scorer accepts MUST start an utterance, and the rest of the machine (hangover
// -> commit -> DECODE) then runs untouched on top of it.
TEST_F(SpeechPipelineVadScorer, QuietSpeechAcceptedByScorerDrivesTheFullCycle) {
    install_scorer();
    scorer_.probability = 0.95f;  // "speech", at -60 dBFS

    push_ms(0.001f, 50);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
    EXPECT_EQ(control_.cancels, 1);
    EXPECT_EQ(control_.rewinds, 1);

    // Now the scorer reports silence over LOUD audio: the hangover must still
    // run off the scorer's verdict and commit, proving the silence accounting
    // is driven by the decision and not by energy.
    scorer_.probability = 0.01f;
    push_ms(0.5f, 400);

    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

// The sensitivity knob: the same probability lands on either side of the
// decision depending on the live threshold.
TEST_F(SpeechPipelineVadScorer, ThresholdRetuneMovesTheDecision) {
    install_scorer();
    scorer_.probability = 0.45f;

    pipe_->set_vad_threshold(0.8f);  // insensitive: 0.45 is not speech
    push_ms(0.001f, 100);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_IDLE);

    pipe_->set_vad_threshold(0.2f);  // sensitive: 0.45 is speech
    push_ms(0.001f, 100);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    EXPECT_FLOAT_EQ(pipe_->vad_threshold(), 0.2f);
}

// Out-of-range thresholds clamp rather than disabling the detector outright.
TEST_F(SpeechPipelineVadScorer, ThresholdClamps) {
    pipe_->set_vad_threshold(5.0f);
    EXPECT_FLOAT_EQ(pipe_->vad_threshold(), 1.0f);
    pipe_->set_vad_threshold(-3.0f);
    EXPECT_FLOAT_EQ(pipe_->vad_threshold(), 0.0f);
}

// Hysteresis: the release threshold sits 0.15 below onset, so a probability in
// the band between them sustains an utterance already in progress but would not
// have started one. Without this the decision chatters block to block.
TEST_F(SpeechPipelineVadScorer, ProbabilityInHysteresisBandSustainsButDoesNotOnset) {
    install_scorer();
    pipe_->set_vad_threshold(0.5f);  // release = 0.35

    scorer_.probability = 0.42f;  // inside the band
    push_ms(0.001f, 200);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_IDLE) << "band value must not start an utterance";

    scorer_.probability = 0.9f;  // clear onset
    push_ms(0.001f, 20);
    ASSERT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    // Back into the band, for far longer than the hangover: still speaking, so
    // no commit may fire.
    scorer_.probability = 0.42f;
    push_ms(0.001f, 600);
    EXPECT_EQ(control_.commits, 0) << "band value must sustain, not time out";
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    scorer_.probability = 0.1f;  // below release: the silence clock runs
    push_ms(0.001f, 400);
    EXPECT_EQ(control_.commits, 1);
}

// A negative score is "no opinion": the previous verdict stands, rather than
// falling back to RMS and interleaving two detectors' hysteresis.
TEST_F(SpeechPipelineVadScorer, NegativeScoreHoldsThePreviousVerdict) {
    install_scorer();

    scorer_.probability = 0.9f;  // speech
    push_ms(0.001f, 20);
    ASSERT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    // "No opinion" over LOUD audio for well past the hangover. Holding the
    // previous (speaking) verdict means no commit; falling back to RMS would
    // also read this as speech, so the discriminating part is the next block.
    scorer_.probability = -1.0f;
    push_ms(0.5f, 600);
    EXPECT_EQ(control_.commits, 0);

    // Same "no opinion", but now over SILENCE. The RMS path would call this
    // silence and commit; holding the previous verdict must not.
    push_ms(0.0f, 600);
    EXPECT_EQ(control_.commits, 0) << "no-opinion must hold, not defer to RMS";
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
}

// Uninstalling restores the built-in detector, so the neural path can be a live
// UI toggle rather than a launch-time commitment.
TEST_F(SpeechPipelineVadScorer, ClearingScorerRestoresTheRmsPath) {
    install_scorer();
    scorer_.probability = 0.0f;

    push_ms(0.5f, 200);  // loud, rejected by the scorer
    ASSERT_EQ(pipe_->state(), SPEECH_STATE_IDLE);

    pipe_->set_vad_scorer(nullptr, nullptr);
    EXPECT_FALSE(pipe_->has_vad_scorer());

    const int calls_before = scorer_.calls;
    push_ms(0.5f, 50);  // same loud audio: the RMS path fires on it
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
    EXPECT_EQ(scorer_.calls, calls_before) << "a cleared scorer must not be called";
}

// Manual / push-to-talk mode outranks the scorer exactly as it outranks the RMS
// threshold: while held, NO automatic verdict may move the state machine.
TEST_F(SpeechPipelineVadScorer, ManualModeStillMutesTheScorer) {
    install_scorer();
    pipe_->set_manual_mode(true);
    scorer_.probability = 1.0f;  // maximally confident speech

    push_ms(0.5f, 200);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_IDLE);
    EXPECT_EQ(control_.cancels, 0);

    pipe_->on_speech_start();  // explicit press still works
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    scorer_.probability = 0.0f;  // would auto-commit in automatic mode
    push_ms(0.0f, 600);
    EXPECT_EQ(control_.commits, 0);

    pipe_->on_silence_timeout();  // explicit release
    EXPECT_EQ(control_.commits, 1);
}

}  // namespace
