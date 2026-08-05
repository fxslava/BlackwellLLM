// =============================================================================
// tests/bridge/speech_pipeline_vad_suppressor_test.cpp
//
// Tier-1 regression for the VAD SUPPRESSOR — the hard mask the app holds while
// its own speech is audible (see speech_pipeline_set_vad_suppressed and the
// playback time-lock in audio_sandbox/include/audio_playback.h).
//
// THE BUG THIS PINS. The microphone hears the loudspeaker. The echo canceller
// removes most of it, but its residue is speech-shaped and the VAD scores it as
// speech — so the assistant fired barge-in on its own sentence and cancelled the
// answer it was in the middle of speaking, which surfaced as a `chunks
// cancelled` count with nobody in the room talking. The previous guard raised
// the probability bar while speaking; it is a statistical bound and it failed
// whenever WASAPI clock drift pushed the reference out of alignment. The
// suppressor replaces it with state: while the mask is set the verdict is
// silence, unconditionally.
//
// WHAT IS ASSERTED, and it is deliberately the strong form: NO transition of any
// kind may come from the internal VAD while the mask is set — not onset from
// idle, not barge-in mid-decode, not the silence auto-commit. And clearing it
// must re-arm from a clean slate rather than replaying a verdict formed before
// it was set.
//
// Drives the REAL SpeechPipelineController against a recording IEngineControl,
// exactly as the manual-mode test does. The RMS path is used for the transition
// assertions (no scorer installed) because the mask must hold on BOTH detectors
// and the RMS one needs no model file; one test installs a scorer to prove the
// mask sits above that path too — and that the scorer is not even consulted.
//
// The only CUDA touched is the stream ring's pinned allocation; a machine
// without a CUDA device SKIPs rather than fails.
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

// Counts calls so a test can assert the scorer was NOT consulted while masked.
// Static, because SpeechVadScoreFn is a raw function pointer (the seam refuses
// std::function on purpose — it is called from the audio thread).
int g_scorer_calls = 0;
float g_scorer_reply = 1.0f;
float CountingScorer(void*, const float*, size_t) {
    ++g_scorer_calls;
    return g_scorer_reply;
}

class SpeechPipelineVadSuppressor : public ::testing::Test {
protected:
    static constexpr uint32_t kSampleRate = 16000;
    static constexpr uint32_t kHangoverMs = 300;

    void SetUp() override {
        g_scorer_calls = 0;
        g_scorer_reply = 1.0f;
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
    // 0.5 here stands in for the assistant's own voice arriving back through the
    // microphone: the pipeline cannot tell the difference, which is the point.
    void push_ms(float amplitude, int ms) {
        std::vector<float> pcm(static_cast<size_t>(ms) * (kSampleRate / 1000), amplitude);
        pipe_->push_pcm(pcm.data(), pcm.size());
    }

    RecordingControl control_;
    std::unique_ptr<BridgeAudioStreamOpaque> stream_;
    EngineHandle engine_ = nullptr;
    SpeechPipelineController* pipe_ = nullptr;
};

// Sanity baseline: unmasked, loud audio drives the full cycle. Without this the
// tests below could pass by never producing a transition at all.
TEST_F(SpeechPipelineVadSuppressor, UnmaskedAudioStillDrivesTheStateMachine) {
    EXPECT_FALSE(pipe_->vad_suppressed());
    push_ms(0.5f, 50);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
    EXPECT_EQ(control_.cancels, 1);

    push_ms(0.0f, 400);
    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

// THE FIX. Loud audio while masked -- the assistant's own voice leaking back --
// must produce no onset at all.
TEST_F(SpeechPipelineVadSuppressor, MaskedAudioProducesNoOnset) {
    pipe_->set_vad_suppressed(true);

    push_ms(0.5f, 500);                                  // half a second of "echo"
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_IDLE);
    EXPECT_EQ(control_.cancels, 0);
    EXPECT_EQ(control_.rewinds, 0);
}

// THE ONE THAT COST CHUNKS. The assistant is mid-answer (DECODE_TRANSLATING) and
// speaking it aloud; the microphone is full of that speech. This is exactly the
// path that produced `chunks cancelled` with an empty room, so it is asserted
// separately from the idle case rather than assumed to follow from it.
TEST_F(SpeechPipelineVadSuppressor, MaskedAudioCannotBargeInOnAnAnswerInFlight) {
    push_ms(0.5f, 50);                                   // user speaks
    push_ms(0.0f, 400);                                  // and pauses -> commit
    ASSERT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
    const int cancels_before = control_.cancels;

    // The answer is now being spoken: the app raises the mask for as long as the
    // speaker is audible, plus its acoustic tail.
    pipe_->set_vad_suppressed(true);
    push_ms(0.5f, 2000);                                 // two seconds of our own voice

    EXPECT_EQ(control_.cancels, cancels_before) << "the assistant barged in on itself";
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

// The mask must not fire the OTHER auto-transition either, and this is the test
// that caught the first implementation being wrong.
//
// Masking only the VERDICT is not enough: a masked block scores as silence, and
// silence is exactly what the auto-commit clock counts, so a mask held over an
// open utterance committed it early -- on evidence the pipeline never gathered.
// The fix freezes the clock while masked, the same way manual mode does, so
// "masked" means "made no observation" rather than "observed silence".
//
// The suppressor is only set while the assistant speaks, which in the shipping
// flow cannot overlap an open user utterance -- but a guard that silently
// commits on a state it was never meant to reach is a landmine, so it is pinned.
TEST_F(SpeechPipelineVadSuppressor, MaskedSilenceDoesNotAutoCommit) {
    push_ms(0.5f, 50);                                   // onset -> PREFILL
    ASSERT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    pipe_->set_vad_suppressed(true);
    push_ms(0.0f, 2000);                                 // far past the 300 ms hangover
    EXPECT_EQ(control_.commits, 0);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);
}

// THE FAILURE MODE THAT MATTERS MOST for any suppressor is one that never lifts
// -- an assistant permanently deaf after its first sentence is worse than the
// bug the mask was added to fix. So: clearing it hands control straight back.
//
// And the clock it hands back is a FRESH one. The 2 s of masked silence above
// must not count toward the hangover, or the first block after an answer ends
// would auto-commit whatever the user had not finished saying.
TEST_F(SpeechPipelineVadSuppressor, ClearingTheMaskRestartsTheSilenceClock) {
    push_ms(0.5f, 50);                                   // onset -> PREFILL
    pipe_->set_vad_suppressed(true);
    push_ms(0.0f, 2000);                                 // masked: clock frozen
    ASSERT_EQ(control_.commits, 0);

    pipe_->set_vad_suppressed(false);
    push_ms(0.0f, 290);                                  // 290 < 300 ms FROM THE LIFT
    EXPECT_EQ(control_.commits, 0) << "the mask's own silence leaked into the clock";
    push_ms(0.0f, 20);                                   // crosses the hangover
    EXPECT_EQ(control_.commits, 1);
    EXPECT_EQ(pipe_->state(), SPEECH_STATE_DECODE_TRANSLATING);
}

// Clearing the mask must re-arm from SILENCE, not from the verdict that was in
// force when it was set. The pipeline genuinely did not observe the room while
// masked, and a held `sustain` would let the first ambiguous block after the
// mask lifts resurrect a stale decision.
TEST_F(SpeechPipelineVadSuppressor, ClearingTheMaskReArmsFromSilence) {
    // Install a scorer and drive it into "speaking", so there is a live verdict
    // to go stale. A negative score is the scorer's "no opinion", which is
    // precisely the case that consults the held decision.
    pipe_->set_vad_scorer(&CountingScorer, nullptr);
    pipe_->set_vad_threshold(0.5f);
    g_scorer_reply = 1.0f;
    push_ms(0.5f, 50);
    ASSERT_EQ(pipe_->state(), SPEECH_STATE_PREFILL_SPEAKING);

    pipe_->set_vad_suppressed(true);
    const int calls_at_mask = g_scorer_calls;
    push_ms(0.5f, 200);
    // NOT CONSULTED while masked: Silero is recurrent, and feeding it our own
    // echo would leave its LSTM state primed on a sentence the user never spoke.
    EXPECT_EQ(g_scorer_calls, calls_at_mask);

    pipe_->set_vad_suppressed(false);
    g_scorer_reply = -1.0f;                              // "no opinion" -> held verdict
    push_ms(0.5f, 400);
    EXPECT_GT(g_scorer_calls, calls_at_mask) << "the scorer never resumed";
    // The held verdict is now silence, so the hangover elapses and the utterance
    // commits. Were the pre-mask `sustain` still in force, this would hang open.
    EXPECT_EQ(control_.commits, 1);
}

// PCM MUST KEEP FLOWING. This is what separates the suppressor from the
// microphone gate it replaced (and from manual mode, which gates the ring): the
// pre-roll a real interruption needs, the level meter, and speculative warming
// all read the ring, and a mask that emptied it would break all three the moment
// the assistant opened its mouth.
TEST_F(SpeechPipelineVadSuppressor, MaskedBlocksStillReachTheRing) {
    const auto samples_of_ms = [](int ms) {
        return static_cast<size_t>(ms) * (kSampleRate / 1000);
    };
    pipe_->set_vad_suppressed(true);
    push_ms(0.5f, 100);
    EXPECT_EQ(stream_->ring.available_samples(), samples_of_ms(100));
}

}  // namespace
