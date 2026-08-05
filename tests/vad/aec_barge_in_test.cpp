// -----------------------------------------------------------------------------
// THE FUNCTIONAL CHECK for full-duplex barge-in: the real Silero VAD, scoring
// the real echo canceller's output, on real speech.
//
// Every other test in this repo measures the canceller in decibels. Decibels are
// not the product claim. The product claims are exactly two, and they are the
// two this file asserts:
//
//   (a) while the assistant is speaking and the user is not, the VAD must NOT
//       fire -- otherwise the assistant barges in on itself and cancels the
//       answer it is in the middle of giving;
//   (b) the moment the user talks OVER the assistant, the VAD MUST fire --
//       otherwise there is no barge-in and the microphone might as well have
//       been muted, which is the state this work replaced.
//
// A canceller can satisfy (a) alone by attenuating everything (that is a mic
// gate) and (b) alone by cancelling nothing. Only both together are the feature,
// so both are asserted against the same signal, the same room and the same
// filter instance.
//
// =============================================================================
// WHAT THIS DOES NOT COVER, STATED PLAINLY
// =============================================================================
// The room here is a linear FIR: pure delay, direct path, three reflections. A
// real loudspeaker is NOT linear -- amplifier and cone distortion produce echo
// components no FIR of the reference can generate -- so the linear stage plateaus
// lower in a real room than it does here, and the residual suppressor carries
// more of the load. This test therefore proves the pipeline is correct and the
// decision logic holds; it does not predict the exact ERLE of any particular
// desk. The final check for that is a person talking over the running app.
//
// Both "voices" are real recorded speech rather than synthesis, because that is
// what the VAD was trained on and what makes its verdict meaningful.
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "aec_capture_filter.hpp"
#include "audio_test_wav.hpp"
#include "echo_canceller.hpp"
#include "silero_vad.hpp"

namespace {

using blackwell::audio_rt::BlockFdafEchoCanceller;
using blackwell::audio_rt::EchoCancellerConfig;
using blackwell::vad::SileroVAD;

constexpr std::size_t kRate = 16000;

// Speaker -> room -> microphone. The delay stands in for the playback device
// buffer plus flight time; the taps are the direct path and three reflections.
std::vector<float> MakeRoom(std::size_t delay, std::size_t len) {
    std::vector<float> h(len, 0.0f);
    if (delay < len) h[delay] = 0.55f;
    if (delay + 41 < len) h[delay + 41] = -0.28f;
    if (delay + 127 < len) h[delay + 127] = 0.16f;
    if (delay + 263 < len) h[delay + 263] = -0.07f;
    return h;
}

std::vector<float> Convolve(const std::vector<float>& x, const std::vector<float>& h) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) {
        double acc = 0.0;
        const std::size_t taps = std::min(h.size(), i + 1);
        for (std::size_t k = 0; k < taps; ++k) acc += static_cast<double>(h[k]) * x[i - k];
        y[i] = static_cast<float>(acc);
    }
    return y;
}

void ScaleToRms(std::vector<float>& v, double target) {
    double s = 0.0;
    for (const float x : v) s += static_cast<double>(x) * x;
    const double rms = std::sqrt(s / static_cast<double>(std::max<std::size_t>(1, v.size())));
    if (rms < 1e-9) return;
    const double g = target / rms;
    for (float& x : v) x = static_cast<float>(x * g);
}

// Runs the VAD over a span and reports the fraction of 32 ms chunks it scored as
// speech. A fraction, not a single probability: one chunk over threshold is
// noise, and the pipeline's own decision is a run of them.
double SpeechFraction(SileroVAD& vad, const std::vector<float>& x, std::size_t from,
                      std::size_t to, float threshold) {
    std::size_t chunks = 0, hot = 0;
    for (std::size_t i = from; i + blackwell::vad::kChunkSamples <= to;
         i += blackwell::vad::kChunkSamples) {
        const float p = vad.process_chunk(x.data() + i, blackwell::vad::kChunkSamples);
        if (p >= 0.0f) {
            ++chunks;
            if (p > threshold) ++hot;
        }
    }
    return chunks == 0 ? 0.0 : static_cast<double>(hot) / static_cast<double>(chunks);
}

// Sample index of the first chunk the VAD scores as speech, or `to` if none.
std::size_t FirstDetection(SileroVAD& vad, const std::vector<float>& x, std::size_t from,
                           std::size_t to, float threshold) {
    for (std::size_t i = from; i + blackwell::vad::kChunkSamples <= to;
         i += blackwell::vad::kChunkSamples) {
        if (vad.process_chunk(x.data() + i, blackwell::vad::kChunkSamples) > threshold) {
            return i;
        }
    }
    return to;
}

// The scenario both tests share: an assistant talking continuously, and a user
// who starts talking over it partway through.
struct Scene {
    std::vector<float> far;         // what the speaker emits (the reference)
    std::vector<float> mic;         // echo + near end, what the microphone hears
    std::vector<float> near_only;   // the user alone, for reference
    std::size_t talk_from = 0;      // where the user starts
};

Scene BuildScene(const std::vector<float>& speech) {
    // The assistant's voice: the clip, repeated so it speaks continuously for
    // long enough that the filter converges before it is interrupted -- which is
    // the real sequence, where the assistant has been talking for a second or
    // two before anyone cuts in.
    Scene s;
    for (int i = 0; i < 4; ++i) s.far.insert(s.far.end(), speech.begin(), speech.end());
    ScaleToRms(s.far, 0.08);

    // The user's voice: the SAME recording played backwards. It keeps the
    // spectral and temporal statistics of real speech (so the VAD's verdict
    // means something) while being uncorrelated with the far end -- which
    // matters, because a near-end signal correlated with the reference is one
    // the canceller is entitled to remove.
    std::vector<float> user(speech.rbegin(), speech.rend());
    ScaleToRms(user, 0.06);   // quieter than the echo: the hard case

    const std::vector<float> room = MakeRoom(112, 640);
    const std::vector<float> echo = Convolve(s.far, room);

    s.talk_from = s.far.size() / 2;
    s.mic.assign(s.far.size(), 0.0f);
    s.near_only.assign(s.far.size(), 0.0f);
    for (std::size_t i = 0; i < s.far.size(); ++i) {
        const float n = (i >= s.talk_from && (i - s.talk_from) < user.size())
                            ? user[i - s.talk_from]
                            : 0.0f;
        s.near_only[i] = n;
        s.mic[i] = echo[i] + n;
    }
    return s;
}

std::vector<float> RunAec(const Scene& s) {
    EchoCancellerConfig cfg;
    cfg.block_samples = 128;
    cfg.filter_tail_samples = 4096;   // 256 ms, the shipping default
    BlockFdafEchoCanceller aec(cfg);
    std::vector<float> out(s.mic.size(), 0.0f);
    // Fed in 320-sample blocks: 20 ms, the granularity the capture worker
    // actually delivers. Feeding it in one call would hide any framing bug.
    for (std::size_t i = 0; i < s.mic.size(); i += 320) {
        const std::size_t n = std::min<std::size_t>(320, s.mic.size() - i);
        aec.Process(s.mic.data() + i, s.far.data() + i, out.data() + i, n);
    }
    return out;
}

class AecBargeIn : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string wav = blackwell_test_audio::reference_wav_path();
        if (!blackwell_test_audio::file_exists(wav)) {
            GTEST_SKIP() << "reference clip missing: " << wav;
        }
        speech_ = blackwell_test_audio::load_wav_mono16k(wav);
        if (speech_.size() < kRate) GTEST_SKIP() << "reference clip too short";
    }
    std::vector<float> speech_;
};

}  // namespace

// (a) The assistant must not trigger on itself.
TEST_F(AecBargeIn, AssistantsOwnVoiceDoesNotTriggerTheVad) {
    const Scene s = BuildScene(speech_);
    const std::vector<float> out = RunAec(s);

    SileroVAD vad(BLACKWELL_VAD_MODEL_PATH);
    constexpr float kThreshold = 0.5f;   // the app's default

    // Window: after the filter has had time to converge (the assistant has been
    // speaking for a while) and before the user starts.
    const std::size_t from = kRate * 2;
    const std::size_t to = s.talk_from;
    ASSERT_LT(from, to);

    const double after = SpeechFraction(vad, out, from, to, kThreshold);
    vad.reset_state();
    const double before = SpeechFraction(vad, s.mic, from, to, kThreshold);

    // The microphone genuinely contains speech -- if it did not, this test would
    // pass by measuring nothing.
    // Printed, not just asserted: a bound that passes tells you nothing about
    // how much margin there is, and this is the number someone will want when
    // the same question is asked of a real room.
    std::printf("[aec] self-trigger: VAD hot on %.0f%% of chunks before "
                "cancellation, %.0f%% after\n",
                before * 100.0, after * 100.0);

    EXPECT_GT(before, 0.5) << "the un-cancelled microphone should be full of the "
                              "assistant's voice; the scenario is not set up";
    EXPECT_LT(after, 0.10)
        << "the VAD fires on " << (after * 100.0)
        << "% of chunks of the assistant's OWN voice after cancellation -- it "
           "would barge in on itself";
}

// (b) The user must get through, immediately.
TEST_F(AecBargeIn, UserTalkingOverTheAssistantStillTriggersTheVad) {
    const Scene s = BuildScene(speech_);
    const std::vector<float> out = RunAec(s);

    SileroVAD vad(BLACKWELL_VAD_MODEL_PATH);
    constexpr float kThreshold = 0.5f;

    // Measured from 200 ms after the interruption starts: the suppressor's gain
    // smoother and the leak estimate both need a few blocks, and Silero itself
    // needs 45-110 ms to decide. Anything inside that window is latency, not
    // failure -- but everything after it must be detection.
    const std::size_t from = s.talk_from + kRate / 5;
    const std::size_t to = std::min(s.mic.size(), s.talk_from + kRate * 3);
    ASSERT_LT(from, to);

    const double detected = SpeechFraction(vad, out, from, to, kThreshold);
    EXPECT_GT(detected, 0.5)
        << "the VAD sees the interrupting user in only " << (detected * 100.0)
        << "% of chunks -- barge-in would not fire";

    // ONSET LATENCY, MEASURED AS A COST RATHER THAN AN ABSOLUTE.
    //
    // "How long after the user starts speaking does the VAD fire" is the wrong
    // question to ask of this code, because most of that time belongs to the
    // speech: a real utterance opens with a breath and a quiet first consonant,
    // and Silero needs 45-110 ms of its own regardless. Timing against zero
    // measures the recording.
    //
    // What this component is answerable for is the DIFFERENCE: the same user
    // speech, scored with nothing playing at all, versus scored through a live
    // canceller with the assistant talking over it. That difference is what the
    // canceller costs barge-in, and it is the number that must stay small.
    vad.reset_state();
    const std::size_t onset_cancelled = FirstDetection(vad, out, s.talk_from, to, kThreshold);
    vad.reset_state();
    const std::size_t onset_clean =
        FirstDetection(vad, s.near_only, s.talk_from, to, kThreshold);

    ASSERT_LT(onset_cancelled, to) << "the user was never detected at all";
    ASSERT_LT(onset_clean, to) << "the user is not detectable even in silence; "
                                  "the scenario is not set up";
    const double cost_ms = 1000.0 *
                           (static_cast<double>(onset_cancelled) -
                            static_cast<double>(onset_clean)) /
                           static_cast<double>(kRate);
    std::printf("[aec] barge-in: VAD hot on %.0f%% of double-talk chunks; onset "
                "%.0f ms cancelled vs %.0f ms clean (cost %.0f ms)\n",
                detected * 100.0,
                1000.0 * static_cast<double>(onset_cancelled - s.talk_from) /
                    static_cast<double>(kRate),
                1000.0 * static_cast<double>(onset_clean - s.talk_from) /
                    static_cast<double>(kRate),
                cost_ms);

    EXPECT_LT(cost_ms, 150.0)
        << "cancellation delayed barge-in detection by " << cost_ms
        << " ms versus the same speech with nothing playing";
}

// The control that makes both results mean something: with the canceller
// bypassed, (a) MUST fail. If the assistant's voice does not trigger the VAD
// even without cancellation, the scenario is too quiet and the two tests above
// are measuring nothing.
TEST_F(AecBargeIn, WithoutCancellationTheAssistantWouldTriggerItself) {
    const Scene s = BuildScene(speech_);
    SileroVAD vad(BLACKWELL_VAD_MODEL_PATH);
    const double hot = SpeechFraction(vad, s.mic, kRate * 2, s.talk_from, 0.5f);
    EXPECT_GT(hot, 0.5) << "the raw microphone does not contain convincing echo; "
                           "the cancellation tests above prove nothing";
}
