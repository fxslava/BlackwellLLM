// -----------------------------------------------------------------------------
// silero_vad_test.cpp — unit tests for blackwell::vad::SileroVAD.
//
// Proves the three things that can silently go wrong with an ONNXRuntime
// integration: that it LINKS and loads the pinned model at all, that the
// RECURRENT STATE is actually threaded through Run() (a wrapper that forgets to
// feed `stateN` back into `state` still returns plausible probabilities — it is
// just wrong), and that the streaming adapter's 160->512 sample regrouping is
// equivalent to feeding whole chunks.
//
// CPU-only: no CUDA, no engine, no GPU. Runs under `ctest -L validation`.
//
// The model path is baked in at configure time (BLACKWELL_VAD_MODEL_PATH) so the
// suite is independent of the working directory.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <gtest/gtest.h>

#include "audio_test_wav.hpp"  // reference_wav_path / load_wav_mono16k (real speech clip)
#include "silero_vad.hpp"

namespace {

using blackwell::vad::kChunkSamples;
using blackwell::vad::kInvalidProbability;
using blackwell::vad::kSampleRate;
using blackwell::vad::SileroVAD;

const char* model_path() { return BLACKWELL_VAD_MODEL_PATH; }

// Digital silence. Silero scores true zeros very low, which is the floor every
// other signal in this file is compared against.
std::vector<float> silence(size_t n) { return std::vector<float>(n, 0.0f); }

// Low-level white noise — the "room tone" case. A pure energy threshold would
// trip on this at a high enough gain; a neural VAD should not.
std::vector<float> noise(size_t n, float amplitude) {
    std::vector<float> out(n);
    uint32_t rng = 12345u;  // deterministic: the assertions must be reproducible
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        const float u = static_cast<float>(rng >> 8) / static_cast<float>(1u << 24);
        out[i] = (2.0f * u - 1.0f) * amplitude;
    }
    return out;
}

// A LOUD synthetic tone complex: a 120 Hz harmonic stack shaped by three
// formant-ish resonances and modulated at a 4 Hz syllable rate.
//
// It is NOT heard as speech by Silero (measured ~0.0005), and that is precisely
// why it is useful: it is loud enough in plain RMS terms to sail past the
// pipeline's legacy -40 dBFS onset threshold, so it stands in for exactly the
// false-trigger class the neural VAD is being introduced to reject. It also
// serves as a deterministic, always-available signal for the mechanical tests
// (state advance, feed regrouping) that do not care what the audio means.
std::vector<float> synthetic_tone(size_t n) {
    constexpr float kF0 = 120.0f;
    constexpr float kFormants[] = {700.0f, 1220.0f, 2600.0f};
    constexpr float kGains[]    = {1.0f, 0.5f, 0.25f};
    const float two_pi = 2.0f * std::numbers::pi_v<float>;

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSampleRate);
        float s = 0.0f;
        // Harmonic stack up to ~4 kHz with a -6 dB/octave source spectrum.
        for (int h = 1; h * kF0 < 4000.0f; ++h) {
            const float f = kF0 * static_cast<float>(h);
            float shaped = 0.0f;
            for (int k = 0; k < 3; ++k) {
                const float bw = 120.0f;
                shaped += kGains[k] / (1.0f + std::pow((f - kFormants[k]) / bw, 2.0f));
            }
            s += shaped * std::sin(two_pi * f * t) / static_cast<float>(h);
        }
        // Syllable-rate envelope (never fully closes, so the whole span is voiced).
        const float env = 0.55f + 0.45f * std::sin(two_pi * 4.0f * t);
        out[i] = 0.28f * env * s;
    }
    return out;
}

// Mean probability over a signal fed in whole chunks, from a fresh state.
float mean_probability(SileroVAD& vad, const std::vector<float>& pcm) {
    vad.reset_state();
    double sum = 0.0;
    int n = 0;
    for (size_t off = 0; off + kChunkSamples <= pcm.size(); off += kChunkSamples) {
        sum += vad.process_chunk(pcm.data() + off, kChunkSamples);
        ++n;
    }
    return n == 0 ? 0.0f : static_cast<float>(sum / n);
}

size_t working_set_bytes() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return pmc.WorkingSetSize;
}

// -----------------------------------------------------------------------------

TEST(SileroVad, LoadsPinnedModel) {
    EXPECT_NO_THROW({ SileroVAD vad(model_path()); });
}

// INIT tier: a bad path must throw, not produce a half-built object that
// silently returns zeros forever.
TEST(SileroVad, MissingModelThrows) {
    EXPECT_THROW(SileroVAD("D:/definitely/not/here/silero_vad.onnx"), std::runtime_error);
}

// The model accepts exactly 512 samples; anything else must be refused rather
// than padded behind the caller's back.
TEST(SileroVad, RejectsMalformedChunks) {
    SileroVAD vad(model_path());
    const std::vector<float> pcm = synthetic_tone(kChunkSamples * 2);

    EXPECT_EQ(vad.process_chunk(pcm.data(), 256), kInvalidProbability);
    EXPECT_EQ(vad.process_chunk(pcm.data(), 1024), kInvalidProbability);
    EXPECT_EQ(vad.process_chunk(nullptr, kChunkSamples), kInvalidProbability);
    EXPECT_EQ(vad.inference_errors(), 0u);  // rejected at the door, never ran

    // A well-formed chunk still works afterwards (no state was disturbed).
    const float p = vad.process_chunk(pcm.data(), kChunkSamples);
    EXPECT_GE(p, 0.0f);
    EXPECT_LE(p, 1.0f);
}

// The floor: nothing that is not speech may read as speech.
TEST(SileroVad, NonSpeechScoresLow) {
    SileroVAD vad(model_path());
    const size_t n = kChunkSamples * 30;  // ~1 s

    const float p_silence = mean_probability(vad, silence(n));
    const float p_noise   = mean_probability(vad, noise(n, 0.05f));

    std::printf("[silero] mean p: silence=%.4f noise=%.4f\n", p_silence, p_noise);
    EXPECT_LT(p_silence, 0.2f) << "digital silence must sit near the floor";
    EXPECT_LT(p_noise, 0.5f) << "low-level noise must not read as speech";
}

// THE reason this whole subsystem exists, expressed as an assertion.
//
// The synthetic tone is LOUD: its RMS clears the pipeline's legacy -40 dBFS
// onset threshold with ~30 dB to spare, so the threshold VAD it replaces would
// fire an utterance onset on it, prefill the encoder, and eventually decode a
// hallucinated transcript of a sound nobody spoke. Silero scores it near zero.
// That gap IS the upgrade.
TEST(SileroVad, LoudNonSpeechWouldFoolTheThresholdVadButNotSilero) {
    SileroVAD vad(model_path());
    const std::vector<float> tone = synthetic_tone(kChunkSamples * 30);

    double sumsq = 0.0;
    for (float s : tone) sumsq += static_cast<double>(s) * s;
    const float tone_db =
        static_cast<float>(20.0 * std::log10(std::sqrt(sumsq / tone.size()) + 1e-6));
    const float p_tone = mean_probability(vad, tone);

    std::printf("[silero] loud tone: %.1f dBFS, mean p=%.4f\n", tone_db, p_tone);

    // Precondition: the signal really would have tripped the old detector.
    // -40 dBFS is SpeechPipelineConfig::vad_threshold_db in audio_translator.
    ASSERT_GT(tone_db, -40.0f) << "test signal is too quiet to make the point";
    EXPECT_LT(p_tone, 0.5f) << "the neural VAD must not fire on loud non-speech";
}

// ---- real-speech fixture ----------------------------------------------------
// The clip is the LibriSpeech utterance the Ultravox golden dumps were generated
// from ("MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES / AND WE ARE GLAD
// TO WELCOME HIS GOSPEL"), 5.855 s at 16 kHz. Its measured envelope:
//     0.00-0.50 s  room tone before the first word
//     0.54-5.44 s  continuous speech
//     5.47-5.86 s  decay back to room tone
//
// SKIPS when the clip is absent: it lives under tests/integration/golden_dumps
// (Git LFS), and vad_tests runs in the `validation` label, which must stay green
// on a checkout that never pulled LFS.
class SileroRealSpeech : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string wav = blackwell_test_audio::reference_wav_path();
        pcm = blackwell_test_audio::load_wav_mono16k(wav);
        if (pcm.empty()) {
            GTEST_SKIP() << "reference clip unavailable (" << wav
                         << ") — pull Git LFS or set BLACKWELL_ULTRAVOX_DUMPS";
        }
        ASSERT_GE(pcm.size(), blackwell_test_audio::kClipSamples);
    }

    // Per-chunk probabilities over the whole clip from a cold start; index i
    // covers samples [i*512, (i+1)*512).
    std::vector<float> trace() {
        SileroVAD vad(model_path());
        std::vector<float> out;
        for (size_t off = 0; off + kChunkSamples <= pcm.size(); off += kChunkSamples) {
            out.push_back(vad.process_chunk(pcm.data() + off, kChunkSamples));
        }
        return out;
    }

    // Mean over the chunks whose span falls inside [from_s, to_s).
    static float mean_between(const std::vector<float>& t, double from_s, double to_s) {
        const double chunk_s = static_cast<double>(kChunkSamples) / kSampleRate;
        double sum = 0.0;
        int n = 0;
        for (size_t i = 0; i < t.size(); ++i) {
            const double begin = static_cast<double>(i) * chunk_s;
            if (begin >= from_s && begin + chunk_s <= to_s) {
                sum += t[i];
                ++n;
            }
        }
        return n == 0 ? -1.0f : static_cast<float>(sum / n);
    }

    std::vector<float> pcm;
};

// The headline claim, on real audio: speech reads as speech and the room tone on
// either side of it does not. This is the assertion that would have caught the
// missing 64-sample context prepend — without it, this clip scored 0.0009.
TEST_F(SileroRealSpeech, SpeechReadsHighAndRoomToneReadsLow) {
    const std::vector<float> t = trace();
    ASSERT_FALSE(t.empty());

    const float lead_in = mean_between(t, 0.0, 0.45);   // before the first word
    const float speech  = mean_between(t, 1.0, 5.0);    // mid-utterance
    const float tail    = mean_between(t, 5.65, 5.85);  // after the last word

    std::printf("[silero] real clip: lead-in=%.4f speech=%.4f tail=%.4f\n", lead_in, speech, tail);

    EXPECT_LT(lead_in, 0.2f) << "room tone before the utterance must not read as speech";
    EXPECT_GT(speech, 0.9f) << "fluent speech must read as speech";
    EXPECT_LT(tail, 0.2f) << "room tone after the utterance must not read as speech";
}

// Silero HOLDS THROUGH short inter-word pauses rather than dropping to silence,
// and the pipeline depends on that: a detector that fired a boundary on every
// gap between words would commit half-sentences to the decoder.
//
// audio_test_wav.hpp documents this clip's one true interior silence valley --
// 3.24-3.32 s, the 80 ms pause between "...MIDDLE CLASSES" and "AND WE ARE...",
// measured at -62 dB against ~-20 dB of speech. A pure energy detector sees a
// dead stop there. Silero stays pinned high, which is why utterance boundaries
// remain the silence-hangover's job (800 ms by default), not the VAD's.
TEST_F(SileroRealSpeech, HoldsThroughTheKnownInterWordPause) {
    const std::vector<float> t = trace();
    const double chunk_s = static_cast<double>(kChunkSamples) / kSampleRate;

    float pause_min = 1.0f;
    for (size_t i = 0; i < t.size(); ++i) {
        const double begin = static_cast<double>(i) * chunk_s;
        const double end   = begin + chunk_s;
        if (end > 3.24 && begin < 3.32) pause_min = std::min(pause_min, t[i]);
    }

    std::printf("[silero] min p across the documented 3.24-3.32 s pause = %.4f\n", pause_min);
    EXPECT_GT(pause_min, 0.5f)
        << "an 80 ms inter-word pause must not read as an utterance boundary";
}

// THE test for the RNN state. Feeding the SAME chunk repeatedly must not give
// the same answer every time -- if it does, the state is not being carried and
// the wrapper is silently running a memoryless classifier.
TEST(SileroVad, RecurrentStateAdvancesAcrossChunks) {
    SileroVAD vad(model_path());
    const std::vector<float> chunk = synthetic_tone(kChunkSamples);

    std::vector<float> run;
    for (int i = 0; i < 8; ++i) run.push_back(vad.process_chunk(chunk.data(), kChunkSamples));

    const bool all_identical =
        std::all_of(run.begin(), run.end(), [&](float p) { return p == run.front(); });
    EXPECT_FALSE(all_identical)
        << "identical output for an identical chunk means `stateN` is not being fed back";
}

// reset_state() must return the model to its cold start exactly: the same input
// sequence after a reset has to reproduce the original trajectory.
TEST(SileroVad, ResetStateRestoresColdStart) {
    SileroVAD vad(model_path());
    const std::vector<float> chunk = synthetic_tone(kChunkSamples);

    std::vector<float> first;
    for (int i = 0; i < 6; ++i) first.push_back(vad.process_chunk(chunk.data(), kChunkSamples));

    vad.reset_state();
    EXPECT_FLOAT_EQ(vad.last_probability(), 0.0f);

    std::vector<float> second;
    for (int i = 0; i < 6; ++i) second.push_back(vad.process_chunk(chunk.data(), kChunkSamples));

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_FLOAT_EQ(first[i], second[i]) << "trajectory diverged at step " << i;
    }
}

// Without a reset, continuing the same input must NOT reproduce the cold-start
// trajectory -- the guard that ResetStateRestoresColdStart above is testing a
// real reset rather than a model that happens to be stateless in practice.
TEST(SileroVad, WithoutResetTrajectoryDiffers) {
    SileroVAD vad(model_path());
    const std::vector<float> chunk = synthetic_tone(kChunkSamples);

    std::vector<float> first;
    for (int i = 0; i < 6; ++i) first.push_back(vad.process_chunk(chunk.data(), kChunkSamples));

    std::vector<float> continued;
    for (int i = 0; i < 6; ++i) continued.push_back(vad.process_chunk(chunk.data(), kChunkSamples));

    EXPECT_NE(first, continued);
}

// The streaming adapter must be a pure regrouping: the pipeline hands it 160
// samples at a time, and the resulting state/probability has to match what whole
// 512-sample chunks would have produced.
TEST(SileroVad, FeedIn160BlocksMatchesWholeChunks) {
    SileroVAD reference(model_path());
    SileroVAD streamed(model_path());
    const std::vector<float> pcm = synthetic_tone(kChunkSamples * 10);

    float last_ref = 0.0f;
    for (size_t off = 0; off + kChunkSamples <= pcm.size(); off += kChunkSamples) {
        last_ref = reference.process_chunk(pcm.data() + off, kChunkSamples);
    }

    constexpr size_t kPipelineBlock = 160;  // the controller's 10 ms VAD block
    float last_streamed = 0.0f;
    for (size_t off = 0; off < pcm.size(); off += kPipelineBlock) {
        const size_t take = std::min(kPipelineBlock, pcm.size() - off);
        last_streamed = streamed.feed(pcm.data() + off, take);
    }

    EXPECT_FLOAT_EQ(last_ref, last_streamed);
}

// "Sticky" contract: between inferences feed() reports the previous probability
// rather than a stale zero or a fresh guess.
TEST(SileroVad, FeedIsStickyBetweenInferences) {
    SileroVAD vad(model_path());
    const std::vector<float> pcm = synthetic_tone(kChunkSamples * 4);

    // Complete one chunk, then advance by less than a full chunk.
    const float after_first = vad.feed(pcm.data(), kChunkSamples);
    EXPECT_GE(after_first, 0.0f);

    const float partial = vad.feed(pcm.data() + kChunkSamples, 160);
    EXPECT_FLOAT_EQ(partial, after_first) << "a partial chunk must not change the reported value";
    EXPECT_FLOAT_EQ(vad.last_probability(), after_first);
}

TEST(SileroVad, ThresholdClampsToUnitRange) {
    SileroVAD vad(model_path());
    EXPECT_FLOAT_EQ(vad.threshold(), SileroVAD::kDefaultThreshold);

    vad.set_threshold(0.75f);
    EXPECT_FLOAT_EQ(vad.threshold(), 0.75f);
    vad.set_threshold(3.0f);
    EXPECT_FLOAT_EQ(vad.threshold(), 1.0f);
    vad.set_threshold(-2.0f);
    EXPECT_FLOAT_EQ(vad.threshold(), 0.0f);
}

// Sustained-run smoke test: a long stream must not fault, must not accumulate
// ORT errors, and must not grow the process without bound.
//
// SCOPE: this is a smoke test, not a rigorous leak detector. ONNXRuntime lives
// behind a DLL with its OWN CRT heap, so our /MT debug-heap tooling cannot see
// its allocations at all -- the process working set is the only signal available
// from this side of the boundary, and it is noisy. The bound is therefore
// generous: it catches a per-inference leak (which at ~9400 chunks would run to
// tens of MB), not a small one-time allocation.
TEST(SileroVad, SustainedStreamIsStable) {
    SileroVAD vad(model_path());
    const std::vector<float> chunk = synthetic_tone(kChunkSamples);

    for (int i = 0; i < 200; ++i) (void)vad.process_chunk(chunk.data(), kChunkSamples);  // warm up
    const size_t before = working_set_bytes();

    constexpr int kIterations = 9375;  // ~5 minutes of 32 ms audio
    for (int i = 0; i < kIterations; ++i) {
        const float p = vad.process_chunk(chunk.data(), kChunkSamples);
        ASSERT_GE(p, 0.0f) << "inference returned an invalid probability at step " << i;
        ASSERT_LE(p, 1.0f);
    }

    const size_t after = working_set_bytes();
    EXPECT_EQ(vad.inference_errors(), 0u);
    if (before != 0 && after > before) {
        const double grew_mb = static_cast<double>(after - before) / (1024.0 * 1024.0);
        std::printf("[silero] working set grew %.2f MB over %d inferences\n", grew_mb,
                    kIterations);
        EXPECT_LT(grew_mb, 8.0) << "working set grew without bound across a sustained stream";
    }
}

}  // namespace
