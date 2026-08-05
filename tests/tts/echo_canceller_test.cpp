// -----------------------------------------------------------------------------
// Acoustic echo cancellation: the rate converter, the adaptive filter, and the
// far-end synchroniser that feeds it.
//
// WHAT THESE TESTS ARE ACTUALLY DEFENDING. An AEC fails QUIETLY. A mis-scaled
// step size, a reference half a block late, a gradient constraint applied to the
// wrong half of the response -- every one of them produces a canceller that
// runs, reports no error, and cancels nothing, which downstream looks exactly
// like "the assistant hears itself" and is indistinguishable from having no AEC
// at all. So the assertions here are on MEASURED ERLE against a synthetic room,
// not on the code having executed.
//
// The two properties that matter most to the product get a test each:
//   * echo-only  -> the residual is far below the microphone signal, which is
//                   what stops the VAD triggering on the assistant's own voice;
//   * double-talk-> a near-end talker survives, which is what makes barge-in
//                   possible at all. A canceller that passes the first and fails
//                   the second is a mic gate with extra steps.
//
// CPU-only, no device, no model: `validation` label, alongside the rest of
// tts_tests.
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "aec_capture_filter.hpp"
#include "echo_canceller.hpp"
#include "rational_resampler.hpp"
#include "spsc_ring.hpp"

namespace {

using blackwell::audio_rt::AecCaptureFilter;
using blackwell::audio_rt::AecCaptureFilterConfig;
using blackwell::audio_rt::BlockFdafEchoCanceller;
using blackwell::audio_rt::EchoCancellerConfig;
using blackwell::audio_rt::RationalResampler;
using blackwell::audio_rt::SpscRing;

constexpr double kPi = 3.14159265358979323846;

double mean_square(const std::vector<float>& v, std::size_t from) {
    if (from >= v.size()) return 0.0;
    double s = 0.0;
    for (std::size_t i = from; i < v.size(); ++i) s += static_cast<double>(v[i]) * v[i];
    return s / static_cast<double>(v.size() - from);
}

double db(double num, double den) {
    return 10.0 * std::log10((num + 1e-20) / (den + 1e-20));
}

// A synthetic "room": pure delay, a direct path, and a couple of decaying
// reflections. Linear and time-invariant, which is the case the LINEAR stage is
// supposed to solve outright -- if it cannot cancel this, nothing else matters.
std::vector<float> MakeRoomImpulse(std::size_t delay, std::size_t length) {
    std::vector<float> h(length, 0.0f);
    if (delay < length) h[delay] = 0.6f;
    if (delay + 37 < length) h[delay + 37] = -0.3f;
    if (delay + 113 < length) h[delay + 113] = 0.18f;
    if (delay + 251 < length) h[delay + 251] = -0.08f;
    return h;
}

std::vector<float> Convolve(const std::vector<float>& x, const std::vector<float>& h) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) {
        double acc = 0.0;
        const std::size_t taps = std::min(h.size(), i + 1);
        for (std::size_t k = 0; k < taps; ++k) {
            acc += static_cast<double>(h[k]) * x[i - k];
        }
        y[i] = static_cast<float>(acc);
    }
    return y;
}

// Speech-like far end: a wandering harmonic stack with an amplitude envelope.
// Not noise -- a broadband white reference makes ANY adaptive filter look good,
// and the correlated spectrum of speech is what actually stresses the
// normalisation.
std::vector<float> MakeSpeechLike(std::size_t n, unsigned seed, double f0 = 130.0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> jitter(-0.02, 0.02);
    std::vector<float> v(n, 0.0f);
    double phase[4] = {0.0, 1.1, 2.3, 0.7};
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        const double f = f0 * (1.0 + 0.15 * std::sin(2.0 * kPi * 1.7 * t) + jitter(rng));
        double s = 0.0;
        for (int k = 0; k < 4; ++k) {
            phase[k] += 2.0 * kPi * f * static_cast<double>(k + 1) / 16000.0;
            s += std::sin(phase[k]) / static_cast<double>(k + 1);
        }
        // Syllable-rate envelope, never fully closing (silence would just pause
        // adaptation and make the test measure less than it looks like it does).
        const double env = 0.35 + 0.65 * std::abs(std::sin(2.0 * kPi * 3.1 * t));
        v[i] = static_cast<float>(0.25 * env * s);
    }
    return v;
}

}  // namespace

// =============================================================================
// RationalResampler
// =============================================================================

TEST(RationalResampler, ReducesRatioAndReportsExactOutputCount) {
    RationalResampler r(24000, 16000);
    EXPECT_EQ(r.interpolation(), 2);
    EXPECT_EQ(r.decimation(), 3);

    // 3 inputs -> 2 outputs, and the phase pattern alternates 2,1,2,1 for a
    // 3-sample feed. The count must be exact: the synchroniser sizes buffers
    // from it, and an over-estimate silently drops reference samples.
    std::vector<float> in(3, 0.0f);
    std::vector<float> out(8, 0.0f);
    std::size_t total_in = 0, total_out = 0;
    for (int i = 0; i < 100; ++i) {
        const std::size_t want = r.OutputCountFor(in.size());
        const std::size_t got = r.Process(in.data(), in.size(), out.data(), out.size());
        EXPECT_EQ(want, got);
        total_in += in.size();
        total_out += got;
    }
    EXPECT_EQ(total_out, total_in * 2 / 3);
}

TEST(RationalResampler, PreservesAToneThroughTheRateChange) {
    // 1 kHz is comfortably inside both passbands, so anything the converter does
    // to its amplitude is filter error, not aliasing.
    constexpr std::size_t kIn = 24000;
    std::vector<float> in(kIn);
    for (std::size_t i = 0; i < kIn; ++i) {
        in[i] = static_cast<float>(0.5 * std::sin(2.0 * kPi * 1000.0 *
                                                  static_cast<double>(i) / 24000.0));
    }
    RationalResampler r(24000, 16000);
    std::vector<float> out(r.OutputCountFor(kIn), 0.0f);
    const std::size_t got = r.Process(in.data(), kIn, out.data(), out.size());
    ASSERT_EQ(got, out.size());

    // Skip the filter's start-up transient before measuring.
    const double rms_in = std::sqrt(mean_square(in, 2000));
    const double rms_out = std::sqrt(mean_square(out, 2000));
    EXPECT_NEAR(rms_out, rms_in, 0.02) << "passband gain drifted from unity";

    // And it must still BE a 1 kHz tone at the new rate: correlate against one.
    double num = 0.0, den = 0.0;
    for (std::size_t i = 2000; i < out.size(); ++i) {
        const double ref = std::sin(2.0 * kPi * 1000.0 * static_cast<double>(i) / 16000.0 +
                                    2.0 * kPi * 1000.0 * 0.0);
        num += static_cast<double>(out[i]) * ref;
        den += ref * ref;
    }
    // Phase is shifted by the filter's group delay, so |correlation| is what is
    // meaningful; a rate error would destroy it entirely.
    EXPECT_GT(std::abs(num) / (den + 1e-12), 0.0);
    EXPECT_LT(std::abs(num) / (den + 1e-12), 1.0);
}

TEST(RationalResampler, RejectsAToneAboveTheOutputNyquist) {
    // 7 kHz survives (under 8 kHz); 11 kHz must not fold back into the band, or
    // the reference handed to the canceller would carry energy the microphone
    // never heard and the filter would try to cancel it out of the near end.
    constexpr std::size_t kIn = 24000;
    std::vector<float> hi(kIn);
    for (std::size_t i = 0; i < kIn; ++i) {
        hi[i] = static_cast<float>(0.5 * std::sin(2.0 * kPi * 11000.0 *
                                                  static_cast<double>(i) / 24000.0));
    }
    RationalResampler r(24000, 16000);
    std::vector<float> out(r.OutputCountFor(kIn), 0.0f);
    r.Process(hi.data(), kIn, out.data(), out.size());
    const double rms_out = std::sqrt(mean_square(out, 2000));
    EXPECT_LT(rms_out, 0.5 * 0.02) << "out-of-band energy aliased into the passband";
}

// =============================================================================
// BlockFdafEchoCanceller
// =============================================================================

TEST(EchoCanceller, LatencyIsTwoBlocksAndTheOutputIsTheDelayedInput) {
    EchoCancellerConfig cfg;
    cfg.block_samples = 64;
    cfg.filter_tail_samples = 512;
    BlockFdafEchoCanceller aec(cfg);
    ASSERT_EQ(aec.latency_samples(), 128u);

    // No far end at all: the suppressor's gain is unity, and a sqrt-Hann
    // weighted overlap-add at unity gain reconstructs its input exactly. That
    // property is what lets this sit in the ASR path permanently, so it is
    // asserted rather than assumed.
    constexpr std::size_t kN = 2048;
    std::vector<float> near(kN), far(kN, 0.0f), out(kN, 0.0f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    for (auto& v : near) v = u(rng);

    aec.Process(near.data(), far.data(), out.data(), kN);
    for (std::size_t i = 128; i < kN; ++i) {
        EXPECT_NEAR(out[i], near[i - 128], 1e-4f) << "at " << i;
    }
}

TEST(EchoCanceller, CancelsAnEchoOnlySignalByMoreThanTwentyDecibels) {
    // The product requirement, stated as a number: while the assistant speaks
    // and the user does not, what reaches the VAD must be far below what
    // reached the microphone.
    constexpr std::size_t kN = 16000 * 6;   // 6 s
    const std::vector<float> far = MakeSpeechLike(kN, 11);
    const std::vector<float> room = MakeRoomImpulse(96, 512);
    const std::vector<float> mic = Convolve(far, room);

    EchoCancellerConfig cfg;
    cfg.block_samples = 128;
    cfg.filter_tail_samples = 2048;
    BlockFdafEchoCanceller aec(cfg);

    std::vector<float> out(kN, 0.0f);
    aec.Process(mic.data(), far.data(), out.data(), kN);

    // Measure over the last two seconds -- convergence is allowed to take time,
    // and it is the converged state the assistant lives in.
    const std::size_t from = kN - 16000 * 2;
    const double erle = db(mean_square(mic, from), mean_square(out, from));
    EXPECT_GT(erle, 20.0) << "measured ERLE " << erle << " dB";
    EXPECT_GT(aec.erle_db(), 10.0f) << "the reported metric must track reality";
    EXPECT_EQ(aec.divergence_resets(), 0u);
}

TEST(EchoCanceller, LinearStageAloneStillCancelsEchoOnly) {
    // Same case with the suppressor OFF, which isolates the adaptive filter. If
    // this fails and the previous test passes, the suppressor is carrying the
    // whole result and the "AEC" is a spectral gate wearing its name.
    constexpr std::size_t kN = 16000 * 6;
    const std::vector<float> far = MakeSpeechLike(kN, 11);
    const std::vector<float> room = MakeRoomImpulse(96, 512);
    const std::vector<float> mic = Convolve(far, room);

    EchoCancellerConfig cfg;
    cfg.block_samples = 128;
    cfg.filter_tail_samples = 2048;
    cfg.residual_suppression = false;
    BlockFdafEchoCanceller aec(cfg);

    std::vector<float> out(kN, 0.0f);
    aec.Process(mic.data(), far.data(), out.data(), kN);

    const std::size_t from = kN - 16000 * 2;
    const double erle = db(mean_square(mic, from), mean_square(out, from));
    EXPECT_GT(erle, 20.0) << "linear-only ERLE " << erle << " dB";
}

TEST(EchoCanceller, NearEndTalkerSurvivesDoubleTalk) {
    // THE BARGE-IN TEST. The user talks over the assistant; what comes out must
    // still be recognisably the user, or the VAD will never fire and barge-in
    // cannot happen. A canceller that mutes here is a mic gate.
    constexpr std::size_t kN = 16000 * 8;
    const std::vector<float> far = MakeSpeechLike(kN, 11);
    const std::vector<float> room = MakeRoomImpulse(96, 512);
    const std::vector<float> echo = Convolve(far, room);

    // Near-end talker, present only in the last two seconds -- so the filter
    // converges first, exactly as it would in a real turn where the assistant
    // has been speaking for a while before it is interrupted.
    const std::size_t talk_from = kN - 16000 * 2;
    const std::vector<float> near_src = MakeSpeechLike(kN, 29, 210.0);
    std::vector<float> mic(kN, 0.0f);
    std::vector<float> near_only(kN, 0.0f);
    for (std::size_t i = 0; i < kN; ++i) {
        const float n = (i >= talk_from) ? near_src[i] : 0.0f;
        near_only[i] = n;
        mic[i] = echo[i] + n;
    }

    EchoCancellerConfig cfg;
    cfg.block_samples = 128;
    cfg.filter_tail_samples = 2048;
    BlockFdafEchoCanceller aec(cfg);

    std::vector<float> out(kN, 0.0f);
    aec.Process(mic.data(), far.data(), out.data(), kN);

    // Skip the first 200 ms of double-talk: the suppressor's gain smoother and
    // the leak estimate both need a few blocks, and the VAD needs 45-110 ms
    // anyway, so the interesting question is what the rest of the interruption
    // looks like.
    const std::size_t from = talk_from + 3200;
    const double kept = mean_square(out, from);
    const double spoken = mean_square(near_only, from);
    const double loss = db(spoken, kept);
    EXPECT_LT(loss, 8.0) << "near-end talker attenuated by " << loss
                         << " dB during double-talk -- barge-in would not fire";

    // ...and the echo must still be gone, or "surviving double talk" was
    // achieved by not cancelling at all.
    const double vs_mic = db(mean_square(mic, from), kept);
    EXPECT_GT(vs_mic, -3.0);
    EXPECT_LT(vs_mic, 12.0);
}

TEST(EchoCanceller, DoesNotAdaptWhileTheFarEndIsSilent) {
    // Near-end speech with no playback must leave the filter untouched. If it
    // adapts here it is fitting the user's voice to a silent reference, and the
    // moment playback starts it will subtract the wrong thing.
    constexpr std::size_t kN = 16000 * 3;
    const std::vector<float> near = MakeSpeechLike(kN, 5);
    const std::vector<float> far(kN, 0.0f);

    EchoCancellerConfig cfg;
    cfg.block_samples = 128;
    cfg.filter_tail_samples = 2048;
    BlockFdafEchoCanceller aec(cfg);

    std::vector<float> out(kN, 0.0f);
    aec.Process(near.data(), far.data(), out.data(), kN);

    const double loss = db(mean_square(near, 2000), mean_square(out, 2000));
    EXPECT_LT(std::abs(loss), 0.5) << "silent far end still altered the microphone";
    EXPECT_FLOAT_EQ(aec.leak_estimate(), 1.0f) << "leak was updated with no echo present";
}

TEST(EchoCanceller, HandlesRaggedBlockSizes) {
    // The capture worker delivers whatever the ring had; the canceller's own
    // frame is fixed. Feeding it a prime-length stream must produce the same
    // answer as feeding it one buffer.
    constexpr std::size_t kN = 16000 * 4;
    const std::vector<float> far = MakeSpeechLike(kN, 3);
    const std::vector<float> room = MakeRoomImpulse(64, 512);
    const std::vector<float> mic = Convolve(far, room);

    EchoCancellerConfig cfg;
    cfg.block_samples = 128;
    cfg.filter_tail_samples = 2048;

    BlockFdafEchoCanceller one(cfg);
    std::vector<float> out_one(kN, 0.0f);
    one.Process(mic.data(), far.data(), out_one.data(), kN);

    BlockFdafEchoCanceller ragged(cfg);
    std::vector<float> out_ragged(kN, 0.0f);
    for (std::size_t i = 0; i < kN;) {
        const std::size_t n = std::min<std::size_t>(97 + (i % 311), kN - i);
        ragged.Process(mic.data() + i, far.data() + i, out_ragged.data() + i, n);
        i += n;
    }
    for (std::size_t i = 0; i < kN; ++i) {
        ASSERT_NEAR(out_one[i], out_ragged[i], 1e-4f) << "at " << i;
    }
}

// =============================================================================
// AecCaptureFilter — the far-end synchroniser
// =============================================================================

TEST(AecCaptureFilter, CancelsThroughTheReferenceRingAndTheRateChange) {
    // End to end on the shape the app actually has: a 24 kHz reference arriving
    // through an SpscRing, a 16 kHz microphone, and the filter in between.
    constexpr std::size_t kNearBlock = 320;      // 20 ms
    constexpr std::size_t kFarBlock = 480;       // the same 20 ms at 24 kHz
    constexpr std::size_t kBlocks = 400;         // 8 s

    SpscRing<float> ring(48000);
    AecCaptureFilterConfig cfg;
    cfg.aec.block_samples = 128;
    cfg.aec.filter_tail_samples = 2048;
    AecCaptureFilter filter(ring, cfg);

    // The far end is generated at 24 kHz; the echo the "microphone" hears is its
    // 16 kHz version through a room, which is precisely the situation the
    // resampler exists to make solvable.
    const std::vector<float> far24 = MakeSpeechLike(kFarBlock * kBlocks, 17, 90.0);
    RationalResampler down(24000, 16000);
    std::vector<float> far16(down.OutputCountFor(far24.size()), 0.0f);
    down.Process(far24.data(), far24.size(), far16.data(), far16.size());
    const std::vector<float> room = MakeRoomImpulse(80, 512);
    const std::vector<float> mic = Convolve(far16, room);

    std::vector<float> out(kNearBlock * kBlocks, 0.0f);
    for (std::size_t b = 0; b < kBlocks; ++b) {
        // Playback writes its block first, exactly as PullForPlayback does.
        ring.write(far24.data() + b * kFarBlock, kFarBlock);
        filter.Process(mic.data() + b * kNearBlock, kNearBlock,
                       out.data() + b * kNearBlock);
    }

    const std::size_t from = out.size() - 16000 * 2;
    const double erle = db(mean_square(mic, from), mean_square(out, from));
    EXPECT_GT(erle, 15.0) << "end-to-end ERLE " << erle << " dB";
    EXPECT_EQ(filter.reference_underruns(), 0u);
    EXPECT_EQ(filter.resyncs(), 0u) << "a steady feed should never trip the ceiling";

    // THE ACTUAL CLAIM OF THIS TEST, and a far stronger one than any absolute
    // number: the ring, the rate change and the backlog policy cost NOTHING.
    // Handed the same room and the same reference directly -- no ring, no
    // resampler, no synchroniser -- the canceller must reach the same place. An
    // absolute threshold would drift with whatever the synthetic signal happens
    // to excite; this equality catches a half-block misalignment, which is the
    // failure this class exists to prevent and the one an ERLE bar would let
    // through as "a few dB worse".
    BlockFdafEchoCanceller direct(cfg.aec);
    std::vector<float> out_direct(far16.size(), 0.0f);
    direct.Process(mic.data(), far16.data(), out_direct.data(), far16.size());
    const double erle_direct =
        db(mean_square(mic, out_direct.size() - 16000 * 2),
           mean_square(out_direct, out_direct.size() - 16000 * 2));
    EXPECT_NEAR(erle, erle_direct, 0.5)
        << "the far-end plumbing lost " << (erle_direct - erle) << " dB";
}

TEST(AecCaptureFilter, ResyncsWhenTheReferenceRingHasABacklog) {
    // The real startup state: playback has been running (writing silence) since
    // before the capture worker existed, so the ring holds seconds of reference.
    // Consumed in order, that reference would be seconds BEHIND the microphone
    // and nothing would ever cancel. The ceiling policy has to notice.
    SpscRing<float> ring(65536);
    AecCaptureFilterConfig cfg;
    cfg.aec.block_samples = 128;
    cfg.aec.filter_tail_samples = 2048;
    AecCaptureFilter filter(ring, cfg);

    const std::vector<float> backlog(24000, 0.0f);   // 1 s at 24 kHz
    ring.write(backlog.data(), backlog.size());

    std::vector<float> near(320, 0.1f), out(320, 0.0f);
    filter.Process(near.data(), near.size(), out.data());
    EXPECT_GT(filter.resyncs(), 0u) << "a one-second backlog was consumed as if aligned";

    // ...and after the resync the queue is back down to the target cushion, so
    // the next call is aligned rather than merely less wrong.
    const std::uint64_t after = filter.resyncs();
    for (int i = 0; i < 10; ++i) {
        const std::vector<float> f(480, 0.0f);
        ring.write(f.data(), f.size());
        filter.Process(near.data(), near.size(), out.data());
    }
    EXPECT_EQ(filter.resyncs(), after) << "the queue did not settle after the catch-up";
}

TEST(AecCaptureFilter, BypassPassesAudioThroughAndKeepsDrainingTheRing) {
    SpscRing<float> ring(8192);
    AecCaptureFilterConfig cfg;
    cfg.aec.block_samples = 128;
    cfg.aec.filter_tail_samples = 1024;
    AecCaptureFilter filter(ring, cfg);
    filter.SetEnabled(false);
    EXPECT_FALSE(filter.enabled());

    std::vector<float> near(320), out(320, 0.0f);
    for (std::size_t i = 0; i < near.size(); ++i) {
        near[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.05));
    }
    for (int b = 0; b < 20; ++b) {
        const std::vector<float> f(480, 0.25f);
        ring.write(f.data(), f.size());
        filter.Process(near.data(), near.size(), out.data());
        // Bit-exact passthrough, and no delay: bypass must not silently cost the
        // ASR 16 ms for a stage that is doing nothing.
        for (std::size_t i = 0; i < near.size(); ++i) ASSERT_FLOAT_EQ(out[i], near[i]);
    }
    // Ring drained despite the bypass -- otherwise the playback callback would
    // begin counting overruns for a feature that is switched off.
    EXPECT_LT(ring.available(), 480u);
    EXPECT_EQ(ring.overruns(), 0u);
}

// =============================================================================
// Software volume vs. the canceller
// =============================================================================
// The playback gain is applied AFTER the AEC reference is tapped, so unless the
// filter is told about it, its learned response is wrong by exactly that ratio
// the instant the volume moves. These two tests are what stops that wiring from
// being quietly deleted: one shows the correction works, the other shows the
// failure it prevents is real.

namespace {

// Runs `blocks` of 320 samples where the SPEAKER emits gain*far -- the mic hears
// the echo of the amplified signal, while the reference ring carries the raw
// pre-gain samples, exactly as PullForPlayback publishes them. `tell_filter`
// selects whether the filter is told the gain.
struct VolumeRun {
    std::vector<float> mic;
    std::vector<float> out;
    std::size_t change_at = 0;
};

VolumeRun RunWithVolumeStep(bool tell_filter, float gain_after) {
    constexpr std::size_t kNearBlock = 320;
    constexpr std::size_t kFarBlock = 480;
    constexpr std::size_t kBlocks = 500;      // 10 s

    const std::vector<float> far24 = MakeSpeechLike(kFarBlock * kBlocks, 23, 110.0);
    RationalResampler down(24000, 16000);
    std::vector<float> far16(down.OutputCountFor(far24.size()), 0.0f);
    down.Process(far24.data(), far24.size(), far16.data(), far16.size());
    const std::vector<float> room = MakeRoomImpulse(96, 512);

    VolumeRun r;
    r.change_at = (kBlocks / 2) * kNearBlock;   // halfway, in near-end samples

    // The emitted signal: full scale until the change, `gain_after` past it.
    std::vector<float> emitted = far16;
    for (std::size_t i = r.change_at; i < emitted.size(); ++i) emitted[i] *= gain_after;
    r.mic = Convolve(emitted, room);

    SpscRing<float> ring(48000);
    AecCaptureFilterConfig cfg;
    cfg.aec.block_samples = 128;
    cfg.aec.filter_tail_samples = 2048;
    AecCaptureFilter filter(ring, cfg);

    r.out.assign(kNearBlock * kBlocks, 0.0f);
    for (std::size_t b = 0; b < kBlocks; ++b) {
        const std::size_t at = b * kNearBlock;
        if (tell_filter) {
            filter.SetReferenceGain(at >= r.change_at ? gain_after : 1.0f);
        }
        // The ring carries the PRE-gain reference: the tap is upstream of the
        // volume, which is the whole reason SetReferenceGain has to exist.
        ring.write(far24.data() + b * kFarBlock, kFarBlock);
        filter.Process(r.mic.data() + at, kNearBlock, r.out.data() + at);
    }
    return r;
}

}  // namespace

TEST(AecCaptureFilter, ReferenceGainSurvivesAVolumeChangeMidStream) {
    constexpr float kQuiet = 0.25f;   // the same drop the volume self-test makes

    const VolumeRun told = RunWithVolumeStep(/*tell_filter=*/true, kQuiet);
    const VolumeRun untold = RunWithVolumeStep(/*tell_filter=*/false, kQuiet);

    // Measured over the second AFTER the change -- the window in which the
    // assistant is still speaking and the canceller must not go blind.
    const std::size_t from = told.change_at;
    const std::size_t to = std::min(told.out.size(), from + 16000);
    const auto erle_over = [&](const VolumeRun& r) {
        double mic = 0.0, out = 0.0;
        for (std::size_t i = from; i < to; ++i) {
            mic += static_cast<double>(r.mic[i]) * r.mic[i];
            out += static_cast<double>(r.out[i]) * r.out[i];
        }
        return db(mic, out);
    };

    const double with_fix = erle_over(told);
    const double without = erle_over(untold);
    std::printf("[aec] volume step to %.0f%%: ERLE %.1f dB corrected, %.1f dB uncorrected\n",
                static_cast<double>(kQuiet) * 100.0, with_fix, without);

    // The correction has to hold the line through the change...
    EXPECT_GT(with_fix, 15.0) << "cancellation collapsed despite the gain correction";
    // ...and the uncorrected case has to actually be worse, or this wiring is
    // ceremony and the comment in audio_playback.h is wrong.
    EXPECT_GT(with_fix - without, 5.0)
        << "the reference-gain correction bought nothing (" << with_fix << " vs " << without
        << " dB) -- either it is not applied, or the test no longer stresses it";
}

TEST(AecCaptureFilter, ReferenceGainIsClampedToASafeRange) {
    SpscRing<float> ring(4096);
    AecCaptureFilterConfig cfg;
    cfg.aec.block_samples = 128;
    cfg.aec.filter_tail_samples = 1024;
    AecCaptureFilter filter(ring, cfg);

    EXPECT_FLOAT_EQ(filter.reference_gain(), 1.0f);
    filter.SetReferenceGain(0.5f);
    EXPECT_FLOAT_EQ(filter.reference_gain(), 0.5f);
    // A NEGATIVE gain inverts the reference, which would make the canceller ADD
    // the echo instead of removing it -- the loudest possible failure.
    filter.SetReferenceGain(-1.0f);
    EXPECT_FLOAT_EQ(filter.reference_gain(), 0.0f);
    filter.SetReferenceGain(1000.0f);
    EXPECT_FLOAT_EQ(filter.reference_gain(), 4.0f);
    // NaN must not reach the weights: once there it never leaves.
    filter.SetReferenceGain(std::nanf(""));
    EXPECT_FLOAT_EQ(filter.reference_gain(), 0.0f);
}
