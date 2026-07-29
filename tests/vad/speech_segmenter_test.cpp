// =============================================================================
// tests/vad/speech_segmenter_test.cpp
//
// T1 of the continuous-streaming ladder (docs/CONTINUOUS_STREAMING.md): the
// segmentation POLICY that decides when the re-translation pipeline redrafts
// (Partial) and when it commits (Final).
//
// CPU-only, deterministic, no ONNXRuntime and no model file: the segmenter
// consumes probabilities, so the tests drive it with scripted ones. What Silero
// actually reports on real speech is pinned separately in silero_vad_test.cpp;
// what is under test HERE is the state machine on top of that.
//
// The load-bearing assertion is GrowingWindow: every Partial of an utterance and
// its Final share one begin_sample. A consumer that rewinds the KV cache to the
// commit pointer before each redraft is only correct because of that — if the
// left edge ever moved, every redraft would become a splice.
// =============================================================================
#include "speech_segmenter.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::vad::SegmentKind;
using blackwell::vad::SegmenterConfig;
using blackwell::vad::SpeechSegment;
using blackwell::vad::SpeechSegmenter;

// Defaults, in the units the assertions are written in. 16 kHz, 160-sample
// (10 ms) blocks, so one block is 160 samples and 100 blocks is one second.
constexpr std::uint64_t kBlock       = 160;
constexpr std::uint64_t kPreroll     = 4000;    // 250 ms
constexpr std::uint64_t kHangoverBlk = 40;      // 400 ms
constexpr std::uint64_t kTailPad     = 3200;    // 200 ms
constexpr std::uint64_t kPartialBlk  = 50;      // 500 ms
constexpr std::uint64_t kMinSpeechBlk = 20;     // 200 ms
constexpr std::uint64_t kMaxSpanBlk  = 2000;    // 20 s

constexpr float kSpeech  = 0.95f;
constexpr float kSilence = 0.02f;
constexpr float kBetween = 0.40f;   // above release (0.35), below onset (0.50)

// Feeds `blocks` blocks at probability `p`, appending any events to `out`.
void feed(SpeechSegmenter& s, int blocks, float p, std::vector<SpeechSegment>* out) {
    for (int i = 0; i < blocks; ++i) {
        if (auto ev = s.on_block(p)) out->push_back(*ev);
    }
}

std::vector<SpeechSegment> feed(SpeechSegmenter& s, int blocks, float p) {
    std::vector<SpeechSegment> out;
    feed(s, blocks, p, &out);
    return out;
}

std::vector<SpeechSegment> of_kind(const std::vector<SpeechSegment>& all, SegmentKind k) {
    std::vector<SpeechSegment> out;
    for (const SpeechSegment& s : all) {
        if (s.kind == k) out.push_back(s);
    }
    return out;
}

// ---- onset / hysteresis -----------------------------------------------------

TEST(SpeechSegmenter, SilenceEmitsNothing) {
    SpeechSegmenter seg;
    EXPECT_TRUE(feed(seg, 1000, kSilence).empty());
    EXPECT_FALSE(seg.in_speech());
    EXPECT_EQ(seg.utterance_id(), 0u);
}

// The onset threshold is higher than the sustain threshold, and an utterance may
// only START on the higher one. Without this, audio parked in the hysteresis band
// would open a turn that nothing ever closes.
TEST(SpeechSegmenter, ProbabilityInsideTheHysteresisBandNeverStartsAnUtterance) {
    SpeechSegmenter seg;
    ASSERT_GT(kBetween, seg.release_threshold());
    ASSERT_LT(kBetween, seg.config().onset_threshold);

    EXPECT_TRUE(feed(seg, 500, kBetween).empty());
    EXPECT_FALSE(seg.in_speech());
}

// ---- the pre-roll -----------------------------------------------------------

// The encoder is weakest at a window's left edge, so a segment must never start
// on the first phoneme: the range reaches back before the onset block.
TEST(SpeechSegmenter, SegmentBeginsOnePreRollBeforeTheOnsetBlock) {
    SpeechSegmenter seg;
    feed(seg, 100, kSilence);                   // onset lands at sample 100*160
    const auto ev = feed(seg, 400, kSpeech);
    ASSERT_FALSE(ev.empty());
    EXPECT_EQ(ev.front().begin_sample, 100 * kBlock - kPreroll);
}

TEST(SpeechSegmenter, PreRollClampsAtTheStartOfTheStream) {
    SpeechSegmenter seg;
    const auto ev = feed(seg, 400, kSpeech);    // onset on the very first block
    ASSERT_FALSE(ev.empty());
    EXPECT_EQ(ev.front().begin_sample, 0ull);
}

// ---- the growing-window invariant (the reason re-translation is correct) ----

TEST(SpeechSegmenter, GrowingWindowPartialsAndFinalShareOneBegin) {
    SpeechSegmenter seg;
    std::vector<SpeechSegment> ev;
    feed(seg, 400, kSpeech, &ev);                       // 4 s of speech
    feed(seg, static_cast<int>(kHangoverBlk) + 5, kSilence, &ev);

    const auto partials = of_kind(ev, SegmentKind::Partial);
    const auto finals   = of_kind(ev, SegmentKind::Final);
    ASSERT_GE(partials.size(), 3u);
    ASSERT_EQ(finals.size(), 1u);

    const std::uint64_t begin = partials.front().begin_sample;
    for (const SpeechSegment& p : partials) EXPECT_EQ(p.begin_sample, begin);
    EXPECT_EQ(finals.front().begin_sample, begin);

    // Only the right edge moves, and it only ever moves forward.
    for (std::size_t i = 1; i < partials.size(); ++i)
        EXPECT_GT(partials[i].end_sample, partials[i - 1].end_sample);
    EXPECT_GT(finals.front().end_sample, partials.back().end_sample);
}

TEST(SpeechSegmenter, PartialsArriveOnTheConfiguredCadence) {
    SpeechSegmenter seg;
    const auto partials = of_kind(feed(seg, 400, kSpeech), SegmentKind::Partial);
    ASSERT_GE(partials.size(), 3u);
    for (std::size_t i = 1; i < partials.size(); ++i) {
        EXPECT_EQ(partials[i].end_sample - partials[i - 1].end_sample,
                  kPartialBlk * kBlock);
    }
}

TEST(SpeechSegmenter, EveryEventOfAnUtteranceCarriesOneUtteranceId) {
    SpeechSegmenter seg;
    std::vector<SpeechSegment> ev;
    feed(seg, 400, kSpeech, &ev);
    feed(seg, static_cast<int>(kHangoverBlk) + 5, kSilence, &ev);
    ASSERT_FALSE(ev.empty());
    for (const SpeechSegment& s : ev) EXPECT_EQ(s.utterance_id, 1u);
}

// ---- ending an utterance ----------------------------------------------------

TEST(SpeechSegmenter, FinalArrivesExactlyOneHangoverAfterTheLastSpeechBlock) {
    SpeechSegmenter seg;
    feed(seg, 300, kSpeech);
    const std::uint64_t last_speech_end = seg.stream_position();

    // One block short of the hangover: still nothing.
    EXPECT_TRUE(of_kind(feed(seg, static_cast<int>(kHangoverBlk) - 1, kSilence),
                        SegmentKind::Final).empty());
    const auto finals = of_kind(feed(seg, 1, kSilence), SegmentKind::Final);
    ASSERT_EQ(finals.size(), 1u);
    EXPECT_EQ(seg.stream_position(), last_speech_end + kHangoverBlk * kBlock);
    EXPECT_FALSE(seg.in_speech());
}

// The Final keeps a short tail past the last speech block, NOT the whole
// hangover — otherwise every utterance would be padded with the full 400 ms of
// silence that ended it, for the encoder to chew on.
TEST(SpeechSegmenter, FinalKeepsTheTailPadAndNotTheWholeHangover) {
    SpeechSegmenter seg;
    feed(seg, 300, kSpeech);
    const std::uint64_t last_speech_end = seg.stream_position();
    const auto finals = of_kind(feed(seg, static_cast<int>(kHangoverBlk), kSilence),
                                SegmentKind::Final);
    ASSERT_EQ(finals.size(), 1u);
    EXPECT_EQ(finals.front().end_sample, last_speech_end + kTailPad);
    EXPECT_LT(finals.front().end_sample, seg.stream_position());
}

// A breath is not the end of a sentence.
TEST(SpeechSegmenter, DipShorterThanTheHangoverDoesNotEndTheUtterance) {
    SpeechSegmenter seg;
    std::vector<SpeechSegment> ev;
    feed(seg, 200, kSpeech, &ev);
    feed(seg, static_cast<int>(kHangoverBlk) - 1, kSilence, &ev);   // just short
    feed(seg, 200, kSpeech, &ev);
    EXPECT_TRUE(of_kind(ev, SegmentKind::Final).empty());
    EXPECT_TRUE(seg.in_speech());
    EXPECT_EQ(seg.utterance_id(), 1u);   // still the same utterance

    // ...and the silence clock was re-armed by the resumed speech.
    feed(seg, static_cast<int>(kHangoverBlk), kSilence, &ev);
    ASSERT_EQ(of_kind(ev, SegmentKind::Final).size(), 1u);
}

// A door slam must not open a turn. Nothing is emitted at all — including no
// Partial, so the consumer has no draft to unwind.
TEST(SpeechSegmenter, BurstShorterThanMinUtteranceEmitsNothing) {
    SpeechSegmenter seg;
    std::vector<SpeechSegment> ev;
    feed(seg, static_cast<int>(kMinSpeechBlk) - 1, kSpeech, &ev);
    feed(seg, static_cast<int>(kHangoverBlk) + 5, kSilence, &ev);
    EXPECT_TRUE(ev.empty());
    EXPECT_FALSE(seg.in_speech());
    EXPECT_EQ(seg.committed_end(), 0ull);
}

// ---- utterance to utterance -------------------------------------------------

TEST(SpeechSegmenter, UtteranceIdIncrementsAcrossUtterances) {
    SpeechSegmenter seg;
    std::vector<SpeechSegment> ev;
    for (int i = 0; i < 2; ++i) {
        feed(seg, 300, kSpeech, &ev);
        feed(seg, static_cast<int>(kHangoverBlk) + 5, kSilence, &ev);
    }
    const auto finals = of_kind(ev, SegmentKind::Final);
    ASSERT_EQ(finals.size(), 2u);
    EXPECT_EQ(finals[0].utterance_id, 1u);
    EXPECT_EQ(finals[1].utterance_id, 2u);
    EXPECT_GE(finals[1].begin_sample, finals[0].end_sample);
}

// The pre-roll may not reach back into audio a Final already committed: re-feeding
// those samples would duplicate them in the KV cache on the next redraft.
TEST(SpeechSegmenter, PreRollNeverReachesIntoCommittedAudio) {
    SpeechSegmenter seg;
    std::vector<SpeechSegment> ev;
    feed(seg, 300, kSpeech, &ev);
    feed(seg, static_cast<int>(kHangoverBlk), kSilence, &ev);   // Final for #1
    const auto first = of_kind(ev, SegmentKind::Final);
    ASSERT_EQ(first.size(), 1u);

    // Resume immediately — an unclamped 250 ms pre-roll would reach back past the
    // first Final's end.
    ev.clear();
    feed(seg, 300, kSpeech, &ev);
    ASSERT_FALSE(ev.empty());
    EXPECT_EQ(ev.front().utterance_id, 2u);
    EXPECT_EQ(ev.front().begin_sample, first.front().end_sample);
}

// A speaker who never pauses still has to be committed: the ring and the encoder
// both have ceilings. The continuation is contiguous — no gap, no overlap, and no
// pre-roll to double-count, because the audio never stopped.
TEST(SpeechSegmenter, MaxUtteranceForcesAFinalAndContinuesContiguously) {
    SpeechSegmenter seg;
    const auto ev = feed(seg, static_cast<int>(kMaxSpanBlk) + 200, kSpeech);
    const auto finals = of_kind(ev, SegmentKind::Final);
    ASSERT_EQ(finals.size(), 1u);
    EXPECT_EQ(finals.front().begin_sample, 0ull);
    EXPECT_EQ(finals.front().end_sample, kMaxSpanBlk * kBlock);
    EXPECT_EQ(finals.front().utterance_id, 1u);

    EXPECT_TRUE(seg.in_speech());
    EXPECT_EQ(seg.utterance_id(), 2u);

    // Everything after the cut belongs to utterance 2 and starts exactly at it.
    for (const SpeechSegment& s : ev) {
        if (s.end_sample > finals.front().end_sample) {
            EXPECT_EQ(s.utterance_id, 2u);
            EXPECT_EQ(s.begin_sample, finals.front().end_sample);
        }
    }
}

TEST(SpeechSegmenter, ResetDropsTheInFlightUtteranceWithoutEmitting) {
    SpeechSegmenter seg;
    feed(seg, 300, kSpeech);
    ASSERT_TRUE(seg.in_speech());

    seg.reset();
    EXPECT_FALSE(seg.in_speech());
    EXPECT_EQ(seg.stream_position(), 0ull);
    EXPECT_EQ(seg.utterance_id(), 0u);
    EXPECT_EQ(seg.committed_end(), 0ull);

    // And the clock really did restart: the next utterance is #1 again, from 0.
    const auto ev = feed(seg, 300, kSpeech);
    ASSERT_FALSE(ev.empty());
    EXPECT_EQ(ev.front().utterance_id, 1u);
    EXPECT_EQ(ev.front().begin_sample, 0ull);
}

// ---- configurability --------------------------------------------------------

TEST(SpeechSegmenter, ReleaseThresholdIsDerivedAndFloored) {
    SegmenterConfig cfg{};
    cfg.onset_threshold = 0.5f;
    EXPECT_FLOAT_EQ(SpeechSegmenter(cfg).release_threshold(), 0.35f);

    cfg.onset_threshold = 0.1f;   // would go negative without the floor
    EXPECT_FLOAT_EQ(SpeechSegmenter(cfg).release_threshold(), cfg.release_floor);
}

TEST(SpeechSegmenter, PartialsCanBeDisabledByAVeryLongPeriod) {
    SegmenterConfig cfg{};
    cfg.partial_period_ms = 1000000;   // effectively "never redraft"
    SpeechSegmenter seg(cfg);
    std::vector<SpeechSegment> ev;
    feed(seg, 400, kSpeech, &ev);
    feed(seg, static_cast<int>(kHangoverBlk) + 5, kSilence, &ev);
    EXPECT_TRUE(of_kind(ev, SegmentKind::Partial).empty());
    EXPECT_EQ(of_kind(ev, SegmentKind::Final).size(), 1u);
}

}  // namespace
