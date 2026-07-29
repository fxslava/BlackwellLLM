// =============================================================================
// tests/bridge/segment_job_queue_test.cpp
//
// The audio-thread -> engine-thread handoff for segmenter events. CPU-only.
//
// The queue exists to encode one asymmetry, and these tests are that asymmetry:
// a Partial is only worth its recency (collapse it, drop it under load), a Final
// is a commit (never collapsed, never dropped, never reordered past a Partial).
// Losing a Final leaves the commit pointer behind forever — the utterance stays
// in the draft zone and the very next rewind deletes it.
// =============================================================================
#include "segment_job_queue.hpp"

#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::bridge::SegmentJobQueue;
using blackwell::vad::SegmentKind;
using blackwell::vad::SpeechSegment;

SpeechSegment partial(std::uint32_t utt, std::uint64_t end) {
    return SpeechSegment{SegmentKind::Partial, 0, end, utt};
}
SpeechSegment final_seg(std::uint32_t utt, std::uint64_t end) {
    return SpeechSegment{SegmentKind::Final, 0, end, utt};
}

std::vector<SpeechSegment> drain(SegmentJobQueue& q) {
    std::vector<SpeechSegment> out;
    SpeechSegment s{};
    while (q.take(&s)) out.push_back(s);
    return out;
}

TEST(SegmentJobQueue, EmptyQueueYieldsNothing) {
    SegmentJobQueue q;
    SpeechSegment s{};
    EXPECT_FALSE(q.take(&s));
    EXPECT_EQ(q.pending(), 0u);
}

TEST(SegmentJobQueue, PreservesOrderAcrossUtterances) {
    SegmentJobQueue q;
    q.post(final_seg(1, 100));
    q.post(partial(2, 200));
    q.post(final_seg(2, 300));

    const auto got = drain(q);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0].utterance_id, 1u);
    EXPECT_EQ(got[0].kind, SegmentKind::Final);
    EXPECT_EQ(got[1].utterance_id, 2u);
    EXPECT_EQ(got[1].kind, SegmentKind::Partial);
    EXPECT_EQ(got[2].kind, SegmentKind::Final);
}

// ---- collapse ---------------------------------------------------------------

// The newer Partial covers a strictly longer prefix of the SAME audio from the
// same first sample, so drafting the older one first would draft stale audio and
// immediately throw it away.
TEST(SegmentJobQueue, ConsecutivePartialsOfOneUtteranceCollapseToTheNewest) {
    SegmentJobQueue q;
    q.post(partial(1, 100));
    q.post(partial(1, 200));
    q.post(partial(1, 300));

    EXPECT_EQ(q.pending(), 1u);
    EXPECT_EQ(q.collapsed(), 2u);
    const auto got = drain(q);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].end_sample, 300u);   // the newest right edge survives
}

TEST(SegmentJobQueue, PartialsOfDifferentUtterancesDoNotCollapse) {
    SegmentJobQueue q;
    q.post(partial(1, 100));
    q.post(partial(2, 200));
    EXPECT_EQ(q.pending(), 2u);
    EXPECT_EQ(q.collapsed(), 0u);
}

// A Partial may never absorb anything sitting behind a Final: the Final has to be
// processed first for the commit pointer to advance past its utterance.
TEST(SegmentJobQueue, APartialNeverCollapsesAcrossAFinal) {
    SegmentJobQueue q;
    q.post(partial(1, 100));
    q.post(final_seg(1, 200));
    q.post(partial(1, 300));   // same utterance id, but a Final stands between

    EXPECT_EQ(q.collapsed(), 0u);
    const auto got = drain(q);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[1].kind, SegmentKind::Final);
}

TEST(SegmentJobQueue, FinalsAreNeverCollapsedIntoEachOther) {
    SegmentJobQueue q;
    q.post(final_seg(1, 100));
    q.post(final_seg(2, 200));
    EXPECT_EQ(q.pending(), 2u);
    EXPECT_EQ(q.collapsed(), 0u);
}

// ---- overflow ---------------------------------------------------------------

// Under a backlog the queue must degrade by losing draft FRESHNESS, never by
// losing a commit.
TEST(SegmentJobQueue, OverflowShedsTheOldestPartialAndKeepsEveryFinal) {
    SegmentJobQueue q(/*soft_capacity=*/4);
    // Alternating so nothing collapses (different utterance ids).
    for (std::uint32_t i = 1; i <= 12; ++i) {
        if (i % 3 == 0) q.post(final_seg(i, i * 10));
        else            q.post(partial(i, i * 10));
    }

    const auto got = drain(q);
    EXPECT_LE(got.size(), 8u);
    EXPECT_GT(q.dropped(), 0u);

    int finals = 0;
    for (const SpeechSegment& s : got) {
        if (s.kind == SegmentKind::Final) ++finals;
    }
    EXPECT_EQ(finals, 4) << "a Final was dropped under backlog — the commit pointer would "
                            "never advance past that utterance";

    // Order is still monotone in utterance id: shedding never reorders.
    for (size_t i = 1; i < got.size(); ++i)
        EXPECT_GT(got[i].utterance_id, got[i - 1].utterance_id);
}

// A queue of nothing but Finals grows rather than dropping one: the engine is
// catastrophically behind, and losing translations is strictly worse than memory.
TEST(SegmentJobQueue, AllFinalsBacklogGrowsRatherThanDroppingACommit) {
    SegmentJobQueue q(/*soft_capacity=*/2);
    for (std::uint32_t i = 1; i <= 10; ++i) q.post(final_seg(i, i * 10));
    EXPECT_EQ(q.pending(), 10u);
    EXPECT_EQ(q.dropped(), 0u);
}

TEST(SegmentJobQueue, ClearDropsEverythingPending) {
    SegmentJobQueue q;
    q.post(partial(1, 100));
    q.post(final_seg(1, 200));
    q.clear();
    EXPECT_EQ(q.pending(), 0u);
    SpeechSegment s{};
    EXPECT_FALSE(q.take(&s));
}

// ---- threading --------------------------------------------------------------

// The real usage: one producer thread posting while one consumer drains. Every
// Final must arrive, exactly once, in order.
TEST(SegmentJobQueue, EveryFinalSurvivesConcurrentProduceAndConsume) {
    SegmentJobQueue q(/*soft_capacity=*/4);
    constexpr std::uint32_t kUtterances = 500;

    std::vector<std::uint32_t> seen_finals;
    std::atomic<bool> done{false};

    std::thread producer([&] {
        for (std::uint32_t u = 1; u <= kUtterances; ++u) {
            q.post(partial(u, u * 10));
            q.post(partial(u, u * 10 + 1));   // collapses with the one above
            q.post(final_seg(u, u * 10 + 2));
        }
        done.store(true, std::memory_order_release);
    });

    SpeechSegment s{};
    while (!done.load(std::memory_order_acquire) || q.pending() > 0) {
        if (q.take(&s)) {
            if (s.kind == SegmentKind::Final) seen_finals.push_back(s.utterance_id);
        }
    }
    producer.join();
    while (q.take(&s)) {
        if (s.kind == SegmentKind::Final) seen_finals.push_back(s.utterance_id);
    }

    ASSERT_EQ(seen_finals.size(), kUtterances);
    for (std::uint32_t i = 0; i < kUtterances; ++i) EXPECT_EQ(seen_finals[i], i + 1);
}

}  // namespace
