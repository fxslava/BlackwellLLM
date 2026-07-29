// =============================================================================
// tests/bridge/kv_ledger_test.cpp
//
// T2 of the continuous-streaming ladder (docs/CONTINUOUS_STREAMING.md): the KV
// bookkeeping behind re-translation. Three zones, one commit pointer:
//
//   [0, S)      frozen prefix   never rewound, never evicted
//   [S, C)      committed       the only zone head eviction may touch
//   [C, tail)   draft           thrown away on every Partial
//
// CPU-only and engine-free by construction — KvLedger holds no KV data, it only
// decides WHERE the cache should be cut. That is the point of testing it here:
// the two ways this design can be expensively wrong are cutting the draft in the
// wrong place and cutting the head in the wrong place, and both are decidable
// without a GPU.
//
// The two load-bearing assertions are DraftCycleDoesNotGrowTheCache (a redraft
// loop must be a fixed point, or a long utterance leaks context on every Partial)
// and EvictionNeverEntersTheDraftZone (background housekeeping must not be able
// to corrupt an in-flight redraft).
// =============================================================================
#include "kv_ledger.hpp"

#include <cstdint>

#include <gtest/gtest.h>

namespace {

using blackwell::bridge::KvLedger;
using blackwell::bridge::KvSpanKind;

constexpr std::uint32_t kPrefix = 64;   // a system prompt's worth of frozen tokens

// One finalized exchange: turn header, its audio, its translation.
void append_committed_turn(KvLedger& led, std::uint32_t uid,
                           std::uint32_t audio = 20, std::uint32_t text = 30) {
    led.append(KvSpanKind::TurnMarker, 5, uid);
    led.append(KvSpanKind::Audio, audio, uid);
    led.append(KvSpanKind::Text, text, uid);
    led.commit();
}

// ---- zones and construction -------------------------------------------------

TEST(KvLedger, StartsEmptyWithEveryPointerAtTheFrozenPrefix) {
    const KvLedger led(kPrefix);
    EXPECT_EQ(led.frozen_prefix(), kPrefix);
    EXPECT_EQ(led.commit_point(), kPrefix);
    EXPECT_EQ(led.tail(), kPrefix);
    EXPECT_EQ(led.committed_tokens(), 0u);
    EXPECT_EQ(led.draft_tokens(), 0u);
    EXPECT_FALSE(led.has_draft());
    EXPECT_TRUE(led.spans().empty());
}

TEST(KvLedger, AppendAdvancesTheTailAndOnlyTheTail) {
    KvLedger led(kPrefix);
    led.append(KvSpanKind::Audio, 12, 1);
    EXPECT_EQ(led.tail(), kPrefix + 12);
    EXPECT_EQ(led.commit_point(), kPrefix);
    EXPECT_EQ(led.draft_tokens(), 12u);
    EXPECT_TRUE(led.has_draft());
    ASSERT_EQ(led.spans().size(), 1u);
    EXPECT_EQ(led.spans().front().begin, kPrefix);
    EXPECT_EQ(led.spans().front().end(), kPrefix + 12);
}

// A chunk that produced no soft tokens is a normal outcome, not a bug.
TEST(KvLedger, ZeroLengthAppendIsANoOp) {
    KvLedger led(kPrefix);
    led.append(KvSpanKind::Audio, 0, 1);
    EXPECT_TRUE(led.spans().empty());
    EXPECT_EQ(led.tail(), kPrefix);
}

// ---- the re-translation cycle -----------------------------------------------

TEST(KvLedger, RewindDropsTheDraftAndReturnsTheCommitPoint) {
    KvLedger led(kPrefix);
    append_committed_turn(led, 1);
    const std::uint32_t c = led.commit_point();

    led.append(KvSpanKind::TurnMarker, 5, 2);
    led.append(KvSpanKind::Audio, 18, 2);
    led.append(KvSpanKind::Text, 9, 2);
    ASSERT_TRUE(led.has_draft());

    EXPECT_EQ(led.rewind_to_commit(), c);
    EXPECT_EQ(led.tail(), c);
    EXPECT_FALSE(led.has_draft());
    EXPECT_EQ(led.committed_tokens(), c - kPrefix);
    for (const auto& s : led.spans()) EXPECT_LE(s.end(), c);   // nothing above C survives
}

// A redraft loop must not need to test has_draft() first.
TEST(KvLedger, RewindIsIdempotent) {
    KvLedger led(kPrefix);
    append_committed_turn(led, 1);
    const std::uint32_t c = led.commit_point();
    const std::size_t spans = led.spans().size();

    EXPECT_EQ(led.rewind_to_commit(), c);
    EXPECT_EQ(led.rewind_to_commit(), c);
    EXPECT_EQ(led.tail(), c);
    EXPECT_EQ(led.spans().size(), spans);
}

// THE load-bearing property: N Partials for one utterance must leave the cache
// exactly where one would. If this drifts, a long sentence grows the context on
// every redraft until it evicts its own committed history.
TEST(KvLedger, DraftCycleDoesNotGrowTheCache) {
    KvLedger led(kPrefix);
    append_committed_turn(led, 1);
    const std::uint32_t c = led.commit_point();
    const std::size_t committed_spans = led.spans().size();

    for (std::uint32_t growing = 4; growing < 40; growing += 4) {   // a growing window
        led.rewind_to_commit();
        led.append(KvSpanKind::TurnMarker, 5, 2);
        led.append(KvSpanKind::Audio, growing, 2);
        led.append(KvSpanKind::Text, growing / 2, 2);
        EXPECT_EQ(led.commit_point(), c);                 // C never moves during drafting
        EXPECT_EQ(led.spans().size(), committed_spans + 3u);
    }

    led.rewind_to_commit();
    EXPECT_EQ(led.tail(), c);
    EXPECT_EQ(led.spans().size(), committed_spans);
}

TEST(KvLedger, CommitMakesTheDraftPermanent) {
    KvLedger led(kPrefix);
    led.append(KvSpanKind::TurnMarker, 5, 1);
    led.append(KvSpanKind::Audio, 20, 1);
    const std::uint32_t tail = led.tail();

    led.commit();
    EXPECT_EQ(led.commit_point(), tail);
    EXPECT_FALSE(led.has_draft());
    EXPECT_EQ(led.rewind_to_commit(), tail);   // a later rewind cannot undo it
    EXPECT_EQ(led.spans().size(), 2u);
}

// ---- head eviction ----------------------------------------------------------

TEST(KvLedger, EvictionCutsOnATurnBoundary) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);

    const auto plan = led.plan_evict_head(/*min_delta=*/60);
    ASSERT_TRUE(plan.viable);
    EXPECT_EQ(plan.cut, kPrefix + plan.delta);
    EXPECT_GE(plan.delta, 60u);

    // The cut lands on a TurnMarker's first token, so what survives begins with a
    // well-formed turn header rather than a stranded audio span.
    bool cut_is_turn_start = false;
    for (const auto& s : led.spans()) {
        if (s.begin == plan.cut && s.kind == KvSpanKind::TurnMarker) cut_is_turn_start = true;
    }
    EXPECT_TRUE(cut_is_turn_start);
}

TEST(KvLedger, EvictionShiftsEverySurvivingSpanAndBothPointers) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);
    led.append(KvSpanKind::TurnMarker, 5, 6);      // a draft is outstanding
    led.append(KvSpanKind::Audio, 11, 6);

    const std::uint32_t c0 = led.commit_point();
    const std::uint32_t t0 = led.tail();
    const std::size_t   n0 = led.spans().size();

    const auto plan = led.plan_evict_head(60);
    ASSERT_TRUE(plan.viable);
    ASSERT_TRUE(led.apply_evict_head(plan));

    EXPECT_EQ(led.commit_point(), c0 - plan.delta);
    EXPECT_EQ(led.tail(), t0 - plan.delta);
    EXPECT_EQ(led.frozen_prefix(), kPrefix);                  // the sink is untouched
    EXPECT_EQ(led.spans().size(), n0 - plan.spans_dropped);
    EXPECT_EQ(led.evicted_tokens(), plan.delta);

    // Contiguity survives: the first surviving span sits flush against the prefix
    // and the map has no holes.
    ASSERT_FALSE(led.spans().empty());
    EXPECT_EQ(led.spans().front().begin, kPrefix);
    for (std::size_t i = 1; i < led.spans().size(); ++i)
        EXPECT_EQ(led.spans()[i].begin, led.spans()[i - 1].end());
    EXPECT_EQ(led.spans().back().end(), led.tail());
}

// Background housekeeping must never be able to corrupt an in-flight redraft.
TEST(KvLedger, EvictionNeverEntersTheDraftZone) {
    KvLedger led(kPrefix);
    append_committed_turn(led, 1);
    const std::uint32_t c = led.commit_point();

    // A large draft, full of turn boundaries that WOULD satisfy the request if the
    // planner were willing to look past C.
    for (std::uint32_t uid = 2; uid <= 6; ++uid) {
        led.append(KvSpanKind::TurnMarker, 5, uid);
        led.append(KvSpanKind::Audio, 20, uid);
        led.append(KvSpanKind::Text, 30, uid);
    }
    ASSERT_GT(led.tail(), c + 200u);

    const auto plan = led.plan_evict_head(/*min_delta=*/200);
    EXPECT_FALSE(plan.viable);           // no boundary that big exists BELOW C
    EXPECT_FALSE(led.apply_evict_head(plan));
    EXPECT_EQ(led.commit_point(), c);

    // Anything it does accept stays at or below C.
    const auto small = led.plan_evict_head(/*min_delta=*/1);
    if (small.viable) EXPECT_LE(small.cut, c);
}

TEST(KvLedger, EvictionNeverCutsIntoTheFrozenPrefix) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);
    const auto plan = led.plan_evict_head(1);
    ASSERT_TRUE(plan.viable);
    EXPECT_GT(plan.cut, kPrefix);
    ASSERT_TRUE(led.apply_evict_head(plan));
    EXPECT_EQ(led.frozen_prefix(), kPrefix);
    EXPECT_GE(led.commit_point(), kPrefix);
}

TEST(KvLedger, NotViableWhenThereIsNoCommittedTurnBoundaryBigEnough) {
    KvLedger led(kPrefix);
    append_committed_turn(led, 1);
    EXPECT_FALSE(led.plan_evict_head(/*min_delta=*/10000).viable);

    KvLedger empty(kPrefix);
    EXPECT_FALSE(empty.plan_evict_head(1).viable);
}

// A plan is a decision about a specific state, and every position in it is
// absolute. Nothing STRUCTURAL about a spent plan goes stale on its own — cut is
// still S + delta afterwards — so replay protection has to come from the epoch.
TEST(KvLedger, SpentPlanCannotBeReplayed) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);
    const auto plan = led.plan_evict_head(60);
    ASSERT_TRUE(plan.viable);
    ASSERT_TRUE(led.apply_evict_head(plan));

    const std::uint32_t c = led.commit_point();
    const std::uint64_t evicted = led.evicted_tokens();
    EXPECT_FALSE(led.apply_evict_head(plan));
    EXPECT_EQ(led.commit_point(), c);
    EXPECT_EQ(led.evicted_tokens(), evicted);
}

// Same for a plan overtaken by ordinary drafting between plan and apply.
TEST(KvLedger, PlanIsRefusedAfterAnInterveningMutation) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);
    const auto plan = led.plan_evict_head(60);
    ASSERT_TRUE(plan.viable);

    led.append(KvSpanKind::Audio, 7, 6);         // a redraft got there first
    const std::uint32_t c = led.commit_point();
    EXPECT_FALSE(led.apply_evict_head(plan));
    EXPECT_EQ(led.commit_point(), c);

    // Re-planning against the current state succeeds.
    const auto fresh = led.plan_evict_head(60);
    ASSERT_TRUE(fresh.viable);
    EXPECT_TRUE(led.apply_evict_head(fresh));
}

TEST(KvLedger, MalformedPlanIsRefused) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);
    const auto plan = led.plan_evict_head(60);
    ASSERT_TRUE(plan.viable);
    const std::uint32_t c = led.commit_point();

    auto bad_cut = plan;
    bad_cut.cut += 1;                            // cut no longer == S + delta
    EXPECT_FALSE(led.apply_evict_head(bad_cut));

    auto bad_spans = plan;
    bad_spans.spans_dropped = static_cast<std::uint32_t>(led.spans().size() + 1);
    EXPECT_FALSE(led.apply_evict_head(bad_spans));

    EXPECT_EQ(led.commit_point(), c);            // nothing moved
    EXPECT_EQ(led.evicted_tokens(), 0u);
}

// An idempotent no-op must not invalidate a plan: a redraft loop calls
// rewind_to_commit() unconditionally, and doing so with no draft outstanding
// changes nothing about where the head may be cut.
TEST(KvLedger, NoOpRewindDoesNotInvalidateAPlan) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 5; ++uid) append_committed_turn(led, uid);
    const auto plan = led.plan_evict_head(60);
    ASSERT_TRUE(plan.viable);
    ASSERT_FALSE(led.has_draft());

    led.rewind_to_commit();                      // no draft: nothing to drop
    led.commit();                                // C already == tail
    EXPECT_TRUE(led.apply_evict_head(plan));
}

TEST(KvLedger, EvictedTokensAccumulatesAcrossEvictions) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 12; ++uid) append_committed_turn(led, uid);

    std::uint64_t total = 0;
    for (int i = 0; i < 3; ++i) {
        const auto plan = led.plan_evict_head(60);
        ASSERT_TRUE(plan.viable);
        ASSERT_TRUE(led.apply_evict_head(plan));
        total += plan.delta;
        EXPECT_EQ(led.evicted_tokens(), total);
    }
    EXPECT_GT(total, 0u);
}

// A long session: interleaved drafting and eviction must keep the zones ordered.
TEST(KvLedger, ZonesStayOrderedAcrossManyDraftAndEvictCycles) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 40; ++uid) {
        led.rewind_to_commit();
        led.append(KvSpanKind::TurnMarker, 5, uid);
        led.append(KvSpanKind::Audio, 20, uid);      // draft
        led.rewind_to_commit();
        append_committed_turn(led, uid);             // commit

        if (led.committed_tokens() > 400) {
            const auto plan = led.plan_evict_head(120);
            if (plan.viable) ASSERT_TRUE(led.apply_evict_head(plan));
        }
        ASSERT_LE(led.frozen_prefix(), led.commit_point());
        ASSERT_LE(led.commit_point(), led.tail());
    }
    EXPECT_GT(led.evicted_tokens(), 0u);
    EXPECT_LE(led.committed_tokens(), 600u);   // the window really is bounded
}

TEST(KvLedger, ResetKeepsTheFrozenPrefixResident) {
    KvLedger led(kPrefix);
    for (std::uint32_t uid = 1; uid <= 3; ++uid) append_committed_turn(led, uid);
    led.append(KvSpanKind::Audio, 10, 4);

    led.reset();
    EXPECT_EQ(led.frozen_prefix(), kPrefix);   // its KV is still resident and correct
    EXPECT_EQ(led.commit_point(), kPrefix);
    EXPECT_EQ(led.tail(), kPrefix);
    EXPECT_TRUE(led.spans().empty());
    EXPECT_EQ(led.evicted_tokens(), 0u);
}

}  // namespace
