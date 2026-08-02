// =============================================================================
// tests/tts/transcript_model_test.cpp
//
// Phase 1 of docs/TTS_INTEGRATION_AUDIT.md: the stable per-utterance identity a
// "speak this line" request will be keyed on. It rides in the TTS suite because
// that is the consumer the identity exists for — the model itself is app-side UI
// storage (audio_sandbox/translator), STL-only and ImGui-free by construction,
// which is exactly what makes it testable here with no GUI and no engine.
//
// THE LOAD-BEARING TEST IS IdsSurviveTheHistoryCap. history_ is capped and drops
// from the front, so under a positional key every entry renames itself once the
// cap starts biting: a button wired to index 7 speaks utterance 7 for the first
// 200 utterances and a different one forever after. That bug cannot reproduce in
// a short session and cannot be missed in a long one.
// =============================================================================
#include "transcript_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using rt::kInvalidUtteranceId;
using rt::TranscriptModel;
using rt::Utterance;

// Drive one complete utterance through the model the way the speech pipeline's
// callbacks do: a run of token pieces, then a clean finalisation.
void say(TranscriptModel& m, std::uint64_t gen, const std::string& text) {
    m.on_token(text.c_str(), gen);
    m.on_final(gen);
}

// ---- commit semantics -------------------------------------------------------

TEST(TranscriptModel, StartsEmpty) {
    TranscriptModel m;
    TranscriptModel::Snapshot s;
    m.snapshot(s);
    EXPECT_TRUE(s.history.empty());
    EXPECT_TRUE(s.live.empty());
    EXPECT_EQ(m.next_id(), 1u) << "0 is reserved as the invalid sentinel";
}

TEST(TranscriptModel, TokensAccumulateIntoTheLiveLineThenCommit) {
    TranscriptModel m;
    m.on_token("Hello", 1);
    m.on_token(", ", 1);
    m.on_token("world", 1);

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    EXPECT_TRUE(s.history.empty()) << "nothing is committed until on_final";
    EXPECT_EQ(s.live, "Hello, world");

    m.on_final(1);
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 1u);
    EXPECT_EQ(s.history[0].text, "Hello, world");
    EXPECT_EQ(s.history[0].gen, 1u);
    EXPECT_FALSE(s.history[0].interrupted);
    EXPECT_TRUE(s.live.empty());
}

TEST(TranscriptModel, IdsAreMonotoneAndStartAtOne) {
    TranscriptModel m;
    say(m, 1, "one");
    say(m, 2, "two");
    say(m, 3, "three");

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 3u);
    EXPECT_EQ(s.history[0].id, 1u);
    EXPECT_EQ(s.history[1].id, 2u);
    EXPECT_EQ(s.history[2].id, 3u);
}

TEST(TranscriptModel, NullTokenTextIsIgnored) {
    TranscriptModel m;
    m.on_token(nullptr, 1);
    m.on_token("ok", 1);
    m.on_token(nullptr, 1);
    m.on_final(1);

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 1u);
    EXPECT_EQ(s.history[0].text, "ok");
}

// A generation with no tokens commits nothing — an empty line must never take an
// id, or ids would advance on every silent turn and the transcript would grow
// blank entries.
TEST(TranscriptModel, EmptyGenerationCommitsNothing) {
    TranscriptModel m;
    m.on_final(1);
    m.on_final(2);
    EXPECT_EQ(m.history_size(), 0u);
    EXPECT_EQ(m.next_id(), 1u);
}

TEST(TranscriptModel, FinalForAStaleGenerationDoesNotDuplicate) {
    TranscriptModel m;
    m.on_token("first", 1);
    m.on_token("second", 2);      // gen bump: flushes "first" as interrupted
    m.on_final(1);                // late/duplicate final for the flushed line

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 1u) << "the interrupted line must not commit twice";
    EXPECT_EQ(s.history[0].text, "first");
    EXPECT_EQ(s.live, "second");
}

// ---- barge-in ---------------------------------------------------------------

TEST(TranscriptModel, GenerationBumpFlushesTheLineAsInterrupted) {
    TranscriptModel m;
    m.on_token("I was saying", 1);
    m.on_token("but now this", 2);      // the speaker barged in

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 1u);
    EXPECT_EQ(s.history[0].text, "I was saying");
    EXPECT_EQ(s.history[0].gen, 1u);
    EXPECT_TRUE(s.history[0].interrupted)
        << "a partial translation must be distinguishable from a finished one";
    EXPECT_EQ(s.live, "but now this");
    EXPECT_EQ(s.live_gen, 2u);
}

TEST(TranscriptModel, InterruptedThenFinalisedLinesAreBothKept) {
    TranscriptModel m;
    m.on_token("partial", 1);
    m.on_token("complete", 2);
    m.on_final(2);

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 2u);
    EXPECT_TRUE(s.history[0].interrupted);
    EXPECT_FALSE(s.history[1].interrupted);
    EXPECT_EQ(s.history[0].id, 1u);
    EXPECT_EQ(s.history[1].id, 2u);
}

// ---- the cap: the reason this class exists ----------------------------------

TEST(TranscriptModel, IdsSurviveTheHistoryCap) {
    TranscriptModel m;
    constexpr std::size_t kOverflow = TranscriptModel::kMaxHistory + 50;
    for (std::size_t i = 0; i < kOverflow; ++i) {
        say(m, static_cast<std::uint64_t>(i + 1), "utterance " + std::to_string(i + 1));
    }

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), TranscriptModel::kMaxHistory);

    // The oldest 50 were trimmed, so the window now starts at id 51 — and each
    // surviving entry's id still names the text it was committed with. Under a
    // positional key, index 0 would now silently mean a different utterance than
    // it did before the cap engaged.
    EXPECT_EQ(s.history.front().id, 51u);
    EXPECT_EQ(s.history.front().text, "utterance 51");
    EXPECT_EQ(s.history.back().id, static_cast<std::uint64_t>(kOverflow));
    EXPECT_EQ(s.history.back().text, "utterance " + std::to_string(kOverflow));

    for (const Utterance& u : s.history) {
        EXPECT_EQ(u.text, "utterance " + std::to_string(u.id));
    }
}

TEST(TranscriptModel, IdsAreNeverReused) {
    TranscriptModel m;
    constexpr std::size_t kOverflow = TranscriptModel::kMaxHistory * 2;
    for (std::size_t i = 0; i < kOverflow; ++i) say(m, 1, "x");

    // Every id handed out is distinct, including across trims.
    EXPECT_EQ(m.next_id(), static_cast<std::uint64_t>(kOverflow) + 1u);

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    std::vector<std::uint64_t> ids;
    for (const Utterance& u : s.history) ids.push_back(u.id);
    for (std::size_t i = 1; i < ids.size(); ++i) {
        EXPECT_GT(ids[i], ids[i - 1]) << "ids must be strictly increasing";
    }
}

// ---- lookup by id -----------------------------------------------------------

TEST(TranscriptModel, FindResolvesALiveId) {
    TranscriptModel m;
    say(m, 1, "alpha");
    say(m, 2, "beta");

    Utterance u;
    ASSERT_TRUE(m.find(2, u));
    EXPECT_EQ(u.text, "beta");
    EXPECT_EQ(u.gen, 2u);
}

// A consumer holding an id across time MUST cope with the line ageing out. This
// is the case that a positional key would answer with the WRONG text instead of
// with "gone".
TEST(TranscriptModel, FindFailsCleanlyForTrimmedAndUnknownIds) {
    TranscriptModel m;
    for (std::size_t i = 0; i < TranscriptModel::kMaxHistory + 10; ++i) {
        say(m, 1, "x");
    }
    Utterance u;
    EXPECT_FALSE(m.find(1, u)) << "id 1 was trimmed; it must not resolve to a survivor";
    EXPECT_FALSE(m.find(999999, u));
    EXPECT_FALSE(m.find(kInvalidUtteranceId, u));
}

TEST(TranscriptModel, FindDoesNotSeeTheUncommittedLiveLine) {
    TranscriptModel m;
    m.on_token("still going", 1);
    Utterance u;
    // The live line has no id yet, by design: it is still changing, so nothing
    // may hold a reference to it.
    EXPECT_FALSE(m.find(1, u));
}

// ---- reuse / teardown -------------------------------------------------------

TEST(TranscriptModel, SnapshotReusesTheCallersBuffers) {
    TranscriptModel m;
    say(m, 1, "one");

    TranscriptModel::Snapshot s;
    s.history.assign(9, Utterance{});        // stale contents from a prior frame
    s.live = "stale";
    m.snapshot(s);
    ASSERT_EQ(s.history.size(), 1u);
    EXPECT_EQ(s.history[0].text, "one");
    EXPECT_TRUE(s.live.empty());
}

TEST(TranscriptModel, ClearDropsContentButNotTheIdSequence) {
    TranscriptModel m;
    say(m, 1, "one");
    say(m, 2, "two");
    const std::uint64_t next_before = m.next_id();

    m.clear();
    EXPECT_EQ(m.history_size(), 0u);
    // Ids continue: a cleared transcript must not start re-issuing ids that an
    // in-flight synthesis request already refers to.
    EXPECT_EQ(m.next_id(), next_before);

    say(m, 3, "three");
    Utterance u;
    ASSERT_TRUE(m.find(next_before, u));
    EXPECT_EQ(u.text, "three");
}

TEST(TranscriptModel, ClearDiscardsTheLiveLine) {
    TranscriptModel m;
    m.on_token("in progress", 1);
    m.clear();

    TranscriptModel::Snapshot s;
    m.snapshot(s);
    EXPECT_TRUE(s.live.empty());

    // The generation sentinel was reset too, so the next token of the SAME
    // generation starts a fresh line rather than silently appending to a line
    // that no longer exists.
    m.on_token("after clear", 1);
    m.snapshot(s);
    EXPECT_EQ(s.live, "after clear");
    EXPECT_TRUE(s.history.empty());
}

}  // namespace
