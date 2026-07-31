// =============================================================================
// tests/bridge/retranslation_session_test.cpp
//
// T4's core: the draft-and-commit loop (docs/CONTINUOUS_STREAMING.md). CPU-only,
// driven against a recording fake engine, so the MECHANIC is pinned without a
// GPU, a checkpoint or an audio device.
//
// What is actually worth asserting here is ORDER and ATOMICITY, not arithmetic:
//
//   * every redraft begins with a rewind to C and nothing else ever precedes it;
//   * a Partial leaves C exactly where it found it, so N redrafts of a growing
//     utterance cost the same cache as one;
//   * a Final — and only a Final — advances C;
//   * a failure anywhere leaves the ledger byte-identical to where it started,
//     because a half-appended ledger describes a cache that does not exist and
//     every later eviction plan would be computed against that fiction;
//   * eviction happens only after a commit, only behind C, and only if the
//     engine actually performed it.
// =============================================================================
#include "retranslation_session.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::bridge::ContinuousStreamingConfig;
using blackwell::bridge::IRetranslationEngine;
using blackwell::bridge::KvLedger;
using blackwell::bridge::KvSpanKind;
using blackwell::bridge::RedraftResult;
using blackwell::bridge::RetranslationSession;
using blackwell::vad::SegmentKind;
using blackwell::vad::SpeechSegment;

// Records the exact call sequence so ORDER is assertable, not just outcomes.
class FakeEngine : public IRetranslationEngine {
public:
    bool rewind_to(std::uint32_t pos) override {
        calls.push_back("rewind:" + std::to_string(pos));
        cursor = pos;
        return !fail_rewind;
    }
    std::uint32_t prefill_turn_prefix() override {
        calls.push_back("prefix");
        cursor += prefix_tokens;
        return prefix_tokens;
    }
    std::uint32_t prefill_audio(std::uint64_t begin, std::uint64_t end) override {
        calls.push_back("audio:" + std::to_string(begin) + "-" + std::to_string(end));
        last_begin = begin;
        last_end = end;
        cursor += audio_tokens;
        return audio_tokens;
    }
    std::uint32_t prefill_turn_suffix() override {
        calls.push_back("suffix");
        cursor += suffix_tokens;
        return suffix_tokens;
    }
    std::uint32_t decode_draft(std::uint32_t, std::string* out) override {
        calls.push_back("decode");
        if (out != nullptr) *out = draft_text;
        cursor += decode_tokens;
        return decode_tokens;
    }
    bool evict_head(std::uint32_t keep_from, std::uint32_t delta, std::uint32_t len) override {
        calls.push_back("evict:" + std::to_string(keep_from) + "+" + std::to_string(delta) +
                        "/" + std::to_string(len));
        evict_keep = keep_from;
        evict_delta = delta;
        evict_len = len;
        return !fail_evict;
    }
    std::uint32_t context_capacity() const override { return capacity; }

    std::vector<std::string> calls;
    std::uint32_t prefix_tokens = 5, suffix_tokens = 4, audio_tokens = 12, decode_tokens = 9;
    std::uint32_t capacity = 4096;
    std::uint32_t cursor = 0;
    std::uint64_t last_begin = 0, last_end = 0;
    std::uint32_t evict_keep = 0, evict_delta = 0, evict_len = 0;
    std::string draft_text = "draft";
    bool fail_rewind = false;
    bool fail_evict = false;
};

SpeechSegment partial(std::uint32_t utt, std::uint64_t begin, std::uint64_t end) {
    return SpeechSegment{SegmentKind::Partial, begin, end, utt};
}
SpeechSegment final_seg(std::uint32_t utt, std::uint64_t begin, std::uint64_t end) {
    return SpeechSegment{SegmentKind::Final, begin, end, utt};
}

constexpr std::uint32_t kPrefix = 40;   // frozen system prompt

class Session : public ::testing::Test {
protected:
    void SetUp() override {
        ledger_ = std::make_unique<KvLedger>(kPrefix);
        cfg_.clamp();
        session_ = std::make_unique<RetranslationSession>(&engine_, ledger_.get(), cfg_);
    }
    void rebuild(const ContinuousStreamingConfig& cfg) {
        cfg_ = cfg;
        cfg_.clamp();
        session_ = std::make_unique<RetranslationSession>(&engine_, ledger_.get(), cfg_);
    }

    FakeEngine engine_;
    ContinuousStreamingConfig cfg_{};
    std::unique_ptr<KvLedger> ledger_;
    std::unique_ptr<RetranslationSession> session_;
    int evictions_seen_ = 0;
};

// ---- the redraft order ------------------------------------------------------

TEST_F(Session, RedraftRunsRewindFramingAudioFramingDecodeInThatOrder) {
    const RedraftResult r = session_->on_segment(partial(1, 0, 16000));
    ASSERT_TRUE(r.ok);
    const std::vector<std::string> want = {"rewind:40", "prefix", "audio:0-16000", "suffix",
                                           "decode"};
    EXPECT_EQ(engine_.calls, want);
    EXPECT_EQ(r.text, "draft");
    EXPECT_EQ(r.audio_tokens, engine_.audio_tokens);
    EXPECT_EQ(r.text_tokens, engine_.decode_tokens);
    EXPECT_FALSE(r.committed);
}

// The rewind is the FIRST thing, always — including on the very first segment of
// a session, where it is a no-op. No special case, no ordering to get wrong.
TEST_F(Session, EveryRedraftBeginsWithARewindToTheCommitPointer) {
    session_->on_segment(partial(1, 0, 8000));
    session_->on_segment(partial(1, 0, 16000));
    session_->on_segment(final_seg(1, 0, 24000));

    int rewinds = 0;
    for (size_t i = 0; i < engine_.calls.size(); ++i) {
        if (engine_.calls[i].rfind("rewind:", 0) == 0) {
            ++rewinds;
            EXPECT_EQ(engine_.calls[i], "rewind:40") << "redraft " << rewinds;
            // Nothing may precede a rewind within its redraft.
            if (i > 0) EXPECT_EQ(engine_.calls[i - 1], "decode");
        }
    }
    EXPECT_EQ(rewinds, 3);
}

// The audio window always starts at the utterance's FIRST sample: the loop feeds
// whole utterances, never deltas, which is what the segmenter's growing-window
// invariant exists to guarantee.
TEST_F(Session, AudioIsAlwaysFedFromTheUtteranceFirstSample) {
    session_->on_segment(partial(1, 4000, 12000));
    EXPECT_EQ(engine_.last_begin, 4000u);
    session_->on_segment(partial(1, 4000, 20000));
    EXPECT_EQ(engine_.last_begin, 4000u);
    session_->on_segment(final_seg(1, 4000, 28000));
    EXPECT_EQ(engine_.last_begin, 4000u);
    EXPECT_EQ(engine_.last_end, 28000u);
}

// ---- draft vs commit --------------------------------------------------------

// THE load-bearing property: redrafting is a fixed point. N Partials for one
// utterance must leave the cache exactly where one would, or a long sentence
// grows the context on every redraft until it evicts its own history.
TEST_F(Session, PartialsNeverMoveTheCommitPointerAndDoNotAccumulate) {
    const std::uint32_t c0 = ledger_->commit_point();
    std::uint32_t tail_after_one = 0;

    for (int i = 1; i <= 8; ++i) {
        const RedraftResult r = session_->on_segment(partial(1, 0, 8000u * i));
        ASSERT_TRUE(r.ok);
        EXPECT_EQ(ledger_->commit_point(), c0) << "redraft " << i;
        if (i == 1) tail_after_one = ledger_->tail();
        EXPECT_EQ(ledger_->tail(), tail_after_one) << "redraft " << i;
    }
    EXPECT_EQ(session_->drafts(), 8u);
    EXPECT_EQ(session_->commits(), 0u);
}

TEST_F(Session, FinalAdvancesTheCommitPointerPastTheWholeTurn) {
    const std::uint32_t c0 = ledger_->commit_point();
    const RedraftResult r = session_->on_segment(final_seg(1, 0, 24000));
    ASSERT_TRUE(r.ok);
    EXPECT_TRUE(r.committed);

    // The turn itself, PLUS the next turn's prefix armed on the way out (see THE
    // RESIDENT TURN PREFIX): C deliberately ends with an open user turn, which is
    // what lets the next utterance's redrafts skip step 2 entirely.
    const std::uint32_t turn = engine_.prefix_tokens + engine_.audio_tokens +
                               engine_.suffix_tokens + engine_.decode_tokens;
    EXPECT_EQ(ledger_->commit_point(), c0 + turn + engine_.prefix_tokens);
    EXPECT_EQ(ledger_->tail(), ledger_->commit_point());
    EXPECT_FALSE(ledger_->has_draft());
    EXPECT_EQ(session_->commits(), 1u);
    EXPECT_EQ(session_->resident_prefix(), engine_.prefix_tokens);
}

// THE OPTIMIZATION, stated as a property: once a prefix is resident, a redraft
// must not call prefill_turn_prefix at all. At ~20.5 ms/token on the 8B backbone
// those ~29 tokens are ~600 ms of pure recomputation per redraft.
TEST_F(Session, AnArmedPrefixIsNotRePrefilledByLaterRedrafts) {
    session_->on_segment(final_seg(1, 0, 16000));      // commits, then arms
    ASSERT_EQ(session_->resident_prefix(), engine_.prefix_tokens);

    engine_.calls.clear();
    const RedraftResult r = session_->on_segment(partial(2, 20000, 28000));
    ASSERT_TRUE(r.ok);
    for (const std::string& call : engine_.calls) EXPECT_NE(call, "prefix");
    // What it paid for vs. what it was handed.
    EXPECT_EQ(r.framing_tokens, engine_.suffix_tokens);
    EXPECT_EQ(r.resident_prefix_tokens, engine_.prefix_tokens);
    // The draft is still exactly what the ledger says it is.
    EXPECT_EQ(ledger_->draft_tokens(),
              r.framing_tokens + r.audio_tokens + r.text_tokens);
}

// begin_session() extends the same saving to utterance 1, which otherwise pays
// for its own framing on every one of its redrafts.
TEST_F(Session, BeginSessionArmsThePrefixBeforeTheFirstSegment) {
    session_->begin_session();
    EXPECT_EQ(session_->resident_prefix(), engine_.prefix_tokens);

    engine_.calls.clear();
    const RedraftResult r = session_->on_segment(partial(1, 0, 16000));
    ASSERT_TRUE(r.ok);
    const std::vector<std::string> want = {"rewind:45", "audio:0-16000", "suffix", "decode"};
    EXPECT_EQ(engine_.calls, want);
    EXPECT_EQ(r.framing_tokens, engine_.suffix_tokens);
}

// The optimization must DEGRADE, never corrupt: an engine that cannot arm leaves
// the loop on its original cold path rather than desyncing the ledger.
TEST_F(Session, AFailedArmFallsBackToPrefillingThePrefixPerRedraft) {
    engine_.prefix_tokens = 0;          // arming yields nothing
    session_->begin_session();
    EXPECT_EQ(session_->resident_prefix(), 0u);

    engine_.calls.clear();
    const RedraftResult r = session_->on_segment(partial(1, 0, 16000));
    ASSERT_TRUE(r.ok);
    // It still went looking for a prefix, exactly as the pre-optimization loop did.
    EXPECT_NE(std::find(engine_.calls.begin(), engine_.calls.end(), "prefix"),
              engine_.calls.end());
    EXPECT_EQ(ledger_->draft_tokens(),
              r.framing_tokens + r.audio_tokens + r.text_tokens);
}

// Drafts of utterance N+1 build ON TOP of N's committed translation.
TEST_F(Session, SuccessiveUtterancesStackOnCommittedHistory) {
    session_->on_segment(final_seg(1, 0, 16000));
    const std::uint32_t c1 = ledger_->commit_point();

    session_->on_segment(partial(2, 20000, 28000));
    EXPECT_EQ(ledger_->commit_point(), c1);
    EXPECT_GT(ledger_->tail(), c1);
    EXPECT_EQ(engine_.calls.back(), "decode");

    session_->on_segment(final_seg(2, 20000, 32000));
    EXPECT_GT(ledger_->commit_point(), c1);
    EXPECT_EQ(session_->commits(), 2u);
}

TEST_F(Session, LedgerSpansDescribeTheTurnStructure) {
    session_->on_segment(final_seg(1, 0, 16000));
    // Four spans for the turn, plus a fifth: the NEXT turn's prefix, armed on the
    // way out of the commit. It carries utterance_id 0 because the utterance it
    // frames has not arrived yet.
    ASSERT_EQ(ledger_->spans().size(), 5u);
    EXPECT_EQ(ledger_->spans()[0].kind, KvSpanKind::TurnMarker);
    EXPECT_EQ(ledger_->spans()[1].kind, KvSpanKind::Audio);
    EXPECT_EQ(ledger_->spans()[2].kind, KvSpanKind::TurnMarker);
    EXPECT_EQ(ledger_->spans()[3].kind, KvSpanKind::Text);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(ledger_->spans()[i].utterance_id, 1u);

    EXPECT_EQ(ledger_->spans()[4].kind, KvSpanKind::TurnMarker);
    EXPECT_EQ(ledger_->spans()[4].len, engine_.prefix_tokens);
    EXPECT_EQ(ledger_->spans()[4].utterance_id, 0u);
}

// ---- atomicity --------------------------------------------------------------

TEST_F(Session, FailedRewindLeavesTheLedgerUntouched) {
    session_->on_segment(final_seg(1, 0, 16000));
    const std::uint32_t c = ledger_->commit_point();
    const std::size_t spans = ledger_->spans().size();

    engine_.fail_rewind = true;
    const RedraftResult r = session_->on_segment(partial(2, 20000, 24000));
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(ledger_->commit_point(), c);
    EXPECT_EQ(ledger_->tail(), c);
    EXPECT_EQ(ledger_->spans().size(), spans);
}

// A Partial too short to produce a soft token is normal, not an error — but it
// must unwind rather than leave framing tokens stranded above C.
TEST_F(Session, EmptyAudioUnwindsToTheCommitPointer) {
    session_->on_segment(final_seg(1, 0, 16000));
    const std::uint32_t c = ledger_->commit_point();
    const std::size_t spans = ledger_->spans().size();

    engine_.audio_tokens = 0;
    engine_.calls.clear();
    const RedraftResult r = session_->on_segment(partial(2, 20000, 20100));
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(ledger_->tail(), c);
    EXPECT_EQ(ledger_->spans().size(), spans);
    // It rewound on the way in AND on the way out; it never decoded.
    EXPECT_EQ(engine_.calls.front(), "rewind:" + std::to_string(c));
    EXPECT_EQ(engine_.calls.back(), "rewind:" + std::to_string(c));
    for (const std::string& call : engine_.calls) EXPECT_NE(call, "decode");
}

// A redraft whose worst case cannot fit is refused BEFORE any work, rather than
// discovered part-way through a decode with no way back.
TEST_F(Session, RedraftThatCannotFitTheContextIsRefusedUpFront) {
    engine_.capacity = kPrefix + 10;   // far too small for a maximal draft
    engine_.calls.clear();
    const RedraftResult r = session_->on_segment(partial(1, 0, 16000));
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(session_->overflow_refusals(), 1u);
    EXPECT_EQ(ledger_->tail(), ledger_->commit_point());
    for (const std::string& call : engine_.calls) {
        EXPECT_NE(call, "decode");
        EXPECT_NE(call, "prefix");
    }
}

// ---- eviction ---------------------------------------------------------------

// Eviction runs only after a commit, where tail == C, so it can never interact
// with a draft in flight.
TEST_F(Session, EvictionTriggersAtTheHighWaterMarkAndOnlyAfterACommit) {
    ContinuousStreamingConfig cfg{};
    cfg.eviction_high_water_mark = 128;
    cfg.eviction_target_tokens = 32;
    rebuild(cfg);

    int committed_turns = 0;
    for (std::uint32_t utt = 1; utt <= 12; ++utt) {
        session_->on_segment(partial(utt, utt * 1000ull, utt * 1000ull + 500));
        // A Partial may never evict: the high water is about COMMITTED tokens.
        EXPECT_EQ(session_->evictions(), static_cast<std::uint64_t>(evictions_seen_));
        const RedraftResult r = session_->on_segment(final_seg(utt, utt * 1000ull, utt * 1000ull + 900));
        ASSERT_TRUE(r.ok);
        ++committed_turns;
        evictions_seen_ = static_cast<int>(session_->evictions());
        ASSERT_LE(ledger_->frozen_prefix(), ledger_->commit_point());
        ASSERT_EQ(ledger_->tail(), ledger_->commit_point());
    }
    EXPECT_GT(session_->evictions(), 0u);
    EXPECT_GT(ledger_->evicted_tokens(), 0u);
    // The window really is bounded: without eviction 12 turns of 30 tokens would
    // sail past the high water and keep going.
    EXPECT_LE(ledger_->committed_tokens(), cfg.eviction_high_water_mark);
    EXPECT_EQ(committed_turns, 12);
}

TEST_F(Session, EvictionAsksTheEngineBeforeTheLedgerAndSkipsTheFrozenPrefix) {
    ContinuousStreamingConfig cfg{};
    cfg.eviction_high_water_mark = 128;
    cfg.eviction_target_tokens = 32;
    rebuild(cfg);

    for (std::uint32_t utt = 1; utt <= 12 && session_->evictions() == 0; ++utt) {
        session_->on_segment(final_seg(utt, utt * 1000ull, utt * 1000ull + 900));
    }
    ASSERT_GT(session_->evictions(), 0u);
    EXPECT_EQ(engine_.evict_keep, kPrefix) << "eviction must start above the frozen prefix";
    EXPECT_GT(engine_.evict_delta, 0u);
    EXPECT_GT(ledger_->frozen_prefix(), 0u);
}

// If the cache could not be compacted, the map must not claim it was — otherwise
// every later plan is computed against a fiction.
TEST_F(Session, LedgerIsNotAdvancedWhenTheEngineRefusesToEvict) {
    ContinuousStreamingConfig cfg{};
    cfg.eviction_high_water_mark = 128;
    cfg.eviction_target_tokens = 32;
    rebuild(cfg);
    engine_.fail_evict = true;

    for (std::uint32_t utt = 1; utt <= 12; ++utt) {
        session_->on_segment(final_seg(utt, utt * 1000ull, utt * 1000ull + 900));
    }
    EXPECT_EQ(session_->evictions(), 0u);
    EXPECT_EQ(ledger_->evicted_tokens(), 0u);
    // ...and the committed zone grew past the high water, which is the honest
    // consequence of a cache that refuses to compact.
    EXPECT_GT(ledger_->committed_tokens(), cfg.eviction_high_water_mark);
}

}  // namespace
