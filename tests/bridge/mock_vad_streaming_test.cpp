// =============================================================================
// tests/bridge/mock_vad_streaming_test.cpp
//
// Tier-1 (CPU, deterministic) mock-VAD streaming regression for the Phase-0 KV
// checkpoint/rollback primitive. No CUDA: it mirrors RealEngineControl's KV
// bookkeeping on a std::vector<int> so the checkpoint / system-prefix-floor /
// rollback / delta-append ARITHMETIC is proven independently of any GPU, while
// still driving the REAL EngineControlBridge barge-in epoch (cancel_generation /
// cancelled) as the interruption signal.
//
// The definitive assertion is self-consistency, not a pinned golden string: a
// speculative draft that is rolled back must leave the KV byte-identical to the
// no-speculation baseline (zero residue). That is exactly the guarantee
// VRAMArena::truncate_kv + kv_cache_rollback exist to provide; here it is checked
// on the mock mirror, and on the real GPU path by the Tier-2 integration test.
//
// Links like engine_control_bridge_test.cpp (blackwell_bridge + blackwell_core_obj
// satisfy the engine symbols the base references; the overrides never call them).
// =============================================================================
#include "engine_control_bridge.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

#include <gtest/gtest.h>

#include "sliding_audio_window.h"  // blackwell::audio::center_slice_plan (pure geometry)

using blackwell::EngineStatus;
using blackwell::bridge::EngineControlBridge;

namespace {

// GPU-free stand-in that mirrors RealEngineControl's KV state (a token vector +
// sequence pointer + verified-prefix boundary) and reproduces its checkpoint /
// rollback logic EXACTLY, including the frozen-system-prefix floor clamp. The KV
// vector plays the role the real Continuous KV slabs + offload mirror play, so a
// resize() down to `safe_pos` is the mock of VRAMArena::truncate_kv.
class MockVadStreamingBridge : public EngineControlBridge {
public:
    struct KVCheckpoint {
        int      pos = 0;
        uint32_t verified_tokens = 0;
    };

    explicit MockVadStreamingBridge(const Config& cfg = {})
        : EngineControlBridge(/*engine=*/nullptr, cfg) {}

    // Freeze `n` system-prefix tokens as the rewind floor (barge-in never drops
    // below this) and anchor the sequence pointer past them.
    void prefill_system_prompt(uint32_t n) {
        set_system_prefix_tokens(n);
        kv_.assign(n, kSysToken);
        pos_ = static_cast<int>(n);
        history_base_ = pos_;   // no retained history yet (== RealEngineControl)
    }

    // Append one sliding-window chunk's delta soft-tokens (the newly generated
    // tokens; the overlapping history is never re-appended — Phase-1 contract).
    void feed_chunk(const std::vector<int>& delta) {
        for (const int t : delta) { kv_.push_back(t); ++pos_; }
    }

    // ---- Phase-0 primitives: arithmetic-identical to RealEngineControl ----
    KVCheckpoint checkpoint() const { return {pos_, verified_tokens_}; }

    void rollback(const KVCheckpoint& cp) {
        const int floor    = static_cast<int>(effective_keep_tokens(0));
        const int safe_pos = std::max(cp.pos, floor);        // never below the prefix
        pos_ = safe_pos;
        verified_tokens_ = cp.verified_tokens;
        kv_.resize(static_cast<std::size_t>(safe_pos));      // == truncate_kv semantics
    }

    const std::vector<int>& kv() const { return kv_; }
    int      pos() const { return pos_; }
    uint32_t verified_tokens() const { return verified_tokens_; }

    // ---- Bounded-history additions: arithmetic-identical to RealEngineControl --

    // The retained-history base (system prefix + retained turns) a barge-in
    // rewind must honour; RealEngineControl advances it at turn boundaries.
    void set_history_base(int base) { history_base_ = base; }
    int  history_base() const { return history_base_; }

    // Mirrors RealEngineControl::do_rewind's keep resolution: the requested keep
    // is clamped up to the system-prefix floor AND the retained-history base.
    uint32_t resolve_rewind_keep(uint32_t requested) const {
        uint32_t keep = effective_keep_tokens(requested);
        const auto base = static_cast<uint32_t>(history_base_);
        return keep < base ? base : keep;
    }

    // Stateless end-of-turn flush: rollback to the base (== system floor when no
    // history is retained). Mirrors finalize_turn's non-bounded branch.
    void flush_turn() { rollback({history_base_, verified_tokens_}); }

private:
    static constexpr int kSysToken = -1;   // sentinel for a frozen system-prefix slot
    std::vector<int> kv_;
    int      pos_ = 0;
    int      history_base_ = 0;
    uint32_t verified_tokens_ = 0;
};

// Mirrors finalize_turn's deque pruning: drop OLDEST records until the total is
// within the budget. Returns the retained total.
std::size_t prune_to_budget(std::deque<std::vector<int>>& turns, std::size_t budget) {
    std::size_t total = 0;
    for (const auto& t : turns) total += t.size();
    while (!turns.empty() && total > budget) {
        total -= turns.front().size();
        turns.pop_front();
    }
    return total;
}

// A scripted utterance fed as 320 ms sliding-window hops: each chunk contributes
// two delta soft-tokens (the default hop_tokens of SlidingWindowConfig).
const std::vector<std::vector<int>> kChunks = {
    {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19},
};

std::vector<int> flatten(const std::vector<std::vector<int>>& cs, std::size_t upto) {
    std::vector<int> v;
    for (std::size_t i = 0; i < upto; ++i) v.insert(v.end(), cs[i].begin(), cs[i].end());
    return v;
}

constexpr uint32_t kSystemPrefix = 44;   // matches the translator's real prefix length

}  // namespace

// Baseline: feed every chunk straight through, no speculation. The KV past the
// frozen prefix is exactly the concatenation of every delta.
TEST(MockVadStreaming, BaselineAppendsEveryDelta) {
    MockVadStreamingBridge b;
    b.prefill_system_prompt(kSystemPrefix);
    for (const auto& c : kChunks) b.feed_chunk(c);

    EXPECT_EQ(b.pos(), static_cast<int>(kSystemPrefix) + 10);
    const std::vector<int> tail(b.kv().begin() + kSystemPrefix, b.kv().end());
    EXPECT_EQ(tail, flatten(kChunks, kChunks.size()));
}

// The definitive guard: a mid-speech SPECULATIVE hop, interrupted by a barge-in
// (real epoch bump) and rolled back, must leave the KV byte-identical to the
// no-speculation baseline — zero speculative residue.
TEST(MockVadStreaming, SpeculativeDraftRolledBackLeavesNoResidue) {
    MockVadStreamingBridge base;
    base.prefill_system_prompt(kSystemPrefix);
    for (const auto& c : kChunks) base.feed_chunk(c);        // reference

    MockVadStreamingBridge stream;
    stream.prefill_system_prompt(kSystemPrefix);
    for (std::size_t i = 0; i < kChunks.size(); ++i) {
        stream.feed_chunk(kChunks[i]);                      // committed hop
        if (i == 2) {                                       // mid-speech speculation
            const auto cp = stream.checkpoint();
            stream.feed_chunk({900, 901, 902});             // bogus draft (3 tokens)
            EXPECT_GT(stream.pos(), cp.pos);                // draft advanced the pointer

            stream.cancel_generation(i + 1);                // VAD barge-in: real epoch bump
            EXPECT_TRUE(stream.cancelled(i));               // older gen is now superseded

            stream.rollback(cp);                            // discard the draft
            EXPECT_EQ(stream.pos(), cp.pos);                // pointer restored exactly
        }
    }

    EXPECT_EQ(stream.pos(), base.pos());
    EXPECT_EQ(stream.kv(), base.kv());                      // ← "matches golden perfectly"
}

// Rollback must never drop below the frozen system prefix, even when asked to
// rewind past it (the effective_keep_tokens floor clamp).
TEST(MockVadStreaming, RollbackRespectsSystemPrefixFloor) {
    MockVadStreamingBridge b;
    b.prefill_system_prompt(kSystemPrefix);
    b.feed_chunk({1, 2});

    b.rollback({/*pos=*/10, /*verified_tokens=*/0});        // 10 < 44 floor
    EXPECT_EQ(b.pos(), static_cast<int>(kSystemPrefix));    // clamped up to the floor
    EXPECT_EQ(b.kv().size(), static_cast<std::size_t>(kSystemPrefix));
}

// The verified-prefix boundary is restored from the checkpoint on rollback.
TEST(MockVadStreaming, RollbackRestoresVerifiedTokenBoundary) {
    MockVadStreamingBridge b;
    b.prefill_system_prompt(kSystemPrefix);
    b.feed_chunk({7, 8});
    const MockVadStreamingBridge::KVCheckpoint cp{b.pos(), /*verified_tokens=*/2};

    b.feed_chunk({99});                                     // speculative
    b.rollback(cp);
    EXPECT_EQ(b.verified_tokens(), 2u);
    EXPECT_EQ(b.pos(), cp.pos);
}

// ---- Stateless text-context mode (Live Translator) --------------------------
// Every turn (audio soft-tokens + generated reply) is flushed after decode: N
// turns later the KV must be byte-identical to the freshly-prefilled prefix —
// zero accumulation, which is the whole point of the mode (no transliteration
// feedback loops, no silent dead-stop at max_context).
TEST(MockVadStreaming, StatelessModeFlushesEveryTurnWithNoResidue) {
    MockVadStreamingBridge b;
    b.prefill_system_prompt(kSystemPrefix);
    const std::vector<int> pristine = b.kv();

    for (int turn = 0; turn < 50; ++turn) {
        for (const auto& c : kChunks) b.feed_chunk(c);      // audio user turn
        b.feed_chunk({500 + turn, 501 + turn, 502 + turn}); // generated reply
        b.flush_turn();                                     // finalize_turn (stateless)
        EXPECT_EQ(b.pos(), static_cast<int>(kSystemPrefix));
        EXPECT_EQ(b.kv(), pristine);
    }
}

// ---- Bounded-history mode (Voice Assistant) ---------------------------------
// A barge-in rewind must clamp UP to the retained-history base, not just the
// system prefix: retained turns are committed context and must survive the
// micro-rewind (RealEngineControl::do_rewind's second clamp).
TEST(MockVadStreaming, BargeInRewindHonoursHistoryBase) {
    MockVadStreamingBridge b;
    b.prefill_system_prompt(kSystemPrefix);
    b.feed_chunk({70, 71, 72, 73});                         // a retained turn's tokens
    b.set_history_base(b.pos());                            // turn boundary advanced the base

    // A raw barge-in asks to keep only the verified prompt (often 0/the system
    // prefix); both must resolve to the history base, never below it.
    EXPECT_EQ(b.resolve_rewind_keep(0), static_cast<uint32_t>(kSystemPrefix) + 4u);
    EXPECT_EQ(b.resolve_rewind_keep(kSystemPrefix), static_cast<uint32_t>(kSystemPrefix) + 4u);
    // A keep ABOVE the base still wins (a rewind may keep more, never less).
    EXPECT_EQ(b.resolve_rewind_keep(kSystemPrefix + 10u), static_cast<uint32_t>(kSystemPrefix) + 10u);
}

// ---- CenterSlice geometry (append-only speculative center-slicing) ----------
// Pure-arithmetic pins for blackwell::audio::center_slice_plan and the
// pause-commit / resume bookkeeping AudioEmbeddingPipeline::center_hop /
// commit_pending_edge implement on top of it. Geometry: the validated defaults
// W=14, hop=2, L=2, K=3 (L+K+hop <= W holds).

namespace {

constexpr int kCsW   = 14;  // window_tokens
constexpr int kCsHop = 2;   // hop_tokens
constexpr int kCsK   = 3;   // right_edge_tokens (the pause-commit margin)

// The soft-tokens a window starting at absolute token w0 "projects": token i's
// identity IS its absolute index, so KV byte-identity across scenarios is exact.
std::vector<int> cs_window(int w0) {
    std::vector<int> v(kCsW);
    for (int i = 0; i < kCsW; ++i) v[static_cast<std::size_t>(i)] = w0 + i;
    return v;
}

// Apply one center hop to a mock stream: slice per center_slice_plan, feed the
// new rows, advance the committed counter exactly as center_hop does.
int cs_hop(MockVadStreamingBridge& b, int committed, int w0) {
    const auto p = blackwell::audio::center_slice_plan(committed, w0, kCsW, kCsK);
    const std::vector<int> win = cs_window(w0);
    b.feed_chunk(std::vector<int>(win.begin() + p.start_row,
                                  win.begin() + p.start_row + p.count));
    return std::max(committed, w0 + std::max(0, kCsW - kCsK));
}

}  // namespace

// Cold start: the first window commits its ENTIRE stable span [0, W-K) — the
// utterance opening has full left context by definition (no earlier audio), so
// no left-edge drop applies at row 0.
TEST(MockVadStreaming, CenterSliceColdStartCommitsWholeStableSpan) {
    const auto p = blackwell::audio::center_slice_plan(/*committed=*/0, /*w0=*/0,
                                                       kCsW, kCsK);
    EXPECT_EQ(p.start_row, 0);
    EXPECT_EQ(p.count, kCsW - kCsK);   // 11
}

// Warm hops: each slide by `hop` appends exactly hop new rows, contiguous with
// the committed history, and every appended row sits >= L rows from the window's
// left edge (the continuity invariant L+K+hop <= W at work).
TEST(MockVadStreaming, CenterSliceWarmHopsAppendExactlyHopTokens) {
    constexpr int kL = 2;              // left_edge_tokens of the validated defaults
    int committed = kCsW - kCsK;       // after the cold window
    for (int w0 = kCsHop; w0 <= 8; w0 += kCsHop) {
        const auto p = blackwell::audio::center_slice_plan(committed, w0, kCsW, kCsK);
        EXPECT_EQ(p.count, kCsHop);                    // exactly the hop, no more
        EXPECT_EQ(w0 + p.start_row, committed);        // contiguous, no gap/overlap
        EXPECT_GE(p.start_row, kL);                    // full left context in-window
        committed = std::max(committed, w0 + kCsW - kCsK);
    }
}

// A window too short to clear the right edge yields nothing (no negative counts).
TEST(MockVadStreaming, CenterSliceShortWindowYieldsNothing) {
    EXPECT_EQ(blackwell::audio::center_slice_plan(0, 0, kCsK, kCsK).count, 0);
    EXPECT_EQ(blackwell::audio::center_slice_plan(0, 0, kCsK - 1, kCsK).count, 0);
    // And a window already fully covered by the committed history yields nothing.
    EXPECT_EQ(blackwell::audio::center_slice_plan(/*committed=*/20, /*w0=*/2,
                                                  kCsW, kCsK).count, 0);
}

// The definitive CenterSlice guard: a VAD pause commits the K edge tokens, speech
// resumes (pointer-only rollback + counter un-commit), and streaming continues —
// the final KV must be byte-identical to the never-paused baseline: zero
// tentative-edge residue, and pos_ monotone during speech in both scenarios.
TEST(MockVadStreaming, CenterSlicePauseResumeLeavesNoEdgeResidue) {
    // Baseline: continuous speech across windows w0 = 0, 2, 4, 6 — append-only.
    MockVadStreamingBridge base;
    base.prefill_system_prompt(kSystemPrefix);
    int base_committed = 0;
    for (int w0 = 0; w0 <= 6; w0 += kCsHop) base_committed = cs_hop(base, base_committed, w0);

    // Paused stream: same speech, but a pause after w0 = 4 commits the K edge
    // tokens (checkpoint first), then speech resumes and w0 = 6 continues.
    MockVadStreamingBridge stream;
    stream.prefill_system_prompt(kSystemPrefix);
    int committed = 0;
    int prev_pos = stream.pos();
    for (int w0 = 0; w0 <= 4; w0 += kCsHop) {
        committed = cs_hop(stream, committed, w0);
        EXPECT_GE(stream.pos(), prev_pos);             // append-only during speech
        prev_pos = stream.pos();
    }

    // VAD pause: commit everything the last window (w0=4) still withholds —
    // exactly commit_pending_edge's slice [committed - w0, W).
    const auto cp = stream.checkpoint();               // pause_cp_ analogue
    const int pre_commit = committed;                  // pre_commit_committed_ analogue
    {
        const std::vector<int> win = cs_window(4);
        const int start_row = committed - 4;
        stream.feed_chunk(std::vector<int>(win.begin() + start_row, win.end()));
        committed = 4 + kCsW;
        EXPECT_EQ(stream.pos(), cp.pos + kCsK);        // exactly K tentative tokens
    }

    // Speech resumes: pointer-only rollback (resume_after_pause analogue).
    stream.rollback(cp);
    committed = pre_commit;                            // uncommit_pending_edge analogue
    EXPECT_EQ(stream.pos(), cp.pos);

    committed = cs_hop(stream, committed, 6);          // streaming continues

    EXPECT_EQ(committed, base_committed);
    EXPECT_EQ(stream.pos(), base.pos());
    EXPECT_EQ(stream.kv(), base.kv());                 // zero edge residue
}

// The budget pruning drops the OLDEST turns first and retains the maximal
// most-recent suffix that fits — exactly finalize_turn's deque arithmetic.
TEST(MockVadStreaming, HistoryBudgetPrunesOldestTurnsFirst) {
    std::deque<std::vector<int>> turns;
    turns.push_back(std::vector<int>(100, 1));   // oldest
    turns.push_back(std::vector<int>(90, 2));
    turns.push_back(std::vector<int>(80, 3));    // newest

    const std::size_t total = prune_to_budget(turns, /*budget=*/200);
    ASSERT_EQ(turns.size(), 2u);                 // the 100-token turn was evicted
    EXPECT_EQ(turns.front().front(), 2);         // retention is a most-recent suffix
    EXPECT_EQ(turns.back().front(), 3);
    EXPECT_EQ(total, 170u);

    // A budget smaller than the newest turn empties the history entirely (the
    // while loop has no "keep at least one" exemption — mirrors the impl).
    const std::size_t total2 = prune_to_budget(turns, /*budget=*/50);
    EXPECT_TRUE(turns.empty());
    EXPECT_EQ(total2, 0u);
}
