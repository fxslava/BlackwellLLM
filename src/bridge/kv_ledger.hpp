#pragma once
// -----------------------------------------------------------------------------
// KvLedger — the token-level map of the streaming KV cache, and the bookkeeping
// half of the re-translation (draft-and-commit) loop. See
// docs/CONTINUOUS_STREAMING.md.
//
// THREE ZONES, in KV position order:
//
//   [0, S)        FROZEN PREFIX  the system prompt. Never rewound, never evicted.
//                                It is also the attention sink that keeps RoPE
//                                phase anchored at true position 0, which is what
//                                lets head eviction stay transparent to the model.
//   [S, C)        COMMITTED      finalized utterances and their translations. The
//                                ONLY zone head eviction may touch.
//   [C, tail)     DRAFT          the in-flight re-translation. Disposable: thrown
//                                away and rebuilt on every Partial.
//
// C is the COMMIT POINTER. Each Partial rewinds the KV cache to exactly C and
// re-feeds the utterance from its first sample; the Final does the same once more
// and then advances C past it. The model therefore always reads a well-formed
// [prefix][committed turns][one whole utterance] — it is never shown a fragment
// and never asked to revise its own draft.
//
// THIS CLASS HOLDS NO KV DATA. It decides WHERE the cache should be cut; the
// caller performs the cut. The two pairings are contractual:
//
//   rewind_to_commit()        pair with IKVCacheManager::rewind(seq, <returned pos>)
//                             (-> VRAMArena::truncate_kv: bookkeeping only, O(1),
//                              no data movement — which is what makes redrafting
//                              on every Partial affordable at all)
//   plan/apply_evict_head()   pair with the RoPE-aware compaction kernel (T3)
//
// Call the ledger side only once the cache side has succeeded, so a failed cut
// cannot leave the map describing a cache that was never changed.
//
// WHY SPANS AND NOT A TOKEN COUNT. Eviction has to cut on a TURN BOUNDARY or it
// strands a dangling <|start_header_id|> or half an utterance in the context.
// A running total cannot answer "where may I cut?"; an ordered span list answers
// it in one scan and costs nothing to maintain.
//
// THREADING: not thread-safe. Owned by the single engine thread — the same thread
// that performs the rewinds and prefills it describes (CLAUDE.md single-threaded
// engine doctrine).
//
// DEPENDENCIES: STL only. No CUDA, no engine, no bridge ABI.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <vector>

namespace blackwell::bridge {

enum class KvSpanKind : std::uint8_t {
    TurnMarker,  // chat-template scaffolding; the ONLY legal head-eviction cut point
    Audio,       // Ultravox soft tokens
    Text,        // decoded translation folded back into the context on commit
};

// A half-open KV position range [begin, begin + len).
struct KvSpan {
    std::uint32_t begin        = 0;
    std::uint32_t len          = 0;
    KvSpanKind    kind         = KvSpanKind::Text;
    std::uint32_t utterance_id = 0;

    std::uint32_t end() const noexcept { return begin + len; }
};

class KvLedger {
public:
    // frozen_prefix_tokens is what prefill_system_prompt() returned: the ledger
    // starts empty with C == tail == S, so the first append lands in the draft.
    explicit KvLedger(std::uint32_t frozen_prefix_tokens = 0)
        : frozen_(frozen_prefix_tokens),
          commit_(frozen_prefix_tokens),
          tail_(frozen_prefix_tokens) {}

    // ---- draft construction -------------------------------------------------
    // Appends at the tail. Zero-length is a no-op rather than an error: a chunk
    // that produced no soft tokens is a normal outcome, not a bug to assert on.
    void append(KvSpanKind kind, std::uint32_t len, std::uint32_t utterance_id = 0) {
        if (len == 0) return;
        spans_.push_back(KvSpan{tail_, len, kind, utterance_id});
        tail_ += len;
        ++epoch_;
    }

    // ---- the two re-translation primitives ----------------------------------
    // Drop the draft. Returns the KV position to rewind the cache to (== C), so
    // the call site reads: kv.rewind(seq, ledger.rewind_to_commit()).
    //
    // Idempotent: with no draft outstanding this is a no-op that still returns C,
    // so a redraft loop does not need to test has_draft() first.
    std::uint32_t rewind_to_commit() noexcept {
        if (tail_ == commit_) return commit_;    // no-op: leave the epoch alone
        while (!spans_.empty() && spans_.back().begin >= commit_) spans_.pop_back();
        tail_ = commit_;
        ++epoch_;
        return commit_;
    }

    // The draft becomes permanent. Only ever called for a Final — a Partial's
    // audio and its generated text are both discarded by the next rewind, which
    // is exactly why a wrong early guess cannot poison the session.
    void commit() noexcept {
        if (commit_ == tail_) return;            // no-op: leave the epoch alone
        commit_ = tail_;
        ++epoch_;
    }

    // ---- head eviction (background; strictly behind C) ----------------------
    struct EvictionPlan {
        bool          viable        = false;
        std::uint32_t delta         = 0;  // tokens dropped from [S, S + delta)
        std::uint32_t cut           = 0;  // absolute cut point (== S + delta)
        std::uint32_t spans_dropped = 0;
        // The ledger state this plan describes. Every position in the plan is
        // absolute, so ANY intervening mutation invalidates it — including the
        // plan's own application, which is what stops a spent plan from being
        // replayed (nothing structural about it goes stale on its own).
        std::uint64_t epoch = 0;
    };

    // Smallest turn boundary at or past S + min_delta that still lies within the
    // committed zone. Not viable when there is no such boundary — the caller then
    // has no legal cut and must wait for more commits (or raise max_context).
    //
    // The cut lands ON a TurnMarker's first token, so what survives eviction
    // begins with a well-formed turn header.
    EvictionPlan plan_evict_head(std::uint32_t min_delta) const noexcept {
        EvictionPlan plan{};
        plan.epoch = epoch_;
        std::uint32_t dropped = 0;
        for (const KvSpan& s : spans_) {
            // Never cut into the draft. `>` not `>=`: cutting exactly at C is
            // legal and simply evicts the whole committed zone.
            if (s.begin > commit_) break;
            if (s.kind == KvSpanKind::TurnMarker && s.begin > frozen_ &&
                s.begin - frozen_ >= min_delta) {
                plan.viable = true;
                plan.delta = s.begin - frozen_;
                plan.cut = s.begin;
                plan.spans_dropped = dropped;
                return plan;
            }
            ++dropped;
        }
        return plan;
    }

    // Applies a plan from plan_evict_head() on THIS ledger with no mutation in
    // between. Returns false on a stale or malformed plan rather than corrupting
    // the map — the caller is mid-session on the engine thread and a throw here
    // would unwind the decode loop (CLAUDE.md runtime tier).
    bool apply_evict_head(const EvictionPlan& plan) noexcept {
        if (!plan.viable || plan.delta == 0) return false;
        if (plan.epoch != epoch_) return false;                 // stale
        if (plan.cut != frozen_ + plan.delta) return false;     // malformed
        if (plan.cut > commit_) return false;
        if (plan.spans_dropped > spans_.size()) return false;

        spans_.erase(spans_.begin(),
                     spans_.begin() + static_cast<std::ptrdiff_t>(plan.spans_dropped));
        for (KvSpan& s : spans_) s.begin -= plan.delta;
        commit_ -= plan.delta;
        tail_ -= plan.delta;
        evicted_ += plan.delta;
        ++epoch_;
        return true;
    }

    // ---- observers ----------------------------------------------------------
    std::uint32_t frozen_prefix() const noexcept { return frozen_; }
    std::uint32_t commit_point() const noexcept { return commit_; }
    std::uint32_t tail() const noexcept { return tail_; }
    std::uint32_t draft_tokens() const noexcept { return tail_ - commit_; }
    bool          has_draft() const noexcept { return tail_ > commit_; }
    std::uint32_t committed_tokens() const noexcept { return commit_ - frozen_; }
    std::uint64_t evicted_tokens() const noexcept { return evicted_; }
    const std::vector<KvSpan>& spans() const noexcept { return spans_; }

    // Session restart. Keeps the frozen prefix: the system prompt's KV is still
    // resident and still correct, so re-prefilling it would be pure waste.
    void reset() noexcept {
        spans_.clear();
        commit_ = frozen_;
        tail_ = frozen_;
        evicted_ = 0;
        ++epoch_;
    }

private:
    std::vector<KvSpan> spans_;          // ordered by begin; spans never straddle C
    std::uint32_t       frozen_  = 0;    // S
    std::uint32_t       commit_  = 0;    // C
    std::uint32_t       tail_    = 0;
    std::uint64_t       evicted_ = 0;    // cumulative, for telemetry
    std::uint64_t       epoch_   = 0;    // bumped on every state change; see EvictionPlan
};

}  // namespace blackwell::bridge
