#pragma once
// -----------------------------------------------------------------------------
// RetranslationSession — the draft-and-commit loop itself
// (docs/CONTINUOUS_STREAMING.md). Turns the segmenter's Partial/Final events into
// the exact sequence of engine operations the architecture calls for, and keeps
// the KvLedger's map of the cache in step with them.
//
// ONE REDRAFT, in order:
//
//   1. LOCAL REWIND   the KV is rewound to the commit pointer C. This discards
//                     the previous draft's audio AND its generated text — bounded,
//                     O(1) (VRAMArena::truncate_kv is bookkeeping only), and the
//                     reason redrafting on every Partial is affordable at all.
//   2. TURN PREFIX    chat-template framing that opens a user turn.
//   3. AUDIO          the utterance's soft tokens, from its FIRST sample. The
//                     segmenter guarantees every Partial and the Final of one
//                     utterance share begin_sample, so this is always the whole
//                     utterance and never a splice of deltas.
//   4. TURN SUFFIX    close the user turn, open the assistant turn.
//   5. DECODE         generate the draft translation.
//
// A Final runs the identical five steps and then ADVANCES C, moving the draft out
// of the disposable zone and into committed history. That asymmetry is the entire
// difference between a draft and a commit: nothing else in the loop changes.
//
// ATOMICITY. A redraft is all-or-nothing against the ledger. Every step's token
// count is accumulated locally and only written to the ledger once the whole
// redraft has succeeded; on any failure the KV is rewound back to C and the
// ledger is left exactly where it started. A half-appended ledger would describe
// a cache that does not exist, and every later eviction plan would be computed
// against that fiction.
//
// EVICTION runs only immediately AFTER a commit, never mid-draft. At that instant
// tail == C, so there is no draft for it to interact with, and the "strictly
// behind C" rule holds trivially rather than by argument. The engine is asked
// first and the ledger updated only if the engine succeeded — the ledger must
// never claim a cut the cache did not make.
//
// THREADING: engine-thread only. Every call here performs engine work; the
// segments arrive from the audio thread through a SegmentJobSlot, which is the
// thread-crossing seam (CLAUDE.md single-threaded engine doctrine).
//
// DEPENDENCIES: STL + KvLedger + ContinuousStreamingConfig + SpeechSegment. No
// CUDA, no engine headers — the engine is reached through IRetranslationEngine,
// which is what makes the loop testable at desk speed.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <string>

#include "continuous_streaming_config.hpp"
#include "kv_ledger.hpp"
#include "speech_segmenter.hpp"

namespace blackwell::bridge {

// The engine operations a redraft needs, and nothing else. Implementations do the
// tokenizing, the chat templating and the CUDA; the loop above stays ignorant of
// all three.
//
// Every method returns the number of KV tokens it appended (0 is a legitimate
// answer), except rewind/evict which report success. A method that FAILS must
// leave the cache in a state the caller can recover from with rewind_to().
class IRetranslationEngine {
public:
    virtual ~IRetranslationEngine() = default;

    // Truncate the KV back to `pos` and set the engine's write cursor there.
    virtual bool rewind_to(std::uint32_t pos) = 0;

    // Chat-template framing that opens the user turn holding this utterance.
    virtual std::uint32_t prefill_turn_prefix() = 0;

    // The utterance's audio, ALWAYS from its first sample: encode [begin, end)
    // and prefill the resulting soft tokens. Returns the soft-token count.
    virtual std::uint32_t prefill_audio(std::uint64_t begin_sample,
                                        std::uint64_t end_sample) = 0;

    // Close the user turn and open the assistant turn.
    virtual std::uint32_t prefill_turn_suffix() = 0;

    // Generate the draft, appending its text to *out. Returns tokens generated.
    virtual std::uint32_t decode_draft(std::uint32_t max_new_tokens, std::string* out) = 0;

    // RoPE-aware head eviction over [keep_from, keep_from + delta) with the live
    // region currently `cache_len` long. Returns false if it could not be applied.
    virtual bool evict_head(std::uint32_t keep_from, std::uint32_t delta,
                            std::uint32_t cache_len) = 0;

    // Total KV capacity, so a redraft that cannot fit is refused before it starts
    // rather than running off the end of the cache mid-decode.
    virtual std::uint32_t context_capacity() const = 0;
};

struct RedraftResult {
    bool          ok             = false;  // the redraft completed and the ledger advanced
    bool          committed      = false;  // a Final advanced the commit pointer
    std::uint32_t audio_tokens   = 0;
    std::uint32_t text_tokens    = 0;
    std::uint32_t framing_tokens = 0;
    std::uint32_t evicted_tokens = 0;      // non-zero only on a commit
    std::uint32_t utterance_id   = 0;
    std::string   text;                    // the draft (or final) translation
};

class RetranslationSession {
public:
    // `engine` and `ledger` are non-owning and must outlive the session. The
    // ledger arrives already carrying the frozen prefix length, because the system
    // prompt is prefilled before a session exists.
    RetranslationSession(IRetranslationEngine* engine, KvLedger* ledger,
                         const ContinuousStreamingConfig& cfg,
                         std::uint32_t max_new_tokens = 128) noexcept
        : engine_(engine), ledger_(ledger), cfg_(cfg), max_new_tokens_(max_new_tokens) {
        cfg_.clamp();
    }

    // Drives one segment to completion. Partial redrafts; Final redrafts, commits,
    // and then considers eviction.
    RedraftResult on_segment(const vad::SpeechSegment& seg) {
        RedraftResult r;
        r.utterance_id = seg.utterance_id;
        if (engine_ == nullptr || ledger_ == nullptr) return r;

        // ---- 1. LOCAL REWIND -------------------------------------------------
        // Unconditional and idempotent: with no draft outstanding this is a no-op
        // that still leaves the cursor at C, so the loop needs no special case for
        // the first segment of an utterance.
        const std::uint32_t commit = ledger_->commit_point();
        if (!engine_->rewind_to(commit)) return r;
        ledger_->rewind_to_commit();

        // Refuse a redraft that cannot fit rather than discovering it mid-decode.
        // The draft's worst case is one maximal utterance plus the decode cap.
        if (commit + cfg_.draft_headroom_tokens(static_cast<int>(max_new_tokens_)) >
            engine_->context_capacity()) {
            ++overflow_refusals_;
            return r;
        }

        // ---- 2-5. framing, audio, framing, decode ----------------------------
        // Counts are accumulated locally and committed to the ledger only once the
        // whole redraft succeeded (see ATOMICITY above).
        const std::uint32_t prefix = engine_->prefill_turn_prefix();
        const std::uint32_t audio = engine_->prefill_audio(seg.begin_sample, seg.end_sample);
        if (audio == 0) {
            // No soft tokens means no utterance to translate. Not a failure — a
            // very short Partial can legitimately produce none — but there is
            // nothing to draft, so unwind to C and leave the ledger untouched.
            (void)engine_->rewind_to(commit);
            return r;
        }
        const std::uint32_t suffix = engine_->prefill_turn_suffix();

        std::string text;
        const std::uint32_t generated = engine_->decode_draft(max_new_tokens_, &text);

        // ---- ledger: one atomic batch ---------------------------------------
        ledger_->append(KvSpanKind::TurnMarker, prefix, seg.utterance_id);
        ledger_->append(KvSpanKind::Audio, audio, seg.utterance_id);
        ledger_->append(KvSpanKind::TurnMarker, suffix, seg.utterance_id);
        ledger_->append(KvSpanKind::Text, generated, seg.utterance_id);

        r.ok = true;
        r.audio_tokens = audio;
        r.text_tokens = generated;
        r.framing_tokens = prefix + suffix;
        r.text = std::move(text);

        if (seg.kind == vad::SegmentKind::Final) {
            // THE COMMIT. Everything above moves from the disposable zone into
            // committed history, and the next utterance drafts on top of it.
            ledger_->commit();
            r.committed = true;
            ++commits_;
            r.evicted_tokens = maybe_evict();
        } else {
            ++drafts_;
        }
        return r;
    }

    // ---- observables (any thread that owns the session) ----------------------
    std::uint64_t drafts() const noexcept { return drafts_; }
    std::uint64_t commits() const noexcept { return commits_; }
    std::uint64_t evictions() const noexcept { return evictions_; }
    std::uint64_t overflow_refusals() const noexcept { return overflow_refusals_; }

    void set_config(const ContinuousStreamingConfig& cfg) noexcept {
        cfg_ = cfg;
        cfg_.clamp();
    }
    const ContinuousStreamingConfig& config() const noexcept { return cfg_; }

private:
    // Called ONLY straight after a commit, where tail == C and no draft can be
    // disturbed. Reclaims down to the low watermark in one pass, so compactions
    // are rare and large instead of one per commit.
    std::uint32_t maybe_evict() {
        const std::uint32_t committed = ledger_->committed_tokens();
        if (!cfg_.should_evict(committed)) return 0;

        const KvLedger::EvictionPlan plan =
            ledger_->plan_evict_head(cfg_.eviction_min_delta(committed));
        if (!plan.viable) return 0;   // no turn boundary far enough back yet

        // Engine first: the ledger must never claim a cut the cache did not make.
        if (!engine_->evict_head(ledger_->frozen_prefix(), plan.delta, ledger_->tail()))
            return 0;
        if (!ledger_->apply_evict_head(plan)) return 0;

        ++evictions_;
        return plan.delta;
    }

    IRetranslationEngine*     engine_ = nullptr;   // non-owning
    KvLedger*                 ledger_ = nullptr;   // non-owning
    ContinuousStreamingConfig cfg_{};
    std::uint32_t             max_new_tokens_ = 128;

    std::uint64_t drafts_ = 0;
    std::uint64_t commits_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t overflow_refusals_ = 0;
};

}  // namespace blackwell::bridge
