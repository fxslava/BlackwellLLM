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
//   2. TURN PREFIX    chat-template framing that opens a user turn. USUALLY FREE:
//                     see THE RESIDENT TURN PREFIX below.
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
// THE RESIDENT TURN PREFIX (the 30% the profile handed us).
// Step 2's tokens are byte-identical on every redraft of every utterance, yet the
// naive loop re-prefilled them each time. Measured in Release on the 8B backbone
// the redraft is linear in TOTAL draft tokens at ~20.5 ms/token, so those ~29
// framing tokens cost ~600 ms PER REDRAFT — 30% of a short draft's whole budget,
// spent recomputing a constant.
//
// So the prefix is prefilled ONCE, immediately after each commit, and lands in the
// COMMITTED zone rather than the draft zone. A redraft then rewinds to a C that
// already ends with an open user turn, and step 2 is a no-op. The committed
// sequence is unchanged — [prefix][audio][suffix][text] per turn, exactly as
// before — the prefix is simply written one commit early.
//
// Two consequences worth stating because they look like bugs otherwise:
//   * C ends with a DANGLING open user turn between utterances. That is deliberate
//     and it is what makes the rewind target correct.
//   * arming runs AFTER maybe_evict(), never before, so eviction still happens at
//     the instant tail == C and its "no draft can be disturbed" argument holds
//     unchanged.
// If arming ever fails (a faulted engine returning 0), resident_prefix_ stays 0
// and the next redraft silently takes the old cold path. Self-healing by
// construction: the optimization can degrade, never corrupt.
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
    // Framing tokens prefilled INTO THE DRAFT by this redraft. Normally just the
    // suffix: the prefix is resident below C (see THE RESIDENT TURN PREFIX). The
    // invariant draft_tokens == framing + audio + text holds either way.
    std::uint32_t framing_tokens = 0;
    // Framing tokens the redraft did NOT have to pay for because they were already
    // resident. This is the optimization's yield, reported rather than inferred.
    std::uint32_t resident_prefix_tokens = 0;
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
        //
        // Step 2 is skipped entirely when the prefix is already resident below C —
        // the rewind above landed on top of it. Zero is the FAST path here, not a
        // failure; `prefix` is what this redraft actually paid for.
        const std::uint32_t prefix =
            resident_prefix_ != 0 ? 0u : engine_->prefill_turn_prefix();
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
        // The resident prefix is NOT re-appended: it was recorded when it was armed
        // and already lives below C. Appending it again would double-count it and
        // desync the ledger from the cache.
        if (prefix != 0) ledger_->append(KvSpanKind::TurnMarker, prefix, seg.utterance_id);
        ledger_->append(KvSpanKind::Audio, audio, seg.utterance_id);
        ledger_->append(KvSpanKind::TurnMarker, suffix, seg.utterance_id);
        ledger_->append(KvSpanKind::Text, generated, seg.utterance_id);

        r.ok = true;
        r.audio_tokens = audio;
        r.text_tokens = generated;
        r.framing_tokens = prefix + suffix;
        r.resident_prefix_tokens = resident_prefix_;
        r.text = std::move(text);

        if (seg.kind == vad::SegmentKind::Final) {
            // THE COMMIT. Everything above moves from the disposable zone into
            // committed history, and the next utterance drafts on top of it.
            ledger_->commit();
            r.committed = true;
            ++commits_;
            r.evicted_tokens = maybe_evict();
            // Arm the NEXT turn's prefix. Strictly after eviction, so eviction
            // still runs at the instant tail == C.
            arm_turn_prefix();
        } else {
            ++drafts_;
        }
        return r;
    }

    // Arm the FIRST turn prefix, before any segment arrives. Optional: without it
    // utterance 1 takes the cold path and pays for its own framing on every
    // redraft, and every utterance after it is armed by its predecessor's commit.
    // Not folded into the constructor because arming calls the engine, and the
    // engine is only ready once the system prompt has been prefilled — a
    // sequencing requirement a constructor cannot enforce.
    void begin_session() { arm_turn_prefix(); }

    // Framing tokens currently resident below C (0 == the next redraft pays).
    std::uint32_t resident_prefix() const noexcept { return resident_prefix_; }

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

    // Prefill the next turn's framing into the COMMITTED zone, so the redrafts
    // that follow rewind on top of it instead of recomputing it. A zero return
    // (faulted engine) leaves resident_prefix_ at 0 and the next redraft takes the
    // cold path — the optimization degrades, it never corrupts.
    void arm_turn_prefix() {
        resident_prefix_ = 0;
        if (engine_ == nullptr || ledger_ == nullptr) return;
        const std::uint32_t n = engine_->prefill_turn_prefix();
        if (n == 0) return;
        // utterance_id 0: this framing belongs to the utterance that has not
        // arrived yet, so it is attributed to none of them.
        ledger_->append(KvSpanKind::TurnMarker, n, /*utterance_id=*/0);
        ledger_->commit();
        resident_prefix_ = n;
    }

    IRetranslationEngine*     engine_ = nullptr;   // non-owning
    KvLedger*                 ledger_ = nullptr;   // non-owning
    ContinuousStreamingConfig cfg_{};
    std::uint32_t             max_new_tokens_ = 128;

    std::uint32_t resident_prefix_ = 0;   // framing tokens already below C
    std::uint64_t drafts_ = 0;
    std::uint64_t commits_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t overflow_refusals_ = 0;
};

}  // namespace blackwell::bridge
