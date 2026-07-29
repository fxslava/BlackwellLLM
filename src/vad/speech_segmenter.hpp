#pragma once
// -----------------------------------------------------------------------------
// SpeechSegmenter — turns a stream of per-block speech PROBABILITIES into the two
// events the re-translation (draft-and-commit) pipeline runs on. See
// docs/CONTINUOUS_STREAMING.md.
//
//   Partial — the utterance SO FAR, emitted on a fixed cadence while the speaker
//             is still going. The consumer rewinds the KV cache to the commit
//             pointer C, re-feeds this whole range, and re-drafts.
//   Final   — the utterance COMPLETE, emitted once the release hangover expires.
//             The consumer rewinds to C one last time, feeds this range, decodes,
//             and only then advances C past it.
//
// THE GROWING-WINDOW INVARIANT — the property that makes re-translation correct:
// every Partial of an utterance, AND its Final, carry the SAME begin_sample. Only
// the right edge moves. A consumer that always rewinds to C before feeding can
// therefore never double-count audio and never needs to diff two ranges against
// each other: what it is handed is always the complete utterance from its first
// sample. Breaking this turns every redraft into a splice, which is precisely the
// failure mode this architecture exists to avoid.
//
// WHY SEGMENTS AND NOT BLOCKS. Gating the ring per 10 ms block would let a
// keyboard clack between two words excise a phoneme, and Whisper's encoder is
// bidirectional over a Conv1D receptive field — it would then rebuild that word
// from two spliced acoustic contexts. Real pauses do not need the help: Silero
// holds p >= 0.95 across the reference clip's 3.24-3.32 s inter-word pause (see
// tests/vad/silero_vad_test.cpp). So the gate is per UTTERANCE, with a pre-roll
// at the onset and a hangover at the release.
//
// TIME AND COORDINATES. Sample-accurate, never wall-clock: the caller drives
// on_block() once per fixed-size VAD block and the segmenter's own clock advances
// by exactly that block. Every position is an ABSOLUTE sample index since
// construction (or the last reset()), which is the same coordinate the audio ring
// is indexed by. uint64_t: at 16 kHz this does not wrap in any realistic session.
//
// THREADING: not thread-safe and not required to be. One owner — the DSP worker
// that already owns the SileroVAD — calls on_block()/reset(); the observers are
// for that same thread.
//
// DEPENDENCIES: STL only. No CUDA, no ONNXRuntime, no engine. It consumes
// probabilities, so it is equally drivable by Silero, by the built-in RMS
// detector, or by a scripted fake in a test.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cstdint>
#include <optional>

namespace blackwell::vad {

enum class SegmentKind : std::uint8_t {
    Partial,  // redraft point: the utterance so far
    Final,    // commit point: the utterance complete
};

// A half-open absolute sample range [begin_sample, end_sample).
struct SpeechSegment {
    SegmentKind   kind         = SegmentKind::Partial;
    std::uint64_t begin_sample = 0;   // inclusive; INCLUDES the pre-roll
    std::uint64_t end_sample   = 0;   // exclusive
    std::uint32_t utterance_id = 0;   // 1-based; shared by an utterance's Partials + Final

    std::uint64_t samples() const noexcept { return end_sample - begin_sample; }
};

struct SegmenterConfig {
    int sample_rate   = 16000;
    int block_samples = 160;     // 10 ms — the pipeline's VAD block size

    // Hysteresis, deliberately the SAME shape as SpeechPipelineController's
    // probability path: one sensitivity knob, release derived from it. An
    // utterance STARTS above onset_threshold and SUSTAINS above the derived
    // release, so a probability hovering at the threshold cannot chatter.
    float onset_threshold = 0.5f;
    float release_drop    = 0.15f;
    float release_floor   = 0.05f;

    // Audio prepended before the onset block. The encoder is weakest at a
    // window's left edge (Phase 6), so it must never be handed a segment that
    // starts on the first phoneme. Clamped so it can never reach back into audio
    // a previous Final already committed.
    int preroll_ms = 250;

    // Continuous sub-release audio that ends an utterance.
    int hangover_ms = 400;

    // Audio kept PAST the last speech block in a Final. Not the whole hangover:
    // that would pad every utterance with the full silence that ended it.
    int tail_pad_ms = 200;

    // Redraft cadence while speaking. Every Partial costs a rewind + re-prefill
    // + re-decode, so this is a GPU-budget knob, not a detection knob.
    int partial_period_ms = 500;

    // Bursts with less speech than this emit NOTHING — no Partial, no Final.
    // Door slams and single keystrokes must not open a turn.
    int min_utterance_ms = 200;

    // A speaker who never pauses still has to be committed eventually: the ring
    // and the encoder both have ceilings. On reaching this the utterance is cut
    // and, if speech is still going, the next one continues CONTIGUOUSLY from the
    // cut (no pre-roll — the audio never stopped, so there is no cold edge).
    int max_utterance_ms = 20000;
};

class SpeechSegmenter {
public:
    explicit SpeechSegmenter(const SegmenterConfig& cfg = {}) noexcept : cfg_(cfg) {
        block_        = static_cast<std::uint64_t>(cfg_.block_samples < 1 ? 1 : cfg_.block_samples);
        preroll_      = to_samples(cfg_.preroll_ms);
        hangover_     = to_samples(cfg_.hangover_ms);
        tail_pad_     = to_samples(cfg_.tail_pad_ms);
        partial_      = to_samples(cfg_.partial_period_ms);
        min_speech_   = to_samples(cfg_.min_utterance_ms);
        max_span_     = to_samples(cfg_.max_utterance_ms);
        release_      = std::max(cfg_.onset_threshold - cfg_.release_drop, cfg_.release_floor);
    }

    // Feed EXACTLY one VAD block's speech probability. At most one event per
    // block: a Final is only reachable from silence and a Partial only from
    // speech, so the two can never collide.
    std::optional<SpeechSegment> on_block(float probability) noexcept {
        const std::uint64_t block_begin = stream_pos_;
        const std::uint64_t block_end   = stream_pos_ + block_;
        stream_pos_ = block_end;

        const bool onset   = probability > cfg_.onset_threshold;
        const bool sustain = probability > release_;

        if (!speaking_) {
            if (!onset) return std::nullopt;
            start_utterance(block_begin, block_end);
        }

        if (sustain) {
            speech_samples_ += block_;
            last_speech_end_ = block_end;
            silence_run_ = 0;
        } else {
            silence_run_ += block_;
        }

        // The cap is checked BEFORE the hangover: an utterance that reaches it has
        // to be cut on this block regardless of what the last few blocks scored.
        if (block_end - begin_ >= max_span_) return force_final(block_end, sustain);
        if (silence_run_ >= hangover_) return end_utterance(block_end);

        // Redraft cadence. Gated on the same min_utterance_ms as the Final, so a
        // burst that ends up being discarded as a blip never emitted a Partial
        // either — there is no draft for the consumer to unwind.
        if (speech_samples_ >= min_speech_ && block_end - last_partial_end_ >= partial_) {
            last_partial_end_ = block_end;
            return SpeechSegment{SegmentKind::Partial, begin_, block_end, utterance_id_};
        }
        return std::nullopt;
    }

    // Audio discontinuity (mode flip, seek, device change). Drops any in-flight
    // utterance without emitting: its audio is gone, so a Final over it would
    // name a range the ring can no longer produce.
    void reset() noexcept {
        speaking_ = false;
        stream_pos_ = 0;
        committed_end_ = 0;
        begin_ = 0;
        last_speech_end_ = 0;
        last_partial_end_ = 0;
        speech_samples_ = 0;
        silence_run_ = 0;
        utterance_id_ = 0;
    }

    bool          in_speech() const noexcept { return speaking_; }
    std::uint64_t stream_position() const noexcept { return stream_pos_; }
    std::uint64_t committed_end() const noexcept { return committed_end_; }
    std::uint32_t utterance_id() const noexcept { return utterance_id_; }
    float         release_threshold() const noexcept { return release_; }
    const SegmenterConfig& config() const noexcept { return cfg_; }

private:
    std::uint64_t to_samples(int ms) const noexcept {
        const std::int64_t clamped = ms < 0 ? 0 : ms;
        return static_cast<std::uint64_t>(clamped) *
               static_cast<std::uint64_t>(cfg_.sample_rate) / 1000ull;
    }

    void start_utterance(std::uint64_t block_begin, std::uint64_t block_end) noexcept {
        speaking_ = true;
        ++utterance_id_;
        const std::uint64_t pre = block_begin > preroll_ ? block_begin - preroll_ : 0;
        // The pre-roll may not reach into audio a previous Final already covered:
        // re-feeding committed samples would duplicate them in the KV cache.
        begin_            = std::max(pre, committed_end_);
        last_speech_end_  = block_end;
        last_partial_end_ = block_end;
        speech_samples_   = 0;
        silence_run_      = 0;
    }

    // Hangover expiry — the ordinary end of an utterance.
    std::optional<SpeechSegment> end_utterance(std::uint64_t now_end) noexcept {
        speaking_ = false;
        if (speech_samples_ < min_speech_) return std::nullopt;   // blip
        const std::uint64_t end = std::min(last_speech_end_ + tail_pad_, now_end);
        committed_end_ = end;
        return SpeechSegment{SegmentKind::Final, begin_, end, utterance_id_};
    }

    // max_utterance_ms reached. Hard cut at the cap with no tail pad (there is no
    // trailing silence to pad WITH — the speaker is still going).
    std::optional<SpeechSegment> force_final(std::uint64_t now_end, bool sustain) noexcept {
        speaking_ = false;
        if (speech_samples_ < min_speech_) return std::nullopt;
        committed_end_ = now_end;
        const SpeechSegment out{SegmentKind::Final, begin_, now_end, utterance_id_};
        if (sustain) {
            // Contiguous continuation. start_utterance clamps begin_ to
            // committed_end_, so the new utterance starts exactly where this one
            // ended: no gap, no overlap, and no pre-roll to double-count.
            start_utterance(now_end, now_end);
        }
        return out;
    }

    SegmenterConfig cfg_{};

    // Resolved from cfg_ once (ms -> samples), so the hot path does no division.
    std::uint64_t block_      = 160;
    std::uint64_t preroll_    = 0;
    std::uint64_t hangover_   = 0;
    std::uint64_t tail_pad_   = 0;
    std::uint64_t partial_    = 0;
    std::uint64_t min_speech_ = 0;
    std::uint64_t max_span_   = 0;
    float         release_    = 0.0f;

    bool          speaking_         = false;
    std::uint64_t stream_pos_       = 0;   // absolute; advances one block per call
    std::uint64_t committed_end_    = 0;   // end of the last emitted Final
    std::uint64_t begin_            = 0;   // current utterance start (with pre-roll)
    std::uint64_t last_speech_end_  = 0;   // end of the last sustaining block
    std::uint64_t last_partial_end_ = 0;   // right edge of the last Partial
    std::uint64_t speech_samples_   = 0;   // sustaining audio in this utterance
    std::uint64_t silence_run_      = 0;   // consecutive sub-release audio
    std::uint32_t utterance_id_     = 0;
};

}  // namespace blackwell::vad
