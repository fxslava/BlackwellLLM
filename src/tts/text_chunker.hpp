#pragma once
// -----------------------------------------------------------------------------
// TextChunker — splits a streaming LLM token feed into speakable spans, ON
// PUNCTUATION AND NOTHING ELSE.
//
// WHY THIS EXISTS. F5-TTS is not a streaming model: the DiT denoises a whole
// mel jointly, so no prefix of the audio exists until the last solver step, and
// the total duration is decided before any sample is produced. Time-to-first-
// audio therefore equals full synthesis time for whatever text you hand it. The
// only lever is to hand it LESS text, sooner. Splitting a four-sentence reply on
// sentence boundaries turns TTFB from a function of the whole reply into a
// function of the first clause, and the rest synthesises in the shadow of
// playback (synthesis runs several times faster than realtime, so the sink does
// not starve after the first chunk).
//
// =============================================================================
// PUNCTUATION IS THE ONLY BOUNDARY. THERE IS NO CHARACTER-COUNT SPLIT.
// =============================================================================
// An earlier version also broke at `max_chunk_chars` (at the next space) and at
// a hard codepoint cap. Both are gone, and their absence is the design:
//
//   * A cut taken at "the next space after 150 characters" lands MID-PHRASE by
//     construction -- it knows nothing about syntax. F5 re-anchors the voice to
//     the reference clip at every chunk boundary, so a mid-phrase cut is heard
//     as the sentence restarting: pitch resets, the trailing word is clipped
//     short, and the seam is exactly where a listener expects continuity.
//   * A boundary chosen at punctuation is a boundary the SPEAKER would have
//     taken anyway. Re-conditioning across it is close to inaudible, because a
//     comma is already a place where prosody resets.
//
// So the rule is: emit at `.` `!` `?` `;` `\n` (STRONG) and at `,` `:` `—` `–`
// (CLAUSE). Those are the only boundaries a chunk is CHOSEN at.
//
// There is one exception and it is a valve rather than a policy: text that has
// produced no punctuation for `runaway_guard_chars` is broken at the next
// whitespace. Punctuation-free output is routine for an LLM (enumerations,
// code, a run-on answer), and buffering it until EndOfStream means the user
// hears nothing at all for the whole generation -- a worse outcome than one
// break in an odd place. The valve still never lands mid-word, and never
// between a stress mark and its vowel (see SplitAllowedAt).
//
// STRESS MARKS SURVIVE CHUNK BOUNDARIES. Marked text can reach this class --
// the user typed it, or the model emitted it -- and a '+' separated from the
// vowel it binds to is not a cosmetic problem: the two halves land in different
// utterances, so the mark becomes a bare '+' token in one chunk and the vowel
// loses its stress in the other. No split is ever taken adjacent to a '+',
// under either placement convention.
//
// FIRST CHUNK GOES AT THE FIRST PUNCTUATION MARK, WHATEVER IT IS. The first
// chunk is the only one whose synthesis the user waits through -- every later
// one is produced while the previous plays. So the length floor is suspended
// for it: "Да," becomes an utterance on purpose, because 300 ms of "Да," while
// the rest generates beats 2 s of silence. From the second chunk on the floor
// applies again (min_chunk_chars), since by then latency is hidden and the only
// thing left to optimise is prosody.
//
// NUMBERS AND ABBREVIATIONS ARE NOT SENTENCE ENDS. "3.14", "12:30" and "1,5"
// contain characters from the boundary set between digits, and "т. д." ends a
// token that is not a sentence. Splitting there would hand the normaliser
// (text_normalizer.hpp) half a number, which it would then read as two. Both
// are guarded; the digit guard WAITS for the next codepoint rather than
// guessing, so a '.' arriving as the last byte of a token holds until the token
// after it settles the question.
//
// UTF-8 IS HANDLED AT THE BYTE LEVEL, DELIBERATELY. An LLM stream splits
// wherever the tokenizer decided, which routinely lands MID-CODEPOINT for
// Cyrillic (2 bytes/char) — a token can end with the first byte of "и". So the
// scanner distinguishes "incomplete, wait for more bytes" from "malformed", and
// never emits a chunk that ends inside a codepoint. Splitting on a byte count
// instead would corrupt roughly one character per token boundary, and the
// downstream tokenizer would map each to the unknown id without complaint.
//
// All lengths are in CODEPOINTS, not bytes. min_chunk_chars = 20 against a
// Cyrillic sentence is 20 characters, not 10.
//
// THREADING: NOT thread-safe, by design. One producer thread (whatever consumes
// the LLM stream) calls PushToken/Flush; the same thread pops. Reset() races
// everything, so a barge-in arriving on another thread must marshal — which is
// exactly what TTSDuplexBridge does with its mutex. Putting a lock in here would
// tax the common path for a case the layer above already has to serialise.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>

namespace blackwell::tts {

struct ChunkerConfig {
    // Strong boundaries: '.', '!', '?', ';', '\n'. A chunk ending here is a
    // complete thought, so re-conditioning the voice across it is barely
    // audible -- which is why these are NOT subject to min_chunk_chars.
    bool split_on_sentence_ends = true;

    // Clause boundaries: ',', ':', U+2014 em dash, U+2013 en dash. These cut
    // latency further but split WITHIN a sentence, where a prosody
    // discontinuity is more noticeable, so they ARE subject to the floor below.
    // Turn off if the voice sounds choppy.
    //
    // ASCII '-' is deliberately NOT here: in Russian it is a word-internal
    // hyphen far more often than a dash ("кто-то", "из-за"), and splitting
    // there produces a chunk ending mid-word.
    bool split_on_commas = true;

    // Floor for CLAUSE splits, from the second chunk of a reply onward. Without
    // it, "Так," in the middle of a sentence becomes its own utterance: a
    // ~200 ms synthesis whose reference-conditioning overhead dwarfs its
    // content, and which lands as a clipped bark. 20 is roughly a short clause.
    //
    // Does NOT apply to strong boundaries (a complete short sentence is a
    // legitimate utterance) and does not apply to the first chunk at all.
    std::size_t min_chunk_chars = 20;

    // Suspend the floor for the FIRST chunk after a reset/flush, so the reply
    // starts speaking at its first punctuation mark of any kind. This is the
    // time-to-first-audio knob; turning it off costs a clause of latency at the
    // start of every reply and buys slightly smoother opening prosody.
    bool first_chunk_asap = true;

    // THE SAFETY VALVE, and it is ON. Punctuation-free output is not a
    // pathology in an LLM, it is a routine shape: enumerations, code, a
    // transcript read back, a run-on answer. Without a valve none of it is
    // spoken until EndOfStream, so the user hears nothing for the whole
    // generation and then the entire reply at once -- which is precisely the
    // latency chunking exists to remove.
    //
    // At this many buffered codepoints the chunker splits at the NEXT
    // WHITESPACE, so the break still lands between words. 180 is about ten
    // seconds of Russian speech: long enough that ordinary punctuated prose
    // never reaches it (a sentence end always fires first), short enough that
    // the wait is a pause rather than an outage. 0 disables it.
    std::size_t runaway_guard_chars = 180;

    // The valve's own valve: text with NO whitespace at all (a pasted URL, a
    // CJK run, a decode that lost the space token) would otherwise sail past
    // the guard above and buffer without bound. At this many codepoints the
    // split is taken at the next codepoint boundary, accepting a mid-word cut
    // because the alternative is no audio at all. Must be > runaway_guard_chars
    // to mean anything; 0 disables it.
    std::size_t runaway_hard_cap_chars = 400;
};

class TextChunker {
public:
    explicit TextChunker(ChunkerConfig cfg = ChunkerConfig{});

    // Appends one LLM stream token and extracts whatever chunks became complete.
    // Safe to call with a token that ends mid-codepoint; the partial bytes are
    // retained until the rest arrives.
    void PushToken(std::string_view token);

    bool HasPendingChunk() const noexcept { return !pending_.empty(); }

    // Oldest complete chunk, or "" when none. Whitespace-trimmed; punctuation is
    // preserved, because it is what the model reads as prosody.
    std::string PopChunk();

    // End of stream: emit whatever is buffered as a final chunk, even if it is
    // shorter than min_chunk_chars. Any trailing incomplete codepoint is emitted
    // as-is rather than dropped — at end-of-stream it is genuinely truncated
    // input, and the tokenizer counts it as one unknown character instead of
    // silently losing it. Also re-arms the first-chunk rule for the next reply.
    void Flush();

    // Drops the buffer AND every queued chunk, immediately. This is barge-in:
    // the user talked over the assistant, so nothing still unspoken should ever
    // be spoken. Not a graceful drain — that would be exactly wrong here.
    void Reset() noexcept;

    // ---- observers -----------------------------------------------------------
    std::size_t pending_chunks() const noexcept { return pending_.size(); }
    std::size_t buffered_bytes() const noexcept { return buf_.size(); }
    const ChunkerConfig& config() const noexcept { return cfg_; }

    // Total chunks emitted since construction (not reduced by Pop or Reset).
    std::size_t total_emitted() const noexcept { return total_emitted_; }

private:
    // What a boundary codepoint is worth. Strong ends a thought; Clause ends a
    // phrase inside one and is what the length floor guards.
    enum class Boundary { None, Clause, Strong };

    // Whether a boundary candidate may be taken, or whether the decision needs
    // bytes that have not arrived (the digit guard around "3.14").
    enum class Guard { Take, Skip, NeedMore };

    // Pulls every chunk the current buffer can complete.
    void Extract();
    // Trims whitespace and queues; drops spans with nothing speakable in them.
    void Emit(std::string_view span);
    // Decides whether the punctuation at [at, next) really ends a span.
    Guard GuardAt(std::uint32_t cp, std::size_t at, std::size_t next) const;
    // Whether a chunk may end at byte `pos` without cutting a stress mark off
    // its vowel. Returns NeedMore when the byte after `pos` has not arrived,
    // because that byte is what decides it.
    Guard SplitAllowedAt(std::size_t pos) const;
    // Applies the length policy: first chunk free, strong always, clause floored.
    bool AcceptsSplit(Boundary kind, std::size_t chunk_cps) const noexcept;

    ChunkerConfig cfg_;
    std::string buf_;                  // bytes not yet emitted
    std::deque<std::string> pending_;  // complete chunks, oldest first
    std::size_t total_emitted_ = 0;

    // Chunks emitted since the last Reset/Flush. Only "is this the first one"
    // is ever asked, but a counter costs the same as a flag and reads better.
    std::size_t emitted_in_stream_ = 0;

    // Resume point for Extract(): bytes/codepoints of buf_ already scanned with
    // no boundary found. Without it, appending N tokens rescans the buffer N
    // times, which with no character cap is O(reply^2) rather than O(reply).
    // Always reset to 0 when buf_ is modified anywhere but the tail.
    std::size_t scan_bytes_ = 0;
    std::size_t scan_cps_ = 0;
};

}  // namespace blackwell::tts
