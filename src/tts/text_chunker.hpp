#pragma once
// -----------------------------------------------------------------------------
// TextChunker — splits a streaming LLM token feed into speakable spans.
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
// The cost is real and worth stating: each chunk is re-conditioned on the
// reference audio rather than on what was just spoken, so prosody continuity
// breaks at chunk boundaries. Splitting more aggressively lowers latency and
// flattens intonation. That is what min_chunk_chars exists to bound.
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
#include <deque>
#include <string>
#include <string_view>

namespace blackwell::tts {

struct ChunkerConfig {
    // Hard boundaries: '.', '!', '?', '\n'. A chunk ending here is a complete
    // thought, and re-conditioning the voice across it is barely audible.
    bool split_on_sentence_ends = true;

    // Soft boundaries: ',', ';', ':', '-', U+2014 em dash. These cut latency
    // further but split WITHIN a sentence, where a prosody discontinuity is more
    // noticeable. Turn off if the voice sounds choppy.
    bool split_on_commas = true;

    // A split is only taken once the pending chunk has at least this many
    // codepoints. Without it, "Да," or "Так," becomes its own utterance: a
    // ~200 ms synthesis whose reference-conditioning overhead dwarfs its
    // content, and which lands as a clipped bark. 20 is roughly a short clause.
    std::size_t min_chunk_chars = 20;

    // Once the pending chunk reaches this many codepoints, split at the NEXT
    // whitespace — never mid-word. A run-on with no punctuation would otherwise
    // buffer until end-of-stream and give back the very latency chunking exists
    // to remove.
    std::size_t max_chunk_chars = 150;

    // Safety valve for input with no whitespace at all (a pasted URL, a CJK run,
    // a decode that ran away). At this many codepoints the chunker splits on the
    // next codepoint boundary regardless. Not a tuning knob: it exists so a
    // pathological stream cannot buffer without bound. 0 disables it.
    std::size_t hard_cap_chars = 400;
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
    // silently losing it.
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
    // Pulls every chunk the current buffer can complete.
    void Extract();
    // Trims whitespace and queues; drops whitespace-only spans.
    void Emit(std::string_view span);

    ChunkerConfig cfg_;
    std::string buf_;                  // bytes not yet emitted
    std::deque<std::string> pending_;  // complete chunks, oldest first
    std::size_t total_emitted_ = 0;
};

}  // namespace blackwell::tts
