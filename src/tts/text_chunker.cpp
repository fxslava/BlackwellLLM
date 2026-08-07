// -----------------------------------------------------------------------------
// TextChunker implementation — see text_chunker.hpp for the punctuation-only
// argument, the first-chunk rule and the UTF-8 contract.
// -----------------------------------------------------------------------------
#include "text_chunker.hpp"

#include <cstdint>

namespace blackwell::tts {
namespace {

constexpr std::uint32_t kEmDash = 0x2014;   // U+2014, 3 bytes in UTF-8
constexpr std::uint32_t kEnDash = 0x2013;   // U+2013, the one people paste instead

enum class Scan { Ok, Incomplete, Malformed };

// Decodes the codepoint at `i`. Distinguishes INCOMPLETE (a valid prefix that
// needs more bytes — the normal case at an LLM token boundary) from MALFORMED
// (bytes that can never be valid). f5_tokenizer.cpp has a similar decoder but
// deliberately not the same one: there, a truncated tail is garbage to be
// counted, whereas here it is data that has not arrived yet, and conflating the
// two would either corrupt characters or stall the stream forever.
Scan decode(std::string_view s, std::size_t i, std::uint32_t& cp, std::size_t& len) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const std::size_t avail = s.size() - i;

    auto cont = [&](std::size_t k) {
        return (static_cast<unsigned char>(s[i + k]) & 0xC0u) == 0x80u;
    };
    auto tail = [&](std::size_t k) {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + k]) & 0x3Fu);
    };

    if (b0 < 0x80u) { cp = b0; len = 1; return Scan::Ok; }
    if ((b0 & 0xE0u) == 0xC0u) {
        if (avail < 2) return Scan::Incomplete;
        if (!cont(1)) return Scan::Malformed;
        cp = ((b0 & 0x1Fu) << 6) | tail(1); len = 2; return Scan::Ok;
    }
    if ((b0 & 0xF0u) == 0xE0u) {
        if (avail < 3) {
            // Only "incomplete" if what we DO have is a valid prefix; otherwise
            // the stream is broken and waiting would hang.
            for (std::size_t k = 1; k < avail; ++k) if (!cont(k)) return Scan::Malformed;
            return Scan::Incomplete;
        }
        if (!cont(1) || !cont(2)) return Scan::Malformed;
        cp = ((b0 & 0x0Fu) << 12) | (tail(1) << 6) | tail(2); len = 3; return Scan::Ok;
    }
    if ((b0 & 0xF8u) == 0xF0u) {
        if (avail < 4) {
            for (std::size_t k = 1; k < avail; ++k) if (!cont(k)) return Scan::Malformed;
            return Scan::Incomplete;
        }
        if (!cont(1) || !cont(2) || !cont(3)) return Scan::Malformed;
        cp = ((b0 & 0x07u) << 18) | (tail(1) << 12) | (tail(2) << 6) | tail(3);
        len = 4; return Scan::Ok;
    }
    return Scan::Malformed;
}

inline bool is_space(std::uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
}

// STRONG: the end of a thought. ';' sits here rather than with the commas
// because it separates independent clauses -- prosodically it is a full stop
// with a shorter tail, not a pause inside a phrase.
inline bool is_sentence_end(std::uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == ';' || cp == '\n';
}

// CLAUSE: a pause inside a sentence. No ASCII '-' -- see ChunkerConfig.
inline bool is_clause_end(std::uint32_t cp) {
    return cp == ',' || cp == ':' || cp == kEmDash || cp == kEnDash;
}

inline bool is_ascii_digit(char c) {
    return c >= '0' && c <= '9';
}

// A letter for the abbreviation guard: ASCII alpha or the Cyrillic block. Only
// used to answer "is the thing before this dot a single letter", so the coarse
// range is enough and a full Unicode table would be dead weight.
inline bool is_letter(std::uint32_t cp) {
    if (cp < 0x80u) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (cp == 0x401u || cp == 0x451u) return true;             // Ё ё
    return cp >= 0x410u && cp <= 0x44Fu;                       // А..я
}

// Decodes the codepoint ENDING at byte `pos` (exclusive). Returns false when
// there is none or the bytes do not form one -- callers treat that as "no
// letter there", which is the safe answer for every guard below.
bool decode_prev(std::string_view s, std::size_t pos, std::uint32_t& cp, std::size_t& start) {
    if (pos == 0) return false;
    std::size_t b = pos - 1;
    // Walk back over continuation bytes to the lead byte (max 3 steps for
    // well-formed UTF-8; the bound also stops a corrupt run from walking away).
    std::size_t steps = 0;
    while (b > 0 && (static_cast<unsigned char>(s[b]) & 0xC0u) == 0x80u && steps < 3) {
        --b;
        ++steps;
    }
    std::size_t len = 0;
    if (decode(s, b, cp, len) != Scan::Ok) return false;
    if (b + len != pos) return false;   // not a clean codepoint ending here
    start = b;
    return true;
}

// Anything that carries sound. A span of pure punctuation ("," on its own,
// which the first-chunk rule can otherwise produce from a stray leading comma)
// is not worth a synthesis and tokenises to little more than a pause.
bool has_speakable(std::string_view s) {
    for (std::size_t i = 0; i < s.size();) {
        std::uint32_t cp = 0;
        std::size_t len = 0;
        const Scan st = decode(s, i, cp, len);
        if (st != Scan::Ok) { ++i; continue; }
        i += len;
        if (cp < 0x80u) {
            if ((cp >= '0' && cp <= '9') || is_letter(cp)) return true;
            continue;
        }
        // Non-ASCII: letters and CJK are speakable; the dash/quote/ellipsis
        // ranges are the punctuation we deliberately keep for prosody.
        if (cp >= 0x2010u && cp <= 0x205Eu) continue;   // general punctuation
        if (cp == 0x00A0u) continue;                    // NBSP
        return true;
    }
    return false;
}

}  // namespace

TextChunker::TextChunker(ChunkerConfig cfg) : cfg_(cfg) {
    // A zero floor would let a lone "," become an utterance in the middle of a
    // reply. That is a configuration mistake rather than an interesting edge
    // case, so clamp instead of failing a stream at runtime.
    if (cfg_.min_chunk_chars == 0) cfg_.min_chunk_chars = 1;
}

void TextChunker::PushToken(std::string_view token) {
    if (token.empty()) return;
    buf_.append(token.data(), token.size());
    Extract();
}

// The two guards that keep punctuation-only splitting from cutting a number or
// an abbreviation in half. Both are cheap and both are LOCAL: they look at the
// codepoint before and the byte after, never at the sentence.
TextChunker::Guard TextChunker::GuardAt(std::uint32_t cp, std::size_t at,
                                        std::size_t next) const {
    // ---- digits: "3.14", "1,5", "12:30" ------------------------------------
    // Only these three can appear between digits. The check is on RAW BYTES
    // because an ASCII digit can never be a UTF-8 continuation byte, so
    // "previous byte is a digit" is exact without decoding.
    if (cp == '.' || cp == ',' || cp == ':') {
        const bool digit_before = at > 0 && is_ascii_digit(buf_[at - 1]);
        if (digit_before) {
            // The answer depends on a byte that may not have arrived. WAIT --
            // guessing "split" would hand the normaliser "3." and "14" as two
            // numbers, and guessing "keep" would stall a real sentence end.
            // Flush() is what resolves this at end of stream.
            if (next >= buf_.size()) return Guard::NeedMore;
            if (is_ascii_digit(buf_[next])) return Guard::Skip;
        }
    }

    // ---- initials and abbreviations: "т. д.", "г. Москва", "И. Иванов" -----
    // A dot straight after a SINGLE letter that is itself at a word start is
    // not a sentence end. Deliberately not a dictionary: the shape is what
    // identifies these, and a list would be wrong for the next language.
    if (cp == '.') {
        std::uint32_t prev = 0;
        std::size_t prev_start = 0;
        if (decode_prev(buf_, at, prev, prev_start) && is_letter(prev)) {
            std::uint32_t before = 0;
            std::size_t before_start = 0;
            const bool have_before = decode_prev(buf_, prev_start, before, before_start);
            if (!have_before || is_space(before) || before == '(' || before == '"') {
                return Guard::Skip;
            }
        }
    }

    return Guard::Take;
}

// A stress mark binds to ONE vowel, and which side it sits on is a per-voice
// convention (stress_marker.hpp): '+' may precede its vowel ("р+ыбка", what
// RUAccent and the Russian F5 finetunes use) or follow it ("ры+бка"). Rather
// than know which, refuse to end a chunk on either side of a '+' -- that is
// correct under both, and costs one byte comparison.
//
// A split here would not merely look wrong. The two halves become separate
// UTTERANCES: one ends with a dangling '+' the model reads as a token of its
// own, the other opens with a vowel that has silently lost its stress.
//
// ONLY THE VALVE CALLS THIS, and that is not an oversight. A punctuation split
// point sits immediately after a `.` `,` `;` `:` `!` `?` or dash, so the byte
// before it is never a '+' -- and a chunk that BEGINS with '+' has kept the
// mark together with the vowel that follows it. The punctuation path therefore
// cannot orphan a mark, and making it ask would add a one-token stall to every
// chunk boundary in the common case, which is TTFA paid for nothing.
TextChunker::Guard TextChunker::SplitAllowedAt(std::size_t pos) const {
    if (pos > 0 && buf_[pos - 1] == '+') return Guard::Skip;
    // The byte AFTER the split point decides the other convention, and it may
    // not have arrived. Wait rather than guess -- Flush() resolves it at end of
    // stream, and one token of delay is invisible next to a synthesis.
    if (pos >= buf_.size()) return Guard::NeedMore;
    if (buf_[pos] == '+') return Guard::Skip;
    return Guard::Take;
}

bool TextChunker::AcceptsSplit(Boundary kind, std::size_t chunk_cps) const noexcept {
    // The first chunk of a reply is the only one whose synthesis the user waits
    // through. Everything after it is produced while the previous one plays, so
    // the floor buys prosody at no latency cost -- but here it would buy
    // prosody with the one delay the listener actually perceives.
    if (cfg_.first_chunk_asap && emitted_in_stream_ == 0) return true;
    if (kind == Boundary::Strong) return true;      // a complete thought
    return chunk_cps >= cfg_.min_chunk_chars;
}

void TextChunker::Extract() {
    for (;;) {
        // Resumes where the last scan stopped. buf_ only ever grows at the tail
        // between emissions, so everything before scan_bytes_ is settled.
        std::size_t i = scan_bytes_;
        std::size_t cps = scan_cps_;
        std::size_t split_at = 0;
        bool found = false;

        while (i < buf_.size()) {
            std::uint32_t cp = 0;
            std::size_t len = 0;
            const Scan s = decode(buf_, i, cp, len);
            if (s == Scan::Incomplete) break;      // wait for the rest of the char
            if (s == Scan::Malformed) {
                // Consume one byte so a corrupt stream cannot wedge the scanner.
                // It reaches the tokenizer as one unknown character.
                cp = 0xFFFDu;
                len = 1;
            }
            const std::size_t next = i + len;

            const Boundary kind =
                (cfg_.split_on_sentence_ends && is_sentence_end(cp)) ? Boundary::Strong
                : (cfg_.split_on_commas && is_clause_end(cp))        ? Boundary::Clause
                                                                     : Boundary::None;
            if (kind != Boundary::None) {
                const Guard g = GuardAt(cp, i, next);
                if (g == Guard::NeedMore) break;   // undecidable until more bytes
                if (g == Guard::Take && AcceptsSplit(kind, cps + 1)) {
                    split_at = next;               // keep the punctuation
                    found = true;
                    break;
                }
            }

            // ---- the safety valve, not a chunking rule (see ChunkerConfig) --
            // Checked BEFORE the cursor advances, exactly like the punctuation
            // boundary above: a NeedMore must leave i/cps pointing AT this
            // codepoint so the next token re-examines it rather than skipping
            // the split point entirely.
            const std::size_t cps_after = cps + 1;
            // Preferred form: break at whitespace, so the cut still lands
            // between words.
            const bool over_guard =
                cfg_.runaway_guard_chars != 0 && cps_after >= cfg_.runaway_guard_chars;
            // Last resort: no whitespace has appeared at all. A mid-word cut is
            // accepted here because the alternative is buffering without bound
            // and speaking nothing.
            const bool over_hard_cap = cfg_.runaway_hard_cap_chars != 0 &&
                                       cps_after >= cfg_.runaway_hard_cap_chars;

            if ((over_guard && is_space(cp)) || over_hard_cap) {
                const Guard g = SplitAllowedAt(next);
                if (g == Guard::NeedMore) break;   // the deciding byte is missing
                if (g == Guard::Take) {
                    split_at = next;
                    found = true;
                    break;
                }
                // Skip: this position would orphan a stress mark. Keep scanning;
                // the next whitespace (or codepoint) is offered on the following
                // iteration, so the valve fires a character or two later.
            }

            i = next;
            ++cps;
        }

        if (!found) {
            // Park the cursor exactly where the scan stopped: on an incomplete
            // codepoint or on an undecided boundary, both of which must be
            // re-examined against the next token rather than skipped.
            scan_bytes_ = i;
            scan_cps_ = cps;
            return;
        }
        Emit(std::string_view(buf_).substr(0, split_at));
        buf_.erase(0, split_at);
        scan_bytes_ = 0;
        scan_cps_ = 0;
    }
}

void TextChunker::Emit(std::string_view span) {
    // Trim ASCII whitespace only. Every byte of a multi-byte codepoint has the
    // high bit set, so this can never cut into one.
    std::size_t b = 0;
    std::size_t e = span.size();
    while (b < e && (span[b] == ' ' || span[b] == '\t' ||
                     span[b] == '\n' || span[b] == '\r')) ++b;
    while (e > b && (span[e - 1] == ' ' || span[e - 1] == '\t' ||
                     span[e - 1] == '\n' || span[e - 1] == '\r')) --e;
    if (e <= b) return;   // whitespace-only: nothing to speak

    const std::string_view trimmed = span.substr(b, e - b);
    // Punctuation with no word in it is not an utterance. This is what stops a
    // reply opening with ", " from spending a whole synthesis on a comma under
    // the first-chunk rule.
    if (!has_speakable(trimmed)) return;

    pending_.emplace_back(trimmed);
    ++total_emitted_;
    ++emitted_in_stream_;
}

void TextChunker::Flush() {
    if (!buf_.empty()) {
        // Emitted whole, including a trailing partial codepoint if the stream
        // ended mid-character -- see the header. The length floor does not
        // apply: this is the last of the reply and dropping it would silently
        // truncate speech.
        Emit(buf_);
        buf_.clear();
        scan_bytes_ = 0;
        scan_cps_ = 0;
    }
    // Re-arm the first-chunk rule LAST: the next token belongs to the next
    // reply, whose opening clause deserves the same latency treatment this one
    // got. Emit() has just incremented the counter, so this must follow it.
    emitted_in_stream_ = 0;
}

void TextChunker::Reset() noexcept {
    buf_.clear();
    pending_.clear();
    scan_bytes_ = 0;
    scan_cps_ = 0;
    emitted_in_stream_ = 0;
}

std::string TextChunker::PopChunk() {
    if (pending_.empty()) return std::string();
    std::string out = std::move(pending_.front());
    pending_.pop_front();
    return out;
}

}  // namespace blackwell::tts
