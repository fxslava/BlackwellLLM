// -----------------------------------------------------------------------------
// TextChunker implementation — see text_chunker.hpp for the latency argument and
// the UTF-8 contract.
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
inline bool is_sentence_end(std::uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == '\n';
}
inline bool is_clause_end(std::uint32_t cp) {
    return cp == ',' || cp == ';' || cp == ':' || cp == '-' ||
           cp == kEmDash || cp == kEnDash;
}

}  // namespace

TextChunker::TextChunker(ChunkerConfig cfg) : cfg_(cfg) {
    // A zero minimum would let a lone "," become an utterance; a zero maximum
    // would split every codepoint. Both are configuration mistakes rather than
    // interesting edge cases, so clamp instead of failing a stream at runtime.
    if (cfg_.max_chunk_chars == 0) cfg_.max_chunk_chars = 1;
    if (cfg_.min_chunk_chars > cfg_.max_chunk_chars) {
        cfg_.min_chunk_chars = cfg_.max_chunk_chars;
    }
}

void TextChunker::PushToken(std::string_view token) {
    if (token.empty()) return;
    buf_.append(token.data(), token.size());
    Extract();
}

void TextChunker::Extract() {
    // Rescans from the start each time. That is O(buffer) per token and the
    // buffer is bounded by max_chunk_chars (plus one word), so it is a few
    // hundred bytes — not worth the state a resumable cursor would cost.
    for (;;) {
        std::size_t i = 0;      // byte cursor
        std::size_t cps = 0;    // codepoints consumed so far
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
            i += len;
            ++cps;

            // A split point only counts once the chunk is long enough to be worth
            // speaking on its own -- this is what stops "Да," from becoming an
            // utterance.
            const bool boundary =
                (cfg_.split_on_sentence_ends && is_sentence_end(cp)) ||
                (cfg_.split_on_commas && is_clause_end(cp));
            if (boundary && cps >= cfg_.min_chunk_chars) {
                split_at = i;                      // keep the punctuation
                found = true;
                break;
            }

            // Over the soft cap: break at the next whitespace, never mid-word.
            if (cps >= cfg_.max_chunk_chars && is_space(cp)) {
                split_at = i;
                found = true;
                break;
            }

            // Over the hard cap: no whitespace has appeared at all, so split on
            // this codepoint boundary rather than buffer without bound.
            if (cfg_.hard_cap_chars != 0 && cps >= cfg_.hard_cap_chars) {
                split_at = i;
                found = true;
                break;
            }
        }

        if (!found) return;
        Emit(std::string_view(buf_).substr(0, split_at));
        buf_.erase(0, split_at);
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

    pending_.emplace_back(span.substr(b, e - b));
    ++total_emitted_;
}

void TextChunker::Flush() {
    if (buf_.empty()) return;
    // Emitted whole, including a trailing partial codepoint if the stream ended
    // mid-character -- see the header. min_chunk_chars does not apply: this is
    // the last of the reply and dropping it would silently truncate speech.
    Emit(buf_);
    buf_.clear();
}

void TextChunker::Reset() noexcept {
    buf_.clear();
    pending_.clear();
}

std::string TextChunker::PopChunk() {
    if (pending_.empty()) return std::string();
    std::string out = std::move(pending_.front());
    pending_.pop_front();
    return out;
}

}  // namespace blackwell::tts
