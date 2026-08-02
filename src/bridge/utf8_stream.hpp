#pragma once
// =============================================================================
// bridge/utf8_stream.hpp — reassembles UTF-8 code points across streaming chunk
// boundaries. Header-only, CUDA-free, no engine dependency.
//
// WHY THIS EXISTS. A tokenizer's vocabulary is built over BYTES, not characters,
// so a single multi-byte code point is routinely split across two consecutive
// tokens -- the norm, not an edge case, for Cyrillic, CJK, emoji, and any
// accented Latin text. Decoding those tokens one at a time therefore produces
// chunks that are individually INVALID UTF-8:
//
//     token N     "...\xD0"          <- lead byte of 'б', nothing else
//     token N+1   "\xB1..."          <- its continuation byte
//
// Concatenated they are correct; delivered separately they are two broken
// strings. Every consumer downstream then has to cope: a JSON serializer throws,
// a web view renders U+FFFD, a log file fills with question marks. The damage is
// permanent by the time it is visible, because the replacement character has
// already replaced the bytes that would have completed the sequence.
//
// So the split is healed HERE, at the boundary where the bytes are still intact:
// push() returns only the prefix that ends on a complete code point and holds
// the trailing partial sequence until the bytes that finish it arrive.
//
// WHAT IT IS NOT. This is not a validator. Genuinely malformed input (a stray
// continuation byte, a 5-byte lead) is passed through unchanged rather than
// held: the assembler must never grow without bound or swallow output because a
// model emitted a byte sequence no one can complete. At most 3 bytes are ever
// buffered, and flush() surrenders them at end of stream.
//
// CONCURRENCY. None, deliberately. One assembler belongs to one stream and is
// touched only by the thread decoding it (the engine thread, per CLAUDE.md).
// =============================================================================
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace blackwell::bridge {

// How many bytes the sequence starting with `b` occupies; 0 if `b` is not a
// legal lead byte (a continuation byte 10xxxxxx, or an invalid 11111xxx).
[[nodiscard]] inline constexpr int utf8_sequence_length(unsigned char b) noexcept {
    if (b < 0x80) return 1;                 // 0xxxxxxx  ASCII
    if ((b & 0xE0) == 0xC0) return 2;       // 110xxxxx
    if ((b & 0xF0) == 0xE0) return 3;       // 1110xxxx
    if ((b & 0xF8) == 0xF0) return 4;       // 11110xxx
    return 0;                               // 10xxxxxx continuation, or invalid
}

class Utf8StreamAssembler {
public:
    // Append `chunk` and return the longest prefix ending on a COMPLETE code
    // point. A trailing partial sequence is retained for the next call, so the
    // returned string is always valid UTF-8 whenever the input stream is.
    //
    // Returning empty is normal and means "this token was only part of a
    // character" -- the caller should emit nothing rather than an empty event.
    [[nodiscard]] std::string push(std::string_view chunk) {
        pending_.append(chunk);

        // Walk back from the end over at most 3 continuation bytes looking for
        // the lead byte of the final sequence. Bounded at 4 steps because that
        // is the longest legal sequence: past it the data is malformed, and the
        // loop falls through to "emit everything" rather than buffering junk.
        const std::size_t n = pending_.size();
        std::size_t i = n;
        std::size_t back = 0;
        while (i > 0 && back < 4) {
            --i;
            ++back;
            const int len = utf8_sequence_length(static_cast<unsigned char>(pending_[i]));
            if (len == 0) continue;   // continuation byte -- keep walking back
            if (len > static_cast<int>(back)) {
                // Lead byte of a sequence whose remaining bytes have not arrived.
                // Hand back everything before it and hold the fragment.
                std::string out = pending_.substr(0, i);
                pending_.erase(0, i);
                return out;
            }
            break;                    // the final sequence is complete
        }

        std::string out = std::move(pending_);
        pending_.clear();
        return out;
    }

    // End of stream: surrender any held fragment. Non-empty only when the
    // stream really did stop mid-character (a truncated generation), in which
    // case the bytes are the caller's to render or drop -- losing them silently
    // would be worse than one visible replacement glyph.
    [[nodiscard]] std::string flush() {
        std::string out = std::move(pending_);
        pending_.clear();
        return out;
    }

    // Drop any held fragment without emitting it. For starting a new generation:
    // a superseded turn's dangling bytes must not prepend themselves to the next
    // one's first token.
    void reset() noexcept { pending_.clear(); }

    [[nodiscard]] bool holding() const noexcept { return !pending_.empty(); }

private:
    std::string pending_;   // at most 3 bytes in well-formed operation
};

}  // namespace blackwell::bridge
