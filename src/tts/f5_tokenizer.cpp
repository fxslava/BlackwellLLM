// -----------------------------------------------------------------------------
// F5Tokenizer implementation — see f5_tokenizer.hpp for the parity contract and
// for the two silent failure modes this file exists to prevent.
// -----------------------------------------------------------------------------
#include "f5_tokenizer.hpp"

#include <fstream>
#include <stdexcept>

namespace blackwell::tts {
namespace {

// f5_tts.model.utils.convert_char_to_pinyin's `custom_trans`, transcribed. The
// only part of that function that has any effect on non-CJK text (measured, not
// assumed — see the header block). Applied BEFORE the vocab lookup, exactly as
// Python applies it before list_str_to_idx.
std::uint32_t apply_custom_trans(std::uint32_t cp) {
    switch (cp) {
        case 0x003B: return 0x002C;   // ';'  -> ','
        case 0x201C: return 0x0022;   // '"'  left  double curly -> '"'
        case 0x201D: return 0x0022;   // '"'  right double curly -> '"'
        case 0x2018: return 0x0027;   // '''  left  single curly -> '\''
        case 0x2019: return 0x0027;   // '''  right single curly -> '\''
        default:     return cp;
    }
}

// Minimal UTF-8 decoder. Returns the codepoint and advances `i`; on a malformed
// or truncated sequence it consumes ONE byte and reports kBadCodepoint, so a
// corrupt input degrades to a few unknown characters instead of desynchronising
// the whole stream (or looping forever).
constexpr std::uint32_t kBadCodepoint = 0xFFFFFFFFu;

std::uint32_t next_codepoint(std::string_view s, std::size_t& i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const std::size_t n = s.size();

    auto cont = [&](std::size_t k) {
        return i + k < n && (static_cast<unsigned char>(s[i + k]) & 0xC0u) == 0x80u;
    };
    auto tail = [&](std::size_t k) {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + k]) & 0x3Fu);
    };

    if (b0 < 0x80u) {                                   // 0xxxxxxx
        i += 1;
        return b0;
    }
    if ((b0 & 0xE0u) == 0xC0u && cont(1)) {             // 110xxxxx (Cyrillic lives here)
        const std::uint32_t cp = ((b0 & 0x1Fu) << 6) | tail(1);
        i += 2;
        return cp;
    }
    if ((b0 & 0xF0u) == 0xE0u && cont(1) && cont(2)) {  // 1110xxxx
        const std::uint32_t cp = ((b0 & 0x0Fu) << 12) | (tail(1) << 6) | tail(2);
        i += 3;
        return cp;
    }
    if ((b0 & 0xF8u) == 0xF0u && cont(1) && cont(2) && cont(3)) {  // 11110xxx
        // 4-byte sequences are not hypothetical here: the shipped vocab.txt
        // contains U+20BB6, so a decoder that stopped at 3 bytes would both
        // mis-read the vocab and mis-read any input above the BMP.
        const std::uint32_t cp = ((b0 & 0x07u) << 18) | (tail(1) << 12) |
                                 (tail(2) << 6) | tail(3);
        i += 4;
        return cp;
    }
    i += 1;
    return kBadCodepoint;
}

// Encode back to UTF-8, for reporting unknown characters to a human.
std::string encode_utf8(std::uint32_t cp) {
    std::string out;
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
    return out;
}

}  // namespace

F5Tokenizer::F5Tokenizer(const std::string& vocab_path) {
    std::ifstream f(vocab_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("F5Tokenizer: cannot open vocab: " + vocab_path);
    }

    std::string line;
    std::int32_t id = 0;
    while (std::getline(f, line)) {
        // Strip BOTH terminators. getline already consumed '\n'; the '\r' of a
        // CRLF file survives, and the shipped vocab.txt IS CRLF. Leaving it on
        // makes every key "<char>\r", which matches nothing and silently turns
        // all text into the unknown id. See the header block.
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }

        // One codepoint per line is the only usable form: F5 looks the vocab up
        // one CHARACTER at a time, so a multi-codepoint line could never match
        // there either. Such lines still consume their id (the ids are line
        // indices and must not shift) but are not registered.
        std::size_t i = 0;
        if (!line.empty()) {
            const std::uint32_t cp = next_codepoint(line, i);
            if (cp != kBadCodepoint && i == line.size()) {
                map_.emplace(cp, id);
            }
        }
        ++id;
    }

    size_ = static_cast<std::size_t>(id);
    if (size_ == 0) {
        throw std::runtime_error("F5Tokenizer: vocab is empty: " + vocab_path);
    }

    // THE MIS-PARSE GUARD. get_tokenizer asserts vocab_char_map[" "] == 0, so
    // every valid F5 vocab has a space on line 0. If the file was read with the
    // wrong newline handling, the wrong encoding, or is simply not an F5 vocab,
    // this is the cheap check that catches it — and it catches the CRLF bug
    // above exactly, since " \r" is not " ".
    const auto it = map_.find(0x20u);
    if (it == map_.end() || it->second != 0) {
        throw std::runtime_error(
            "F5Tokenizer: '" + vocab_path + "' does not look like an F5 vocab: entry 0 "
            "must be the space character (get_tokenizer asserts vocab_char_map[' '] == 0). "
            "Read " + std::to_string(size_) + " lines. If the file looks right, suspect "
            "line endings or encoding — every F5 character id is a LINE INDEX, so a "
            "mis-parse shifts or voids the whole table and the model then speaks nonsense "
            "without any error.");
    }
}

std::int32_t F5Tokenizer::IdForCodepoint(std::uint32_t cp) const {
    const auto it = map_.find(cp);
    return it == map_.end() ? kUnknownId : it->second;
}

void F5Tokenizer::TokenizeInto(std::string_view utf8_text, std::vector<std::int32_t>& out,
                               std::size_t* out_unknown) const {
    out.clear();
    out.reserve(utf8_text.size());   // upper bound: 1 codepoint per byte

    std::size_t unknown = 0;
    for (std::size_t i = 0; i < utf8_text.size();) {
        const std::uint32_t raw = next_codepoint(utf8_text, i);
        if (raw == kBadCodepoint) {
            // Malformed bytes are not silently dropped: they are exactly as
            // unpronounceable as an out-of-vocabulary character, and counting
            // them is what tells a caller its encoding is wrong.
            out.push_back(kUnknownId);
            ++unknown;
            continue;
        }
        const std::uint32_t cp = apply_custom_trans(raw);
        const auto it = map_.find(cp);
        if (it == map_.end()) {
            out.push_back(kUnknownId);
            ++unknown;
        } else {
            out.push_back(it->second);
        }
    }
    if (out_unknown != nullptr) *out_unknown = unknown;
}

std::vector<std::int32_t> F5Tokenizer::Tokenize(std::string_view utf8_text,
                                               std::size_t* out_unknown) const {
    std::vector<std::int32_t> out;
    TokenizeInto(utf8_text, out, out_unknown);
    return out;
}

std::vector<std::string> F5Tokenizer::UnknownCharacters(std::string_view utf8_text) const {
    std::vector<std::string> out;
    std::vector<std::uint32_t> seen;
    for (std::size_t i = 0; i < utf8_text.size();) {
        const std::uint32_t raw = next_codepoint(utf8_text, i);
        const std::uint32_t cp = (raw == kBadCodepoint) ? raw : apply_custom_trans(raw);
        if (cp != kBadCodepoint && map_.count(cp) != 0) continue;
        bool dup = false;
        for (const std::uint32_t s : seen) {
            if (s == cp) { dup = true; break; }
        }
        if (dup) continue;
        seen.push_back(cp);
        out.push_back(cp == kBadCodepoint ? std::string("<malformed utf-8>") : encode_utf8(cp));
    }
    return out;
}

}  // namespace blackwell::tts
