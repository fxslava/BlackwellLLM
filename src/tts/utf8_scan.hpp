#pragma once
// -----------------------------------------------------------------------------
// utf8_scan.hpp — the codepoint walk shared by the two text stages that need
// the SAME error contract: text_normalizer.cpp and stress_marker.cpp.
//
// WHY NOT ONE DECODER FOR THE WHOLE FRONTEND. There are three error contracts
// in this directory and they are genuinely different, so folding them together
// would mean a decoder with a mode flag -- which is how the wrong mode gets
// picked:
//
//   text_chunker.cpp   must distinguish INCOMPLETE (bytes that have not
//                      arrived yet, the normal case at an LLM token boundary)
//                      from MALFORMED. It is the only stage reading a live
//                      stream, and conflating the two either corrupts
//                      characters or stalls the stream forever.
//   f5_tokenizer.cpp   counts a truncated tail as one unknown character,
//                      because by then the text is final and the count is a
//                      diagnostic.
//   HERE               a malformed byte is copied through unchanged, one byte
//                      at a time (`ok == false`, `len == 1`). These stages
//                      rewrite text; code-point integrity is the chunker's
//                      contract, and a stage that silently ate bytes would
//                      hide a break in it.
//
// Header-only, no state, no allocation beyond what the caller's string does.
// Cyrillic-specific by design: the classifiers below answer questions about
// Russian orthography (is this a vowel, what is its lowercase form), which is
// what both consumers are actually asking.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace blackwell::tts::utf8 {

struct Char {
    std::uint32_t cp = 0;    // the codepoint, or the raw byte when !ok
    std::size_t   len = 1;   // bytes consumed; ALWAYS >= 1 so scans advance
    bool          ok = false;
};

// One codepoint at `i`, or the single raw byte there when it does not start a
// valid sequence. Returns len == 1 / ok == false at or past the end.
inline Char Decode(std::string_view s, std::size_t i) noexcept {
    if (i >= s.size()) return {};
    const auto b0 = static_cast<unsigned char>(s[i]);
    std::size_t need = 0;
    std::uint32_t cp = 0;
    if (b0 < 0x80u)                 { return {b0, 1, true}; }
    else if ((b0 & 0xE0u) == 0xC0u) { need = 2; cp = b0 & 0x1Fu; }
    else if ((b0 & 0xF0u) == 0xE0u) { need = 3; cp = b0 & 0x0Fu; }
    else if ((b0 & 0xF8u) == 0xF0u) { need = 4; cp = b0 & 0x07u; }
    else                            { return {b0, 1, false}; }

    if (s.size() - i < need) return {b0, 1, false};
    for (std::size_t k = 1; k < need; ++k) {
        const auto bk = static_cast<unsigned char>(s[i + k]);
        if ((bk & 0xC0u) != 0x80u) return {b0, 1, false};
        cp = (cp << 6) | (bk & 0x3Fu);
    }
    return {cp, need, true};
}

inline void Encode(std::string& out, std::uint32_t cp) {
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
}

// ASCII-range test FIRST in every classifier below. Masking a codepoint down to
// seven bits lands Cyrillic squarely in the digit range (U+0430 'а' -> '0'),
// which is how a naive spelling classifies half the alphabet as digits.
inline bool IsAsciiDigit(std::uint32_t cp) noexcept { return cp >= '0' && cp <= '9'; }

inline bool IsLetter(std::uint32_t cp) noexcept {
    if (cp < 0x80u) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (cp == 0x401u || cp == 0x451u) return true;             // Ё ё
    return cp >= 0x410u && cp <= 0x44Fu;                       // А..я
}

inline bool IsCyrillic(std::uint32_t cp) noexcept {
    return cp == 0x401u || cp == 0x451u || (cp >= 0x410u && cp <= 0x44Fu);
}

// The ten Russian vowels, both cases. THE class stress placement is defined
// against: a mark goes on a vowel or it goes nowhere.
inline bool IsVowel(std::uint32_t cp) noexcept {
    switch (cp) {
        case 0x0430: case 0x0435: case 0x0451: case 0x0438: case 0x043E:   // а е ё и о
        case 0x0443: case 0x044B: case 0x044D: case 0x044E: case 0x044F:   // у ы э ю я
        case 0x0410: case 0x0415: case 0x0401: case 0x0418: case 0x041E:   // А Е Ё И О
        case 0x0423: case 0x042B: case 0x042D: case 0x042E: case 0x042F:   // У Ы Э Ю Я
            return true;
        default:
            return false;
    }
}

// Length-preserving in UTF-8 for both scripts, which is what lets callers fold
// case without re-encoding the whole string.
inline std::uint32_t Lower(std::uint32_t cp) noexcept {
    if (cp >= 'A' && cp <= 'Z') return cp + 32u;
    if (cp == 0x401u) return 0x451u;                           // Ё -> ё
    if (cp >= 0x410u && cp <= 0x42Fu) return cp + 32u;         // А..Я -> а..я
    return cp;
}

inline std::string ToLower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const Char c = Decode(s, i);
        if (c.ok) {
            Encode(out, Lower(c.cp));
        } else {
            out.push_back(s[i]);
        }
        i += c.len;
    }
    return out;
}

// Codepoint ENDING at `pos` (exclusive). False when there is none or the bytes
// before `pos` do not form one -- callers read that as "nothing there", which
// is the safe answer for every boundary test.
inline bool DecodePrev(std::string_view s, std::size_t pos, Char& out,
                       std::size_t& start) noexcept {
    if (pos == 0 || pos > s.size()) return false;
    std::size_t b = pos - 1;
    std::size_t steps = 0;
    // Walk back over continuation bytes to the lead byte. Three is the maximum
    // for well-formed UTF-8 and also stops a corrupt run from walking away.
    while (b > 0 && (static_cast<unsigned char>(s[b]) & 0xC0u) == 0x80u && steps < 3) {
        --b;
        ++steps;
    }
    const Char c = Decode(s, b);
    if (!c.ok || b + c.len != pos) return false;
    out = c;
    start = b;
    return true;
}

}  // namespace blackwell::tts::utf8
