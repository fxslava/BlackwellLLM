// -----------------------------------------------------------------------------
// speech_text.cpp — see speech_text.hpp for why this layer exists at all.
//
// TWO PASSES, and the split is not arbitrary. Markdown syntax is entirely ASCII,
// so it can be removed by a byte scan that never has to know what a code point
// is; stress, symbol classification and whitespace all need the code point.
// Running them as one pass would mean the Markdown rules had to be re-expressed
// in code-point terms for no benefit.
// -----------------------------------------------------------------------------
#include "speech_text.hpp"

#include <cstdint>

namespace blackwell::tts {
namespace {

// ---- UTF-8 decoding ---------------------------------------------------------
// One code point, or a single RAW BYTE when the sequence is malformed. The raw
// case is what implements the header's pass-through promise: a byte that is not
// part of a valid sequence is copied out unchanged rather than repaired or
// dropped, because code-point integrity is the chunker's contract, not this
// file's, and silently eating bytes here would hide a break in it.
constexpr std::uint32_t kRawByte = 0xFFFFFFFFu;

struct Decoded {
    std::uint32_t cp = 0;
    std::size_t   len = 1;
};

Decoded decode(std::string_view s, std::size_t i) noexcept {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const std::size_t avail = s.size() - i;

    std::size_t need = 0;
    std::uint32_t cp = 0;
    if (b0 < 0x80)              { return {b0, 1}; }
    else if ((b0 & 0xE0) == 0xC0) { need = 2; cp = b0 & 0x1Fu; }
    else if ((b0 & 0xF0) == 0xE0) { need = 3; cp = b0 & 0x0Fu; }
    else if ((b0 & 0xF8) == 0xF0) { need = 4; cp = b0 & 0x07u; }
    else                        { return {kRawByte, 1}; }

    if (avail < need) return {kRawByte, 1};
    for (std::size_t k = 1; k < need; ++k) {
        const auto bk = static_cast<unsigned char>(s[i + k]);
        if ((bk & 0xC0) != 0x80) return {kRawByte, 1};
        cp = (cp << 6) | (bk & 0x3Fu);
    }
    return {cp, need};
}

void encode(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ---- character classes ------------------------------------------------------

constexpr std::uint32_t kCombiningAcute = 0x0301;

// Cyrillic and Latin vowels, both cases. This is what tells a stress '+' from an
// arithmetic one: "хорош+о" marks a vowel, "2 + 2" does not, and only the first
// may be rewritten.
bool is_vowel(std::uint32_t cp) noexcept {
    switch (cp) {
        // Latin
        case U'a': case U'e': case U'i': case U'o': case U'u': case U'y':
        case U'A': case U'E': case U'I': case U'O': case U'U': case U'Y':
        // Cyrillic
        case U'а': case U'е': case U'ё': case U'и': case U'о':
        case U'у': case U'ы': case U'э': case U'ю': case U'я':
        case U'А': case U'Е': case U'Ё': case U'И': case U'О':
        case U'У': case U'Ы': case U'Э': case U'Ю': case U'Я':
            return true;
        default:
            return false;
    }
}

bool is_space(std::uint32_t cp) noexcept {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x00A0;
}

// Reads as noise or as a pause, never as a word. Everything here tokenises to
// the unknown id -- which in F5 IS the space character -- so keeping it would
// insert a silence rather than produce a sound (f5_tokenizer.hpp).
bool is_unspeakable(std::uint32_t cp) noexcept {
    if (cp == 0x200B || (cp >= 0x200C && cp <= 0x200F)) return true;   // zero-width
    if (cp >= 0xFE00 && cp <= 0xFE0F) return true;                     // variation selectors
    if (cp >= 0x2190 && cp <= 0x21FF) return true;                     // arrows
    if (cp >= 0x2500 && cp <= 0x25FF) return true;                     // box / geometric
    if (cp >= 0x2600 && cp <= 0x27BF) return true;                     // symbols / dingbats
    if (cp >= 0x2B00 && cp <= 0x2BFF) return true;                     // more arrows/symbols
    if (cp >= 0xE000 && cp <= 0xF8FF) return true;                     // private use
    if (cp >= 0x1F000 && cp <= 0x1FAFF) return true;                   // emoji planes
    return false;
    // NOT listed, and on purpose: the dashes, guillemets, curly quotes and the
    // ellipsis. They are punctuation the model reads as prosody, which is the
    // one thing worth keeping from a decorated reply.
}

// ---- pass 1: Markdown -------------------------------------------------------
// ASCII-only by construction, so this never needs to decode. It removes what is
// AUDIBLE as noise and leaves the prose intact -- see the header on why this is
// not, and must not become, a Markdown parser.
std::string strip_markdown(std::string_view s) {
    std::string out;
    out.reserve(s.size());

    bool at_line_start = true;
    for (std::size_t i = 0; i < s.size();) {
        const char c = s[i];

        if (at_line_start) {
            // Leading indentation is layout; it says nothing out loud.
            std::size_t j = i;
            while (j < s.size() && (s[j] == ' ' || s[j] == '\t')) ++j;

            // "### Heading" and "> quoted" -- the marker goes, the words stay.
            std::size_t k = j;
            while (k < s.size() && (s[k] == '#' || s[k] == '>')) ++k;
            if (k > j && k < s.size() && (s[k] == ' ' || s[k] == '\t')) {
                i = k + 1;
                continue;
            }

            // "- item" / "* item" / "+ item". The trailing space is REQUIRED,
            // which is what keeps this rule off a '+' stress mark ("+о") and off
            // a hyphenated word.
            if (j + 1 < s.size() && (s[j] == '-' || s[j] == '*' || s[j] == '+') &&
                (s[j + 1] == ' ' || s[j + 1] == '\t')) {
                i = j + 2;
                continue;
            }

            // A rule ("---", "***"): a whole line of one marker. Silent.
            if (j < s.size() && (s[j] == '-' || s[j] == '*' || s[j] == '_')) {
                std::size_t r = j;
                while (r < s.size() && s[r] == s[j]) ++r;
                std::size_t e = r;
                while (e < s.size() && (s[e] == ' ' || s[e] == '\t')) ++e;
                if (r - j >= 3 && (e >= s.size() || s[e] == '\n')) {
                    i = e;
                    continue;
                }
            }
            at_line_start = false;
        }

        switch (c) {
            case '\n':
                out.push_back('\n');
                at_line_start = true;
                ++i;
                continue;
            case '*':
            case '~':
            case '`':
                // Emphasis, strikethrough and code delimiters: the run goes, the
                // text it wrapped stays. A fence's language tag ("```python") is
                // left behind and reads as a word -- rare enough, in a block that
                // should not have been in <voice> in the first place, that
                // inventing a rule for it would cost more than it saves.
                ++i;
                continue;
            case '_': {
                // ONLY at a word edge. Dropping every underscore would rewrite
                // identifiers ("max_new_tokens") into something the model reads
                // as one run-on word, and those appear in technical answers far
                // more often than "_emphasis_" does.
                const bool left_edge = out.empty() || out.back() == ' ' || out.back() == '\n' ||
                                       out.back() == '(' || out.back() == '"';
                const bool right_edge = i + 1 >= s.size() || s[i + 1] == ' ' || s[i + 1] == '\n' ||
                                        s[i + 1] == ',' || s[i + 1] == '.' || s[i + 1] == '!' ||
                                        s[i + 1] == '?' || s[i + 1] == ')' || s[i + 1] == '"';
                if (left_edge || right_edge) { ++i; continue; }
                out.push_back('_');
                ++i;
                continue;
            }
            case '|':
                // A table cell wall. Reading a table aloud is a product question
                // nobody has answered; a pause between the cells is the least
                // wrong approximation available here.
                out.push_back(' ');
                ++i;
                continue;
            case '!':
                // "![alt](url)" -- the bang belongs to the image, not the
                // sentence. Only when a link actually follows.
                if (i + 1 < s.size() && s[i + 1] == '[') { ++i; continue; }
                out.push_back('!');
                ++i;
                continue;
            case '[': {
                // "[text](url)" -> "text". The URL is unspeakable in the literal
                // sense: reading "https colon slash slash" is worse than silence.
                const std::size_t close = s.find(']', i + 1);
                if (close != std::string_view::npos && close + 1 < s.size() &&
                    s[close + 1] == '(') {
                    const std::size_t end = s.find(')', close + 2);
                    if (end != std::string_view::npos) {
                        out.append(s.substr(i + 1, close - i - 1));
                        i = end + 1;
                        continue;
                    }
                }
                out.push_back('[');
                ++i;
                continue;
            }
            default:
                out.push_back(c);
                ++i;
                continue;
        }
    }
    return out;
}

}  // namespace

std::string NormalizeForSpeech(std::string_view utf8, const SpeechTextOptions& opts) {
    const std::string staged = opts.strip_markdown ? strip_markdown(utf8) : std::string(utf8);
    const std::string_view s(staged);

    std::string out;
    out.reserve(s.size());

    // Where the most recently emitted code point starts. Converting U+0301 into
    // the '+' convention means inserting AHEAD of the vowel that has already
    // been written, and this is the only bookkeeping that needs.
    std::size_t last_cp_start = std::string::npos;
    // Set when a '+' was consumed under StressPolicy::Combining: the accent it
    // stood for is emitted AFTER the vowel that follows.
    bool pending_combining = false;
    bool pending_space = false;   // whitespace seen, not yet emitted (collapsing)

    for (std::size_t i = 0; i < s.size();) {
        const Decoded d = decode(s, i);
        i += d.len;

        if (d.cp == kRawByte) {
            if (pending_space && !out.empty()) { out.push_back(' '); pending_space = false; }
            last_cp_start = out.size();
            out.push_back(s[i - 1]);
            continue;
        }

        if (opts.collapse_whitespace && is_space(d.cp)) {
            pending_space = true;   // emitted lazily, so a trailing run vanishes
            continue;
        }

        if (opts.drop_unspeakable_symbols && is_unspeakable(d.cp)) continue;

        // ---- stress: the combining accent, sitting AFTER its vowel -----------
        if (d.cp == kCombiningAcute) {
            if (opts.stress == StressPolicy::Combining) {
                encode(out, d.cp);   // already in the wanted notation
            } else if (opts.stress == StressPolicy::Plus && last_cp_start != std::string::npos) {
                // Ahead of the vowel already written. `last_cp_start` is a byte
                // offset into `out`, so this cannot land mid-sequence.
                out.insert(last_cp_start, 1, '+');
            }
            // Strip: dropped. So is a stray accent with no vowel before it.
            continue;
        }

        // ---- stress: '+' PRECEDING its vowel ---------------------------------
        if (d.cp == '+' && i < s.size()) {
            const Decoded next = decode(s, i);
            if (next.cp != kRawByte && is_vowel(next.cp)) {
                if (opts.stress == StressPolicy::Plus) {
                    if (pending_space && !out.empty()) { out.push_back(' '); pending_space = false; }
                    last_cp_start = out.size();
                    out.push_back('+');
                } else if (opts.stress == StressPolicy::Combining) {
                    pending_combining = true;   // emitted after the vowel below
                }
                // Strip: dropped, and the vowel is emitted on the next iteration
                // exactly as if the mark had never been there.
                continue;
            }
            // Arithmetic, a signed number, a "C++": not a stress mark. Falls
            // through and is emitted verbatim.
        }

        if (pending_space && !out.empty()) { out.push_back(' '); pending_space = false; }
        last_cp_start = out.size();
        encode(out, d.cp);
        if (pending_combining) {
            encode(out, kCombiningAcute);
            pending_combining = false;
        }
    }

    if (!opts.collapse_whitespace) return out;

    // Leading whitespace never got emitted (the guard is `!out.empty()`), and a
    // trailing run is still pending and deliberately dropped -- so the result is
    // trimmed at both ends without a second scan.
    return out;
}

}  // namespace blackwell::tts
