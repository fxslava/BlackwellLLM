#include "tokenizer/unicode.h"

#include <array>
#include <unordered_map>

namespace blackwell::unicode {

std::vector<uint32_t> decode_utf8(const std::string& text,
                                  std::vector<size_t>* byte_offsets) {
    std::vector<uint32_t> cps;
    cps.reserve(text.size());
    if (byte_offsets) {
        byte_offsets->clear();
        byte_offsets->reserve(text.size());
    }

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        const size_t start = i;
        const unsigned char c = static_cast<unsigned char>(text[i]);
        uint32_t cp = 0xFFFD;
        size_t len = 1;

        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }

        if (len > 1) {
            if (start + len <= n) {
                static constexpr uint32_t kFirstMask[5] = {0, 0x7F, 0x1F, 0x0F, 0x07};
                uint32_t acc = c & kFirstMask[len];
                bool ok = true;
                for (size_t k = 1; k < len; ++k) {
                    const unsigned char cc = static_cast<unsigned char>(text[start + k]);
                    if ((cc & 0xC0) != 0x80) { ok = false; break; }
                    acc = (acc << 6) | (cc & 0x3F);
                }
                if (ok) cp = acc; else len = 1;
            } else {
                len = 1;
            }
        }

        cps.push_back(cp);
        if (byte_offsets) byte_offsets->push_back(start);
        i = start + len;
    }
    return cps;
}

void append_utf8(std::string& out, uint32_t cp) {
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

namespace {

struct Range { uint32_t lo, hi; };

bool in_ranges(const Range* ranges, size_t count, uint32_t cp) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid].lo) hi = mid;
        else if (cp > ranges[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

// Subset of \p{L}: Latin (incl. supplements/extended), IPA, Greek, Cyrillic,
// Armenian, Hebrew, Arabic, Devanagari, Thai, Georgian, kana, CJK, Hangul.
constexpr Range kLetterRanges[] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00AA, 0x00AA}, {0x00B5, 0x00B5},
    {0x00BA, 0x00BA}, {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02C1},
    {0x02C6, 0x02D1}, {0x02E0, 0x02E4}, {0x0370, 0x0374}, {0x0376, 0x0377},
    {0x037A, 0x037D}, {0x037F, 0x037F}, {0x0386, 0x0386}, {0x0388, 0x03F5},
    {0x03F7, 0x0481}, {0x048A, 0x052F}, {0x0531, 0x0556}, {0x0560, 0x0588},
    {0x05D0, 0x05EA}, {0x05EF, 0x05F2}, {0x0620, 0x064A}, {0x066E, 0x066F},
    {0x0671, 0x06D3}, {0x06FA, 0x06FC}, {0x0710, 0x072F}, {0x0750, 0x077F},
    {0x0904, 0x0939}, {0x093D, 0x093D}, {0x0950, 0x0950}, {0x0958, 0x0961},
    {0x0971, 0x097F}, {0x0E01, 0x0E30}, {0x0E32, 0x0E33}, {0x0E40, 0x0E46},
    {0x10A0, 0x10C5}, {0x10D0, 0x10FA}, {0x1100, 0x11FF}, {0x1E00, 0x1F15},
    {0x1F18, 0x1F1D}, {0x1F20, 0x1F45}, {0x1F48, 0x1F4D}, {0x1F50, 0x1F7D},
    {0x1F80, 0x1FBC}, {0x1FC2, 0x1FCC}, {0x1FD0, 0x1FDB}, {0x1FE0, 0x1FEC},
    {0x1FF2, 0x1FFC}, {0x2C60, 0x2C7F}, {0x3041, 0x3096}, {0x30A1, 0x30FA},
    {0x30FC, 0x30FF}, {0x3105, 0x312F}, {0x31A0, 0x31BF}, {0x31F0, 0x31FF},
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA717, 0xA71F}, {0xA722, 0xA788},
    {0xAC00, 0xD7A3}, {0xF900, 0xFAD9}, {0xFB00, 0xFB06}, {0xFB1D, 0xFB1D},
    {0xFB1F, 0xFB28}, {0xFB2A, 0xFB36}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
    {0xFF66, 0xFFDC}, {0x20000, 0x2A6DF},
};

// Subset of \p{N}: ASCII digits, super/subscripts, vulgar fractions, common
// script digit blocks, Roman numerals, fullwidth digits.
constexpr Range kNumberRanges[] = {
    {0x0030, 0x0039}, {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE},
    {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x0966, 0x096F}, {0x0E50, 0x0E59},
    {0x2070, 0x2070}, {0x2074, 0x2079}, {0x2080, 0x2089}, {0x2150, 0x2182},
    {0x2185, 0x2189}, {0x2460, 0x2468}, {0xFF10, 0xFF19},
};

constexpr Range kWhitespaceRanges[] = {
    {0x0009, 0x000D}, {0x0020, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0},
    {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F},
    {0x205F, 0x205F}, {0x3000, 0x3000},
};

template <size_t N>
constexpr size_t count_of(const Range (&)[N]) { return N; }

// GPT-2 alphabet: printable bytes map to themselves; the remaining 68 bytes
// map, in increasing byte order, to U+0100, U+0101, ...
struct ByteAlphabet {
    std::array<uint32_t, 256> to_cp{};
    std::unordered_map<uint32_t, int> to_byte;

    ByteAlphabet() {
        auto printable = [](int b) {
            return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        };
        uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            to_cp[b] = printable(b) ? static_cast<uint32_t>(b) : next++;
            to_byte.emplace(to_cp[b], b);
        }
    }
};

const ByteAlphabet& byte_alphabet() {
    static const ByteAlphabet table;
    return table;
}

} // namespace

bool is_letter(uint32_t cp)     { return in_ranges(kLetterRanges, count_of(kLetterRanges), cp); }
bool is_number(uint32_t cp)     { return in_ranges(kNumberRanges, count_of(kNumberRanges), cp); }
bool is_whitespace(uint32_t cp) { return in_ranges(kWhitespaceRanges, count_of(kWhitespaceRanges), cp); }

const uint32_t* byte_to_unicode_table() { return byte_alphabet().to_cp.data(); }

int unicode_to_byte(uint32_t cp) {
    const auto& m = byte_alphabet().to_byte;
    const auto it = m.find(cp);
    return it == m.end() ? -1 : it->second;
}

} // namespace blackwell::unicode
