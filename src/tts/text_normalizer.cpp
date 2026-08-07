// -----------------------------------------------------------------------------
// text_normalizer.cpp — see text_normalizer.hpp for why digits cannot be left
// for the model to interpret, and for the year heuristic's evidence rule.
//
// THREE LAYERS, bottom up:
//   1. number -> words        (RussianCardinal / RussianOrdinalMasculine)
//   2. token readers          (a digit run + its unit, one symbol, one
//                              abbreviation) -- each returns how many bytes it
//                              consumed, or 0 for "not mine"
//   3. the scan               (ExpandForSpeech), which is nothing but a loop
//                              offering each byte to the readers in order
// The split is what keeps the grammar testable: layer 1 is a pure function of
// an integer and takes no view of the surrounding text.
// -----------------------------------------------------------------------------
#include "text_normalizer.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include "utf8_scan.hpp"

namespace blackwell::tts {
namespace {

using Decoded = utf8::Char;

// Local spellings, so the readers below stay readable. The shared header owns
// the error contract (a malformed byte comes back as one raw byte, which is
// what implements this file's pass-through promise).
inline Decoded decode(std::string_view s, std::size_t i) noexcept {
    return utf8::Decode(s, i);
}

constexpr std::uint32_t kNbsp     = 0x00A0;
constexpr std::uint32_t kDegree   = 0x00B0;
constexpr std::uint32_t kMultiply = 0x00D7;
constexpr std::uint32_t kNumero   = 0x2116;   // №
constexpr std::uint32_t kRuble    = 0x20BD;   // ₽
constexpr std::uint32_t kEuro     = 0x20AC;   // €
constexpr std::uint32_t kMinusSgn = 0x2212;   // U+2212 MINUS SIGN

bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }
bool is_digit_cp(std::uint32_t cp) noexcept { return utf8::IsAsciiDigit(cp); }
bool is_letter_cp(std::uint32_t cp) noexcept { return utf8::IsLetter(cp); }

bool is_space_cp(std::uint32_t cp) noexcept {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == kNbsp;
}

// Case-insensitive (ASCII + Cyrillic) prefix match of `pat` at s[i]. Returns
// the bytes consumed, or 0. Both sides are UTF-8 and the case folding above is
// length-preserving for every character in range, so a byte count is a valid
// answer.
std::size_t match_ci(std::string_view s, std::size_t i, std::string_view pat) noexcept {
    std::size_t a = i;
    std::size_t b = 0;
    while (b < pat.size()) {
        if (a >= s.size()) return 0;
        const Decoded ds = decode(s, a);
        const Decoded dp = decode(pat, b);
        if (!ds.ok || !dp.ok) return 0;
        if (utf8::Lower(ds.cp) != utf8::Lower(dp.cp)) return 0;
        a += ds.len;
        b += dp.len;
    }
    return a - i;
}

void append_word(std::string& out, std::string_view w) {
    if (w.empty()) return;
    if (!out.empty()) {
        const char last = out.back();
        if (last != ' ' && last != '\n' && last != '\t' && last != '(' && last != '"') {
            out.push_back(' ');
        }
    }
    out.append(w);
}

// ---- layer 1: number -> words ------------------------------------------
// Index 0 is unused throughout so the arrays can be subscripted by the digit.

constexpr std::array<const char*, 10> kOnesM = {
    "", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"};
constexpr std::array<const char*, 10> kOnesF = {
    "", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"};
constexpr std::array<const char*, 10> kOnesN = {
    "", "одно", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"};
// 10..19, subscripted by (value - 10).
constexpr std::array<const char*, 10> kTeens = {
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"};
// 20..90, subscripted by tens digit.
constexpr std::array<const char*, 10> kTens = {
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто"};
constexpr std::array<const char*, 10> kHundreds = {
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот"};

struct Scale {
    const char* one;
    const char* few;
    const char* many;
    NumberGender gender;   // "две тысячи" but "два миллиона" -- see the header
};

// Index 0 is the bare units group: no scale word, gender comes from the caller.
// Runs to квинтиллион so that the table covers the whole int64 range and
// cardinal_prefix has no "and the rest" case to get wrong.
constexpr std::array<Scale, 7> kScales = {{
    {"", "", "", NumberGender::Masculine},
    {"тысяча", "тысячи", "тысяч", NumberGender::Feminine},
    {"миллион", "миллиона", "миллионов", NumberGender::Masculine},
    {"миллиард", "миллиарда", "миллиардов", NumberGender::Masculine},
    {"триллион", "триллиона", "триллионов", NumberGender::Masculine},
    {"квадриллион", "квадриллиона", "квадриллионов", NumberGender::Masculine},
    {"квинтиллион", "квинтиллиона", "квинтиллионов", NumberGender::Masculine},
}};

const char* ones_word(int digit, NumberGender g) noexcept {
    switch (g) {
        case NumberGender::Feminine:  return kOnesF[static_cast<std::size_t>(digit)];
        case NumberGender::Neuter:    return kOnesN[static_cast<std::size_t>(digit)];
        case NumberGender::Masculine:
        default:                      return kOnesM[static_cast<std::size_t>(digit)];
    }
}

// One 000..999 group. Appends nothing when the group is zero.
void append_group(std::string& out, int group, NumberGender g) {
    const int h = group / 100;
    const int t = (group / 10) % 10;
    const int o = group % 10;
    if (h != 0) append_word(out, kHundreds[static_cast<std::size_t>(h)]);
    if (t == 1) {
        append_word(out, kTeens[static_cast<std::size_t>(o)]);
        return;
    }
    if (t != 0) append_word(out, kTens[static_cast<std::size_t>(t)]);
    if (o != 0) append_word(out, ones_word(o, g));
}

// Cardinal without the "ноль" special case -- returns "" for 0, which is what
// the ordinal builder needs for its prefix.
std::string cardinal_prefix(std::int64_t value, NumberGender gender) {
    if (value == 0) return std::string();

    std::array<int, kScales.size()> groups{};
    std::size_t count = 0;
    std::int64_t rest = value;
    while (rest > 0 && count < groups.size()) {
        groups[count++] = static_cast<int>(rest % 1000);
        rest /= 1000;
    }

    std::string out;
    for (std::size_t k = count; k-- > 0;) {
        const int g = groups[k];
        if (g == 0) continue;
        const Scale& sc = kScales[k];
        // "тысяча девятьсот", NOT "одна тысяча девятьсот". The numeral is
        // dropped before a bare тысяча -- and only there: "один миллион" is
        // the ordinary reading. This is the difference between a year that
        // sounds like a date and one that sounds like a count.
        const bool bare_thousand = (k == 1 && g == 1);
        if (!bare_thousand) {
            append_group(out, g, k == 0 ? gender : sc.gender);
        }
        if (k != 0) {
            append_word(out, PluralForm(g, sc.one, sc.few, sc.many));
        }
    }
    // kScales covers 10^18, so an int64 always decomposes completely and this
    // is unreachable. It stays because "silently dropped the leading digits"
    // is the one failure mode of this function nobody would hear as wrong.
    if (rest > 0) out.clear();
    return out;
}

// ---- fractions ----------------------------------------------------------
// "3,14" -> "три целых четырнадцать сотых". The integer part agrees with
// "целая" (feminine), which is why RussianCardinal takes a gender at all.
struct FractionNoun {
    const char* one;
    const char* many;
};
constexpr std::array<FractionNoun, 5> kFractionNouns = {{
    {"", ""},                                   // 0 digits -- unused
    {"десятая", "десятых"},
    {"сотая", "сотых"},
    {"тысячная", "тысячных"},
    {"десятитысячная", "десятитысячных"},
}};

// ---- units and symbols ---------------------------------------------------

struct Unit {
    const char* token;   // as written, matched case-insensitively AFTER a number
    const char* one;
    const char* few;
    const char* many;
    NumberGender gender;
};

// LONGEST FIRST within a shared prefix ("тыс." before "тыс", "млрд" before
// "млн" is irrelevant -- different prefixes -- but the rule is cheap to keep).
// Single-letter units are deliberately absent: "г" is год or грамм, "м" is
// метр or минута, and a wrong expansion is worse than an unread symbol.
constexpr std::array<Unit, 18> kUnits = {{
    {"%",    "процент",    "процента",    "процентов",   NumberGender::Masculine},
    {"тыс.", "тысяча",     "тысячи",      "тысяч",       NumberGender::Feminine},
    {"тыс",  "тысяча",     "тысячи",      "тысяч",       NumberGender::Feminine},
    {"млрд", "миллиард",   "миллиарда",   "миллиардов",  NumberGender::Masculine},
    {"млн",  "миллион",    "миллиона",    "миллионов",   NumberGender::Masculine},
    {"руб.", "рубль",      "рубля",       "рублей",      NumberGender::Masculine},
    {"руб",  "рубль",      "рубля",       "рублей",      NumberGender::Masculine},
    {"км",   "километр",   "километра",   "километров",  NumberGender::Masculine},
    {"кг",   "килограмм",  "килограмма",  "килограммов", NumberGender::Masculine},
    {"мм",   "миллиметр",  "миллиметра",  "миллиметров", NumberGender::Masculine},
    {"см",   "сантиметр",  "сантиметра",  "сантиметров", NumberGender::Masculine},
    {"мл",   "миллилитр",  "миллилитра",  "миллилитров", NumberGender::Masculine},
    {"шт.",  "штука",      "штуки",       "штук",        NumberGender::Feminine},
    {"ГБ",   "гигабайт",   "гигабайта",   "гигабайт",    NumberGender::Masculine},
    {"МБ",   "мегабайт",   "мегабайта",   "мегабайт",    NumberGender::Masculine},
    {"КБ",   "килобайт",   "килобайта",   "килобайт",    NumberGender::Masculine},
    {"ТБ",   "терабайт",   "терабайта",   "терабайт",    NumberGender::Masculine},
    {"°",    "градус",     "градуса",     "градусов",    NumberGender::Masculine},
}};

// Currency signs written after the amount. Separate from kUnits only because
// they are single codepoints with no letter-boundary requirement.
constexpr std::array<Unit, 3> kCurrencySuffixes = {{
    {"₽", "рубль",  "рубля",   "рублей",    NumberGender::Masculine},
    {"$", "доллар", "доллара", "долларов",  NumberGender::Masculine},
    {"€", "евро",   "евро",    "евро",      NumberGender::Masculine},
}};

struct Abbrev {
    const char* token;
    const char* words;
};

// Matched only at a WORD START, longest first. Everything here is unambiguous
// in isolation; anything that is not (единственная "г.", "в.", "с.") is absent
// on purpose -- see the header.
constexpr std::array<Abbrev, 20> kAbbrevs = {{
    {"и т. д.", "и так далее"},
    {"и т.д.",  "и так далее"},
    {"т. д.",   "так далее"},
    {"т.д.",    "так далее"},
    {"т. п.",   "тому подобное"},
    {"т.п.",    "тому подобное"},
    {"т. е.",   "то есть"},
    {"т.е.",    "то есть"},
    {"т. к.",   "так как"},
    {"т.к.",    "так как"},
    {"др.",     "другие"},
    {"гг.",     "годов"},
    {"проф.",   "профессор"},
    {"напр.",   "например"},
    {"стр.",    "страница"},
    {"рис.",    "рисунок"},
    {"чел.",    "человек"},
    {"мес.",    "месяцев"},
    {"ок.",     "около"},
    {"им.",     "имени"},
}};

struct YearCue {
    const char* word;
    OrdinalCase grammatical_case;
};

// Prepositions that make a following four-digit number a YEAR -- and that
// GOVERN ITS CASE, which is why the table carries one. "в 2026" is
// prepositional, "с 2026" genitive, "к 2026" dative, and reading all three in
// the nominative is a grammatical error in the middle of the sentence.
constexpr std::array<YearCue, 10> kYearCuePrepositions = {{
    {"в", OrdinalCase::Prepositional},
    {"во", OrdinalCase::Prepositional},
    {"на", OrdinalCase::Prepositional},
    {"с", OrdinalCase::Genitive},
    {"со", OrdinalCase::Genitive},
    {"до", OrdinalCase::Genitive},
    {"от", OrdinalCase::Genitive},
    {"за", OrdinalCase::Genitive},
    {"к", OrdinalCase::Dative},
    {"по", OrdinalCase::Dative},
}};

// год-forms that make a PRECEDING number a year. Consulted for the case only
// when there is no preposition, since the preposition wins ("к 2026 году" is
// dative even though "году" on its own is prepositional).
constexpr std::array<YearCue, 6> kYearCueNouns = {{
    {"году", OrdinalCase::Prepositional},
    {"года", OrdinalCase::Genitive},
    {"годов", OrdinalCase::Genitive},
    {"годам", OrdinalCase::Dative},
    {"гг.", OrdinalCase::Genitive},
    {"год", OrdinalCase::Nominative},
}};

// ---- lookahead helpers on the raw text ----------------------------------

std::size_t skip_spaces(std::string_view s, std::size_t i) noexcept {
    while (i < s.size()) {
        const Decoded d = decode(s, i);
        if (!d.ok || !is_space_cp(d.cp)) break;
        i += d.len;
    }
    return i;
}

// True when the codepoint at `i` cannot continue a word -- the boundary test
// every table match needs so "кг" does not fire inside "кгм".
bool at_token_end(std::string_view s, std::size_t i) noexcept {
    if (i >= s.size()) return true;
    const Decoded d = decode(s, i);
    if (!d.ok) return true;
    return !is_letter_cp(d.cp) && !is_digit_cp(d.cp);
}

// The lowercased word ENDING at `end` (exclusive), for the preposition cue.
// Empty when what precedes is punctuation or the start of the chunk, which the
// caller reads as "no cue" -- the correct answer for a number that opens a
// sentence.
std::string preceding_word_lower(std::string_view s, std::size_t end) {
    std::size_t p = end;
    Decoded c{};
    std::size_t start = 0;
    // Back over the separating spaces, then over the letters of the word.
    while (utf8::DecodePrev(s, p, c, start) && is_space_cp(c.cp)) p = start;
    std::size_t word_end = p;
    while (utf8::DecodePrev(s, p, c, start) && is_letter_cp(c.cp)) p = start;
    return utf8::ToLower(s.substr(p, word_end - p));
}

// ---- digit-run readers ---------------------------------------------------

void append_digits_individually(std::string& out, std::string_view digits) {
    for (const char c : digits) {
        const int d = c - '0';
        append_word(out, d == 0 ? "ноль" : kOnesM[static_cast<std::size_t>(d)]);
    }
}

// Everything the scan learned about one digit run, so the emit step reads as
// one decision rather than as five nested ifs.
struct NumberSpan {
    std::string_view int_digits;
    std::string_view frac_digits;      // empty when there is no fraction
    std::size_t end = 0;               // first byte after the run (and its unit)
    const Unit* unit = nullptr;
    bool year = false;
    OrdinalCase year_case = OrdinalCase::Nominative;
};

bool digits_are_spoken(std::string_view digits) noexcept {
    if (digits.empty() || digits.size() > kMaxSpokenDigits) return false;
    // A leading zero means the run is an identifier, a time or a version, not
    // a quantity: "007" is read as three digits by any human reader too.
    return !(digits.size() > 1 && digits[0] == '0');
}

std::int64_t to_int64(std::string_view digits) noexcept {
    std::int64_t v = 0;
    for (const char c : digits) v = v * 10 + (c - '0');
    return v;
}

}  // namespace

// -----------------------------------------------------------------------------
// Public number formatting
// -----------------------------------------------------------------------------
std::string_view PluralForm(std::int64_t value, std::string_view one,
                            std::string_view few, std::string_view many) {
    const std::int64_t v = value < 0 ? -value : value;
    const std::int64_t mod100 = v % 100;
    if (mod100 >= 11 && mod100 <= 14) return many;
    switch (v % 10) {
        case 1:  return one;
        case 2:
        case 3:
        case 4:  return few;
        default: return many;
    }
}

std::string RussianCardinal(std::int64_t value, NumberGender gender) {
    if (value == 0) return "ноль";
    std::string out;
    std::int64_t v = value;
    if (v < 0) {
        out = "минус";
        // INT64_MIN has no positive counterpart; clamping loses one value at
        // the very bottom of the range and keeps the negation defined.
        v = (v == INT64_MIN) ? INT64_MAX : -v;
    }
    const std::string body = cardinal_prefix(v, gender);
    if (body.empty()) return "ноль";   // out of table range; caller falls back
    append_word(out, body);
    return out;
}

namespace {

// Declines a masculine ordinal by rewriting its ENDING. Sound for this closed
// set: every ordinal the year path can produce ends in -ый / -ой / -ий, and the
// only stem change among them is "третий", which is listed.
std::string decline_ordinal(std::string nominative, OrdinalCase c) {
    if (c == OrdinalCase::Nominative || nominative.empty()) return nominative;

    // "третий" -> "треть|его/ему/ем": the soft stem the -ий rule cannot derive.
    const std::string_view third = "третий";
    const bool is_third = nominative.size() >= third.size() &&
                          nominative.compare(nominative.size() - third.size(),
                                             third.size(), third) == 0;
    const std::string_view yy = "ый";
    const std::string_view oy = "ой";
    const std::string_view iy = "ий";
    auto ends_with = [&nominative](std::string_view suffix) {
        return nominative.size() >= suffix.size() &&
               nominative.compare(nominative.size() - suffix.size(), suffix.size(),
                                  suffix) == 0;
    };

    std::string_view ending;
    std::size_t drop = 0;
    if (is_third) {
        drop = 4;   // "ий" (4 bytes) plus nothing; the ь comes from the ending
        ending = c == OrdinalCase::Genitive ? "ьего"
                 : c == OrdinalCase::Dative ? "ьему"
                                            : "ьем";
    } else if (ends_with(yy) || ends_with(oy)) {
        drop = 4;   // both endings are two 2-byte codepoints
        ending = c == OrdinalCase::Genitive ? "ого"
                 : c == OrdinalCase::Dative ? "ому"
                                            : "ом";
    } else if (ends_with(iy)) {
        drop = 4;
        ending = c == OrdinalCase::Genitive ? "его"
                 : c == OrdinalCase::Dative ? "ему"
                                            : "ем";
    } else {
        return nominative;   // not an ordinal shape; leave it alone
    }
    nominative.erase(nominative.size() - drop);
    nominative.append(ending);
    return nominative;
}

}  // namespace

namespace {

// The nominative form. Split out so the case rewrite above has exactly one
// input to work on rather than one per early return.
std::string ordinal_nominative(std::int64_t value) {
    static constexpr std::array<const char*, 10> kOrdUnits = {
        "", "первый", "второй", "третий", "четвёртый", "пятый",
        "шестой", "седьмой", "восьмой", "девятый"};
    static constexpr std::array<const char*, 10> kOrdTeens = {
        "десятый", "одиннадцатый", "двенадцатый", "тринадцатый", "четырнадцатый",
        "пятнадцатый", "шестнадцатый", "семнадцатый", "восемнадцатый", "девятнадцатый"};
    static constexpr std::array<const char*, 10> kOrdTens = {
        "", "", "двадцатый", "тридцатый", "сороковой", "пятидесятый",
        "шестидесятый", "семидесятый", "восьмидесятый", "девяностый"};
    static constexpr std::array<const char*, 10> kOrdHundreds = {
        "", "сотый", "двухсотый", "трёхсотый", "четырёхсотый", "пятисотый",
        "шестисотый", "семисотый", "восьмисотый", "девятисотый"};
    static constexpr std::array<const char*, 10> kOrdThousands = {
        "", "тысячный", "двухтысячный", "трёхтысячный", "четырёхтысячный",
        "пятитысячный", "шеститысячный", "семитысячный", "восьмитысячный",
        "девятитысячный"};

    if (value < 1 || value > 999999) return RussianCardinal(value);

    // ONLY the final component inflects. Everything to its left stays cardinal,
    // which is what makes "тысяча девятьсот девяностый" come out right.
    std::string out;
    const std::int64_t mod100 = value % 100;
    if (mod100 >= 11 && mod100 <= 19) {
        append_word(out, cardinal_prefix(value - mod100, NumberGender::Masculine));
        append_word(out, kOrdTeens[static_cast<std::size_t>(mod100 - 10)]);
        return out;
    }
    const std::int64_t mod10 = value % 10;
    if (mod10 != 0) {
        append_word(out, cardinal_prefix(value - mod10, NumberGender::Masculine));
        append_word(out, kOrdUnits[static_cast<std::size_t>(mod10)]);
        return out;
    }
    if (mod100 != 0) {
        append_word(out, cardinal_prefix(value - mod100, NumberGender::Masculine));
        append_word(out, kOrdTens[static_cast<std::size_t>(mod100 / 10)]);
        return out;
    }
    const std::int64_t mod1000 = value % 1000;
    if (mod1000 != 0) {
        append_word(out, cardinal_prefix(value - mod1000, NumberGender::Masculine));
        append_word(out, kOrdHundreds[static_cast<std::size_t>(mod1000 / 100)]);
        return out;
    }
    const std::int64_t thousands = value / 1000;
    if (thousands >= 1 && thousands <= 9) {
        append_word(out, kOrdThousands[static_cast<std::size_t>(thousands)]);
        return out;
    }
    // 21000, 150000...: the ordinal ("двадцатитысячный") is a compound this
    // pipeline never needs, and guessing it would be untested grammar on a hot
    // path. Cardinal is wrong-but-intelligible, which is the right failure.
    return RussianCardinal(value);
}

}  // namespace

std::string RussianOrdinalMasculine(std::int64_t value, OrdinalCase grammatical_case) {
    return decline_ordinal(ordinal_nominative(value), grammatical_case);
}

namespace {

// Reads the digit run starting at `i` plus whatever belongs to it (a decimal
// fraction, a trailing unit) and decides whether it is a year.
NumberSpan read_number(std::string_view s, std::size_t i, const NormalizerOptions& opts) {
    NumberSpan n;
    std::size_t j = i;
    while (j < s.size() && is_digit(s[j])) ++j;
    n.int_digits = s.substr(i, j - i);
    n.end = j;

    // Decimal fraction: the separator must sit BETWEEN digits, which is also
    // the guard the chunker applies so "3.14" is never split across chunks.
    if (j + 1 < s.size() && (s[j] == ',' || s[j] == '.') && is_digit(s[j + 1])) {
        std::size_t k = j + 1;
        while (k < s.size() && is_digit(s[k])) ++k;
        n.frac_digits = s.substr(j + 1, k - j - 1);
        n.end = k;
        j = k;
    }

    // A unit, optionally separated by a space. '%' and '°' bind tight; the
    // word-shaped ones need a token boundary after them.
    if (opts.expand_symbols || opts.expand_abbreviations) {
        const std::size_t u = skip_spaces(s, j);
        for (const Unit& unit : kUnits) {
            const std::size_t got = match_ci(s, u, unit.token);
            if (got != 0 && at_token_end(s, u + got)) {
                n.unit = &unit;
                n.end = u + got;
                break;
            }
        }
        if (n.unit == nullptr) {
            for (const Unit& unit : kCurrencySuffixes) {
                const std::size_t got = match_ci(s, u, unit.token);
                if (got != 0) {
                    n.unit = &unit;
                    n.end = u + got;
                    break;
                }
            }
        }
    }

    // ---- the year decision, on evidence only (see the header) --------------
    if (opts.year_ordinals && n.unit == nullptr && n.frac_digits.empty() &&
        n.int_digits.size() == 4 && n.int_digits[0] != '0') {
        const std::int64_t v = to_int64(n.int_digits);
        if (v >= 1000 && v <= 2999) {
            const std::string prev = preceding_word_lower(s, i);
            bool cue = false;
            for (const YearCue& p : kYearCuePrepositions) {
                if (prev == p.word) {
                    cue = true;
                    n.year_case = p.grammatical_case;
                    break;
                }
            }
            if (!cue) {
                const std::size_t after = skip_spaces(s, n.end);
                for (const YearCue& w : kYearCueNouns) {
                    const std::size_t got = match_ci(s, after, w.word);
                    if (got != 0 && at_token_end(s, after + got)) {
                        cue = true;
                        n.year_case = w.grammatical_case;
                        break;
                    }
                }
            }
            if (!cue && v >= 1900 && v <= 2099) {
                // Nothing follows but punctuation or the end of the chunk, so
                // there is no noun for it to be counting. A bare year is by far
                // the likeliest reading of "2026." in prose.
                const std::size_t after = skip_spaces(s, n.end);
                cue = after >= s.size() || !is_letter_cp(decode(s, after).cp);
            }
            n.year = cue;
        }
    }
    return n;
}

void emit_number(std::string& out, const NumberSpan& n, const NormalizerOptions& opts) {
    // Unspeakable as a quantity: an id, a version, a card number.
    if (!digits_are_spoken(n.int_digits)) {
        append_digits_individually(out, n.int_digits);
        if (!n.frac_digits.empty()) {
            append_word(out, "точка");
            append_digits_individually(out, n.frac_digits);
        }
        if (n.unit != nullptr) append_word(out, n.unit->many);
        return;
    }

    const std::int64_t iv = to_int64(n.int_digits);

    if (!n.frac_digits.empty()) {
        // "три целых четырнадцать сотых". Both halves are feminine because they
        // agree with целая/десятая, not with whatever the unit is.
        const std::size_t fd = n.frac_digits.size();
        append_word(out, RussianCardinal(iv, NumberGender::Feminine));
        append_word(out, PluralForm(iv, "целая", "целых", "целых"));
        if (fd < kFractionNouns.size() && digits_are_spoken(n.frac_digits)) {
            const std::int64_t fv = to_int64(n.frac_digits);
            const FractionNoun& fn = kFractionNouns[fd];
            append_word(out, RussianCardinal(fv, NumberGender::Feminine));
            append_word(out, PluralForm(fv, fn.one, fn.many, fn.many));
        } else {
            // Past the named denominators (or a leading-zero fraction like
            // "0,05"): the digits themselves are unambiguous, the denominator
            // name would not be.
            append_digits_individually(out, n.frac_digits);
        }
        // A fraction takes the genitive singular: "3,5 процента".
        if (n.unit != nullptr) append_word(out, n.unit->few);
        return;
    }

    if (n.year && opts.year_ordinals) {
        append_word(out, RussianOrdinalMasculine(iv, n.year_case));
        return;
    }

    const NumberGender g = n.unit != nullptr ? n.unit->gender : NumberGender::Masculine;
    append_word(out, RussianCardinal(iv, g));
    if (n.unit != nullptr) {
        append_word(out, PluralForm(iv, n.unit->one, n.unit->few, n.unit->many));
    }
}

// One symbol, when it is not already part of a number. Returns bytes consumed.
std::size_t read_symbol(std::string_view s, std::size_t i, std::string& out) {
    const Decoded d = decode(s, i);
    if (!d.ok) return 0;

    switch (d.cp) {
        case kNumero: append_word(out, "номер");        return d.len;
        case kDegree: append_word(out, "градусов");     return d.len;
        case kRuble:  append_word(out, "рублей");       return d.len;
        case kEuro:   append_word(out, "евро");         return d.len;
        case kMultiply: append_word(out, "умножить на"); return d.len;
        case '%':     append_word(out, "процентов");    return d.len;
        case '=':     append_word(out, "равно");        return d.len;
        case '&':     append_word(out, "и");            return d.len;
        default: break;
    }

    // Signs, which are only signs next to a number. '+' in particular MUST NOT
    // be touched otherwise: it is the stress mark this pipeline puts in later
    // (stress_marker.hpp), and "C++" is a name.
    if (d.cp == '+' || d.cp == '-' || d.cp == kMinusSgn) {
        const std::size_t after = skip_spaces(s, i + d.len);
        if (after < s.size() && is_digit(s[after])) {
            if (d.cp == '+') {
                append_word(out, "плюс");
                return d.len;
            }
            // A hyphen between digits is a range ("10-20"), not a minus. Only a
            // sign at a token start is negative.
            const bool token_start = i == 0 || is_space_cp(decode(s, i - 1).cp) ||
                                     s[i - 1] == '(';
            if (token_start) {
                append_word(out, "минус");
                return d.len;
            }
        }
    }
    return 0;
}

}  // namespace

// -----------------------------------------------------------------------------
std::string ExpandForSpeech(std::string_view utf8, const NormalizerOptions& opts) {
    std::string out;
    out.reserve(utf8.size() + utf8.size() / 2);

    bool at_word_start = true;

    for (std::size_t i = 0; i < utf8.size();) {
        // Abbreviations first: "т.д." must be recognised before the '.' inside
        // it reaches any other reader.
        if (opts.expand_abbreviations && at_word_start) {
            std::size_t taken = 0;
            for (const Abbrev& a : kAbbrevs) {
                const std::size_t got = match_ci(utf8, i, a.token);
                if (got != 0 && at_token_end(utf8, i + got)) {
                    append_word(out, a.words);
                    taken = got;
                    break;
                }
            }
            if (taken != 0) {
                i += taken;
                at_word_start = false;
                continue;
            }
        }

        if (opts.expand_numbers && is_digit(utf8[i])) {
            const NumberSpan n = read_number(utf8, i, opts);
            emit_number(out, n, opts);
            i = n.end;
            at_word_start = false;
            continue;
        }

        if (opts.expand_symbols) {
            const std::size_t got = read_symbol(utf8, i, out);
            if (got != 0) {
                i += got;
                at_word_start = false;
                continue;
            }
        }

        // Everything else is copied verbatim, one codepoint (or one malformed
        // byte) at a time.
        const Decoded d = decode(utf8, i);
        out.append(utf8, i, d.len);
        i += d.len;
        at_word_start = !d.ok || (!is_letter_cp(d.cp) && !is_digit_cp(d.cp));
    }
    return out;
}

}  // namespace blackwell::tts
