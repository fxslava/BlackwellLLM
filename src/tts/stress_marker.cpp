// -----------------------------------------------------------------------------
// stress_marker.cpp — see stress_marker.hpp for why a wrongly placed mark is
// worse than no mark at all, and for the two '+' conventions.
//
// THE BUILT-IN TABLE IS WRITTEN IN THE OUTPUT NOTATION. Entries are marked
// words ("два+дцать"), not (word, index) pairs, for three reasons: it is the
// same format LoadDictionary() reads, so there is one parser rather than two;
// a reviewer can check a stress by reading it aloud; and a typo produces a
// dropped entry rather than a mark on the wrong syllable (VowelIndexOfMarked
// returns 0 for a '+' that touches no vowel, and Add ignores that).
// -----------------------------------------------------------------------------
#include "stress_marker.hpp"

#include <cstddef>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "utf8_scan.hpp"

namespace blackwell::tts {
namespace {

// Every word this pipeline can GENERATE, so a normalised number reads correctly
// end to end ("сто два+дцать три проце+нта"), plus the assistant's own
// high-frequency vocabulary. Monosyllables are absent on purpose -- they have
// nowhere for the stress to go, and the accentor short-circuits them.
// A raw array rather than std::array so the size is deduced: this list grows,
// and a hand-maintained count is a compile error waiting to be committed.
constexpr const char* const kSeedDictionary[] = {
    // ---- cardinals ---------------------------------------------------------
    "оди+н", "одна+", "одно+", "одного+", "одну+",
    "четы+ре", "во+семь", "де+вять", "де+сять",
    "оди+ннадцать", "двена+дцать", "трина+дцать", "четы+рнадцать", "пятна+дцать",
    "шестна+дцать", "семна+дцать", "восемна+дцать", "девятна+дцать",
    "два+дцать", "три+дцать", "со+рок", "пятьдеся+т", "шестьдеся+т",
    "се+мьдесят", "во+семьдесят", "девяно+сто",
    "две+сти", "три+ста", "четы+реста", "пятьсо+т", "шестьсо+т", "семьсо+т",
    "восемьсо+т", "девятьсо+т",
    "ты+сяча", "ты+сячи", "ты+сяч",
    "миллио+н", "миллио+на", "миллио+нов",
    "миллиа+рд", "миллиа+рда", "миллиа+рдов",
    "триллио+н", "триллио+на", "триллио+нов",
    "квадриллио+н", "квадриллио+на", "квадриллио+нов",
    "квинтиллио+н", "квинтиллио+на", "квинтиллио+нов",

    // ---- ordinals (the year readings) --------------------------------------
    "пе+рвый", "второ+й", "тре+тий", "пя+тый", "шесто+й", "седьмо+й",
    "восьмо+й", "девя+тый", "деся+тый",
    "оди+ннадцатый", "двена+дцатый", "трина+дцатый", "четы+рнадцатый",
    "пятна+дцатый", "шестна+дцатый", "семна+дцатый", "восемна+дцатый",
    "девятна+дцатый",
    "двадца+тый", "тридца+тый", "сороково+й", "пятидеся+тый", "шестидеся+тый",
    "семидеся+тый", "восьмидеся+тый", "девяно+стый",
    // The ё entries below are exactly the case the trust_yo rule gets WRONG:
    // in a compound the ё carries a secondary stress, not the main one.
    "со+тый", "двухсо+тый", "трёхсо+тый", "четырёхсо+тый", "пятисо+тый",
    "шестисо+тый", "семисо+тый", "восьмисо+тый", "девятисо+тый",
    "ты+сячный", "двухты+сячный", "трёхты+сячный", "четырёхты+сячный",
    "пятиты+сячный", "шеститы+сячный", "семиты+сячный", "восьмиты+сячный",
    "девятиты+сячный",

    // ---- fractions and units the normaliser emits --------------------------
    "проце+нт", "проце+нта", "проце+нтов",
    "це+лая", "це+лых", "деся+тая", "десяты+х", "со+тая", "со+тых",
    "ты+сячная", "ты+сячных", "десятиты+сячная", "десятиты+сячных",
    "гра+дус", "гра+дуса", "гра+дусов",
    "рубля+", "рубле+й", "до+ллар", "до+ллара", "до+лларов", "е+вро",
    "киломе+тр", "киломе+тра", "киломе+тров",
    "килогра+мм", "килогра+мма", "килогра+ммов",
    "миллиме+тр", "миллиме+тра", "миллиме+тров",
    "сантиме+тр", "сантиме+тра", "сантиме+тров",
    "миллили+тр", "миллили+тра", "миллили+тров",
    "гигаба+йт", "гигаба+йта", "мегаба+йт", "мегаба+йта",
    "килоба+йт", "килоба+йта", "тераба+йт", "тераба+йта",
    "шту+ка", "шту+ки", "но+мер", "ми+нус", "равно+", "умно+жить", "то+чка",
    "го+да", "году+", "годо+в", "го+ды",

    // ---- what the abbreviation table expands into --------------------------
    "да+лее", "тому+", "подо+бное", "други+е", "профе+ссор", "наприме+р",
    "страни+ца", "рису+нок", "челове+к", "ме+сяцев", "о+коло", "и+мени",

    // ---- assistant vocabulary ----------------------------------------------
    "приве+т", "спаси+бо", "пожа+луйста", "здра+вствуйте", "хорошо+",
    "коне+чно", "поэ+тому", "потому+", "сейча+с", "сего+дня", "за+втра",
    "вчера+", "всегда+", "никогда+", "мо+жет", "мо+жно", "ну+жно", "на+до",
    "о+чень", "то+лько", "е+сли", "когда+", "что+бы", "тепе+рь", "уже+",
    "ме+ньше", "бо+льше", "лу+чше", "ху+же", "вре+мя", "го+род", "лю+ди",
    "рабо+та", "вопро+с", "отве+т", "зада+ча", "приме+р", "сло+во", "язы+к",
    "систе+ма", "програ+мма", "компью+тер", "интерне+т", "телефо+н",
    "да+нные", "па+пка", "оши+бка", "реше+ние", "значе+ние", "результа+т",
    "поря+док", "но+вый", "ста+рый", "бы+стро", "ме+дленно", "про+сто",
    "сло+жно", "гото+во", "поня+тно", "извини+те", "пра+вильно",
    "непра+вильно", "включи+ть", "вы+ключить", "откры+ть", "закры+ть",
    "сде+лать", "де+лать", "ду+маю", "зна+ю", "хочу+", "могу+", "бу+дет",
    "бы+ло", "се+рвер", "по+иск", "найти+", "показа+ть", "сказа+ть",
    "говори+ть", "слу+шаю", "повтори+те", "золота+я", "ры+бка",
};

std::string strip_marks(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        if (c != '+') out.push_back(c);
    }
    return out;
}

std::size_t count_vowels(std::string_view word) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < word.size();) {
        const utf8::Char c = utf8::Decode(word, i);
        if (c.ok && utf8::IsVowel(c.cp)) ++n;
        i += c.len;
    }
    return n;
}

// 1-based index of the first 'ё', or 0. The orthographic rule "ё is always
// stressed" holds for simple words; the compounds it fails on are in the seed
// table above, which is consulted first.
std::size_t yo_index(std::string_view word) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < word.size();) {
        const utf8::Char c = utf8::Decode(word, i);
        if (c.ok && utf8::IsVowel(c.cp)) {
            ++n;
            if (c.cp == 0x451u || c.cp == 0x401u) return n;
        }
        i += c.len;
    }
    return 0;
}

std::string_view trim(std::string_view s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' ||
                     s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}

}  // namespace

DictionaryAccentor::DictionaryAccentor(AccentOptions opts) : opts_(opts) {
    dict_.reserve(std::size(kSeedDictionary) * 2);
    for (const char* entry : kSeedDictionary) {
        const std::string_view marked(entry);
        Add(strip_marks(marked), VowelIndexOfMarked(marked));
    }
}

int DictionaryAccentor::VowelIndexOfMarked(std::string_view marked) {
    int vowels = 0;
    for (std::size_t i = 0; i < marked.size();) {
        const utf8::Char c = utf8::Decode(marked, i);
        const std::size_t next = i + c.len;

        if (c.ok && c.cp == '+') {
            // Which side of the vowel is this mark on? Both notations are
            // accepted so a ruaccent dump ("р+ыбка") loads unchanged.
            utf8::Char prev{};
            std::size_t prev_start = 0;
            const bool has_prev = utf8::DecodePrev(marked, i, prev, prev_start);
            const utf8::Char after = utf8::Decode(marked, next);

            if (has_prev && prev.ok && utf8::IsVowel(prev.cp)) return vowels;
            if (after.ok && utf8::IsVowel(after.cp)) return vowels + 1;
            // A '+' touching no vowel is not a stress mark (arithmetic, "C++",
            // a typo in a dictionary line). Ignored rather than guessed at.
            i = next;
            continue;
        }
        if (c.ok && utf8::IsVowel(c.cp)) ++vowels;
        i = next;
    }
    return 0;
}

void DictionaryAccentor::Add(std::string_view word, int vowel_index) {
    if (word.empty() || vowel_index < 1) return;
    const std::string key = utf8::ToLower(word);
    // An index past the end of the word would mark nothing; rejecting it here
    // keeps AccentuateWord free of a failure case it cannot report.
    if (static_cast<std::size_t>(vowel_index) > count_vowels(key)) return;
    dict_[key] = vowel_index;
}

std::size_t DictionaryAccentor::LoadDictionary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("DictionaryAccentor: cannot open stress dictionary: " + path);
    }

    std::size_t accepted = 0;
    std::string line;
    while (std::getline(f, line)) {
        const std::string_view t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        // Form 1: a marked word. Form 2: "word <1-based vowel index>".
        const std::size_t sep = t.find_first_of(" \t");
        std::string word;
        int index = 0;
        if (sep == std::string_view::npos) {
            word = strip_marks(t);
            index = VowelIndexOfMarked(t);
        } else {
            word = strip_marks(t.substr(0, sep));
            const std::string_view num = trim(t.substr(sep + 1));
            index = 0;
            for (const char c : num) {
                if (c < '0' || c > '9') { index = 0; break; }
                index = index * 10 + (c - '0');
            }
            // A marked word with a trailing comment still parses: the mark wins.
            if (index == 0) index = VowelIndexOfMarked(t.substr(0, sep));
        }
        // Add's own acceptance rule, restated so the count means "this line
        // took effect" rather than "the map grew" (an override does not grow it).
        const bool usable = !word.empty() && index >= 1 &&
                            static_cast<std::size_t>(index) <= count_vowels(word);
        Add(word, index);
        if (usable) ++accepted;
    }
    return accepted;
}

std::string DictionaryAccentor::AccentuateWord(std::string_view word) const {
    if (word.empty()) return std::string();

    int index = 0;
    const auto it = dict_.find(utf8::ToLower(word));
    if (it != dict_.end()) {
        index = it->second;
    } else if (opts_.trust_yo) {
        index = static_cast<int>(yo_index(word));
    }
    if (index == 0) {
        const std::size_t vowels = count_vowels(word);
        // One vowel: unambiguous, so a mark adds a character and no
        // information. No vowels or several unknown ones: refuse to guess.
        if (!(vowels == 1 && opts_.mark_monosyllables)) return std::string(word);
        index = 1;
    }

    std::string out;
    out.reserve(word.size() + 1);
    int seen = 0;
    for (std::size_t i = 0; i < word.size();) {
        const utf8::Char c = utf8::Decode(word, i);
        const bool vowel = c.ok && utf8::IsVowel(c.cp);
        if (vowel) ++seen;
        if (vowel && seen == index && opts_.placement == PlusPlacement::BeforeVowel) {
            out.push_back('+');
        }
        out.append(word, i, c.len);
        if (vowel && seen == index && opts_.placement == PlusPlacement::AfterVowel) {
            out.push_back('+');
        }
        i += c.len;
    }
    return out;
}

std::string DictionaryAccentor::Accentuate(std::string_view utf8_text) const {
    std::string out;
    out.reserve(utf8_text.size() + utf8_text.size() / 8);

    for (std::size_t i = 0; i < utf8_text.size();) {
        const utf8::Char c = utf8::Decode(utf8_text, i);
        if (!c.ok || !utf8::IsLetter(c.cp)) {
            out.append(utf8_text, i, c.len);
            i += c.len;
            continue;
        }

        // One word run. A '+' INSIDE it means the text arrived already marked
        // (a model that emits stress, or a second pass over our own output);
        // such a word is copied untouched rather than marked twice.
        std::size_t j = i;
        bool already_marked = false;
        bool all_cyrillic = true;
        while (j < utf8_text.size()) {
            const utf8::Char d = utf8::Decode(utf8_text, j);
            if (d.ok && utf8::IsLetter(d.cp)) {
                if (!utf8::IsCyrillic(d.cp)) all_cyrillic = false;
                j += d.len;
                continue;
            }
            if (d.ok && d.cp == '+') {
                const utf8::Char e = utf8::Decode(utf8_text, j + d.len);
                if (e.ok && utf8::IsLetter(e.cp)) {
                    already_marked = true;
                    j += d.len;
                    continue;
                }
            }
            break;
        }

        const std::string_view word = utf8_text.substr(i, j - i);
        // Latin runs are left alone: this table is Russian, and the vowel rule
        // above ("one vowel, no mark") is not even true of English.
        if (already_marked || !all_cyrillic) {
            out.append(word);
        } else {
            out.append(AccentuateWord(word));
        }
        i = j;
    }
    return out;
}

}  // namespace blackwell::tts
