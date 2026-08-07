#pragma once
// -----------------------------------------------------------------------------
// text_normalizer.hpp — ExpandForSpeech: digits and symbols become WORDS.
//
// WHY IT IS A PIPELINE STAGE AND NOT A PROMPT INSTRUCTION. F5 is a
// character-level model (f5_tokenizer.hpp): it is handed "123" as three glyphs
// and has to invent a pronunciation for them from whatever the training data
// happened to contain. Russian TTS checkpoints are trained on normalised text,
// so a bare digit run reliably produces one of three failures -- it is read in
// the wrong language, read digit-by-digit, or skipped as if it were not there.
// Worse, '%' and '№' are usually not in the vocabulary at all, and an
// out-of-vocabulary character maps to id 0, which IS the space character. The
// symptom is not a mispronunciation: it is a hole in the sentence.
//
// The model above cannot be asked to fix this either. It does not know what
// vocabulary the voice pack shipped with, and "write numbers as words" is
// exactly the kind of instruction an 8B backbone follows for two sentences and
// then forgets mid-reply. A normalisation the pipeline can PROVE is worth more
// than an instruction the model may follow -- the same argument speech_text.hpp
// makes for Markdown.
//
// =============================================================================
// RUSSIAN GRAMMAR IS THE HARD PART, AND IT IS WHY THIS IS NOT A LOOKUP TABLE
// =============================================================================
// "2 процента" and "5 процентов" differ; so do "один рубль", "два рубля",
// "пять рублей". Getting the agreement wrong is immediately audible to a native
// speaker, so every unit carries its three forms (one / few / many) and the
// selector is the standard 10/100 rule. Scale words carry GENDER too: it is
// "две тысячи" (feminine) but "два миллиона" (masculine), which is the single
// most common way a naive number-to-words function gives itself away.
//
// YEARS ARE ORDINAL, and that is a semantic judgement no purely lexical rule
// can make: "2026" is "две тысячи двадцать шестой" in a date and "две тысячи
// двадцать шесть" as a count. So the year reading is taken only on EVIDENCE --
// a preposition in front ("в 2026"), a год-form behind ("2026 году"), or a bare
// four-digit number in the plausible range with no noun after it to count. See
// kYearCuePrepositions. Everything else is cardinal, because reading a quantity
// as a date is the more jarring of the two errors.
//
// WHAT IT DELIBERATELY DOES NOT DO. No dates ("12.05.2026"), no phone-number
// formatting, no Roman numerals, no currency-amount grammar beyond the unit
// itself. Each of those is a small parser with its own ambiguities, and a wrong
// expansion is worse than an unexpanded digit run -- the listener can decode
// "twelve oh five", but not a confidently wrong date. Long digit runs (more
// than kMaxSpokenDigits) and runs with a leading zero are therefore read
// DIGIT BY DIGIT, which is what a person does with an account number.
//
// ORDER IN THE PIPELINE: after NormalizeForSpeech (which removes Markdown, so
// this never sees "**2026**") and BEFORE the accentor (stress_marker.hpp),
// which needs words, not digits, to place its marks.
//
// PURE FUNCTION: no state beyond immutable tables, callable from any thread.
// Runs per chunk on the TTS worker, where a ~200 ms diffusion step dwarfs it,
// so it is written for clarity rather than speed.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace blackwell::tts {

// Digit runs longer than this are read one digit at a time. Above it a number
// is almost never a quantity -- it is an id, a card number or a hash, and
// "сто двадцать три миллиарда..." is not what the listener wants to hear.
inline constexpr std::size_t kMaxSpokenDigits = 12;

// Grammatical gender of the noun a numeral agrees with. Russian only inflects
// 1 and 2 ("один/одна/одно", "два/две"), but those two cover most of the damage.
enum class NumberGender : std::int32_t { Masculine = 0, Feminine = 1, Neuter = 2 };

struct NormalizerOptions {
    // "123" -> "сто двадцать три". Off leaves digit runs untouched.
    bool expand_numbers = true;

    // '%', '№', '°', '+', '=', currency signs -> words. These are the entries
    // most likely to be missing from a voice pack's vocabulary, i.e. the ones
    // whose absence is silent.
    bool expand_symbols = true;

    // "т.д." -> "так далее", "млн" -> "миллионов", "руб." -> "рублей".
    // Conservative by construction: only unambiguous forms are listed, so
    // single-letter units ("г" = год or грамм) are left alone.
    bool expand_abbreviations = true;

    // Read a four-digit number as an ordinal when the context says "year".
    // Off makes every number a cardinal, which is the safe reading for text
    // that is mostly quantities.
    bool year_ordinals = true;
};

// UTF-8 in, UTF-8 out. Text outside the constructs above is copied byte for
// byte, malformed input included -- code-point integrity belongs to the chunker
// (text_chunker.hpp), and a normaliser that silently ate bytes would hide a
// break in it.
[[nodiscard]] std::string ExpandForSpeech(std::string_view utf8,
                                          const NormalizerOptions& opts = {});

// ---- exposed for tests, and useful on their own -----------------------------

// Cardinal: 2026 -> "две тысячи двадцать шесть" (Feminine picks "одна"/"две"
// for the units, which is what agreement with a feminine noun needs).
// Negative values are prefixed with "минус"; the range is the full int64.
[[nodiscard]] std::string RussianCardinal(std::int64_t value,
                                          NumberGender gender = NumberGender::Masculine);

// The four cases a year actually appears in. Anything else ("о 2026 годе")
// is rare enough that reading it in the nominative is the honest failure.
enum class OrdinalCase : std::int32_t {
    Nominative = 0,     // "две тысячи двадцать шестой"
    Genitive = 1,       // "с две тысячи двадцать шестого"
    Dative = 2,         // "к две тысячи двадцать шестому"
    Prepositional = 3,  // "в две тысячи двадцать шестом году"
};

// Ordinal, masculine: 2026 -> "две тысячи двадцать шестой".
// ONLY the final component inflects, which is what makes "тысяча девятьсот
// девяностый" come out right. Values outside [1, 999999] fall back to the
// cardinal, because nothing in this pipeline needs "миллиардный" and inventing
// it would be untested code on a hot path.
//
// THE CASE IS NOT DECORATION. "в 2026 году" read as a nominative ("в две
// тысячи двадцать шестой году") is the most common date phrasing in Russian
// and lands as a grammatical error in the middle of the sentence -- exactly the
// kind of wrongness a listener attributes to the assistant rather than to the
// text. ExpandForSpeech picks the case from the preposition, falling back to
// the год-form; see kYearCuePrepositions.
[[nodiscard]] std::string RussianOrdinalMasculine(
    std::int64_t value, OrdinalCase grammatical_case = OrdinalCase::Nominative);

// Picks the one/few/many form for `value` by the standard Russian rule
// (…1 but not …11 -> one; …2-4 but not …12-14 -> few; otherwise many).
// Exposed because callers that format their own counts need the same rule.
[[nodiscard]] std::string_view PluralForm(std::int64_t value, std::string_view one,
                                          std::string_view few, std::string_view many);

}  // namespace blackwell::tts
