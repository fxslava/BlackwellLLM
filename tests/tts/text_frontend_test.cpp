// -----------------------------------------------------------------------------
// text_frontend_test.cpp — the two stages between the chunker and the tokenizer:
// number/symbol expansion (text_normalizer.hpp) and stress placement
// (stress_marker.hpp).
//
// WHY THESE ARE WORTH TESTING AT ALL. Both failures are SILENT at every layer
// below them. An unexpanded "%" is a valid string that tokenises to id 0 -- the
// space character -- so the symptom is a pause, not an error; a mis-stressed
// word is perfectly intelligible to every assertion in the pipeline and wrong
// only to a listener. Neither can be caught downstream, so the grammar has to
// be pinned here.
//
// CPU-only and model-free: both stages are pure functions over UTF-8.
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <string>

#include "stress_marker.hpp"
#include "text_normalizer.hpp"

using blackwell::tts::AccentOptions;
using blackwell::tts::DictionaryAccentor;
using blackwell::tts::ExpandForSpeech;
using blackwell::tts::NormalizerOptions;
using blackwell::tts::NumberGender;
using blackwell::tts::PlusPlacement;
using blackwell::tts::RussianCardinal;
using blackwell::tts::RussianOrdinalMasculine;

// =============================================================================
// Cardinals
// =============================================================================
TEST(RussianNumbers, CardinalBasics) {
    EXPECT_EQ(RussianCardinal(0), "ноль");
    EXPECT_EQ(RussianCardinal(1), "один");
    EXPECT_EQ(RussianCardinal(11), "одиннадцать");
    EXPECT_EQ(RussianCardinal(21), "двадцать один");
    EXPECT_EQ(RussianCardinal(100), "сто");
    EXPECT_EQ(RussianCardinal(123), "сто двадцать три");
    EXPECT_EQ(RussianCardinal(999), "девятьсот девяносто девять");
}

// The thousands group is FEMININE ("две тысячи", not "два тысячи") while the
// millions group is masculine. Getting this wrong is the classic tell of a
// number-to-words function written against English.
TEST(RussianNumbers, ScaleWordsCarryTheirOwnGender) {
    EXPECT_EQ(RussianCardinal(1000), "тысяча");
    EXPECT_EQ(RussianCardinal(2000), "две тысячи");
    EXPECT_EQ(RussianCardinal(5000), "пять тысяч");
    EXPECT_EQ(RussianCardinal(2026), "две тысячи двадцать шесть");
    EXPECT_EQ(RussianCardinal(2000000), "два миллиона");
    EXPECT_EQ(RussianCardinal(21000), "двадцать одна тысяча");
}

TEST(RussianNumbers, CardinalTakesTheCallersGenderForTheUnitsGroup) {
    EXPECT_EQ(RussianCardinal(1, NumberGender::Feminine), "одна");
    EXPECT_EQ(RussianCardinal(2, NumberGender::Feminine), "две");
    EXPECT_EQ(RussianCardinal(22, NumberGender::Feminine), "двадцать две");
    EXPECT_EQ(RussianCardinal(5, NumberGender::Feminine), "пять");   // no inflection
}

TEST(RussianNumbers, NegativeGetsMinus) {
    EXPECT_EQ(RussianCardinal(-5), "минус пять");
}

// =============================================================================
// Ordinals -- only the LAST component inflects
// =============================================================================
TEST(RussianNumbers, OrdinalInflectsOnlyTheFinalComponent) {
    EXPECT_EQ(RussianOrdinalMasculine(2026), "две тысячи двадцать шестой");
    EXPECT_EQ(RussianOrdinalMasculine(2015), "две тысячи пятнадцатый");
    EXPECT_EQ(RussianOrdinalMasculine(1990), "тысяча девятьсот девяностый");
    EXPECT_EQ(RussianOrdinalMasculine(1900), "тысяча девятисотый");
    EXPECT_EQ(RussianOrdinalMasculine(2000), "двухтысячный");
    EXPECT_EQ(RussianOrdinalMasculine(1), "первый");
    EXPECT_EQ(RussianOrdinalMasculine(40), "сороковой");
}

// "в 2026 году" read in the nominative is a grammatical error in the middle of
// the sentence -- and it is the most common date phrasing there is.
TEST(RussianNumbers, OrdinalDeclines) {
    using blackwell::tts::OrdinalCase;
    EXPECT_EQ(RussianOrdinalMasculine(2026, OrdinalCase::Prepositional),
              "две тысячи двадцать шестом");
    EXPECT_EQ(RussianOrdinalMasculine(2026, OrdinalCase::Genitive),
              "две тысячи двадцать шестого");
    EXPECT_EQ(RussianOrdinalMasculine(2026, OrdinalCase::Dative),
              "две тысячи двадцать шестому");
    EXPECT_EQ(RussianOrdinalMasculine(2000, OrdinalCase::Prepositional), "двухтысячном");
    // The one stem the -ий rule cannot derive.
    EXPECT_EQ(RussianOrdinalMasculine(2003, OrdinalCase::Genitive),
              "две тысячи третьего");
}

// =============================================================================
// The scan: numbers in context
// =============================================================================
TEST(ExpandForSpeech, PlainNumbersBecomeWords) {
    EXPECT_EQ(ExpandForSpeech("Всего 123 штуки"), "Всего сто двадцать три штуки");
}

// Agreement with the unit, which is the part a lookup table cannot do.
TEST(ExpandForSpeech, UnitsAgreeWithTheirNumber) {
    EXPECT_EQ(ExpandForSpeech("1%"), "один процент");
    EXPECT_EQ(ExpandForSpeech("2%"), "два процента");
    EXPECT_EQ(ExpandForSpeech("5%"), "пять процентов");
    EXPECT_EQ(ExpandForSpeech("11%"), "одиннадцать процентов");   // NOT "процент"
    EXPECT_EQ(ExpandForSpeech("21%"), "двадцать один процент");
    EXPECT_EQ(ExpandForSpeech("100 км"), "сто километров");
}

TEST(ExpandForSpeech, FeminineUnitsPullTheNumeralWithThem) {
    EXPECT_EQ(ExpandForSpeech("2 тыс."), "две тысячи");
    EXPECT_EQ(ExpandForSpeech("1 шт."), "одна штука");
}

TEST(ExpandForSpeech, Decimals) {
    EXPECT_EQ(ExpandForSpeech("3,14"), "три целых четырнадцать сотых");
    EXPECT_EQ(ExpandForSpeech("1,5"), "одна целая пять десятых");
    // A fraction takes the genitive singular of its unit.
    EXPECT_EQ(ExpandForSpeech("3,5%"), "три целых пять десятых процента");
}

// The year reading needs EVIDENCE. A quantity with a noun after it is a
// quantity, not a date.
TEST(ExpandForSpeech, YearsAreOrdinalOnlyOnEvidence) {
    // The preposition supplies both the evidence AND the case.
    EXPECT_EQ(ExpandForSpeech("в 2026 году"), "в две тысячи двадцать шестом году");
    EXPECT_EQ(ExpandForSpeech("с 2020 года"), "с две тысячи двадцатого года");
    EXPECT_EQ(ExpandForSpeech("Это было в 1990"), "Это было в тысяча девятьсот девяностом");
    // No cue, and a noun follows: a count.
    EXPECT_EQ(ExpandForSpeech("1500 человек"), "тысяча пятьсот человек");
    // Off: everything is a cardinal.
    NormalizerOptions off;
    off.year_ordinals = false;
    EXPECT_EQ(ExpandForSpeech("в 2026 году", off), "в две тысячи двадцать шесть году");
}

// Long runs and leading zeros are identifiers, not quantities -- a human reads
// them digit by digit too.
TEST(ExpandForSpeech, IdentifiersAreReadDigitByDigit) {
    EXPECT_EQ(ExpandForSpeech("007"), "ноль ноль семь");
    EXPECT_EQ(ExpandForSpeech("1234567890123"),
              "один два три четыре пять шесть семь восемь девять ноль один два три");
}

TEST(ExpandForSpeech, Symbols) {
    EXPECT_EQ(ExpandForSpeech("№5"), "номер пять");
    EXPECT_EQ(ExpandForSpeech("2 + 2 = 4"), "два плюс два равно четыре");
    EXPECT_EQ(ExpandForSpeech("-5 градусов"), "минус пять градусов");
}

// A '-' between digits is a range, not a sign, and '+' with no number after it
// is not arithmetic. Both matter because the accentor's marks are '+' too.
TEST(ExpandForSpeech, SignsOnlyCountNextToNumbers) {
    EXPECT_EQ(ExpandForSpeech("C++ и Rust"), "C++ и Rust");
    EXPECT_EQ(ExpandForSpeech("хорош+о"), "хорош+о");
}

TEST(ExpandForSpeech, Abbreviations) {
    EXPECT_EQ(ExpandForSpeech("и т.д."), "и так далее");
    EXPECT_EQ(ExpandForSpeech("т. п."), "тому подобное");
    EXPECT_EQ(ExpandForSpeech("т.е. вот так"), "то есть вот так");
}

TEST(ExpandForSpeech, LeavesOrdinaryTextAlone) {
    const std::string s = "Обычный текст без чисел и символов.";
    EXPECT_EQ(ExpandForSpeech(s), s);
}

TEST(ExpandForSpeech, DisabledStagesAreNoOps) {
    NormalizerOptions off;
    off.expand_numbers = false;
    off.expand_symbols = false;
    off.expand_abbreviations = false;
    EXPECT_EQ(ExpandForSpeech("123 % и т.д.", off), "123 % и т.д.");
}

// =============================================================================
// Stress placement
// =============================================================================
TEST(DictionaryAccentor, MarksAfterTheStressedVowel) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate("золотая рыбка"), "золота+я ры+бка");
}

TEST(DictionaryAccentor, PlacementIsAConvention) {
    AccentOptions before;
    before.placement = PlusPlacement::BeforeVowel;
    const DictionaryAccentor a(before);
    EXPECT_EQ(a.Accentuate("рыбка"), "р+ыбка");
}

// The whole point of the ordering with the normaliser: a number becomes words,
// and those words are exactly the ones the seed dictionary covers.
TEST(DictionaryAccentor, CoversWhatTheNormaliserEmits) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate(ExpandForSpeech("123%")), "сто два+дцать три проце+нта");
    EXPECT_EQ(a.Accentuate(ExpandForSpeech("2026")),
              "две ты+сячи два+дцать шесто+й");
}

// Refusing to guess is the design: an unknown multi-vowel word is left for the
// model's own prior rather than being marked wrongly.
TEST(DictionaryAccentor, LeavesUnknownWordsUnmarked) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate("абракадабра"), "абракадабра");
}

TEST(DictionaryAccentor, MonosyllablesAndLatinAreUntouched) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate("дом"), "дом");
    EXPECT_EQ(a.Accentuate("hello world"), "hello world");
}

// 'ё' carries the stress in simple words...
TEST(DictionaryAccentor, YoIsStressedWhenTheWordIsUnknown) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate("тётя"), "тё+тя");
}

// ...but NOT in the compounds, which is why the dictionary is consulted first.
TEST(DictionaryAccentor, DictionaryBeatsTheYoRule) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate("трёхсотый"), "трёхсо+тый");
}

// Text that already carries marks (a model that emits them, or a second pass
// over our own output) must not be marked twice.
TEST(DictionaryAccentor, AlreadyMarkedWordsPassThrough) {
    const DictionaryAccentor a;
    const std::string once = a.Accentuate("золотая рыбка");
    EXPECT_EQ(a.Accentuate(once), once);
}

TEST(DictionaryAccentor, PunctuationAndSpacingSurvive) {
    const DictionaryAccentor a;
    EXPECT_EQ(a.Accentuate("Привет, рыбка!"), "Приве+т, ры+бка!");
}

TEST(DictionaryAccentor, AcceptsBothNotationsWhenParsing) {
    // "золота+я" marks the 3rd vowel (о-о-А-я); "р+ыбка" marks the 1st. Both
    // must load, so a ruaccent dump can be used unchanged.
    EXPECT_EQ(DictionaryAccentor::VowelIndexOfMarked("золота+я"), 3);
    EXPECT_EQ(DictionaryAccentor::VowelIndexOfMarked("р+ыбка"), 1);
    EXPECT_EQ(DictionaryAccentor::VowelIndexOfMarked("C++"), 0);
}

TEST(DictionaryAccentor, AddedEntriesOverrideTheSeedTable) {
    DictionaryAccentor a;
    a.Add("абракадабра", 4);
    EXPECT_EQ(a.Accentuate("абракадабра"), "абракада+бра");
    // An index past the end of the word is refused rather than clamped.
    a.Add("абракадабра", 99);
    EXPECT_EQ(a.Accentuate("абракадабра"), "абракада+бра");
}
