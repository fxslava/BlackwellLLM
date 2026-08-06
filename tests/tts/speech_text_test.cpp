// -----------------------------------------------------------------------------
// speech_text_test.cpp — NormalizeForSpeech, the last thing that touches a chunk
// before it becomes character ids.
//
// WHAT THESE ASSERT, AND WHY IT MATTERS RATHER THAN BEING A STRING-UTILITY
// SUITE: every case here is a way for text to reach a CHARACTER-LEVEL tokenizer
// carrying a symbol it does not have. F5 maps an unknown character to id 0,
// which IS the space character (f5_tokenizer.hpp) -- so the failure mode under
// test is not a mispronunciation, it is a PAUSE inserted inside a word, and it
// raises no error anywhere.
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <string>

#include "speech_text.hpp"

using blackwell::tts::NormalizeForSpeech;
using blackwell::tts::SpeechTextOptions;
using blackwell::tts::StressPolicy;

namespace {

// ---- Markdown ---------------------------------------------------------------
// The model is ASKED not to put Markdown in the spoken half (reply_split.hpp);
// this is what happens when it does anyway, which for an 8B backbone is often.

TEST(SpeechText, EmphasisMarkersAreRemovedButTheWordsSurvive) {
    EXPECT_EQ(NormalizeForSpeech("This is **very** important"), "This is very important");
    EXPECT_EQ(NormalizeForSpeech("a ~~struck~~ word"), "a struck word");
    EXPECT_EQ(NormalizeForSpeech("call `run()` now"), "call run() now");
}

TEST(SpeechText, UnderscoresSurviveInsideIdentifiers) {
    // The case that makes a blanket strip wrong: technical answers name flags,
    // and "maxnewtokens" is not the word the user asked about.
    EXPECT_EQ(NormalizeForSpeech("set max_new_tokens higher"), "set max_new_tokens higher");
    EXPECT_EQ(NormalizeForSpeech("_emphasised_ text"), "emphasised text");
}

TEST(SpeechText, ListsHeadingsAndQuotesLoseOnlyTheirMarkers) {
    EXPECT_EQ(NormalizeForSpeech("## Heading"), "Heading");
    EXPECT_EQ(NormalizeForSpeech("- first\n- second"), "first second");
    EXPECT_EQ(NormalizeForSpeech("> quoted"), "quoted");
}

TEST(SpeechText, LinkTextIsSpokenAndTheUrlIsNot) {
    // Reading "h t t p s colon slash slash" aloud is worse than silence.
    EXPECT_EQ(NormalizeForSpeech("see [the docs](https://example.com/x) for more"),
              "see the docs for more");
}

TEST(SpeechText, ALiteralLessThanIsNotAMarkdownConstruct) {
    EXPECT_EQ(NormalizeForSpeech("if a < b then"), "if a < b then");
}

// ---- stress marks -----------------------------------------------------------
// Two notations, one meaning, and only one of them can be in any given vocab.

TEST(SpeechText, StripRemovesBothStressNotations) {
    SpeechTextOptions o;
    o.stress = StressPolicy::Strip;
    EXPECT_EQ(NormalizeForSpeech("хорошо́", o), "хорошо");
    EXPECT_EQ(NormalizeForSpeech("хорош+о", o), "хорошо");
}

TEST(SpeechText, PlusPolicyMovesTheAccentAheadOfItsVowel) {
    SpeechTextOptions o;
    o.stress = StressPolicy::Plus;
    // The combining accent follows its vowel; the '+' convention precedes it.
    // Converting is therefore an INSERT ahead of a character already emitted --
    // the one operation that has to be byte-exact or it lands mid-sequence, and
    // Cyrillic is two bytes per character.
    EXPECT_EQ(NormalizeForSpeech("хорошо́", o), "хорош+о");
    EXPECT_EQ(NormalizeForSpeech("хорош+о", o), "хорош+о");   // already correct
}

TEST(SpeechText, CombiningPolicyMovesTheAccentAfterItsVowel) {
    SpeechTextOptions o;
    o.stress = StressPolicy::Combining;
    EXPECT_EQ(NormalizeForSpeech("хорош+о", o), "хорошо́");
    EXPECT_EQ(NormalizeForSpeech("хорошо́", o), "хорошо́");
}

TEST(SpeechText, ArithmeticPlusIsNotAStressMark) {
    // The discriminator is "is a vowel next?", and it has to be, because a
    // rewritten "2 + 2" would be read as "2 2".
    for (const StressPolicy p : {StressPolicy::Strip, StressPolicy::Plus,
                                 StressPolicy::Combining}) {
        SpeechTextOptions o;
        o.stress = p;
        EXPECT_EQ(NormalizeForSpeech("2 + 2 = 4", o), "2 + 2 = 4");
        EXPECT_EQ(NormalizeForSpeech("C++ code", o), "C++ code");
    }
}

TEST(SpeechText, ABulletDashIsNotConfusedWithAStressPlus) {
    // "+ item" is a list marker; "+о" is a stress mark. The space is the whole
    // difference, and both rules are live at once.
    SpeechTextOptions o;
    o.stress = StressPolicy::Plus;
    EXPECT_EQ(NormalizeForSpeech("+ пункт", o), "пункт");
    EXPECT_EQ(NormalizeForSpeech("сл+ово", o), "сл+ово");
}

// ---- symbols and whitespace -------------------------------------------------

TEST(SpeechText, UnspeakableSymbolsGoAndProsodyPunctuationStays) {
    EXPECT_EQ(NormalizeForSpeech("Готово \xF0\x9F\x91\x8D"), "Готово");
    // Dashes, guillemets and the ellipsis are how the model writes prosody.
    EXPECT_EQ(NormalizeForSpeech("Да — конечно…"), "Да — конечно…");
}

TEST(SpeechText, WhitespaceIsCollapsedAndTrimmed) {
    EXPECT_EQ(NormalizeForSpeech("  a\n\n  b\t c  "), "a b c");
}

TEST(SpeechText, DecorationOnlyInputNormalisesToNothing) {
    // The caller treats an empty result as "no words here" and skips synthesis,
    // so a chunk of "**" must not become a chunk of two unknown ids.
    EXPECT_TRUE(NormalizeForSpeech("**  **").empty());
}

TEST(SpeechText, MalformedUtf8IsPassedThroughRatherThanEaten) {
    // Code-point integrity belongs to the chunker; a normaliser that silently
    // dropped bytes would hide a break in it.
    const std::string torn = "a\xD0";
    EXPECT_EQ(NormalizeForSpeech(torn), torn);
}

}  // namespace
