// =============================================================================
// tests/tts/phonemizer_test.cpp
//
// Phase 1 of docs/TTS_INTEGRATION_AUDIT.md: the text frontend behind the
// IPhonemizer seam. CPU-only, no ONNXRuntime, no model, no voice file — the
// voice is a hand-built TtsVoice, which is the point of that struct owning the
// symbol table.
//
// The load-bearing assertions are the Cyrillic ones. Russian is the translator's
// primary target language and Cyrillic is 2-byte UTF-8, so a frontend that walks
// bytes instead of codepoints does not fail loudly — it silently emits a stream
// of unknown symbols and the model speaks nothing. That failure looks like "the
// voice is broken", which is the wrong place to go looking.
//
// This file is compiled with /utf-8 (see tests/CMakeLists.txt): the literals
// below must reach the compiler as the same bytes the phonemizer will see.
// =============================================================================
#include "phonemizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::tts::kNoId;
using blackwell::tts::PassthroughPhonemizer;
using blackwell::tts::TtsStatus;
using blackwell::tts::TtsVoice;
using blackwell::tts::utf8_sequence_length;

using Ids = std::vector<std::int64_t>;

// A minimal English grapheme voice: bare symbols, no decorations. Ids are
// arbitrary but distinct, so a mapping mistake shows up as a wrong value rather
// than a plausible-looking one.
TtsVoice ascii_voice() {
    TtsVoice v;
    v.id = "test-ascii";
    v.language_index = 2;                      // "English" in rt::kLanguages
    v.symbol_to_id = {{"a", 10}, {"b", 11}, {"c", 12}, {" ", 1}, {".", 2}};
    return v;
}

// A Cyrillic grapheme voice — every symbol here is a 2-byte UTF-8 sequence.
TtsVoice cyrillic_voice() {
    TtsVoice v;
    v.id = "test-ru";
    v.language_index = 1;                      // "Russian" in rt::kLanguages
    v.symbol_to_id = {{"п", 20}, {"р", 21}, {"и", 22}, {"в", 23},
                      {"е", 24}, {"т", 25}, {" ", 1}};
    return v;
}

// ---- the UTF-8 scanner ------------------------------------------------------

TEST(Utf8SequenceLength, AcceptsWellFormedSequences) {
    EXPECT_EQ(utf8_sequence_length("a", 0), 1u);          // ASCII
    EXPECT_EQ(utf8_sequence_length("п", 0), 2u);          // U+043F Cyrillic
    EXPECT_EQ(utf8_sequence_length("—", 0), 3u);          // U+2014 em dash
    EXPECT_EQ(utf8_sequence_length("😀", 0), 4u);         // U+1F600
}

TEST(Utf8SequenceLength, RejectsMalformedAndOutOfRange) {
    EXPECT_EQ(utf8_sequence_length("", 0), 0u);           // empty
    EXPECT_EQ(utf8_sequence_length("a", 5), 0u);          // pos past the end
    EXPECT_EQ(utf8_sequence_length("\x80", 0), 0u);       // bare continuation byte
    EXPECT_EQ(utf8_sequence_length("\xFF", 0), 0u);       // invalid lead
    EXPECT_EQ(utf8_sequence_length("\xC3", 0), 0u);       // 2-byte lead, truncated
    EXPECT_EQ(utf8_sequence_length("\xE2\x82", 0), 0u);   // 3-byte lead, truncated
    EXPECT_EQ(utf8_sequence_length("\xC3\x28", 0), 0u);   // bad continuation byte
}

// The scanner must be able to walk INTO a string, not just start at 0 — the
// phonemizer's loop depends on it.
TEST(Utf8SequenceLength, WalksAMixedString) {
    const std::string s = "aпb";
    EXPECT_EQ(utf8_sequence_length(s, 0), 1u);
    EXPECT_EQ(utf8_sequence_length(s, 1), 2u);
    EXPECT_EQ(utf8_sequence_length(s, 3), 1u);
}

// ---- passthrough: the happy paths -------------------------------------------

TEST(PassthroughPhonemizer, MapsAsciiSymbolsInOrder) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids;

    ASSERT_EQ(p.to_ids("abc", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{10, 11, 12}));
    EXPECT_EQ(p.unknown_symbols(), 0u);
    EXPECT_EQ(p.invalid_sequences(), 0u);
}

// THE test of this file: multi-byte codepoints must map one symbol per
// CODEPOINT, not one per byte.
TEST(PassthroughPhonemizer, MapsCyrillicOneIdPerCodepoint) {
    PassthroughPhonemizer p;
    const TtsVoice v = cyrillic_voice();
    Ids ids;

    ASSERT_EQ(p.to_ids("привет", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{20, 21, 22, 23, 24, 25}));
    EXPECT_EQ(ids.size(), 6u) << "6 codepoints, 12 bytes — a byte-wise walk would emit 12";
    EXPECT_EQ(p.unknown_symbols(), 0u);
}

TEST(PassthroughPhonemizer, SpaceIsASymbolLikeAnyOther) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids;
    ASSERT_EQ(p.to_ids("a b", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{10, 1, 11}));
}

// ---- decorations ------------------------------------------------------------

TEST(PassthroughPhonemizer, AppliesBosAndEos) {
    PassthroughPhonemizer p;
    TtsVoice v = ascii_voice();
    v.bos_id = 100;
    v.eos_id = 101;
    Ids ids;

    ASSERT_EQ(p.to_ids("ab", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{100, 10, 11, 101}));
}

// The Piper layout: bos, pad, s0, pad, s1, pad, eos. The LEADING pad (right
// after bos) is the part that is easy to omit and clips the utterance onset.
TEST(PassthroughPhonemizer, InterleavesPadIncludingTheLeadingOne) {
    PassthroughPhonemizer p;
    TtsVoice v = ascii_voice();
    v.bos_id = 100;
    v.eos_id = 101;
    v.pad_id = 0;
    v.interleave_pad = true;
    Ids ids;

    ASSERT_EQ(p.to_ids("ab", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{100, 0, 10, 0, 11, 0, 101}));
}

// interleave_pad without a pad id is a malformed sidecar, not a crash: the flag
// is ignored rather than emitting kNoId (-1) into the graph's embedding lookup.
TEST(PassthroughPhonemizer, InterleavePadWithoutPadIdIsIgnored) {
    PassthroughPhonemizer p;
    TtsVoice v = ascii_voice();
    v.interleave_pad = true;
    ASSERT_EQ(v.pad_id, kNoId);
    Ids ids;

    ASSERT_EQ(p.to_ids("ab", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{10, 11}));
}

// ---- degradation ------------------------------------------------------------

TEST(PassthroughPhonemizer, UnknownSymbolsAreSkippedAndCounted) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids;

    ASSERT_EQ(p.to_ids("axbyc", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{10, 11, 12}));
    EXPECT_EQ(p.unknown_symbols(), 2u);
}

// The misconfiguration this counter exists to expose: Cyrillic text handed to an
// English-only voice pronounces nothing at all.
TEST(PassthroughPhonemizer, WrongScriptYieldsEmptyResultNotSilence) {
    PassthroughPhonemizer p;
    TtsVoice v = ascii_voice();
    v.language_index = TtsVoice::kAnyLanguage;   // isolate the SCRIPT mismatch
    v.bos_id = 100;
    v.eos_id = 101;
    Ids ids;

    EXPECT_EQ(p.to_ids("привет", 1, v, ids), TtsStatus::EmptyResult);
    EXPECT_TRUE(ids.empty()) << "a bare [bos, eos] would cost a forward pass and click";
    EXPECT_EQ(p.unknown_symbols(), 6u);
}

TEST(PassthroughPhonemizer, MalformedUtf8IsSkippedAndTheRestSurvives) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids;

    // A bare continuation byte wedged between two good symbols.
    const std::string text = std::string("a") + "\x80" + "b";
    ASSERT_EQ(p.to_ids(text, v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{10, 11}));
    EXPECT_EQ(p.invalid_sequences(), 1u);
}

// Resynchronisation must advance, or the loop hangs the TTS thread on malformed
// input. A truncated lead byte is the case that would spin forever.
TEST(PassthroughPhonemizer, TruncatedSequenceTerminates) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids;

    const std::string text = std::string("ab") + "\xC3";   // 2-byte lead, nothing after
    ASSERT_EQ(p.to_ids(text, v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{10, 11}));
    EXPECT_EQ(p.invalid_sequences(), 1u);
}

TEST(PassthroughPhonemizer, EmptyAndWhitespaceOnlyInput) {
    PassthroughPhonemizer p;
    TtsVoice v = ascii_voice();
    Ids ids;

    EXPECT_EQ(p.to_ids("", v.language_index, v, ids), TtsStatus::EmptyResult);
    EXPECT_TRUE(ids.empty());

    // A space IS in this voice's table, so whitespace alone is pronounceable —
    // EmptyResult means "nothing this voice can say", not "no visible glyphs".
    ASSERT_EQ(p.to_ids(" ", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids, (Ids{1}));

    // Drop the space from the table and the same input becomes unpronounceable.
    v.symbol_to_id.erase(" ");
    EXPECT_EQ(p.to_ids(" ", v.language_index, v, ids), TtsStatus::EmptyResult);
}

// ---- caller errors ----------------------------------------------------------

TEST(PassthroughPhonemizer, LanguageMismatchIsReportedNotGuessed) {
    PassthroughPhonemizer p;
    const TtsVoice v = cyrillic_voice();        // language_index == 1 (Russian)
    Ids ids;

    EXPECT_EQ(p.to_ids("привет", 2 /*English*/, v, ids), TtsStatus::UnsupportedLanguage);
    EXPECT_TRUE(ids.empty());

    // kAnyLanguage opts out of the check entirely.
    TtsVoice any = v;
    any.language_index = TtsVoice::kAnyLanguage;
    EXPECT_EQ(p.to_ids("привет", 2, any, ids), TtsStatus::Success);
}

TEST(PassthroughPhonemizer, VoiceWithNoSymbolTableIsACallerError) {
    PassthroughPhonemizer p;
    TtsVoice v;                                  // default: empty table
    ASSERT_FALSE(v.is_usable());
    Ids ids;
    EXPECT_EQ(p.to_ids("abc", v.language_index, v, ids), TtsStatus::InvalidArgument);
    EXPECT_TRUE(ids.empty());
}

// ---- the length guard -------------------------------------------------------

TEST(PassthroughPhonemizer, TruncatesAtMaxInputIdsAndStaysWellFormed) {
    PassthroughPhonemizer p;
    TtsVoice v = ascii_voice();
    v.bos_id = 100;
    v.eos_id = 101;
    v.max_input_ids = 6;          // room for bos + 4 symbols + eos
    Ids ids;

    ASSERT_EQ(p.to_ids("abcabcabc", v.language_index, v, ids), TtsStatus::Success);
    EXPECT_EQ(ids.size(), v.max_input_ids);
    EXPECT_EQ(ids.front(), 100);
    EXPECT_EQ(ids.back(), 101) << "a clamped sequence must still be terminated";
    EXPECT_EQ(p.truncations(), 1u);
}

TEST(PassthroughPhonemizer, OutputBufferIsClearedOnEveryPath) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids{7, 7, 7};                             // caller's stale contents

    EXPECT_EQ(p.to_ids("", v.language_index, v, ids), TtsStatus::EmptyResult);
    EXPECT_TRUE(ids.empty());

    ids.assign(3, 7);
    EXPECT_EQ(p.to_ids("abc", 99 /*wrong language*/, v, ids), TtsStatus::UnsupportedLanguage);
    EXPECT_TRUE(ids.empty());
}

TEST(PassthroughPhonemizer, ReusedBufferDoesNotAccumulate) {
    PassthroughPhonemizer p;
    const TtsVoice v = ascii_voice();
    Ids ids;
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(p.to_ids("ab", v.language_index, v, ids), TtsStatus::Success);
        EXPECT_EQ(ids, (Ids{10, 11})) << "iteration " << i;
    }
}

}  // namespace
