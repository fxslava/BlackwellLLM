// -----------------------------------------------------------------------------
// test_duplex_pipeline.cpp — TextChunker splitting + the barge-in path through
// TTSDuplexBridge.
//
// CPU-only and model-free, which is the whole reason ISynthesizer exists: the
// behaviour that most needs a test -- cancelling WHILE the solver is running --
// would otherwise need a 1.3 GB checkpoint and a GPU, and could never live in
// the fast `validation` suite. BlockingSynthesizer stands in for the engine and
// blocks on command so the cancel lands genuinely mid-synthesis rather than
// between chunks, which is the case that actually matters and the easy one to
// accidentally not test.
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "spsc_ring.hpp"
#include "text_chunker.hpp"
#include "tts_duplex_bridge.hpp"

using blackwell::audio_rt::SpscRing;
using blackwell::tts::AecTap;
using blackwell::tts::ChunkerConfig;
using blackwell::tts::DuplexConfig;
using blackwell::tts::F5Tokenizer;
using blackwell::tts::ISynthesizer;
using blackwell::tts::TextChunker;
using blackwell::tts::TtsStatus;
using blackwell::tts::TTSDuplexBridge;

namespace {

std::vector<std::string> DrainAll(TextChunker& c) {
    std::vector<std::string> out;
    while (c.HasPendingChunk()) out.push_back(c.PopChunk());
    return out;
}

}  // namespace

// =============================================================================
// TextChunker
// =============================================================================
TEST(TextChunker, SplitsOnSentenceEnds) {
    TextChunker c;   // defaults: clause floor 20, first chunk unfloored
    c.PushToken("This is a reasonably long first sentence. And here is a second one. ");
    const auto got = DrainAll(c);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "This is a reasonably long first sentence.");
    EXPECT_EQ(got[1], "And here is a second one.");
}

TEST(TextChunker, SplitsOnCommasWhenEnabled) {
    ChunkerConfig cfg;
    cfg.split_on_commas = true;
    TextChunker c(cfg);
    c.PushToken("A clause long enough to pass the minimum, and then another clause here.");
    const auto got = DrainAll(c);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "A clause long enough to pass the minimum,");
    EXPECT_EQ(got[1], "and then another clause here.");
}

TEST(TextChunker, CommaSplittingCanBeDisabled) {
    ChunkerConfig cfg;
    cfg.split_on_commas = false;
    TextChunker c(cfg);
    c.PushToken("A clause long enough to pass the minimum, and then another clause here.");
    const auto got = DrainAll(c);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], "A clause long enough to pass the minimum, and then another clause here.");
}

// THE latency rule: the FIRST punctuation mark of a reply ends the first chunk,
// whatever it is, because that chunk's synthesis is the only one the user waits
// through. "Да," is 3 characters and becomes an utterance on purpose.
TEST(TextChunker, FirstChunkGoesOutAtTheFirstPunctuation) {
    TextChunker c;
    c.PushToken("Да, конечно я могу помочь тебе с этим прямо сейчас.");
    const auto got = DrainAll(c);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "Да,");
    EXPECT_EQ(got[1], "конечно я могу помочь тебе с этим прямо сейчас.");
}

// ...and the floor applies again from the second chunk on, where the latency is
// already hidden behind playback and only prosody is left to protect.
TEST(TextChunker, ClauseFloorAppliesAfterTheFirstChunk) {
    TextChunker c;
    c.PushToken("Начало первой фразы. Да, и дальше идёт длинное продолжение фразы.");
    const auto got = DrainAll(c);
    ASSERT_GE(got.size(), 2u);
    EXPECT_EQ(got[0], "Начало первой фразы.");
    EXPECT_NE(got[1], "Да,") << "the mid-reply clause floor did not suppress a 3-char split";
}

TEST(TextChunker, MinChunkCharsCountsCodepointsNotBytes) {
    // 12 Cyrillic characters = 24 UTF-8 bytes. With min_chunk_chars = 20 the
    // mid-reply split must NOT be taken -- if the implementation counted bytes
    // it would see 24 >= 20 and split, fragmenting the reply.
    ChunkerConfig cfg;
    cfg.min_chunk_chars = 20;
    cfg.first_chunk_asap = false;   // otherwise the floor is suspended here
    TextChunker c(cfg);
    c.PushToken("Привет мир, и снова здравствуйте дорогие друзья.");
    const auto got = DrainAll(c);
    ASSERT_FALSE(got.empty());
    EXPECT_NE(got[0], "Привет мир,");
}

// Punctuation is the only boundary a chunk is CHOSEN at: text under the safety
// valve's threshold is never cut at a character count.
TEST(TextChunker, ShortPunctuationFreeTextNeverSplits) {
    ChunkerConfig cfg;
    cfg.min_chunk_chars = 5;
    TextChunker c(cfg);
    const std::string input = "aaaa bbbb cccc dddd eeee ffff gggg hhhh";
    c.PushToken(input);
    EXPECT_FALSE(c.HasPendingChunk()) << "a character count split a phrase";

    c.Flush();
    const auto got = DrainAll(c);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], input);
}

// THE case the valve exists for: a long reply with no punctuation at all --
// routine LLM output -- must produce audio BEFORE the generation finishes,
// not a silence followed by the whole reply at once.
TEST(TextChunker, LongPunctuationFreeReplySpeaksBeforeFlush) {
    TextChunker c;   // default valve: 180 codepoints
    std::string input;
    for (int i = 0; i < 60; ++i) input += "слово ";   // 6 cp each, no punctuation
    c.PushToken(input);

    ASSERT_TRUE(c.HasPendingChunk()) << "nothing was speakable before Flush";
    const std::string first = c.PopChunk();
    EXPECT_FALSE(first.empty());
    // Broke at a space, so no word was cut in half. Compared as a STRING, not
    // via back(): "о" is two bytes in UTF-8 and a char comparison would be
    // testing the continuation byte.
    const std::string word = "слово";
    ASSERT_GE(first.size(), word.size());
    EXPECT_EQ(first.compare(first.size() - word.size(), word.size(), word), 0)
        << "the valve split mid-word: " << first;

    c.Flush();
    std::string rejoined = first;
    for (const auto& s : DrainAll(c)) {
        rejoined += ' ';
        rejoined += s;
    }
    // Trailing space is trimmed by Emit; compare against the trimmed input.
    std::string expect = input;
    while (!expect.empty() && expect.back() == ' ') expect.pop_back();
    EXPECT_EQ(rejoined, expect) << "characters were lost or reordered";
}

TEST(TextChunker, RunawayGuardSplitsAtWhitespace) {
    ChunkerConfig cfg;
    cfg.runaway_guard_chars = 20;
    TextChunker c(cfg);
    const std::string input = "aaaa bbbb cccc dddd eeee ffff gggg hhhh";
    c.PushToken(input);
    c.Flush();

    const auto got = DrainAll(c);
    EXPECT_GT(got.size(), 1u) << "the runaway guard never fired";
    std::string rejoined;
    for (const auto& s : got) {
        EXPECT_FALSE(s.empty());
        if (!rejoined.empty()) rejoined += ' ';
        rejoined += s;
    }
    EXPECT_EQ(rejoined, input) << "a word was split or characters were lost";
}

// Text with no whitespace at all cannot be broken between words, and the valve
// must not simply give up: the hard cap accepts a mid-word cut rather than
// buffer without bound.
TEST(TextChunker, HardCapFiresWhenThereIsNoWhitespace) {
    ChunkerConfig cfg;
    cfg.runaway_guard_chars = 20;
    cfg.runaway_hard_cap_chars = 40;
    TextChunker c(cfg);
    c.PushToken(std::string(100, 'a'));
    ASSERT_TRUE(c.HasPendingChunk()) << "the hard cap never fired";
    EXPECT_EQ(c.PopChunk().size(), 40u);
}

// A '+' separated from its vowel becomes a bare token in one utterance and a
// silently unstressed vowel in the next. No split may land on either side of
// one, under EITHER placement convention.
TEST(TextChunker, NeverSplitsBetweenAStressMarkAndItsVowel) {
    ChunkerConfig cfg;
    cfg.runaway_guard_chars = 8;
    cfg.runaway_hard_cap_chars = 8;
    TextChunker c(cfg);

    // No whitespace, so every split is taken by the hard cap -- which lands on
    // an arbitrary codepoint boundary and would hit the marks if unguarded.
    std::string input;
    for (int i = 0; i < 12; ++i) input += "р+ыбка";
    c.PushToken(input);
    c.Flush();

    std::string rejoined;
    for (const auto& s : DrainAll(c)) {
        ASSERT_FALSE(s.empty());
        EXPECT_NE(s.back(), '+') << "chunk ended on a stress mark: " << s;
        EXPECT_NE(s.front(), '+') << "chunk began with an orphaned mark: " << s;
        rejoined += s;
    }
    EXPECT_EQ(rejoined, input) << "characters were lost";
}

// The same guard on the whitespace path, where the mark sits next to the space.
TEST(TextChunker, ValveSkipsASplitPointAdjacentToAMark) {
    ChunkerConfig cfg;
    cfg.runaway_guard_chars = 6;
    TextChunker c(cfg);
    // The space at index 6 is followed by '+', so splitting there would open
    // the next chunk with an orphaned mark under the after-vowel convention.
    c.PushToken("абвгде +южя ещё слова");
    c.Flush();

    for (const auto& s : DrainAll(c)) {
        EXPECT_NE(s.back(), '+') << "chunk ended on a stress mark: " << s;
        EXPECT_NE(s.front(), '+') << "chunk began with an orphaned mark: " << s;
    }
}

// A decimal point is not a sentence end. Splitting here would hand the text
// normaliser "3." and "14" and it would read them as two numbers.
TEST(TextChunker, DoesNotSplitInsideNumbers) {
    TextChunker c;
    c.PushToken("Погрешность 3.14 процента, и это устраивает.");
    const auto got = DrainAll(c);
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(got[0], "Погрешность 3.14 процента,");
}

// The digit guard must WAIT rather than guess when the deciding byte has not
// arrived: "3." at a token boundary is undecidable until the next token.
TEST(TextChunker, DigitGuardWaitsForTheDecidingByte) {
    TextChunker c;
    c.PushToken("Значение равно 3.");
    EXPECT_FALSE(c.HasPendingChunk()) << "split on a '.' whose successor was unknown";
    c.PushToken("14 и не больше.");
    const auto got = DrainAll(c);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], "Значение равно 3.14 и не больше.");
}

// An initial or an abbreviation ends a token, not a sentence.
TEST(TextChunker, DoesNotSplitAfterASingleLetterInitial) {
    TextChunker c;
    c.PushToken("Об этом писал А. Пушкин, и не только он.");
    const auto got = DrainAll(c);
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(got[0], "Об этом писал А. Пушкин,");
}

// A reply opening with punctuation must not spend a whole synthesis on it. The
// stray mark is dropped (it is inaudible), and the first REAL clause becomes
// the first chunk.
TEST(TextChunker, PunctuationOnlySpanIsNotAnUtterance) {
    TextChunker c;
    c.PushToken(", хорошо, теперь по существу дела.");
    const auto got = DrainAll(c);
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(got[0], "хорошо,") << "a lone comma became a chunk";
}

// Flush ends the reply, so the NEXT one gets the first-chunk rule again.
TEST(TextChunker, FirstChunkRuleIsReArmedByFlush) {
    TextChunker c;
    c.PushToken("Первая реплика целиком.");
    c.Flush();
    (void)DrainAll(c);

    c.PushToken("Да, вторая реплика начинается здесь.");
    const auto got = DrainAll(c);
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(got[0], "Да,");
}

// An LLM stream splits wherever its tokenizer decided, which for Cyrillic lands
// mid-codepoint constantly. A byte-level chunker corrupts one character per
// token boundary and the tokenizer maps each to the unknown id in silence.
TEST(TextChunker, SurvivesTokensSplitMidCodepoint) {
    const std::string full = "Привет, это проверка локального синтеза русской речи.";
    TextChunker c;
    for (std::size_t i = 0; i < full.size(); ++i) {
        c.PushToken(std::string_view(full).substr(i, 1));   // one BYTE at a time
    }
    c.Flush();
    std::string rejoined;
    for (const auto& s : DrainAll(c)) {
        if (!rejoined.empty()) rejoined += ' ';
        rejoined += s;
    }
    // Every byte must survive, in order: same content, no replacement chars.
    EXPECT_EQ(rejoined, full);
    EXPECT_EQ(rejoined.find("\xEF\xBF\xBD"), std::string::npos) << "codepoint was corrupted";
}

TEST(TextChunker, FlushEmitsRemainderBelowMinimum) {
    TextChunker c;
    c.PushToken("Short tail");           // under min_chunk_chars, no punctuation
    EXPECT_FALSE(c.HasPendingChunk());
    c.Flush();
    ASSERT_TRUE(c.HasPendingChunk());
    EXPECT_EQ(c.PopChunk(), "Short tail");
}

TEST(TextChunker, ResetDropsBufferAndQueue) {
    TextChunker c;
    c.PushToken("A first sentence that is long enough. And a partial second");
    EXPECT_TRUE(c.HasPendingChunk());
    EXPECT_GT(c.buffered_bytes(), 0u);
    c.Reset();
    EXPECT_FALSE(c.HasPendingChunk());
    EXPECT_EQ(c.buffered_bytes(), 0u);
    c.Flush();                                  // nothing left to flush
    EXPECT_FALSE(c.HasPendingChunk());
}

TEST(TextChunker, WhitespaceOnlyInputProducesNothing) {
    TextChunker c;
    c.PushToken("   \n\t  ");
    c.Flush();
    EXPECT_FALSE(c.HasPendingChunk());
    EXPECT_EQ(c.total_emitted(), 0u);
}

// =============================================================================
// Duplex bridge
// =============================================================================
namespace {

// Stands in for F5TtsEngine. Emits `samples_per_chunk` of a recognisable ramp,
// and can be made to block inside Synthesize so a barge-in lands MID-synthesis.
class BlockingSynthesizer final : public ISynthesizer {
public:
    std::atomic<bool> entered{false};      // set once inside Synthesize
    std::atomic<bool> release{true};       // false = block until cancelled
    std::atomic<int>  calls{0};
    std::atomic<int>  interrupted{0};
    std::size_t       samples_per_chunk = 1024;

    TtsStatus Synthesize(const std::vector<std::int64_t>& ids,
                         const std::atomic<bool>* cancel,
                         std::vector<float>& out) noexcept override {
        calls.fetch_add(1);
        out.clear();
        if (ids.empty()) return TtsStatus::EmptyResult;
        entered.store(true);

        // Stand-in for the ODE loop: poll the cancel flag the way the real
        // solver does (once per completed step) instead of running to completion.
        for (int step = 0; step < 64; ++step) {
            if (cancel != nullptr && cancel->load(std::memory_order_relaxed)) {
                interrupted.fetch_add(1);
                out.clear();               // contract: empty on Interrupted
                return TtsStatus::Interrupted;
            }
            if (!release.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                --step;                    // stay in the loop until released
                continue;
            }
        }
        out.resize(samples_per_chunk);
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<float>(i % 100) / 100.0f;
        }
        return TtsStatus::Success;
    }

    int sample_rate() const noexcept override { return 24000; }
};

// The bridge needs a real F5Tokenizer. The vocab that ships with the checkpoint
// is not a build artefact, so synthesise a minimal one: entry 0 MUST be a space
// (F5Tokenizer enforces that as its mis-parse guard), then printable ASCII and
// the Cyrillic block.
std::string WriteTinyVocab() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "blackwell_tts_test_vocab.txt").string();
    std::ofstream f(path, std::ios::binary);
    f << " \n";                                        // id 0 -- required
    for (int c = 0x21; c <= 0x7E; ++c) f << static_cast<char>(c) << "\n";
    for (unsigned cp = 0x0410; cp <= 0x044F; ++cp) {   // А..я
        f << static_cast<char>(0xC0u | (cp >> 6))
          << static_cast<char>(0x80u | (cp & 0x3Fu)) << "\n";
    }
    return path;
}

struct Fixture {
    std::string vocab_path = WriteTinyVocab();
    F5Tokenizer tok{vocab_path};
    SpscRing<float> speaker{8192};
    SpscRing<float> aec{8192};
    BlockingSynthesizer synth;
};

}  // namespace

TEST(DuplexBridge, SpeaksChunksAndFeedsBothSinks) {
    Fixture fx;
    DuplexConfig cfg;
    cfg.aec_tap = AecTap::Playback;
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec, ChunkerConfig{}, cfg);

    bridge.PushToken("This is a long enough first sentence. ");
    ASSERT_TRUE(bridge.HasPendingText());

    std::size_t produced = 0;
    ASSERT_EQ(bridge.PumpOnce(&produced), TtsStatus::Success);
    EXPECT_EQ(produced, fx.synth.samples_per_chunk);
    EXPECT_EQ(bridge.chunks_spoken(), 1u);
    EXPECT_EQ(fx.speaker.available(), fx.synth.samples_per_chunk);

    // Reference is tapped at PLAYBACK, so it is still empty until the device
    // pulls. That is the whole point: the reference must be aligned to what the
    // speaker emits, not to when synthesis happened.
    EXPECT_EQ(fx.aec.available(), 0u);

    std::vector<float> block(256);
    const std::size_t real = bridge.PullForPlayback(block.data(), block.size());
    EXPECT_EQ(real, block.size());
    EXPECT_EQ(fx.aec.available(), block.size());

    // ...and what the AEC got is BYTE-IDENTICAL to what the device got.
    std::vector<float> ref(256);
    ASSERT_EQ(fx.aec.read(ref.data(), ref.size()), ref.size());
    EXPECT_EQ(ref, block);
}

TEST(DuplexBridge, SynthesisTapFeedsReferenceImmediately) {
    Fixture fx;
    DuplexConfig cfg;
    cfg.aec_tap = AecTap::Synthesis;
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec, ChunkerConfig{}, cfg);

    bridge.PushToken("This is a long enough first sentence. ");
    ASSERT_EQ(bridge.PumpOnce(), TtsStatus::Success);
    EXPECT_EQ(fx.aec.available(), fx.synth.samples_per_chunk);
}

TEST(DuplexBridge, CancelBeforePumpSpeaksNothing) {
    Fixture fx;
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec);
    bridge.PushToken("A sentence that would otherwise be spoken aloud. ");
    bridge.Cancel();

    EXPECT_EQ(bridge.PumpOnce(), TtsStatus::Interrupted);
    EXPECT_EQ(fx.synth.calls.load(), 0) << "synthesis started after a cancel";
    EXPECT_FALSE(bridge.HasPendingText()) << "chunker was not reset";
    EXPECT_EQ(bridge.chunks_spoken(), 0u);
}

// THE test this file exists for: the flag flips while the solver is running.
TEST(DuplexBridge, CancelMidSynthesisHaltsAndResetsCleanly) {
    Fixture fx;
    std::atomic<bool> cancel{false};
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec);
    bridge.SetCancelSignal(&cancel);

    bridge.PushToken("A first sentence long enough to be spoken. "
                     "And a second one that will never be reached. ");
    fx.synth.release.store(false);          // block inside Synthesize

    std::atomic<TtsStatus> result{TtsStatus::Success};
    std::thread worker([&] { result.store(bridge.PumpOnce()); });

    // Wait until we are genuinely inside the solver, then barge in.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!fx.synth.entered.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(fx.synth.entered.load()) << "synthesis never started";
    cancel.store(true);

    worker.join();   // must return promptly; a hang here IS the failure

    EXPECT_EQ(result.load(), TtsStatus::Interrupted);
    EXPECT_EQ(fx.synth.interrupted.load(), 1);
    EXPECT_EQ(bridge.chunks_spoken(), 0u);
    EXPECT_GE(bridge.chunks_cancelled(), 1u);

    // State is clean: no queued text, nothing published, cancelled latched.
    EXPECT_TRUE(bridge.cancelled());
    EXPECT_FALSE(bridge.HasPendingText()) << "pending chunks survived barge-in";
    EXPECT_EQ(bridge.samples_published(), 0u);

    // Tokens arriving after the barge-in belong to an abandoned reply.
    bridge.PushToken("More of the abandoned reply. ");
    EXPECT_FALSE(bridge.HasPendingText());

    // ...and the bridge is reusable for the next turn.
    bridge.Resume();
    cancel.store(false);
    fx.synth.release.store(true);
    EXPECT_FALSE(bridge.cancelled());
    bridge.PushToken("A fresh sentence for the next turn entirely. ");
    EXPECT_EQ(bridge.PumpOnce(), TtsStatus::Success);
    EXPECT_EQ(bridge.chunks_spoken(), 1u);
}

// Barge-in must drop audio that is queued but not yet played, and must do it
// without SpscRing::reset() -- which is unsafe while the consumer is live.
TEST(DuplexBridge, BargeInDropsQueuedSpeakerAudioButKeepsAecReference) {
    Fixture fx;
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec);

    bridge.PushToken("A sentence long enough to be spoken aloud now. ");
    ASSERT_EQ(bridge.PumpOnce(), TtsStatus::Success);
    ASSERT_EQ(fx.speaker.available(), fx.synth.samples_per_chunk);

    // Play a little, so the AEC reference holds the tail still in the room.
    std::vector<float> block(256);
    bridge.PullForPlayback(block.data(), block.size());
    const std::size_t ref_before = fx.aec.available();
    ASSERT_GT(ref_before, 0u);

    bridge.Cancel();

    // The drain is consumer-side: it happens on the next pull, not inside
    // Cancel(), because only the consumer may read the ring.
    EXPECT_GT(fx.speaker.available(), 0u) << "producer drained the ring itself";
    bridge.PullForPlayback(block.data(), block.size());
    EXPECT_EQ(fx.speaker.available(), 0u) << "stale speaker audio was not dropped";

    // The reference is deliberately NOT flushed: audio already inside the device
    // buffer is still going to be emitted and the room keeps ringing, which is
    // exactly when the canceller needs its reference.
    EXPECT_GE(fx.aec.available(), ref_before);
}

TEST(DuplexBridge, PullPadsWithSilenceOnUnderrun) {
    Fixture fx;
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec);
    std::vector<float> block(128, 1.0f);
    const std::size_t real = bridge.PullForPlayback(block.data(), block.size());
    EXPECT_EQ(real, 0u);
    for (const float v : block) EXPECT_FLOAT_EQ(v, 0.0f);
    // The reference stream must stay continuous even through silence, or every
    // later sample is shifted and the alignment is lost.
    EXPECT_EQ(fx.aec.available(), block.size());
}

TEST(DuplexBridge, EmptyPumpIsCheapAndHarmless) {
    Fixture fx;
    TTSDuplexBridge bridge(fx.synth, fx.tok, fx.speaker, fx.aec);
    EXPECT_EQ(bridge.PumpOnce(), TtsStatus::EmptyResult);
    EXPECT_EQ(fx.synth.calls.load(), 0);
    EXPECT_EQ(bridge.chunks_spoken(), 0u);
}
