// =============================================================================
// utf8_stream_test.cpp — the streaming UTF-8 contract, and the regression that
// motivated it.
//
// THE BUG THIS PINS DOWN. A vocabulary is built over bytes and a transport
// chunks on its own schedule, so neither one owes a consumer a whole code point.
// Both edges shipped a version that tore them:
//   * the decode loop emitted one detokenized token at a time, and a Cyrillic
//     character routinely spans two tokens;
//   * OfflineTransport sliced its echo every 12 BYTES, cutting two-byte Cyrillic
//     characters in half.
// Either way each orphaned byte reached the UI as its own U+FFFD, which is what
// "Привет" arriving as "<?><?>ривет" actually is: not a font problem, not a
// codepage problem, but a split sequence that was decoded before it was whole.
//
// So these tests assert the property that matters at every seam -- EVERY CHUNK
// HANDED ONWARD IS VALID UTF-8, AND CONCATENATING THEM REPRODUCES THE INPUT
// EXACTLY -- rather than asserting the specific chunk sizes, which are tuning.
//
// The source is compiled with /utf-8, so the Cyrillic literals below are UTF-8
// bytes in the binary.
// =============================================================================
#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#include "utf8_stream.hpp"        // src/bridge — the class under test
#include "offline_transport.hpp"  // src/cloud  — header-only, no libcurl

using blackwell::bridge::Utf8StreamAssembler;
using blackwell::bridge::utf8_sequence_length;

namespace {

// Strict UTF-8 validation, independent of the assembler's own logic (a checker
// that shared code with the thing it checks would agree with its bugs).
bool is_valid_utf8(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto lead = static_cast<unsigned char>(s[i]);
        int len = 0;
        if (lead < 0x80) len = 1;
        else if ((lead & 0xE0) == 0xC0) len = 2;
        else if ((lead & 0xF0) == 0xE0) len = 3;
        else if ((lead & 0xF8) == 0xF0) len = 4;
        else return false;                       // continuation byte, or 11111xxx
        if (i + static_cast<size_t>(len) > s.size()) return false;   // truncated
        for (int k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) {
                return false;                    // missing continuation
            }
        }
        i += static_cast<size_t>(len);
    }
    return true;
}

// The exact utterance from the bug report, plus a 3-byte (CJK) and a 4-byte
// (emoji) case so the walk-back covers every sequence length.
const std::string kRussian = "Привет, расскажи что-то о себе.";
const std::string kMixed   = "Привет 你好 🙂 ok";

}  // namespace

// ---- the assembler in isolation ---------------------------------------------

TEST(Utf8StreamAssembler, HoldsAPartialSequenceUntilItIsComplete) {
    Utf8StreamAssembler a;
    // "П" is D0 9F. Split between the two bytes: the first push can emit nothing.
    EXPECT_EQ(a.push("\xD0"), "");
    EXPECT_TRUE(a.holding());
    EXPECT_EQ(a.push("\x9F"), "\xD0\x9F");
    EXPECT_FALSE(a.holding());
}

TEST(Utf8StreamAssembler, EmitsTheCompletePrefixAndHoldsOnlyTheTail) {
    Utf8StreamAssembler a;
    // "ok" + the lead byte of "П": the ASCII prefix must not be held hostage by
    // the fragment behind it.
    EXPECT_EQ(a.push("ok\xD0"), "ok");
    EXPECT_EQ(a.push("\x9F"), "\xD0\x9F");
}

// Split at EVERY byte offset: whatever the boundary, the pieces that come out
// are individually valid and concatenate back to the original.
TEST(Utf8StreamAssembler, AnySplitPointReconstructsExactlyAndEmitsOnlyValidChunks) {
    for (const std::string& text : {kRussian, kMixed}) {
        for (size_t cut = 0; cut <= text.size(); ++cut) {
            Utf8StreamAssembler a;
            std::string rebuilt;
            for (const std::string& part : {text.substr(0, cut), text.substr(cut)}) {
                const std::string out = a.push(part);
                EXPECT_TRUE(is_valid_utf8(out)) << "cut=" << cut;
                rebuilt += out;
            }
            rebuilt += a.flush();
            EXPECT_EQ(rebuilt, text) << "cut=" << cut;
            EXPECT_FALSE(a.holding()) << "cut=" << cut;
        }
    }
}

// The pathological case: one byte at a time, which is what a byte-level
// tokenizer's worst case looks like.
TEST(Utf8StreamAssembler, ByteAtATimeStillYieldsWholeCharacters) {
    Utf8StreamAssembler a;
    std::string rebuilt;
    for (const char c : kMixed) {
        const std::string out = a.push(std::string_view(&c, 1));
        EXPECT_TRUE(is_valid_utf8(out));
        rebuilt += out;
    }
    rebuilt += a.flush();
    EXPECT_EQ(rebuilt, kMixed);
}

// Malformed input must PASS THROUGH, never accumulate: a model that emits a byte
// sequence no one can complete must not be able to stall the stream or grow the
// buffer without bound.
TEST(Utf8StreamAssembler, MalformedInputIsPassedThroughRatherThanBuffered) {
    Utf8StreamAssembler a;
    EXPECT_EQ(a.push("\x80\x80\x80\x80\x80"), "\x80\x80\x80\x80\x80");
    EXPECT_FALSE(a.holding());
}

// A superseded turn's dangling bytes must never prepend themselves to the next
// turn's first token -- the reason reset() exists at all.
TEST(Utf8StreamAssembler, ResetDropsTheHeldFragment) {
    Utf8StreamAssembler a;
    EXPECT_EQ(a.push("\xD0"), "");
    a.reset();
    EXPECT_FALSE(a.holding());
    EXPECT_EQ(a.push("ok"), "ok");   // not "\xD0ok"
}

// flush() surrenders a genuinely truncated tail rather than dropping bytes the
// producer did emit.
TEST(Utf8StreamAssembler, FlushSurrendersATruncatedTail) {
    Utf8StreamAssembler a;
    EXPECT_EQ(a.push("hi\xD0"), "hi");
    EXPECT_EQ(a.flush(), "\xD0");
    EXPECT_FALSE(a.holding());
}

TEST(Utf8StreamAssembler, SequenceLengthClassifiesLeadAndContinuationBytes) {
    EXPECT_EQ(utf8_sequence_length(0x41), 1);   // 'A'
    EXPECT_EQ(utf8_sequence_length(0xD0), 2);   // Cyrillic lead
    EXPECT_EQ(utf8_sequence_length(0xE4), 3);   // CJK lead
    EXPECT_EQ(utf8_sequence_length(0xF0), 4);   // emoji lead
    EXPECT_EQ(utf8_sequence_length(0x9F), 0);   // continuation
    EXPECT_EQ(utf8_sequence_length(0xFF), 0);   // invalid
}

// ---- the transport that shipped the bug -------------------------------------

namespace {

// Drive OfflineTransport with no simulated latency and collect its pieces.
std::vector<std::string> stream_offline(std::string_view intent, int chunk_chars) {
    blackwell::cloud::OfflineTransport::Config cfg;
    cfg.latency_ms = 0;
    cfg.chunk_delay_ms = 0;
    cfg.chunk_chars = chunk_chars;
    blackwell::cloud::OfflineTransport transport(cfg);

    std::vector<std::string> pieces;
    blackwell::cloud::Callbacks cb;
    cb.on_text = [&pieces](std::string_view s) { pieces.emplace_back(s); };

    const std::string body = "{}";
    const blackwell::cloud::TransportRequest req{body, intent, /*sequence=*/1};
    const blackwell::cloud::Result r = transport.send(req, cb);
    EXPECT_EQ(r.status, blackwell::cloud::Status::Ok);
    return pieces;
}

}  // namespace

// THE REGRESSION. Every piece the offline transport hands its caller must be
// valid UTF-8 on its own -- a live SSE text_delta always is, and a stand-in that
// is worse than the thing it stands in for tests the wrong contract.
TEST(OfflineTransport, NeverEmitsAPieceThatEndsInsideACharacter) {
    // Sweep the chunk size: the bug only appeared when a boundary happened to
    // land mid-character, so pinning one size would pass by luck.
    for (int chunk = 1; chunk <= 24; ++chunk) {
        const std::vector<std::string> pieces = stream_offline(kRussian, chunk);
        std::string rebuilt;
        for (const std::string& p : pieces) {
            EXPECT_TRUE(is_valid_utf8(p))
                << "chunk_chars=" << chunk << " piece=" << ::testing::PrintToString(p);
            rebuilt += p;
        }
        // The echo wraps the intent, so assert the intent survives inside it.
        EXPECT_NE(rebuilt.find(kRussian), std::string::npos) << "chunk_chars=" << chunk;
        EXPECT_TRUE(is_valid_utf8(rebuilt)) << "chunk_chars=" << chunk;
    }
}

TEST(OfflineTransport, HandlesThreeAndFourByteSequencesToo) {
    for (int chunk = 1; chunk <= 24; ++chunk) {
        std::string rebuilt;
        for (const std::string& p : stream_offline(kMixed, chunk)) {
            EXPECT_TRUE(is_valid_utf8(p)) << "chunk_chars=" << chunk;
            rebuilt += p;
        }
        EXPECT_NE(rebuilt.find(kMixed), std::string::npos) << "chunk_chars=" << chunk;
    }
}

// Defence in depth: even if some future transport DOES split a character, the
// assembler the view puts in front of it must repair the stream.
TEST(OfflineTransport, AssemblerRepairsAStreamThatWasSplitAnyway) {
    // Simulate the OLD, broken behaviour explicitly -- a fixed byte stride.
    const std::string reply = kRussian;
    Utf8StreamAssembler a;
    std::string rebuilt;
    for (size_t i = 0; i < reply.size(); i += 12) {
        const std::string chunk = reply.substr(i, 12);
        const std::string out = a.push(chunk);
        EXPECT_TRUE(is_valid_utf8(out));
        rebuilt += out;
    }
    rebuilt += a.flush();
    EXPECT_EQ(rebuilt, reply);
}
