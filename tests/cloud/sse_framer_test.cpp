// =============================================================================
// sse_framer_test.cpp — text/event-stream framing, protocol-independent.
//
// THE INVARIANT UNDER TEST is the one stated in sse_framer.hpp: feed() must
// tolerate arbitrary split points, because TCP will split a `data:` line
// mid-JSON, mid-UTF-8, and between the \r and the \n. Every test that matters
// here asserts the byte-at-a-time feed produces exactly what the whole-buffer
// feed produced -- that is the chunk-boundary bug class, and it is the one that
// survives to production because a local test with one big buffer never sees it.
//
// The parser is a recorder rather than either real decoder: framing owns "where
// does an event start and end", and testing it through a JSON parser would mean
// a framing regression could be masked by a parser that happened to cope.
// =============================================================================
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sse_framer.hpp"

namespace {

using blackwell::cloud::detail::SseFramer;

// Records the raw `data:` payload of every event the framer completes.
struct RecordingParser {
    std::vector<std::string> events;
    bool fail_next = false;

    bool parse_event(std::string& ev) {
        if (fail_next) return false;
        events.push_back(ev);
        return true;
    }
};

// Feed `wire` one byte at a time. Returns false if the framer ever refused.
bool feed_by_byte(SseFramer<RecordingParser>& f, const std::string& wire) {
    for (const char c : wire) {
        if (!f.feed(&c, 1)) return false;
    }
    return true;
}

std::vector<std::string> frame_whole(const std::string& wire) {
    RecordingParser p;
    SseFramer<RecordingParser> f(p);
    EXPECT_TRUE(f.feed(wire.data(), wire.size()));
    return p.events;
}

std::vector<std::string> frame_by_byte(const std::string& wire) {
    RecordingParser p;
    SseFramer<RecordingParser> f(p);
    EXPECT_TRUE(feed_by_byte(f, wire));
    return p.events;
}

TEST(SseFramer, BlankLineTerminatesAnEvent) {
    const auto ev = frame_whole("data: one\n\ndata: two\n\n");
    ASSERT_EQ(ev.size(), 2u);
    EXPECT_EQ(ev[0], "one");
    EXPECT_EQ(ev[1], "two");
}

// The one optional space after the colon is stripped; anything beyond it is
// payload, per the SSE spec.
TEST(SseFramer, StripsExactlyOneLeadingSpace) {
    EXPECT_EQ(frame_whole("data: x\n\n").at(0), "x");
    EXPECT_EQ(frame_whole("data:x\n\n").at(0), "x");
    EXPECT_EQ(frame_whole("data:  x\n\n").at(0), " x");
}

TEST(SseFramer, JoinsMultiLineDataWithNewlines) {
    const auto ev = frame_whole("data: a\ndata: b\ndata: c\n\n");
    ASSERT_EQ(ev.size(), 1u);
    EXPECT_EQ(ev[0], "a\nb\nc");
}

// Keep-alive comments and the fields we deliberately ignore must not produce
// events -- a spurious empty event would reach the parser as malformed JSON.
TEST(SseFramer, IgnoresCommentsAndNonDataFields) {
    const auto ev = frame_whole(": keep-alive\n\nevent: chunk\nid: 7\nretry: 100\ndata: x\n\n");
    ASSERT_EQ(ev.size(), 1u);
    EXPECT_EQ(ev[0], "x");
}

TEST(SseFramer, TrailingBlankLinesProduceNoEmptyEvents) {
    EXPECT_TRUE(frame_whole("\n\n\n").empty());
    EXPECT_EQ(frame_whole("data: x\n\n\n\n").size(), 1u);
}

// A stream that ends without its terminating blank line leaves the event
// UNDELIVERED rather than half-delivered: a truncated `data:` line is not a
// complete JSON document, and emitting it would be a parse error blamed on the
// server rather than on the connection that dropped.
TEST(SseFramer, IncompleteTrailingEventIsNotDelivered) {
    EXPECT_TRUE(frame_whole("data: {\"a\":1}").empty());
}

// ---- the chunk-boundary invariant -------------------------------------------

TEST(SseFramer, ByteAtATimeMatchesWholeBuffer) {
    const std::string wire =
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\r\n"
        "\r\n"
        ": ping\r\n"
        "\r\n"
        "event: chunk\r\n"
        "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\r\n"
        "\r\n"
        "data: [DONE]\r\n"
        "\r\n";
    EXPECT_EQ(frame_by_byte(wire), frame_whole(wire));
    EXPECT_EQ(frame_whole(wire).size(), 3u);
}

// CRLF is what the wire actually uses, and the \r must never survive into the
// payload -- `[DONE]\r` is not `[DONE]`, and a trailing \r inside JSON is a
// parse failure.
TEST(SseFramer, StripsCarriageReturnEvenWhenSplitAcrossChunks) {
    const std::string wire = "data: [DONE]\r\n\r\n";
    const auto whole = frame_whole(wire);
    ASSERT_EQ(whole.size(), 1u);
    EXPECT_EQ(whole[0], "[DONE]");
    // Split exactly between the \r and the \n -- the case a single-buffer test
    // can never reach.
    RecordingParser p;
    SseFramer<RecordingParser> f(p);
    const std::string head = "data: [DONE]\r";
    ASSERT_TRUE(f.feed(head.data(), head.size()));
    const std::string tail = "\n\r\n";
    ASSERT_TRUE(f.feed(tail.data(), tail.size()));
    ASSERT_EQ(p.events.size(), 1u);
    EXPECT_EQ(p.events[0], "[DONE]");
}

// Splitting inside a multi-byte UTF-8 sequence must be invisible: framing is
// byte-oriented and must not interpret the payload at all.
TEST(SseFramer, SurvivesASplitInsideAUtf8Sequence) {
    const std::string wire = "data: \xD1\x81\xD1\x82\xD0\xBE\n\n";  // "сто"
    EXPECT_EQ(frame_by_byte(wire), frame_whole(wire));
    EXPECT_EQ(frame_whole(wire).at(0), "\xD1\x81\xD1\x82\xD0\xBE");
}

// A parser that refuses must stop the transfer immediately -- the caller turns
// this into Status::MalformedStream.
TEST(SseFramer, ParserRefusalPropagates) {
    RecordingParser p;
    p.fail_next = true;
    SseFramer<RecordingParser> f(p);
    const std::string wire = "data: x\n\n";
    EXPECT_FALSE(f.feed(wire.data(), wire.size()));
}

// reset() must drop a half-accumulated event, or the next transfer on a reused
// decoder would begin with the tail of the previous one.
TEST(SseFramer, ResetDropsPartialState) {
    RecordingParser p;
    SseFramer<RecordingParser> f(p);
    const std::string partial = "data: abc";
    ASSERT_TRUE(f.feed(partial.data(), partial.size()));
    f.reset();
    const std::string next = "data: x\n\n";
    ASSERT_TRUE(f.feed(next.data(), next.size()));
    ASSERT_EQ(p.events.size(), 1u);
    EXPECT_EQ(p.events[0], "x");
}

}  // namespace
