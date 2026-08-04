// =============================================================================
// openai_sse_test.cpp — what a /chat/completions `data:` payload MEANS.
//
// Framing is covered separately (sse_framer_test.cpp); this file is about the
// three ways this endpoint differs from Anthropic's, each of which is a bug
// that ships silently if it is wrong:
//
//   * `[DONE]` is a SENTINEL, not JSON. Handing it to a parser fails, and the
//     failure is reported as MalformedStream on every otherwise-perfect
//     request -- i.e. every request looks broken except the broken ones.
//   * There is no `type` discriminator. delta / finish_reason / usage arrive in
//     whatever combination the gateway feels like, including all at once and
//     none at all.
//   * The fields are optional AND the servers vary. This is a de facto standard
//     with many implementations: a missing field is normal traffic, not a
//     protocol error, and treating it as one takes down a working stream.
//
// The byte-at-a-time cases matter for the same reason they do in the framer:
// TCP splits arrive in production and never in a test that feeds one buffer.
// =============================================================================
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "openai_sse.hpp"

namespace {

using blackwell::cloud::detail::OpenAiSseDecoder;

struct Sink {
    std::string text;
    std::vector<std::string> finishes;
    std::uint32_t prompt = 0, completion = 0, cached = 0;
    int usage_calls = 0;
    std::string api_error;

    void on_text_delta(std::string_view s) { text.append(s); }
    void on_usage(std::uint32_t p, std::uint32_t c, std::uint32_t cd) {
        prompt = p;
        completion = c;
        cached = cd;
        ++usage_calls;
    }
    void on_finish(std::string_view r) { finishes.emplace_back(r); }
    void on_api_error(std::string_view t, std::string_view m) {
        api_error.assign(t).append(": ").append(m);
    }
};

// Returns false if the decoder refused (-> Status::MalformedStream).
bool feed_whole(Sink& s, const std::string& wire) {
    OpenAiSseDecoder<Sink> d(s);
    return d.feed(wire.data(), wire.size());
}

bool feed_by_byte(Sink& s, const std::string& wire) {
    OpenAiSseDecoder<Sink> d(s);
    for (const char c : wire) {
        if (!d.feed(&c, 1)) return false;
    }
    return true;
}

// A realistic stream, including the fields gateways disagree about. The third
// chunk carries \uXXXX escapes so the decoder's unescaping is exercised -- this
// app is a Russian/English voice assistant, so that path is the common one.
const char* const kStream =
    "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,"
    "\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\","
    "\"content\":\"\"},\"finish_reason\":null}]}\n\n"
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Paris\"},"
    "\"finish_reason\":null}]}\n\n"
    ": ping\n\n"
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" "
    "\\u0441\\u0442\\u043e\\u043b\\u0438\\u0446\\u0430\"}}]}\n\n"
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":7,"
    "\"total_tokens\":49,\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n"
    "data: [DONE]\n\n";

// "Paris столица"
const char* const kExpectedText = "Paris \xD1\x81\xD1\x82\xD0\xBE\xD0\xBB\xD0\xB8\xD1\x86\xD0\xB0";

TEST(OpenAiSse, DecodesACompleteStream) {
    Sink s;
    ASSERT_TRUE(feed_whole(s, kStream));
    EXPECT_EQ(s.text, kExpectedText);
    ASSERT_EQ(s.finishes.size(), 1u);
    EXPECT_EQ(s.finishes[0], "stop");
    EXPECT_TRUE(s.api_error.empty());
}

TEST(OpenAiSse, ByteAtATimeIsIdentical) {
    Sink whole, split;
    ASSERT_TRUE(feed_whole(whole, kStream));
    ASSERT_TRUE(feed_by_byte(split, kStream));
    EXPECT_EQ(split.text, whole.text);
    EXPECT_EQ(split.finishes, whole.finishes);
    EXPECT_EQ(split.prompt, whole.prompt);
    EXPECT_EQ(split.cached, whole.cached);
}

// The sentinel is why this decoder cannot just hand every payload to simdjson.
TEST(OpenAiSse, DoneSentinelIsNotAParseFailure) {
    Sink s;
    EXPECT_TRUE(feed_whole(s, "data: [DONE]\n\n"));
    EXPECT_TRUE(feed_whole(s, "data:[DONE]\n\n"));
    EXPECT_TRUE(feed_whole(s, "data:  [DONE]  \n\n"));
    EXPECT_TRUE(s.text.empty());
}

// ---- usage ------------------------------------------------------------------

TEST(OpenAiSse, MapsUsageIncludingTheNestedCacheCounter) {
    Sink s;
    ASSERT_TRUE(feed_whole(s, kStream));
    EXPECT_EQ(s.usage_calls, 1);
    EXPECT_EQ(s.prompt, 42u);
    EXPECT_EQ(s.completion, 7u);
    // cached_tokens lives one level down, and mapping it is what keeps the UI's
    // "the prompt cache is working" indicator meaningful on this leg.
    EXPECT_EQ(s.cached, 32u);
}

// Most gateways never send it. Absence is normal traffic.
TEST(OpenAiSse, MissingUsageIsNotAnError) {
    Sink s;
    ASSERT_TRUE(feed_whole(s, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"));
    EXPECT_EQ(s.usage_calls, 0);
    EXPECT_EQ(s.text, "x");
}

TEST(OpenAiSse, UsageWithoutCacheDetailsReportsZeroCached) {
    Sink s;
    ASSERT_TRUE(feed_whole(
        s, "data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n"));
    EXPECT_EQ(s.prompt, 5u);
    EXPECT_EQ(s.cached, 0u);
}

// ---- finish reasons ---------------------------------------------------------

TEST(OpenAiSse, ReportsLengthAndContentFilter) {
    {
        Sink s;
        ASSERT_TRUE(feed_whole(
            s, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"));
        ASSERT_EQ(s.finishes.size(), 1u);
        EXPECT_EQ(s.finishes[0], "length");
    }
    {
        Sink s;
        ASSERT_TRUE(feed_whole(
            s, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\n"));
        ASSERT_EQ(s.finishes.size(), 1u);
        EXPECT_EQ(s.finishes[0], "content_filter");
    }
}

// null is what a mid-stream chunk carries; it is not a finish.
TEST(OpenAiSse, NullFinishReasonIsNotAFinish) {
    Sink s;
    ASSERT_TRUE(feed_whole(
        s, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"));
    EXPECT_TRUE(s.finishes.empty());
}

// ---- the chunks that are not text ------------------------------------------

TEST(OpenAiSse, SkipsNullContentAndToolCallChunks) {
    Sink s;
    ASSERT_TRUE(feed_whole(s,
        "data: {\"choices\":[{\"delta\":{\"content\":null}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"));
    EXPECT_EQ(s.text, "ok");
}

TEST(OpenAiSse, EmptyChoicesArrayIsFine) {
    Sink s;
    EXPECT_TRUE(feed_whole(s, "data: {\"choices\":[]}\n\n"));
    EXPECT_TRUE(s.text.empty());
}

// ---- errors -----------------------------------------------------------------

// Rate limits and upstream failures commonly arrive as an error object INSIDE
// an HTTP 200 stream. The transport reports success, so if this is not parsed
// out of the body nothing downstream ever learns the request failed.
TEST(OpenAiSse, ErrorObjectInsideA200StreamIsSurfaced) {
    Sink s;
    ASSERT_TRUE(feed_whole(
        s,
        "data: {\"error\":{\"message\":\"Rate limit reached\",\"type\":\"rate_limit_error\"}}\n\n"));
    EXPECT_EQ(s.api_error, "rate_limit_error: Rate limit reached");
}

// A gateway that starts emitting an HTML error page mid-stream must surface as
// MalformedStream, not as silence -- silence is indistinguishable from a model
// that answered with nothing.
TEST(OpenAiSse, NonJsonPayloadIsRefused) {
    Sink s;
    EXPECT_FALSE(feed_whole(s, "data: <html>502 Bad Gateway</html>\n\n"));
}

TEST(OpenAiSse, TruncatedJsonIsRefused) {
    Sink s;
    EXPECT_FALSE(feed_whole(s, "data: {\"choices\":[{\"delta\":\n\n"));
}

// ---- the decoder is reusable ------------------------------------------------

TEST(OpenAiSse, ResetClearsPartialFramingBetweenTransfers) {
    Sink s;
    OpenAiSseDecoder<Sink> d(s);
    const std::string partial = "data: {\"choices\":[{\"delta\":{\"content\":\"lost\"";
    ASSERT_TRUE(d.feed(partial.data(), partial.size()));
    d.reset();
    const std::string next = "data: {\"choices\":[{\"delta\":{\"content\":\"kept\"}}]}\n\n";
    ASSERT_TRUE(d.feed(next.data(), next.size()));
    EXPECT_EQ(s.text, "kept");
}

}  // namespace
