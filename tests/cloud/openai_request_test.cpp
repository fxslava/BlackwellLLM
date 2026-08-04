// =============================================================================
// openai_request_test.cpp — the /chat/completions body and the endpoint join.
//
// WHY THIS IS WORTH TESTING AT ALL. The body is hand-assembled string
// concatenation (openai_request.hpp), which is the right call for a hot path
// with a fixed shape -- but it means a missing comma or an unescaped quote is a
// runtime HTTP 400 from a third-party gateway, reported to the user as "the
// assistant did not answer". There is no compiler between this code and that
// symptom, so the tests are the only thing standing there.
//
// The escaping cases are not hypothetical: this app transcribes SPEECH, so the
// intent routinely contains quotes, newlines and non-ASCII text.
// =============================================================================
#include <gtest/gtest.h>

#include <string>

#include "openai_request.hpp"

namespace {

using blackwell::cloud::build_openai_request;
using blackwell::cloud::join_url;
using blackwell::cloud::OpenAiRequestOptions;
using blackwell::cloud::RequestContext;

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

RequestContext basic_ctx() {
    RequestContext c;
    c.instructions = "You are a concise voice assistant.";
    c.glossary = "This is a spoken-language voice assistant session.";
    c.intent = "What is the capital of France?";
    c.model = "gpt-4o-mini";
    return c;
}

// ---- endpoint ---------------------------------------------------------------
// The setting the user edits is the BASE url, because that is what providers
// document and what people paste. Every plausible way of pasting one must land
// on the same endpoint -- a double slash is a 404 on enough gateways to matter.

TEST(JoinUrl, NormalisesTheWaysPeoplePasteABaseUrl) {
    const std::string want = "https://router.cheap/v1/chat/completions";
    EXPECT_EQ(join_url("https://router.cheap/v1", "chat/completions"), want);
    EXPECT_EQ(join_url("https://router.cheap/v1/", "chat/completions"), want);
    EXPECT_EQ(join_url("https://router.cheap/v1//", "chat/completions"), want);
    EXPECT_EQ(join_url("https://router.cheap/v1", "/chat/completions"), want);
    EXPECT_EQ(join_url("  https://router.cheap/v1  ", "chat/completions"), want);
    EXPECT_EQ(join_url("https://router.cheap/v1\n", "chat/completions"), want);
}

TEST(JoinUrl, LeavesAnInteriorPathAlone) {
    EXPECT_EQ(join_url("https://host/openai/deployments/x", "chat/completions"),
              "https://host/openai/deployments/x/chat/completions");
}

// ---- body shape -------------------------------------------------------------

TEST(OpenAiRequest, HasTheDocumentedShape) {
    const std::string b = build_openai_request(basic_ctx());
    EXPECT_EQ(b.rfind(R"({"model":"gpt-4o-mini","messages":[)", 0), 0u);
    EXPECT_TRUE(contains(b, R"("role":"system")"));
    EXPECT_TRUE(contains(b, R"("role":"user")"));
    EXPECT_TRUE(contains(b, R"("stream":true)"));
    EXPECT_EQ(b.back(), '}');
}

TEST(OpenAiRequest, SystemHalvesAreJoinedWithABlankLine) {
    const std::string b = build_openai_request(basic_ctx());
    EXPECT_TRUE(contains(b, "assistant.\\n\\nThis is a spoken-language"));
}

TEST(OpenAiRequest, CommittedPrefixPrecedesTheIntentInTheUserTurn) {
    RequestContext c = basic_ctx();
    c.committed_prefix = "earlier";
    c.intent = "now";
    EXPECT_TRUE(contains(build_openai_request(c), R"("content":"earlier\n\nnow")"));
}

// An empty half must not leave a dangling separator: "\n\nnow" reads to the
// model as a turn that began with a blank line, and the glossary is empty in
// several legitimate configurations.
TEST(OpenAiRequest, EmptyHalvesProduceNoSeparator) {
    RequestContext c;
    c.instructions = "S";
    c.intent = "U";
    c.model = "m";
    const std::string b = build_openai_request(c);
    EXPECT_TRUE(contains(b, R"("content":"S")"));
    EXPECT_TRUE(contains(b, R"("content":"U")"));
}

// ---- optional fields --------------------------------------------------------
// Both default OFF, and that is the tested behaviour rather than an accident:
// a gateway that does not know `stream_options` fails the whole request with a
// 400 rather than ignoring the field, so it must be opt-in.

TEST(OpenAiRequest, OptionalFieldsAreOmittedByDefault) {
    const std::string b = build_openai_request(basic_ctx());
    EXPECT_FALSE(contains(b, "stream_options"));
    EXPECT_FALSE(contains(b, "max_tokens"));
}

TEST(OpenAiRequest, OptionalFieldsAppearWhenAsked) {
    OpenAiRequestOptions o;
    o.max_tokens = 256;
    o.include_usage = true;
    const std::string b = build_openai_request(basic_ctx(), o);
    EXPECT_TRUE(contains(b, R"("max_tokens":256)"));
    EXPECT_TRUE(contains(b, R"("stream_options":{"include_usage":true})"));
}

TEST(OpenAiRequest, ZeroMaxTokensOmitsTheFieldRatherThanSendingZero) {
    OpenAiRequestOptions o;
    o.max_tokens = 0;
    // Sending "max_tokens":0 would be a request for an empty answer, which is
    // the opposite of "no ceiling".
    EXPECT_FALSE(contains(build_openai_request(basic_ctx(), o), "max_tokens"));
}

// ---- escaping ---------------------------------------------------------------

TEST(OpenAiRequest, EscapesWhatSpeechActuallyContains) {
    RequestContext c = basic_ctx();
    c.intent = "He said \"hi\"\nthen left\tquickly\\done";
    const std::string b = build_openai_request(c);
    EXPECT_TRUE(contains(b, R"(He said \"hi\"\nthen left\tquickly\\done)"));
}

TEST(OpenAiRequest, PassesUtf8ThroughUnescaped) {
    RequestContext c = basic_ctx();
    c.intent = "\xD1\x81\xD1\x82\xD0\xBE\xD0\xBB\xD0\xB8\xD1\x86\xD0\xB0";  // "столица"
    // UTF-8 is legal in a JSON string and must survive byte-for-byte: escaping
    // it to \uXXXX would be correct too, but silently mangling it would not.
    EXPECT_TRUE(contains(build_openai_request(c),
                         "\xD1\x81\xD1\x82\xD0\xBE\xD0\xBB\xD0\xB8\xD1\x86\xD0\xB0"));
}

// A raw control byte is invalid inside a JSON string -- the gateway answers 400
// and the user sees "the assistant did not answer". Detokenizers do emit them,
// which is why append_json_string has the \u00xx branch at all.
TEST(OpenAiRequest, EscapesControlCharactersAsUnicodeEscapes) {
    // NAMED LOCAL, not a temporary: every RequestContext field is a
    // std::string_view, so `c.intent = std::string(...)` compiles and leaves a
    // dangling view the moment the full expression ends. The struct is built
    // that way on purpose (it borrows the caller's already-owned buffers), and
    // this is the trap that comes with it.
    const std::string intent("a\x01z\x1F");
    RequestContext c = basic_ctx();
    c.intent = intent;
    const std::string b = build_openai_request(c);
    EXPECT_TRUE(contains(b, "a\\u0001z\\u001f"));
    // And the raw bytes must be gone, not merely accompanied by an escape.
    EXPECT_FALSE(contains(b, std::string("a\x01")));
}

}  // namespace
