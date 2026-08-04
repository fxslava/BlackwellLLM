// =============================================================================
// openai_client_test.cpp — OpenAiStreamClient + OpenAiTransport, driven for
// real: the curl handle is created, the multi poll loop runs, the watchdogs
// fire, and the Result is classified.
//
// NO LIVE ENDPOINT IS CONTACTED, and that is a deliberate limit rather than an
// oversight. A test that spent money would be a test nobody runs; a test that
// depended on a third-party gateway being up would fail for reasons that are
// not this repo's. What is left is still the part worth guarding: everything
// between "construct a client" and "classify what came back" is code we wrote,
// and none of it was exercised at all before this file existed.
//
// The target is 127.0.0.1:9 (the discard port). Nothing listens, so the connect
// budget decides how long these tests take -- hence the deliberately short
// timeouts below. Whether that reads as ConnectTimeout or NetworkError depends
// on whether the host RSTs or drops, so both are accepted; what must NOT happen
// is Ok, a hang, or a crash.
// =============================================================================
#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <string_view>

#include "openai_request.hpp"
#include "openai_stream_client.hpp"
#include "openai_transport.hpp"

namespace {

using namespace blackwell::cloud;

constexpr long kConnectMs = 700;   // nothing listens; this bounds the whole test
constexpr long kTtftMs = 1500;

OpenAiStreamClient::Config unreachable_config() {
    OpenAiStreamClient::Config c;
    c.api_key = "sk-not-a-real-key";
    c.base_url = "http://127.0.0.1:9/v1";
    c.connect_timeout_ms = kConnectMs;
    c.ttft_timeout_ms = kTtftMs;
    c.stall_timeout_ms = kTtftMs;
    return c;
}

long long elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

// ---- construction -----------------------------------------------------------

// The user edits a BASE url; "/chat/completions" is the client's business.
TEST(OpenAiClient, DerivesTheEndpointFromTheBaseUrl) {
    OpenAiStreamClient::Config c = unreachable_config();
    c.base_url = "https://router.cheap/v1/";
    OpenAiStreamClient client(std::move(c));
    EXPECT_EQ(client.endpoint(), "https://router.cheap/v1/chat/completions");
}

// curl_global_init() is nobody's responsibility any more -- the shared core
// does it once, thread-safely. Constructing several clients must be fine.
TEST(OpenAiClient, MultipleClientsCoexist) {
    OpenAiStreamClient a(unreachable_config());
    OpenAiStreamClient b(unreachable_config());
    EXPECT_EQ(a.endpoint(), b.endpoint());
}

// ---- the transport's two jobs ----------------------------------------------

TEST(OpenAiTransportTest, BadgeNamesTheModelAndReportsAsBilled) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "gpt-4o-mini");
    EXPECT_STREQ(t.name(), "Remote: gpt-4o-mini");
    // The cost badge is the only thing telling a user a commit was billed.
    EXPECT_TRUE(t.is_live());
}

// The dispatcher's ContextProvider is shared by every leg, so it supplies the
// Anthropic default. The transport that will actually send the body is what
// decides the model for its own endpoint.
TEST(OpenAiTransportTest, OverridesTheContextsModelWithItsOwn) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "gpt-4o-mini");

    RequestContext ctx;
    ctx.instructions = "be brief";
    ctx.intent = "hello";
    ctx.model = "claude-opus-5";

    const std::string body = t.build_body(ctx);
    EXPECT_NE(body.find(R"("model":"gpt-4o-mini")"), std::string::npos);
    EXPECT_EQ(body.find("claude-opus-5"), std::string::npos);
    EXPECT_NE(body.find(R"("stream":true)"), std::string::npos);
}

TEST(OpenAiTransportTest, PassesItsRequestOptionsThrough) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiRequestOptions opt;
    opt.max_tokens = 128;
    OpenAiTransport t(client, "m", opt);

    RequestContext ctx;
    ctx.intent = "hello";
    EXPECT_NE(t.build_body(ctx).find(R"("max_tokens":128)"), std::string::npos);
}

// Anthropic's prewarm is free (max_tokens:0 writes the cache and bills no
// output). This endpoint has no such shape, so the honest answer is to send
// nothing rather than to silently spend money shaving a TLS handshake.
TEST(OpenAiTransportTest, PrewarmIsAFreeNoOp) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "m");

    RequestContext ctx;
    ctx.intent = "ping";
    EXPECT_TRUE(t.build_prewarm_body(ctx).empty());

    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(t.prewarm("").status, Status::Ok);
    // Returning Ok is only honest if nothing was sent -- so it must be instant,
    // well inside the connect budget an actual request would have paid.
    EXPECT_LT(elapsed_ms(t0), kConnectMs);
}

// ---- the curl core, actually running ---------------------------------------

TEST(OpenAiClient, UnreachableEndpointIsClassifiedAndBounded) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "gpt-4o-mini");

    RequestContext ctx;
    ctx.intent = "hello";
    const std::string body = t.build_body(ctx);

    int deltas = 0;
    Callbacks cb;
    cb.on_text = [&deltas](std::string_view) { ++deltas; };

    const auto t0 = std::chrono::steady_clock::now();
    const TransportRequest req{body, "hello", 1};
    const Result r = t.send(req, cb);
    const long long ms = elapsed_ms(t0);

    EXPECT_NE(r.status, Status::Ok) << "detail: " << r.error_detail;
    EXPECT_TRUE(r.status == Status::ConnectTimeout || r.status == Status::NetworkError)
        << "got " << to_string(r.status) << ": " << r.error_detail;
    // A transport-level failure is worth another attempt on the same frozen
    // body; the dispatcher's retry ladder depends on this classification.
    EXPECT_TRUE(is_retryable(r.status, r.http_status));
    EXPECT_EQ(deltas, 0);
    // The watchdog is the point: no CURLOPT_TIMEOUT is set (it would kill long
    // generations), so if the poll loop's budgets did not work this would hang.
    EXPECT_LT(ms, kConnectMs + kTtftMs + 2000) << "took " << ms << "ms";
}

// A FIRED WATCHDOG MUST SAY WHICH ONE, AND BY HOW MUCH.
//
// This is a regression test for a diagnosis, not for behaviour. The message
// used to be a bare "watchdog fired" -- which reached the UI verbatim, named
// neither budget, and gave no way to tell a genuinely slow endpoint from a
// mistuned timeout. The only way to act on it was to go looking for the
// numbers, and the number that turns up first is the deliberately-tiny one in
// THIS file, which production never uses. That cost a bug report.
//
// A zero TTFT budget fires on the first poll iteration, before the connect
// budget can expire, so this needs no server and no timing tolerance.
TEST(OpenAiClient, TtftWatchdogNamesItselfAndReportsWhatItMeasured) {
    OpenAiStreamClient::Config c = unreachable_config();
    c.ttft_timeout_ms = 0;
    OpenAiStreamClient client(std::move(c));
    OpenAiTransport t(client, "m");

    RequestContext ctx;
    ctx.intent = "hello";
    const std::string body = t.build_body(ctx);
    const TransportRequest req{body, "hello", 1};
    Callbacks cb;

    const Result r = t.send(req, cb);
    EXPECT_EQ(r.status, Status::TtftTimeout) << "got " << to_string(r.status);
    EXPECT_NE(r.error_detail.find("no first byte"), std::string::npos) << r.error_detail;
    EXPECT_NE(r.error_detail.find("TTFT budget"), std::string::npos) << r.error_detail;
    // The bare string is what this test exists to keep out of the UI.
    EXPECT_EQ(r.error_detail.find("watchdog fired"), std::string::npos) << r.error_detail;
}

// Teardown is not a watchdog and must not read as one -- it is the normal way a
// transfer ends when the app is closing, and reporting it as a timeout sends
// people looking for a performance problem that does not exist.
TEST(OpenAiClient, TeardownIsNotReportedAsAWatchdog) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "m");

    RequestContext ctx;
    ctx.intent = "hello";
    const std::string body = t.build_body(ctx);
    const TransportRequest req{body, "hello", 1};
    Callbacks cb;

    t.shutdown();
    const Result r = t.send(req, cb);
    ASSERT_EQ(r.status, Status::ShuttingDown);
    EXPECT_EQ(r.error_detail.find("watchdog"), std::string::npos) << r.error_detail;
}

// The easy handle is REUSED across requests -- that is what keeps the
// connection pooled. A failed transfer must not leave it poisoned (the
// completion message has to be drained even when a watchdog fired).
TEST(OpenAiClient, HandleIsReusableAfterAFailedTransfer) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "m");

    RequestContext ctx;
    ctx.intent = "hello";
    const std::string body = t.build_body(ctx);
    const TransportRequest req{body, "hello", 1};
    Callbacks cb;

    const Result first = t.send(req, cb);
    const Result second = t.send(req, cb);
    EXPECT_NE(first.status, Status::Ok);
    EXPECT_EQ(second.status, first.status) << "second: " << to_string(second.status);
}

// shutdown() is the one method callable from another thread, and it exists so
// teardown is not held up by an in-flight transfer. After it, send() must
// short-circuit rather than pay a full connect budget on the way out.
TEST(OpenAiClient, SendAfterShutdownReturnsImmediately) {
    OpenAiStreamClient client(unreachable_config());
    OpenAiTransport t(client, "m");

    RequestContext ctx;
    ctx.intent = "hello";
    const std::string body = t.build_body(ctx);
    const TransportRequest req{body, "hello", 1};
    Callbacks cb;

    t.shutdown();
    const auto t0 = std::chrono::steady_clock::now();
    const Result r = t.send(req, cb);
    EXPECT_EQ(r.status, Status::ShuttingDown);
    EXPECT_LT(elapsed_ms(t0), kConnectMs);
}

}  // namespace
