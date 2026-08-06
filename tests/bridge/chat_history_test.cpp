// =============================================================================
// chat_history_test.cpp — the remote leg's bounded memory (cloud/chat_history.hpp).
//
// WHAT IS ACTUALLY AT RISK HERE. This store is what makes the cloud model
// remember the user's name, and it is also the only thing standing between a
// long session and an input bill that grows with every turn. Both halves are
// invariants, not behaviour: an unbounded ring is a slow money leak that no test
// downstream would ever notice, and a half-recorded turn is an HTTP 400 on the
// stricter gateways.
//
// It lives in bridge_tests rather than cloud_tests because the header is
// dependency-free (no curl, no simdjson), so it must stay testable in a build
// with BUILD_CLOUD_CLIENT=OFF -- which is also the build in which the store
// still runs, feeding the offline stand-in.
// =============================================================================
#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "chat_history.hpp"
#include "intent_commit.hpp"
#include "intent_dispatcher.hpp"
#include "openai_request.hpp"

namespace {

using blackwell::cloud::ChatHistory;
using blackwell::cloud::ChatTurn;

// One completed exchange, the way the dispatcher's three callbacks drive it.
void turn(ChatHistory& h, const std::string& user, const std::string& reply) {
    h.begin_turn(user);
    h.append_reply(reply);
    (void)h.commit_turn();
}

std::vector<ChatTurn> snap(const ChatHistory& h) {
    std::vector<ChatTurn> v;
    h.snapshot(v);
    return v;
}

TEST(ChatHistory, RecordsCompletedTurnsOldestFirst) {
    ChatHistory h;
    turn(h, "my name is Ann", "Hello, Ann.");
    turn(h, "what is my name", "Ann.");

    const auto v = snap(h);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0].user, "my name is Ann");
    EXPECT_EQ(v[0].assistant, "Hello, Ann.");
    EXPECT_EQ(v[1].user, "what is my name");
}

// Streaming is the real call pattern: on_text fires once per delta.
TEST(ChatHistory, ConcatenatesStreamedDeltasIntoOneReply) {
    ChatHistory h;
    h.begin_turn("hi");
    h.append_reply("Hel");
    h.append_reply("lo");
    h.append_reply("!");
    EXPECT_TRUE(h.commit_turn());
    EXPECT_EQ(snap(h)[0].assistant, "Hello!");
}

// THE MEMORY BOUND. Without it a long session grows the store forever, and the
// window it feeds grows the request with it.
TEST(ChatHistory, DropsTheOldestTurnPastTheCap) {
    ChatHistory::Config cfg;
    cfg.max_turns = 3;
    ChatHistory h(cfg);
    for (int i = 1; i <= 10; ++i) {
        turn(h, "u" + std::to_string(i), "a" + std::to_string(i));
    }
    const auto v = snap(h);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0].user, "u8");
    EXPECT_EQ(v[2].user, "u10");
    EXPECT_EQ(h.size(), 3u);
}

// A half turn on the wire is two consecutive `user` messages, which the strict
// gateways reject outright -- so it must never reach the ring.
TEST(ChatHistory, RefusesToRecordAHalfTurn) {
    ChatHistory h;
    h.begin_turn("asked, never answered");
    EXPECT_FALSE(h.commit_turn());
    EXPECT_EQ(h.size(), 0u);

    h.begin_turn("");
    h.append_reply("a reply to nothing");
    EXPECT_FALSE(h.commit_turn());
    EXPECT_EQ(h.size(), 0u);
}

// A barge-in or a failed request drops the turn WHOLE. Replaying a sentence the
// user talked over would teach the model it was an accepted answer.
TEST(ChatHistory, AbandonedTurnsLeaveNoTrace) {
    ChatHistory h;
    turn(h, "keep me", "kept");
    h.begin_turn("interrupted");
    h.append_reply("half a sen");
    h.abandon_turn();

    const auto v = snap(h);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].user, "keep me");
}

// A delta arriving after the turn was abandoned (the local leg's emit races
// shutdown) must not resurrect it.
TEST(ChatHistory, IgnoresRepliesWhenNoTurnIsOpen) {
    ChatHistory h;
    h.append_reply("stray");
    h.begin_turn("u");
    h.abandon_turn();
    h.append_reply("late");
    EXPECT_FALSE(h.commit_turn());
    EXPECT_EQ(h.size(), 0u);
}

TEST(ChatHistory, ClearForgetsEverythingIncludingThePendingTurn) {
    ChatHistory h;
    turn(h, "u1", "a1");
    h.begin_turn("pending");
    h.clear();
    h.append_reply("late");
    EXPECT_FALSE(h.commit_turn());
    EXPECT_EQ(h.size(), 0u);
}

// THE RUNAWAY BOUND. A model that loops until max_tokens must not poison the
// next several requests with thousands of replayed tokens.
TEST(ChatHistory, ClampsAMessageThatRunsAway) {
    ChatHistory::Config cfg;
    cfg.max_chars_per_message = 16;
    ChatHistory h(cfg);
    h.begin_turn(std::string(100, 'u'));
    for (int i = 0; i < 50; ++i) h.append_reply("aaaaaaaaaa");
    ASSERT_TRUE(h.commit_turn());

    const auto v = snap(h);
    EXPECT_EQ(v[0].user.size(), 16u);
    EXPECT_EQ(v[0].assistant.size(), 16u);
}

// Clamping cuts at a BYTE offset, and this app transcribes Russian: a cut
// through a two-byte codepoint yields invalid UTF-8, which the gateway answers
// with a 400 that surfaces as "the assistant did not answer". The clamp must
// therefore land on a codepoint boundary, even if that means one byte short.
TEST(ChatHistory, NeverClampsInsideAUtf8Codepoint) {
    ChatHistory::Config cfg;
    cfg.max_chars_per_message = 5;  // odd cap vs. 2-byte Cyrillic: cuts mid-char
    ChatHistory h(cfg);
    h.begin_turn("\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82");  // "привет"
    h.append_reply("\xD0\xB4\xD0\xB0");                                // "да"
    ASSERT_TRUE(h.commit_turn());

    // NAMED, not `const std::string& u = snap(h)[0].user`: that binds into a
    // vector destroyed at the end of the full expression, and reference
    // lifetime extension does not reach through the subscript.
    const std::vector<ChatTurn> v = snap(h);
    const std::string& u = v[0].user;
    EXPECT_EQ(u.size(), 4u);          // 2 whole codepoints, not 2.5
    EXPECT_EQ(u, "\xD0\xBF\xD1\x80");  // "пр"
    // And no dangling lead byte, which is the shape that fails on the wire.
    EXPECT_NE(static_cast<unsigned char>(u.back()) & 0xC0, 0xC0);
}

// A 3-byte codepoint clipped with only ONE of its continuation bytes present is
// the other half of the same trap. Nothing whole fits, so the turn is refused
// outright rather than sent as a user message with no text in it.
TEST(ChatHistory, DropsAnIncompleteThreeByteSequence) {
    ChatHistory::Config cfg;
    cfg.max_chars_per_message = 2;
    ChatHistory h(cfg);
    h.begin_turn("\xE2\x82\xAC\xE2\x82\xAC");  // "€€"
    h.append_reply("ok");
    EXPECT_FALSE(h.commit_turn());
    EXPECT_EQ(h.size(), 0u);
}

// The same repair, driven by the streaming path rather than the cap: deltas
// split at TOKEN boundaries, so a reply can legitimately end mid-character.
TEST(ChatHistory, RepairsAReplyWhoseLastDeltaSplitACodepoint) {
    ChatHistory h;
    h.begin_turn("u");
    h.append_reply("\xD0\xB4\xD0\xB0");  // "да"
    h.append_reply("\xD0");              // a lead byte nothing completes
    ASSERT_TRUE(h.commit_turn());
    EXPECT_EQ(snap(h)[0].assistant, "\xD0\xB4\xD0\xB0");
}

// snapshot() overwrites rather than appends: the dispatcher reuses ONE scratch
// buffer for every attempt of every turn (main.cpp), so a growing vector would
// replay the window twice on the second request.
TEST(ChatHistory, SnapshotReplacesTheCallersBuffer) {
    ChatHistory h;
    turn(h, "u1", "a1");
    std::vector<ChatTurn> buf;
    h.snapshot(buf);
    h.snapshot(buf);
    EXPECT_EQ(buf.size(), 1u);

    turn(h, "u2", "a2");
    h.snapshot(buf);
    ASSERT_EQ(buf.size(), 2u);
    EXPECT_EQ(buf[1].user, "u2");
}

// =============================================================================
// THE WIRING, driven through the real dispatcher.
//
// Everything above tests the store in isolation; this tests the thing that
// actually breaks -- the four call sites in main.cpp that have to fire in the
// right order on two different threads. It reproduces that wiring exactly
// (context provider + the three dispatcher callbacks) and asserts on the bytes a
// gateway would receive, because "the model forgot my name" is a payload
// question and nothing below the payload can answer it.
// =============================================================================

// Renders the OpenAI-compatible body the way OpenAiTransport does, and keeps
// every one it was handed. Nothing here touches libcurl -- openai_request.hpp is
// header-only, which is what lets this live in the always-built suite.
class BodyRecordingTransport final : public blackwell::cloud::IIntentTransport {
public:
    const char* name() const noexcept override { return "recording"; }
    bool is_live() const noexcept override { return false; }

    std::string build_body(const blackwell::cloud::RequestContext& ctx) const override {
        return blackwell::cloud::build_openai_request(ctx);
    }

    blackwell::cloud::Result send(const blackwell::cloud::TransportRequest& req,
                                  const blackwell::cloud::Callbacks& cb) noexcept override {
        {
            const std::lock_guard<std::mutex> lk(mu_);
            bodies_.push_back(req.body);
        }
        if (cb.on_text) cb.on_text(reply_);
        blackwell::cloud::Result r;
        r.status = blackwell::cloud::Status::Ok;
        r.stop_reason = "end_turn";
        return r;
    }

    void set_reply(std::string s) {
        const std::lock_guard<std::mutex> lk(mu_);
        reply_ = std::move(s);
    }
    std::vector<std::string> bodies() const {
        const std::lock_guard<std::mutex> lk(mu_);
        return bodies_;
    }

private:
    mutable std::mutex       mu_;
    std::vector<std::string> bodies_;
    std::string              reply_ = "ack";
};

// Reproduces main.cpp's wiring around the dispatcher: ONE store, ONE scratch
// buffer spanned by RequestContext::history, and the three callbacks that open,
// feed and seal a turn.
struct DispatchHarness {
    blackwell::bridge::IntentCommitQueue queue{8};
    BodyRecordingTransport               transport;
    ChatHistory                          history;
    std::vector<ChatTurn>                scratch;  // dispatcher thread only
    blackwell::cloud::IntentDispatcher   dispatcher;

    DispatchHarness()
        : dispatcher(queue, transport, [this] {
              blackwell::cloud::RequestContext c;
              c.instructions = "frozen";
              c.model = "test-model";
              history.snapshot(scratch);
              c.history = scratch;
              return c;
          }) {
        dispatcher.set_on_dispatch_start(
            [this](const blackwell::bridge::IntentRecord& r) { history.begin_turn(r.payload); });
        dispatcher.set_on_text([this](std::string_view s) { history.append_reply(s); });
        dispatcher.set_on_complete(
            [this](const blackwell::bridge::IntentRecord&, const blackwell::cloud::Result& r) {
                if (r.status == blackwell::cloud::Status::Ok) {
                    (void)history.commit_turn();
                } else {
                    history.abandon_turn();
                }
            });
        dispatcher.start();
    }

    // Offers one finished utterance and blocks until its body has been recorded.
    void say(const std::string& utterance, const std::string& reply) {
        const std::size_t before = transport.bodies().size();
        transport.set_reply(reply);
        // offer() takes the payload by SINK reference (it moves it into the
        // record), so hand it a copy rather than the caller's string.
        ASSERT_TRUE(queue.offer(blackwell::bridge::TerminationReason::Eos, ++seq_,
                                std::string(utterance), 4));
        for (int i = 0; i < 400 && transport.bodies().size() == before; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ASSERT_GT(transport.bodies().size(), before) << "the dispatcher never sent " << utterance;
    }

    ~DispatchHarness() { dispatcher.stop(); }

private:
    std::uint64_t seq_ = 0;
};

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// THE BUG THIS CHANGE EXISTS TO FIX, stated as a test: the second request must
// carry the first exchange, or the remote model cannot know the user's name.
TEST(RemoteMemory, TheSecondRequestCarriesTheFirstExchange) {
    DispatchHarness h;
    h.say("my name is Ann", "Nice to meet you, Ann.");
    h.say("what is my name", "Ann.");

    const std::vector<std::string> bodies = h.transport.bodies();
    ASSERT_EQ(bodies.size(), 2u);

    // Turn one is a cold start: system + the utterance, and nothing else.
    EXPECT_FALSE(contains(bodies[0], R"("role":"assistant")"));

    // Turn two replays the pair, in order, ahead of the live question.
    EXPECT_TRUE(contains(bodies[1], R"({"role":"user","content":"my name is Ann"})"));
    EXPECT_TRUE(contains(bodies[1], R"({"role":"assistant","content":"Nice to meet you, Ann."})"));
    EXPECT_LT(bodies[1].find("my name is Ann"), bodies[1].find("what is my name"));
}

// The current utterance must NOT also appear as history: it is still the PENDING
// turn when the body is built, and a duplicated message reads to the model as
// the user saying it twice.
TEST(RemoteMemory, TheLiveUtteranceIsNotAlsoReplayedAsHistory) {
    DispatchHarness h;
    h.say("hello there", "hi");

    // Named copy: bodies() returns BY VALUE, so a reference into the subscript
    // would dangle the moment the full expression ends.
    const std::vector<std::string> bodies = h.transport.bodies();
    ASSERT_EQ(bodies.size(), 1u);
    EXPECT_EQ(bodies[0].find("hello there"), bodies[0].rfind("hello there"));
}

// THE COST BOUND, end to end. Ten turns in and the request still carries three
// pairs -- this is the assertion that would catch a regression to "replay
// everything", which no functional test would ever notice.
TEST(RemoteMemory, TheWindowStopsGrowingAfterTheCap) {
    DispatchHarness h;
    for (int i = 1; i <= 10; ++i) {
        h.say("utterance " + std::to_string(i), "reply " + std::to_string(i));
    }
    const std::vector<std::string> bodies = h.transport.bodies();
    ASSERT_EQ(bodies.size(), 10u);

    const std::string& last = bodies.back();  // `bodies` is a named copy, so this is safe
    // 1 system + 3 replayed pairs + 1 live turn.
    std::size_t roles = 0;
    for (std::size_t p = last.find(R"("role":)"); p != std::string::npos;
         p = last.find(R"("role":)", p + 1)) {
        ++roles;
    }
    EXPECT_EQ(roles, 8u);
    EXPECT_FALSE(contains(last, "utterance 1\""));  // long since dropped
    EXPECT_TRUE(contains(last, "utterance 7"));     // oldest still in the window
}

}  // namespace
