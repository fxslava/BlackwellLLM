// =============================================================================
// tests/bridge/intent_commit_test.cpp — THE COMMIT RULE.
//
// This suite exists to make one financial invariant mechanically checkable:
// a fragment must never reach the paid Cloud API. Everything here is CPU-only
// (no CUDA, no network, no libcurl) — the gate and the decode loop's
// termination reporting are both engine-independent by construction.
//
// The decode-loop tests drive the REAL EngineControlBridge::run_decode_loop
// through a stubbed decode_one/is_eos, so the loop's actual control flow is
// under test rather than a re-implementation of it.
// =============================================================================
#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "engine_control_bridge.hpp"
#include "intent_commit.hpp"

using blackwell::EngineStatus;
using blackwell::bridge::EngineControlBridge;
using blackwell::bridge::IntentCommitQueue;
using blackwell::bridge::IntentRecord;
using blackwell::bridge::TerminationReason;
using blackwell::bridge::TokenSink;

// ---------------------------------------------------------------------------
// The gate itself.
// ---------------------------------------------------------------------------

TEST(IntentCommitQueue, OnlyEosCommits) {
    IntentCommitQueue q(4);

    EXPECT_FALSE(q.offer(TerminationReason::BargeIn, 1, "barged in", 5));
    EXPECT_FALSE(q.offer(TerminationReason::TokenCap, 2, "truncated", 5));
    EXPECT_FALSE(q.offer(TerminationReason::Fault, 3, "faulted", 5));
    EXPECT_FALSE(q.offer(TerminationReason::None, 4, "never ran", 0));
    EXPECT_TRUE(q.empty()) << "a non-EOS termination must never become dispatchable";

    EXPECT_TRUE(q.offer(TerminationReason::Eos, 5, "finished thought", 7));
    EXPECT_EQ(q.size(), 1u);

    EXPECT_EQ(q.committed(), 1u);
    EXPECT_EQ(q.dropped_barge_in(), 1u);
    EXPECT_EQ(q.dropped_token_cap(), 1u);
    EXPECT_EQ(q.dropped_fault(), 1u);
    EXPECT_EQ(q.dropped_none(), 1u);
    EXPECT_EQ(q.dropped_total(), 4u);
}

TEST(IntentCommitQueue, RejectedOfferLeavesPayloadIntact) {
    // The caller must still be able to log/inspect a rejected fragment, so a
    // rejection may not consume the string.
    IntentCommitQueue q(2);
    std::string payload = "half a thought";
    EXPECT_FALSE(q.offer(TerminationReason::TokenCap, 1, std::move(payload), 3));
    EXPECT_EQ(payload, "half a thought");
}

TEST(IntentCommitQueue, PayloadSurvivesTheHandoff) {
    IntentCommitQueue q(2);
    ASSERT_TRUE(q.offer(TerminationReason::Eos, 42, "translate: hello world", 4));

    IntentRecord rec;
    ASSERT_TRUE(q.try_pop(rec));
    EXPECT_EQ(rec.payload, "translate: hello world");
    EXPECT_EQ(rec.generation, 42u);
    EXPECT_EQ(rec.token_count, 4u);
    EXPECT_EQ(rec.sequence, 1u);
    EXPECT_TRUE(q.empty());
}

TEST(IntentCommitQueue, FullQueueRefusesTheNewIntentNotTheOldOnes) {
    // Every record already in the queue passed the gate and is owed a call, so
    // backpressure must reject the arrival rather than evict a commitment.
    IntentCommitQueue q(2);
    ASSERT_TRUE(q.offer(TerminationReason::Eos, 1, "first", 1));
    ASSERT_TRUE(q.offer(TerminationReason::Eos, 2, "second", 1));
    EXPECT_FALSE(q.offer(TerminationReason::Eos, 3, "third", 1));
    EXPECT_EQ(q.dropped_queue_full(), 1u);

    IntentRecord rec;
    ASSERT_TRUE(q.try_pop(rec));
    EXPECT_EQ(rec.payload, "first") << "the oldest commitment must survive";
}

TEST(IntentCommitQueue, WaitPopWakesOnCommitAndOnStop) {
    IntentCommitQueue q(4);
    std::string got;

    std::thread consumer([&] {
        IntentRecord rec;
        if (q.wait_pop(rec)) got = rec.payload;
        // Second call must return false once stopped, not hang.
        IntentRecord ignored;
        EXPECT_FALSE(q.wait_pop(ignored));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ASSERT_TRUE(q.offer(TerminationReason::Eos, 1, "woke up", 2));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    q.stop();
    consumer.join();

    EXPECT_EQ(got, "woke up");
}

// ---------------------------------------------------------------------------
// The decode loop's verdict. This is the half that did not exist before: the
// loop used to exit `is_final=1, BRIDGE_OK` for EOS, barge-in AND token cap
// alike, so no consumer could tell a finished thought from a severed one.
// ---------------------------------------------------------------------------

namespace {

// Drives the REAL run_decode_loop. `eos_at` is the token id treated as EOS;
// kNoEos means the model never stops on its own.
class LoopHarness : public EngineControlBridge {
public:
    static constexpr int kNoEos = -1;

    explicit LoopHarness(int eos_at = kNoEos)
        : EngineControlBridge(nullptr, Config{}), eos_at_(eos_at) {}

    using EngineControlBridge::publish_intent;
    using EngineControlBridge::run_decode_loop;

    // Fail on step N to exercise the Fault branch.
    void fail_at(int step) noexcept { fail_at_ = step; }

    int steps() const noexcept { return steps_; }

protected:
    EngineStatus decode_one(int /*token_id*/, int /*pos*/, int* out_token) noexcept override {
        if (fail_at_ >= 0 && steps_ == fail_at_) return EngineStatus::CudaRuntimeError;
        ++steps_;
        if (out_token != nullptr) *out_token = steps_;  // ids 1,2,3,...
        return EngineStatus::Success;
    }

    bool is_eos(int token_id) const noexcept override {
        return eos_at_ != kNoEos && token_id == eos_at_;
    }

private:
    int eos_at_;
    int fail_at_ = -1;
    int steps_ = 0;
};

TokenSink null_sink() { return TokenSink{}; }

}  // namespace

TEST(RunDecodeLoop, ReportsEosWhenTheModelStopsOnItsOwn) {
    LoopHarness h(/*eos_at=*/3);
    TerminationReason reason = TerminationReason::None;
    const EngineStatus st =
        h.run_decode_loop(/*gen=*/0, /*first_token_id=*/1, /*start_pos=*/0,
                          /*max_new_tokens=*/64, null_sink(), &reason);

    EXPECT_EQ(st, EngineStatus::Success);
    EXPECT_EQ(reason, TerminationReason::Eos);
    EXPECT_EQ(h.steps(), 3) << "the loop must stop AT the EOS token, not past it";
}

TEST(RunDecodeLoop, ReportsTokenCapWhenTruncated) {
    // No EOS reachable within the cap -> a truncated thought, not a finished one.
    LoopHarness h(/*eos_at=*/999);
    TerminationReason reason = TerminationReason::None;
    const EngineStatus st = h.run_decode_loop(0, 1, 0, /*max_new_tokens=*/5, null_sink(), &reason);

    EXPECT_EQ(st, EngineStatus::Success) << "a cap is not an engine fault";
    EXPECT_EQ(reason, TerminationReason::TokenCap);
    EXPECT_FALSE(blackwell::bridge::is_dispatchable(reason));
}

TEST(RunDecodeLoop, ReportsBargeInWhenSuperseded) {
    LoopHarness h(/*eos_at=*/999);
    h.cancel_generation(7);  // bump the epoch past gen 0 before the loop starts

    TerminationReason reason = TerminationReason::None;
    const EngineStatus st = h.run_decode_loop(/*gen=*/0, 1, 0, 64, null_sink(), &reason);

    EXPECT_EQ(st, EngineStatus::Success) << "a supersede is not an engine fault";
    EXPECT_EQ(reason, TerminationReason::BargeIn);
    EXPECT_EQ(h.steps(), 0);
}

TEST(RunDecodeLoop, ReportsFaultOnEngineError) {
    LoopHarness h(/*eos_at=*/999);
    h.fail_at(2);

    TerminationReason reason = TerminationReason::None;
    const EngineStatus st = h.run_decode_loop(0, 1, 0, 64, null_sink(), &reason);

    EXPECT_NE(st, EngineStatus::Success);
    EXPECT_EQ(reason, TerminationReason::Fault);
}

TEST(RunDecodeLoop, BaseTierNeverReportsEos) {
    // The base bridge has no tokenizer. It must fail CLOSED: never EOS, so a
    // bridge whose is_eos() override was forgotten cannot dispatch anything.
    LoopHarness h(LoopHarness::kNoEos);
    TerminationReason reason = TerminationReason::None;
    (void)h.run_decode_loop(0, 1, 0, 4, null_sink(), &reason);
    EXPECT_EQ(reason, TerminationReason::TokenCap);
}

// ---------------------------------------------------------------------------
// End to end: loop verdict -> publish_intent -> queue. The invariant the whole
// pivot is for.
// ---------------------------------------------------------------------------

TEST(LocalRouter, OnlyAnEosGenerationReachesTheDispatchQueue) {
    IntentCommitQueue q(8);

    struct Case {
        int eos_at;
        int cap;
        bool cancel;
        bool expect_dispatch;
    };
    const std::vector<Case> cases = {
        {/*eos_at=*/2, /*cap=*/64, /*cancel=*/false, /*expect=*/true},   // finished
        {/*eos_at=*/999, /*cap=*/3, /*cancel=*/false, /*expect=*/false}, // truncated
        {/*eos_at=*/999, /*cap=*/64, /*cancel=*/true, /*expect=*/false}, // barged in
    };

    int expected_commits = 0;
    for (const Case& c : cases) {
        LoopHarness h(c.eos_at);
        h.set_commit_queue(&q);
        if (c.cancel) h.cancel_generation(1);

        TerminationReason reason = TerminationReason::None;
        (void)h.run_decode_loop(/*gen=*/0, 1, 0, c.cap, null_sink(), &reason);

        const bool committed = h.publish_intent(reason, 0, std::string("intent"), 1);
        EXPECT_EQ(committed, c.expect_dispatch)
            << "termination=" << blackwell::bridge::to_string(reason);
        if (c.expect_dispatch) ++expected_commits;
    }

    EXPECT_EQ(q.size(), static_cast<size_t>(expected_commits));
    EXPECT_EQ(q.committed(), static_cast<uint64_t>(expected_commits));
    EXPECT_EQ(q.dropped_total(), cases.size() - static_cast<size_t>(expected_commits));
}

TEST(LocalRouter, UnboundQueueDispatchesNothing) {
    // Cloud routing is opt-in. With no queue bound the local pipeline runs
    // unchanged and publish_intent is inert — including for a clean EOS.
    LoopHarness h(/*eos_at=*/2);
    TerminationReason reason = TerminationReason::None;
    (void)h.run_decode_loop(0, 1, 0, 64, null_sink(), &reason);
    ASSERT_EQ(reason, TerminationReason::Eos);
    EXPECT_FALSE(h.publish_intent(reason, 0, std::string("intent"), 1));
}

TEST(LocalRouter, TokenCapIsCountedNotJustSkipped) {
    // The DoD's telemetry requirement, and the reason publish_intent is called on
    // EVERY path including the rejected ones. If a caller "skips dispatch" by not
    // offering at all, this counter stays 0 and a mis-sized token ceiling kills
    // the cloud pathway with no signal anywhere in the system.
    IntentCommitQueue q(4);
    LoopHarness h(/*eos_at=*/999);   // EOS unreachable within the cap
    h.set_commit_queue(&q);

    TerminationReason reason = TerminationReason::None;
    (void)h.run_decode_loop(0, 1, 0, /*max_new_tokens=*/4, null_sink(), &reason);
    ASSERT_EQ(reason, TerminationReason::TokenCap);

    EXPECT_FALSE(h.publish_intent(reason, 0, std::string("truncated"), 4));
    EXPECT_EQ(q.dropped_token_cap(), 1u) << "truncation must be observable, not silent";
    EXPECT_EQ(q.committed(), 0u);
    EXPECT_TRUE(q.empty());
}

// ---------------------------------------------------------------------------
// The offline workflow end to end: gate -> dispatcher -> transport, with no
// libcurl, no network and no GPU. This is what makes voice_assistant's offline path
// a tested configuration rather than a demo.
// ---------------------------------------------------------------------------

#include "intent_dispatcher.hpp"
#include "offline_transport.hpp"

namespace {

// Records what actually reached the wire. A fragment appearing here is the
// financial bug the whole design exists to prevent.
class RecordingTransport final : public blackwell::cloud::IIntentTransport {
public:
    const char* name() const noexcept override { return "recording"; }
    bool is_live() const noexcept override { return false; }

    blackwell::cloud::Result send(const blackwell::cloud::TransportRequest& req,
                                  const blackwell::cloud::Callbacks& cb) noexcept override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            sent_.emplace_back(req.intent);
        }
        if (cb.on_text) cb.on_text("ack");
        blackwell::cloud::Result r;
        r.status = blackwell::cloud::Status::Ok;
        r.stop_reason = "end_turn";
        return r;
    }

    std::vector<std::string> sent() const {
        std::lock_guard<std::mutex> lk(mu_);
        return sent_;
    }

private:
    mutable std::mutex mu_;
    std::vector<std::string> sent_;
};

}  // namespace

TEST(OfflineDispatch, OnlyEosIntentsReachTheTransport) {
    IntentCommitQueue q(8);
    RecordingTransport transport;
    std::string streamed;

    blackwell::cloud::IntentDispatcher d(q, transport, [] {
        blackwell::cloud::RequestContext c;
        c.instructions = "frozen";   // byte-stable: no timestamp, no session id
        c.glossary = "none";
        c.committed_prefix = "";
        return c;
    });
    d.set_on_text([&](std::string_view s) { streamed.append(s); });
    d.start();

    // Two fragments and one finished thought, in that order.
    EXPECT_FALSE(q.offer(TerminationReason::BargeIn, 1, "interrupted", 3));
    EXPECT_FALSE(q.offer(TerminationReason::TokenCap, 2, "truncated", 256));
    EXPECT_TRUE(q.offer(TerminationReason::Eos, 3, "what is the weather", 5));

    for (int i = 0; i < 200 && transport.sent().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    d.stop();

    const std::vector<std::string> sent = transport.sent();
    ASSERT_EQ(sent.size(), 1u) << "exactly one intent was dispatchable";
    EXPECT_EQ(sent[0], "what is the weather");
    EXPECT_EQ(streamed, "ack");
    EXPECT_EQ(d.dispatched(), 1u);
    EXPECT_EQ(q.dropped_barge_in(), 1u);
    EXPECT_EQ(q.dropped_token_cap(), 1u);
}

TEST(OfflineDispatch, OfflineTransportEchoesTheCommittedIntent) {
    // The offline stand-in must reproduce the SHAPE of a real exchange (streamed
    // pieces, usage counters), or the GUI's incremental-render path is untested.
    blackwell::cloud::OfflineTransport offline(
        blackwell::cloud::OfflineTransport::Config{/*latency_ms=*/10, /*chunk_delay_ms=*/0,
                                                /*chunk_chars=*/8, /*simulate_usage=*/true});
    std::string got;
    int usage_calls = 0;
    blackwell::cloud::Callbacks cb;
    cb.on_text = [&](std::string_view s) { got.append(s); };
    cb.on_usage = [&](const blackwell::cloud::Usage&) { ++usage_calls; };

    const std::string body = "{\"model\":\"x\"}";
    const blackwell::cloud::TransportRequest req{body, "turn on the lights", 1};
    const blackwell::cloud::Result r = offline.send(req, cb);

    EXPECT_EQ(r.status, blackwell::cloud::Status::Ok);
    EXPECT_NE(got.find("turn on the lights"), std::string::npos);
    EXPECT_EQ(usage_calls, 1);
}
