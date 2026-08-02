// =============================================================================
// local_transport_test.cpp — fully-local inference: routing, marshaling, and the
// one property that makes it safe.
//
// THE PROPERTY. A locally generated reply must NEVER be offered to the commit
// gate. The gate's job is to decide which of the USER's utterances deserve an
// answer; feeding the assistant's own words back into it would commit them as a
// fresh intent, answer that, and never stop -- a runaway that on the real
// backbone is a GPU pinned at 100% and, with a live transport bound, a billing
// incident. It is also completely silent: every individual step looks correct.
//
// So DoesNotFeedItsOwnReplyBackIntoTheGate is the test this file exists for. The
// rest establish that the routing and the thread marshaling around it work.
// =============================================================================
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "intent_commit.hpp"
#include "simulated_engine_control.hpp"
#include "local_transport.hpp"
#include "offline_transport.hpp"

using blackwell::bridge::IntentCommitQueue;
using blackwell::bridge::TerminationReason;
using rt::LocalEngineTransport;
using rt::RoutedTransport;
using rt::SimulatedEngineControl;

namespace {

// Drives the engine-thread pump for the duration of a test, exactly as main()'s
// engine thread does. post_engine_task work only runs while this is alive.
class PumpThread {
public:
    explicit PumpThread(blackwell::bridge::EngineControlBridge& control) : control_(control) {
        thread_ = std::thread([this] {
            while (running_.load(std::memory_order_acquire)) {
                (void)control_.wait_and_pump();
            }
        });
    }
    ~PumpThread() { stop(); }
    void stop() noexcept {
        if (!running_.exchange(false, std::memory_order_acq_rel)) return;
        control_.stop();
        if (thread_.joinable()) thread_.join();
    }

private:
    blackwell::bridge::EngineControlBridge& control_;
    std::atomic<bool> running_{true};
    std::thread thread_;
};

// A transport that records what it was asked to send, so routing is observable.
class RecordingTransport final : public blackwell::cloud::IIntentTransport {
public:
    explicit RecordingTransport(const char* label) : label_(label) {}
    [[nodiscard]] const char* name() const noexcept override { return label_; }
    [[nodiscard]] bool is_live() const noexcept override { return live; }
    [[nodiscard]] blackwell::cloud::Result send(
        const blackwell::cloud::TransportRequest& req,
        const blackwell::cloud::Callbacks& cb) noexcept override {
        sent.fetch_add(1, std::memory_order_relaxed);
        last_intent = std::string(req.intent);
        if (cb.on_text) cb.on_text("remote reply");
        blackwell::cloud::Result r;
        r.status = blackwell::cloud::Status::Ok;
        return r;
    }
    std::atomic<int> sent{0};
    std::string last_intent;
    bool live = true;

private:
    const char* label_;
};

blackwell::cloud::Result run_send(blackwell::cloud::IIntentTransport& t,
                                  std::string_view intent, std::string* out_text) {
    const std::string body = "{}";
    const blackwell::cloud::TransportRequest req{body, intent, /*sequence=*/1};
    blackwell::cloud::Callbacks cb;
    cb.on_text = [out_text](std::string_view s) {
        if (out_text != nullptr) out_text->append(s);
    };
    return t.send(req, cb);
}

}  // namespace

// ---- THE safety property -----------------------------------------------------

TEST(LocalTransport, DoesNotFeedItsOwnReplyBackIntoTheGate) {
    SimulatedEngineControl control;
    control.set_piece_delay_ms(0);
    IntentCommitQueue gate(8);
    control.set_commit_queue(&gate);
    PumpThread pump(control);

    LocalEngineTransport local(&control, [&control](const std::string& intent,
                                                    const std::function<void(std::string_view)>& emit,
                                                    TerminationReason* reason) {
        return control.generate_local_reply(intent, emit, reason);
    });

    const uint64_t committed_before = gate.committed();
    std::string reply;
    const blackwell::cloud::Result r = run_send(local, "user said something", &reply);

    EXPECT_EQ(r.status, blackwell::cloud::Status::Ok);
    EXPECT_FALSE(reply.empty());
    // THE assertion: generating an answer must not commit anything. If this ever
    // fails, the local path has become self-feeding.
    EXPECT_EQ(gate.committed(), committed_before);
    EXPECT_EQ(gate.dropped_barge_in(), 0u);
    EXPECT_EQ(gate.dropped_token_cap(), 0u);
    // And nothing may be waiting to be dispatched either -- a queued intent would
    // start the loop on the dispatcher's next pop rather than immediately.
    blackwell::bridge::IntentRecord rec;
    EXPECT_FALSE(gate.try_pop(rec));
}

// ---- marshaling --------------------------------------------------------------

TEST(LocalTransport, GenerationRunsOnTheEngineThreadNotTheCaller) {
    SimulatedEngineControl control;
    control.set_piece_delay_ms(0);
    PumpThread pump(control);

    std::atomic<std::thread::id> ran_on{};
    LocalEngineTransport local(&control, [&](const std::string& intent,
                                             const std::function<void(std::string_view)>& emit,
                                             TerminationReason* reason) {
        ran_on.store(std::this_thread::get_id(), std::memory_order_release);
        return control.generate_local_reply(intent, emit, reason);
    });

    std::string reply;
    EXPECT_EQ(run_send(local, "hello", &reply).status, blackwell::cloud::Status::Ok);
    // The single-thread doctrine: CUDA work must not run on the caller (the
    // dispatcher thread), which is the whole reason this goes through
    // post_engine_task.
    EXPECT_NE(ran_on.load(std::memory_order_acquire), std::this_thread::get_id());
}

TEST(LocalTransport, ShutdownReleasesACallerWhoseTaskWillNeverRun) {
    SimulatedEngineControl control;
    LocalEngineTransport local(&control, [&control](const std::string& intent,
                                                    const std::function<void(std::string_view)>& emit,
                                                    TerminationReason* reason) {
        return control.generate_local_reply(intent, emit, reason);
    });
    // NO pump thread: the task is queued and nothing will ever execute it. This
    // is the real teardown ordering (main() joins the engine thread before the
    // dispatcher stops), so send() must be releasable from outside.
    std::atomic<bool> returned{false};
    std::thread caller([&] {
        std::string reply;
        const blackwell::cloud::Result r = run_send(local, "hello", &reply);
        EXPECT_EQ(r.status, blackwell::cloud::Status::ShuttingDown);
        returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(returned.load(std::memory_order_acquire)) << "should still be parked";
    local.shutdown();
    caller.join();
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
}

// A barge-in mid-answer is a clean supersede, not a failure -- and must NOT be
// retried, or the user gets a second full decode of the turn they talked over.
TEST(LocalTransport, BargeInReportsShuttingDownRatherThanAnError) {
    SimulatedEngineControl control;
    control.set_piece_delay_ms(5);   // long enough to interrupt mid-stream
    PumpThread pump(control);

    LocalEngineTransport local(&control, [&control](const std::string& intent,
                                                    const std::function<void(std::string_view)>& emit,
                                                    TerminationReason* reason) {
        return control.generate_local_reply(intent, emit, reason);
    });

    std::thread barge([&control] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        control.cancel_generation(control.active_generation() + 1);
    });
    std::string reply;
    const blackwell::cloud::Result r = run_send(local, "a fairly long user utterance", &reply);
    barge.join();

    EXPECT_EQ(r.status, blackwell::cloud::Status::ShuttingDown);
    EXPECT_EQ(r.stop_reason, "barge_in");
}

// The local path carries Cyrillic, so it is held to the same streaming contract
// every other producer is: whole code points, losslessly reassembled.
TEST(LocalTransport, StreamsValidUtf8) {
    SimulatedEngineControl control;
    control.set_piece_delay_ms(0);
    PumpThread pump(control);

    LocalEngineTransport local(&control, [&control](const std::string& intent,
                                                    const std::function<void(std::string_view)>& emit,
                                                    TerminationReason* reason) {
        return control.generate_local_reply(intent, emit, reason);
    });

    std::vector<std::string> chunks;
    const std::string body = "{}";
    const blackwell::cloud::TransportRequest req{body, "Привет", 1};
    blackwell::cloud::Callbacks cb;
    cb.on_text = [&chunks](std::string_view s) { chunks.emplace_back(s); };
    EXPECT_EQ(local.send(req, cb).status, blackwell::cloud::Status::Ok);

    std::string full;
    for (const std::string& c : chunks) full += c;
    // The stand-in replies in Russian on purpose; a reply that lost the intent
    // would mean the marshaled copy did not survive the thread hop.
    EXPECT_NE(full.find("Привет"), std::string::npos);
}

// ---- routing -----------------------------------------------------------------

TEST(RoutedTransport, SendsToWhicheverLegIsSelected) {
    RecordingTransport local_leg("local");
    local_leg.live = false;
    RecordingTransport remote_leg("remote");
    RoutedTransport router(&local_leg, &remote_leg);

    router.set_use_local(false);
    EXPECT_EQ(run_send(router, "one", nullptr).status, blackwell::cloud::Status::Ok);
    EXPECT_EQ(remote_leg.sent.load(), 1);
    EXPECT_EQ(local_leg.sent.load(), 0);

    router.set_use_local(true);
    EXPECT_EQ(run_send(router, "two", nullptr).status, blackwell::cloud::Status::Ok);
    EXPECT_EQ(remote_leg.sent.load(), 1);   // unchanged
    EXPECT_EQ(local_leg.sent.load(), 1);
}

// The cost badge is driven by is_live(), so it has to follow the routing -- a
// user who switched to local must not keep seeing "billed".
TEST(RoutedTransport, NameAndLivenessFollowTheSelectedLeg) {
    RecordingTransport local_leg("Local model");
    local_leg.live = false;
    RecordingTransport remote_leg("Claude");
    RoutedTransport router(&local_leg, &remote_leg);

    router.set_use_local(false);
    EXPECT_STREQ(router.name(), "Claude");
    EXPECT_TRUE(router.is_live());

    router.set_use_local(true);
    EXPECT_STREQ(router.name(), "Local model");
    EXPECT_FALSE(router.is_live());
}

// Teardown must not depend on where the toggle happens to be pointing.
TEST(RoutedTransport, ShutdownReachesBothLegs) {
    struct CountingTransport final : blackwell::cloud::IIntentTransport {
        [[nodiscard]] const char* name() const noexcept override { return "x"; }
        [[nodiscard]] bool is_live() const noexcept override { return false; }
        [[nodiscard]] blackwell::cloud::Result send(
            const blackwell::cloud::TransportRequest&,
            const blackwell::cloud::Callbacks&) noexcept override {
            return {};
        }
        void shutdown() noexcept override { ++stops; }
        int stops = 0;
    };
    CountingTransport a, b;
    RoutedTransport router(&a, &b);
    router.set_use_local(true);
    router.shutdown();
    EXPECT_EQ(a.stops, 1);
    EXPECT_EQ(b.stops, 1);
}
