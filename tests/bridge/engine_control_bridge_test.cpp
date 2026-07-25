// =============================================================================
// tests/bridge/engine_control_bridge_test.cpp
//
// CPU-only integration test for EngineControlBridge (src/bridge). It exercises
// the full control plane WITHOUT a GPU by subclassing the bridge and overriding
// the do_*/decode_one execute hooks — so the SPSC command ring, the monotone
// barge-in epoch, the drop-if-stale rule, the wait-free decode cancel, and the
// 0%-CPU wait/notify path are all verified independently of any CUDA engine.
//
// The engine-facing symbols the base class references (BlackwellEngine::rewind /
// forward_status) are satisfied at link time by blackwell_core_obj; the overrides
// mean they are never actually invoked here.
// =============================================================================
#include "engine_control_bridge.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "bridge/engine_api.h"  // BridgeStatus / BRIDGE_OK

using blackwell::EngineStatus;
using blackwell::bridge::EngineControlBridge;
using blackwell::bridge::TokenSink;

namespace {

// Records how the engine thread's decode loop reported tokens through the sink.
struct SinkState {
    std::atomic<int>          tokens{0};
    std::atomic<int>          finals{0};
    std::atomic<BridgeStatus> last_final{BRIDGE_OK};
};

void sink_cb(void* user, const char* /*utf8*/, int32_t /*index*/, int32_t is_final,
             BridgeStatus status) {
    auto* s = static_cast<SinkState*>(user);
    if (is_final != 0) {
        s->last_final.store(status, std::memory_order_relaxed);
        s->finals.fetch_add(1, std::memory_order_relaxed);
    } else {
        s->tokens.fetch_add(1, std::memory_order_relaxed);
    }
}

// Test double: overrides every execute hook so no engine is needed. Records the
// dispatched command types/gens; for barge-in it drives the REAL run_decode_loop
// with a simulated per-token decode_one.
class TestBridge : public EngineControlBridge {
public:
    explicit TestBridge(const Config& cfg = {}) : EngineControlBridge(nullptr, cfg) {}

    // ---- knobs / observations (test thread) ----
    bool             loop_commit = false;  // do_commit_decode runs run_decode_loop
    int              loop_cap = 0;         // run_decode_loop's max_new_tokens
    int              step_sleep_us = 0;    // pace decode_one so a barge-in can land
    std::atomic<int> steps{0};             // decode_one invocations
    uint32_t         last_keep = 0;        // last REWIND keep_prompt_tokens

    std::vector<int> got_types() {
        std::lock_guard<std::mutex> lk(m_);
        return types_;
    }
    std::vector<uint64_t> got_gens() {
        std::lock_guard<std::mutex> lk(m_);
        return gens_;
    }

protected:
    EngineStatus do_rewind(const Command& cmd) override {
        record(/*Rewind*/ 0, cmd.gen);
        last_keep = cmd.keep_prompt_tokens;
        return EngineStatus::Success;
    }
    EngineStatus do_warm_prefill(const Command& cmd) override {
        record(/*WarmPrefill*/ 1, cmd.gen);
        return EngineStatus::Success;
    }
    EngineStatus do_commit_decode(const Command& cmd) override {
        record(/*CommitDecode*/ 2, cmd.gen);
        if (loop_commit) {
            return run_decode_loop(cmd.gen, /*first_token_id=*/1, /*start_pos=*/0, loop_cap,
                                   cmd.sink);
        }
        clear_in_flight();
        return EngineStatus::Success;
    }
    EngineStatus decode_one(int token_id, int /*pos*/, int* out_token) noexcept override {
        steps.fetch_add(1, std::memory_order_relaxed);
        if (step_sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(step_sleep_us));
        }
        if (out_token != nullptr) *out_token = token_id + 1;
        return EngineStatus::Success;
    }

private:
    void record(int type, uint64_t gen) {
        std::lock_guard<std::mutex> lk(m_);
        types_.push_back(type);
        gens_.push_back(gen);
    }
    std::mutex            m_;
    std::vector<int>      types_;
    std::vector<uint64_t> gens_;
};

}  // namespace

// The engine thread drains queued commands in FIFO order and executes each once.
TEST(EngineControlBridge, DispatchesQueuedCommandsInFifoOrder) {
    TestBridge b;
    EXPECT_EQ(b.rewind_kv(nullptr, /*keep=*/7, /*gen=*/1), EngineStatus::Success);
    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::Success);
    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::Success);

    EXPECT_EQ(b.pump(), 3u);

    const std::vector<int> types = b.got_types();
    ASSERT_EQ(types.size(), 3u);
    EXPECT_EQ(types[0], 0);  // Rewind
    EXPECT_EQ(types[1], 1);  // WarmPrefill
    EXPECT_EQ(types[2], 1);  // WarmPrefill
    EXPECT_EQ(b.last_keep, 7u);
}

// A barge-in (newer epoch) drops both still-queued ops and ops enqueued stale.
TEST(EngineControlBridge, SupersededCommandsAreDropped) {
    TestBridge b;
    EXPECT_EQ(b.rewind_kv(nullptr, /*keep=*/5, /*gen=*/1), EngineStatus::Success);  // queued @ gen 1

    b.cancel_generation(2);  // barge-in: live epoch -> 2
    EXPECT_EQ(b.active_generation(), 2u);

    // Enqueued with a stale gen -> dropped immediately (still reports Success).
    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::Success);

    // The queued gen-1 rewind is now stale too; pump drops it (0 executed).
    EXPECT_EQ(b.pump(), 0u);
    EXPECT_TRUE(b.got_types().empty());
}

// cancelled(gen) is the wait-free barge-in check the decode loop reads.
TEST(EngineControlBridge, CancelledReflectsEpoch) {
    TestBridge b;
    EXPECT_FALSE(b.cancelled(5));
    b.cancel_generation(5);
    EXPECT_FALSE(b.cancelled(5));  // same epoch is still live
    EXPECT_TRUE(b.cancelled(4));   // older epoch superseded
    b.cancel_generation(3);        // non-monotone call is ignored
    EXPECT_EQ(b.active_generation(), 5u);
}

// commit_and_decode latches the single-in-flight guard; the loop clears it.
TEST(EngineControlBridge, CommitLatchesAndClearsInFlight) {
    TestBridge b;
    SinkState ss;
    const TokenSink sink{&sink_cb, &ss};

    EXPECT_FALSE(b.generation_in_flight());
    EXPECT_EQ(b.commit_and_decode(nullptr, sink, /*gen=*/1), EngineStatus::Success);
    EXPECT_TRUE(b.generation_in_flight());  // latched at enqueue, before the engine runs

    EXPECT_EQ(b.pump(), 1u);
    EXPECT_FALSE(b.generation_in_flight());  // cleared on decode exit
}

// A barge-in aborts an in-flight decode within one token (wait-free interrupt).
TEST(EngineControlBridge, BargeInAbortsInFlightDecode) {
    TestBridge b;
    b.loop_commit = true;
    b.loop_cap = 1'000'000;  // effectively unbounded; only a barge-in should stop it
    b.step_sleep_us = 100;   // pace so the cancel lands mid-loop

    SinkState ss;
    const TokenSink sink{&sink_cb, &ss};
    ASSERT_EQ(b.commit_and_decode(nullptr, sink, /*gen=*/5), EngineStatus::Success);
    EXPECT_TRUE(b.generation_in_flight());

    std::thread engine_thread([&b] { b.pump(); });

    // Let a few tokens decode, then barge in.
    while (b.steps.load(std::memory_order_relaxed) < 5) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    b.cancel_generation(6);  // barge-in
    engine_thread.join();

    EXPECT_LT(b.steps.load(std::memory_order_relaxed), 1'000'000);  // aborted early
    EXPECT_EQ(ss.finals.load(std::memory_order_relaxed), 1);        // exactly one final
    EXPECT_EQ(ss.last_final.load(std::memory_order_relaxed), BRIDGE_OK);  // clean supersede
    EXPECT_FALSE(b.generation_in_flight());
}

// wait_and_pump() blocks at 0% CPU on an empty ring and wakes on an enqueue.
TEST(EngineControlBridge, WaitAndPumpBlocksThenWakesOnEnqueue) {
    TestBridge b;
    std::atomic<size_t> ret{~size_t{0}};
    std::atomic<bool>   returned{false};

    std::thread waiter([&] {
        ret.store(b.wait_and_pump(), std::memory_order_release);
        returned.store(true, std::memory_order_release);
    });

    // Give it time to park; it must still be blocked (ring empty).
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(returned.load(std::memory_order_acquire));

    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::Success);  // rings the doorbell
    waiter.join();

    EXPECT_EQ(ret.load(std::memory_order_acquire), 1u);
    const std::vector<int> types = b.got_types();
    ASSERT_EQ(types.size(), 1u);
    EXPECT_EQ(types[0], 1);  // WarmPrefill
}

// stop() unblocks a parked wait_and_pump() for clean shutdown.
TEST(EngineControlBridge, StopUnblocksWaiter) {
    TestBridge b;
    std::atomic<size_t> ret{~size_t{0}};
    std::thread waiter([&] { ret.store(b.wait_and_pump(), std::memory_order_release); });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    b.stop();
    waiter.join();

    EXPECT_EQ(ret.load(std::memory_order_acquire), 0u);
}

// A full command ring reports StateMismatch instead of blocking or allocating.
TEST(EngineControlBridge, FullRingReportsStateMismatch) {
    EngineControlBridge::Config cfg;
    cfg.command_ring_capacity = 2;  // rounds to pow2 = 2
    TestBridge b(cfg);

    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::Success);
    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::Success);
    EXPECT_EQ(b.warm_prefill(nullptr, /*gen=*/1), EngineStatus::StateMismatch);  // full
}
