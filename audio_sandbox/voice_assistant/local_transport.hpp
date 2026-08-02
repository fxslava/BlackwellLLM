#pragma once
// -----------------------------------------------------------------------------
// local_transport.hpp — LocalEngineTransport: the "answer it here" leg of the
// Local Router. Fully local inference, with no cloud call and no network.
//
// WHY IT IS A TRANSPORT AND NOT A NEW PIPELINE
//   Everything between a committed intent and the words on screen already
//   exists: IntentCommitQueue applies the EOS rule, IntentDispatcher drains the
//   gate on its own thread and blocks safely there, AssistantView turns deltas
//   into a bubble. NONE of that cares where the reply came from. Implementing
//   IIntentTransport therefore buys the entire path for free, and -- the part
//   that actually matters -- guarantees the local answer is gated by exactly the
//   same arithmetic the billed one is. A second route from speech to an answer
//   would be a second place for the commit rule to be subtly wrong.
//
//   So "Use the local model for responses" is a transport swap. Nothing above
//   this class branches on it; the dispatcher does not know it happened.
//
// THREADING is the whole difficulty, and it resolves cleanly.
//   send() runs on the DISPATCHER thread. Generation is CUDA work on a 5.3 GB
//   weight set and belongs to the ENGINE thread alone (CLAUDE.md's single-thread
//   doctrine). So send() marshals through post_engine_task() and blocks until
//   the engine thread is done -- which is exactly what the dispatcher thread is
//   for ("somewhere safe to block", intent_dispatcher.hpp). The engine thread
//   runs the task at a command-batch boundary, so a spoken turn already decoding
//   finishes first.
//
//   Consequence worth stating: on_text fires on the ENGINE thread here, not the
//   dispatcher thread. That is safe because the dispatcher is parked inside
//   send() for the whole generation (no concurrency), the handoff is ordered by
//   the state mutex, and AssistantView is thread-safe by construction.
//
// SHUTDOWN. The engine thread is joined BEFORE the dispatcher stops, so a task
// queued at the wrong moment would never run and send() would block forever.
// The wait is therefore bounded and re-checks an abandon flag: shutdown() cuts
// it loose, and the shared state outlives both threads (shared_ptr) so a task
// that runs afterwards writes to memory that is still alive and simply finds
// nobody listening.
// -----------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include "engine_control_bridge.hpp"   // EngineControlBridge (post_engine_task)
#include "intent_commit.hpp"           // TerminationReason
#include "intent_transport.hpp"        // IIntentTransport, TransportRequest, Callbacks

namespace rt {

class LocalEngineTransport final : public blackwell::cloud::IIntentTransport {
public:
    // Generate a reply to `intent`, streaming chunks through `emit`. Runs ON THE
    // ENGINE THREAD. A callable rather than a virtual for the same reason
    // ConversationalMode takes prefill_system_prompt that way: it lives on the
    // CONCRETE control (RealEngineControl / SimulatedEngineControl), and taking
    // it explicitly is what lets the GPU-free backend drive this exact class.
    using GenerateFn = std::function<blackwell::EngineStatus(
        const std::string& intent,
        const std::function<void(std::string_view)>& emit,
        blackwell::bridge::TerminationReason* out_reason)>;

    // `control` is BORROWED and must outlive this object.
    LocalEngineTransport(blackwell::bridge::EngineControlBridge* control, GenerateFn generate)
        : control_(control), generate_(std::move(generate)) {}

    [[nodiscard]] const char* name() const noexcept override { return "Local model"; }
    // FALSE, and this is load-bearing rather than cosmetic: `is_live` drives the
    // UI's cost indicator, and local inference spends GPU time but never money.
    // Reporting it as live would train the user to ignore the one badge that
    // tells them a commit was billed.
    [[nodiscard]] bool is_live() const noexcept override { return false; }

    [[nodiscard]] blackwell::cloud::Result send(
        const blackwell::cloud::TransportRequest& req,
        const blackwell::cloud::Callbacks& cb) noexcept override {
        blackwell::cloud::Result r;
        if (control_ == nullptr || !generate_) {
            r.status = blackwell::cloud::Status::NetworkError;
            r.error_detail = "local transport is not bound to an engine";
            return r;
        }

        // Shared with the engine task and outliving both threads, so a task that
        // runs after send() gave up still writes somewhere valid.
        auto state = std::make_shared<State>();
        state->on_text = cb.on_text;

        const std::string intent(req.intent);
        GenerateFn generate = generate_;

        const bool queued = control_->post_engine_task([state, intent, generate] {
            blackwell::bridge::TerminationReason reason =
                blackwell::bridge::TerminationReason::None;
            blackwell::EngineStatus status = blackwell::EngineStatus::Success;
            try {
                status = generate(
                    intent,
                    [&state](std::string_view chunk) {
                        // Abandoned mid-generation (shutdown): stop emitting, but
                        // let the engine finish its turn cleanly -- aborting a
                        // decode loop from the outside is the barge-in epoch's
                        // job, not a teardown race's.
                        std::lock_guard<std::mutex> lk(state->mu);
                        if (state->abandoned || !state->on_text) return;
                        state->on_text(chunk);
                    },
                    &reason);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(state->mu);
                state->detail = e.what();
                status = blackwell::EngineStatus::CudaRuntimeError;
            } catch (...) {
                status = blackwell::EngineStatus::CudaRuntimeError;
            }
            {
                std::lock_guard<std::mutex> lk(state->mu);
                state->status = status;
                state->reason = reason;
                state->done = true;
            }
            state->cv.notify_all();
        });

        if (!queued) {
            r.status = blackwell::cloud::Status::NetworkError;
            r.error_detail = "could not reach the engine thread";
            return r;
        }

        // Bounded wait, re-checked: shutdown() must be able to cut this loose even
        // if the engine thread is already gone and the task will never run.
        {
            std::unique_lock<std::mutex> lk(state->mu);
            while (!state->done && !shutting_down_.load(std::memory_order_acquire)) {
                state->cv.wait_for(lk, std::chrono::milliseconds(100));
            }
            if (!state->done) {
                state->abandoned = true;    // silence any later emit
                r.status = blackwell::cloud::Status::ShuttingDown;
                return r;
            }
            r.error_detail = state->detail;
            if (state->status != blackwell::EngineStatus::Success) {
                r.status = blackwell::cloud::Status::ApiError;
                if (r.error_detail.empty()) r.error_detail = "local generation failed";
                return r;
            }
            // A barge-in is a clean supersede, not an error: the user interrupted
            // their own answer. Reported as ShuttingDown so the dispatcher does
            // NOT retry -- re-running a turn the user talked over is the last
            // thing they want, and it would cost a second full decode.
            if (state->reason == blackwell::bridge::TerminationReason::BargeIn) {
                r.status = blackwell::cloud::Status::ShuttingDown;
                r.stop_reason = "barge_in";
                return r;
            }
            r.stop_reason =
                state->reason == blackwell::bridge::TerminationReason::TokenCap ? "max_tokens"
                                                                               : "end_turn";
        }
        r.status = blackwell::cloud::Status::Ok;
        r.http_status = 200;
        return r;
    }

    void shutdown() noexcept override {
        shutting_down_.store(true, std::memory_order_release);
        // Every parked send() re-checks the flag inside its 100 ms wait, so no
        // per-request handle has to be tracked here.
    }

private:
    struct State {
        std::mutex              mu;
        std::condition_variable cv;
        bool                    done = false;
        bool                    abandoned = false;
        blackwell::EngineStatus status = blackwell::EngineStatus::Success;
        blackwell::bridge::TerminationReason reason =
            blackwell::bridge::TerminationReason::None;
        std::string             detail;
        std::function<void(std::string_view)> on_text;
    };

    blackwell::bridge::EngineControlBridge* control_ = nullptr;   // borrowed
    GenerateFn                              generate_;
    std::atomic<bool>                       shutting_down_{false};
};

// -----------------------------------------------------------------------------
// RoutedTransport — picks local or remote per intent, so "Use the local model for
// responses" is a LIVE setting rather than a restart.
//
// IntentDispatcher binds ONE transport reference for its lifetime, so without
// this the toggle would have to rebuild the dispatcher (or restart the app) to
// change where an answer comes from -- a heavy price for what is conceptually an
// if. Instead the dispatcher binds this, and the choice is one relaxed atomic
// read taken ONCE PER INTENT, at the top of send().
//
// Per-intent, not per-chunk, is the point: an answer that started locally
// finishes locally even if the user flips the switch while it streams. A reply
// spliced from two models mid-sentence would be a strictly worse outcome than
// applying the change to the next turn.
// -----------------------------------------------------------------------------
class RoutedTransport final : public blackwell::cloud::IIntentTransport {
public:
    // Both are BORROWED and must outlive this object. `remote` may be the offline
    // stand-in or the billed client -- this class does not care which.
    RoutedTransport(blackwell::cloud::IIntentTransport* local,
                    blackwell::cloud::IIntentTransport* remote) noexcept
        : local_(local), remote_(remote) {}

    void set_use_local(bool on) noexcept { use_local_.store(on, std::memory_order_release); }
    [[nodiscard]] bool use_local() const noexcept {
        return use_local_.load(std::memory_order_acquire) && local_ != nullptr;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return use_local() ? local_->name() : remote_->name();
    }
    // Delegated rather than hardcoded false: the cost badge must follow whichever
    // transport would actually run the NEXT intent.
    [[nodiscard]] bool is_live() const noexcept override {
        return use_local() ? local_->is_live() : remote_->is_live();
    }

    [[nodiscard]] blackwell::cloud::Result send(
        const blackwell::cloud::TransportRequest& req,
        const blackwell::cloud::Callbacks& cb) noexcept override {
        // Resolved ONCE, here -- see the header block.
        return use_local() ? local_->send(req, cb) : remote_->send(req, cb);
    }

    [[nodiscard]] blackwell::cloud::Result prewarm(const std::string& body) noexcept override {
        // Only the remote leg has a connection or a prompt cache to warm; the
        // local one is already resident.
        return use_local() ? blackwell::cloud::Result{} : remote_->prewarm(body);
    }

    // BOTH, unconditionally. Teardown must not depend on which leg the toggle
    // happens to be pointing at, or a shutdown during a local answer would leave
    // the dispatcher parked on a transport nobody told to stop.
    void shutdown() noexcept override {
        if (local_ != nullptr) local_->shutdown();
        if (remote_ != nullptr) remote_->shutdown();
    }

private:
    blackwell::cloud::IIntentTransport* local_ = nullptr;    // borrowed
    blackwell::cloud::IIntentTransport* remote_ = nullptr;   // borrowed
    std::atomic<bool> use_local_{false};
};

}  // namespace rt
