#pragma once
// =============================================================================
// cloud/intent_dispatcher.hpp — the seam between the local arbiter and the
// Cloud API. Owns the ONE thread allowed to make a paid request.
//
//   engine thread                     dispatcher thread
//   ─────────────                     ─────────────────
//   run_decode_loop()
//     -> TerminationReason
//   publish_intent(reason, ...)
//        │
//        ▼
//   IntentCommitQueue::offer()  ══>  wait_pop()   (blocks at 0% CPU)
//     [ Eos ? commit : drop ]          │
//                                      ▼
//                                 build_intent_request()
//                                      │
//                                      ▼
//                                 ClaudeStreamClient::send()  <- money spent HERE
//
// WHY A DEDICATED THREAD
//   send() blocks for the whole transfer -- seconds. That must never sit on the
//   engine thread (single-thread engine doctrine: it owns the KV cache and the
//   CUDA context) nor on the audio/VAD thread (it has a 10 ms deadline). The
//   dispatcher thread exists solely to have somewhere safe to block.
//
// SERIALISATION IS THE BACKPRESSURE
//   Exactly one request is in flight at a time, by construction: the loop does
//   not pop the next intent until the previous send() returns. If the user
//   out-talks the cloud, intents pile up in the queue and eventually the queue
//   refuses them (counted as dropped_queue_full). That is the correct failure
//   mode -- refusing to START a request is free, whereas every request already
//   started has been paid for.
//
// RETRY
//   Retrying is safe here in a way it is NOT in a streaming/barge-in design:
//   the payload was frozen by the commit gate, so a retry is byte-identical and
//   hits the warm prompt cache (~0.1x input cost). Only transport-level and
//   overloaded failures are retried -- never a refusal, never a 4xx.
// =============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "cloud_types.hpp"
#include "intent_commit.hpp"
#include "intent_request.hpp"
#include "intent_transport.hpp"

namespace blackwell::cloud {

class IntentDispatcher {
public:
    struct Config {
        // Retry budget for ONE intent. Deliberately small: this pathway is
        // interactive, and a user who has stopped talking will not wait through
        // a long backoff ladder. Exhausting it degrades to local-only output.
        int  max_attempts = 3;
        long base_backoff_ms = 400;   // doubled per attempt, jittered
        long max_backoff_ms = 4000;
        // Re-issue the pre-warm when the queue has been idle this long, so the
        // TLS connection and the 1h prompt cache are hot on the next hotkey.
        // 0 disables. Keep it under the cache TTL you set in intent_request.hpp.
        long idle_prewarm_ms = 0;
    };

    // Supplies the stable (cacheable) half of the prompt. Called on the
    // dispatcher thread, once per attempt.
    //
    // CONTRACT: this MUST be byte-stable across calls within a session. A
    // timestamp or counter in here silently destroys the prompt cache and
    // raises no error -- see intent_request.hpp.
    using ContextProvider = std::function<RequestContext()>;

    // Fires on the dispatcher thread once per intent, after all attempts.
    using CompletionFn = std::function<void(const bridge::IntentRecord&, const Result&)>;

    // `transport` is BORROWED (not owned) and must outlive the dispatcher.
    // Swapping offline for live is a matter of which one you pass here --
    // nothing else changes, which is the point of the seam.
    //
    // Borrowed rather than a unique_ptr on purpose: every call site already owns
    // its transport as a stack object or a member that outlives the dispatcher,
    // and taking ownership would force a heap allocation to express a lifetime
    // relationship that is already static. It also matches how the rest of this
    // tree passes collaborators (IEngineControl, BlackwellEngine, the projector
    // -- all non-owning).
    IntentDispatcher(bridge::IntentCommitQueue& queue, IIntentTransport& transport,
                     ContextProvider ctx, Config cfg = {})
        : queue_(queue), transport_(transport), ctx_(std::move(ctx)), cfg_(cfg) {}

    ~IntentDispatcher() { stop(); }

    IntentDispatcher(const IntentDispatcher&) = delete;
    IntentDispatcher& operator=(const IntentDispatcher&) = delete;

    void set_on_text(std::function<void(std::string_view)> fn) { on_text_ = std::move(fn); }
    void set_on_usage(std::function<void(const Usage&)> fn) { on_usage_ = std::move(fn); }
    void set_on_complete(CompletionFn fn) { on_complete_ = std::move(fn); }

    // Fires the moment an intent is popped, BEFORE the first byte is sent, so
    // the UI can show "remote is working" during the round trip rather than
    // appearing frozen until the first delta lands.
    void set_on_dispatch_start(std::function<void(const bridge::IntentRecord&)> fn) {
        on_dispatch_start_ = std::move(fn);
    }

    void start() {
        if (running_.exchange(true, std::memory_order_acq_rel)) return;
        thread_ = std::thread([this] { run(); });
    }

    // Idempotent. Wakes the blocked wait_pop() and aborts any in-flight
    // transfer, then joins. Safe from any thread.
    void stop() noexcept {
        if (!running_.exchange(false, std::memory_order_acq_rel)) return;
        queue_.stop();
        transport_.shutdown();
        if (thread_.joinable()) thread_.join();
    }

    [[nodiscard]] const char* transport_name() const noexcept { return transport_.name(); }
    [[nodiscard]] bool transport_is_live() const noexcept { return transport_.is_live(); }

    [[nodiscard]] uint64_t dispatched() const noexcept {
        return dispatched_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t failed() const noexcept {
        return failed_.load(std::memory_order_relaxed);
    }

private:
    void run() {
        bridge::IntentRecord rec;
        // Prime the connection + cache before the first real intent, so the
        // first hotkey press does not pay a cold TLS handshake AND a cold
        // prefill on the critical path.
        prewarm();

        while (queue_.wait_pop(rec)) {
            // Everything past this point is a request that the local arbiter
            // certified as EOS-terminated. There is no further admission check
            // and no way to un-send it -- the gate was the last word.
            if (on_dispatch_start_) on_dispatch_start_(rec);
            const Result r = send_with_retry(rec);
            if (r.status == Status::Ok) {
                dispatched_.fetch_add(1, std::memory_order_relaxed);
            } else {
                failed_.fetch_add(1, std::memory_order_relaxed);
            }
            if (on_complete_) on_complete_(rec, r);
            rec.payload.clear();
        }
    }

    void prewarm() noexcept {
        if (!ctx_) return;
        RequestContext c = ctx_();
        c.intent = "ping";  // placeholder; it sits AFTER the last breakpoint
        // Rendered BY THE TRANSPORT: what a warm-up looks like is protocol
        // knowledge (Anthropic's is a free max_tokens:0 request; an
        // OpenAI-compatible endpoint has no such shape and declines to send).
        (void)transport_.prewarm(transport_.build_prewarm_body(c));
    }

    Result send_with_retry(const bridge::IntentRecord& rec) {
        Callbacks cb;
        cb.on_text = on_text_;
        cb.on_usage = on_usage_;

        Result last{};
        for (int attempt = 0; attempt < cfg_.max_attempts; ++attempt) {
            if (!running_.load(std::memory_order_acquire)) {
                last.status = Status::ShuttingDown;
                return last;
            }

            RequestContext c = ctx_ ? ctx_() : RequestContext{};
            c.intent = rec.payload;
            // The TRANSPORT renders the body, so the same committed context
            // becomes an Anthropic Messages payload or an OpenAI-compatible
            // one without this loop knowing which (intent_transport.hpp).
            const std::string body = transport_.build_body(c);
            // `body` is a local: send() is synchronous, so it outlives the
            // transfer. That is the whole reason CURLOPT_POSTFIELDS is safe
            // in the live transport without COPYPOSTFIELDS.
            const TransportRequest req{body, rec.payload, rec.sequence};
            last = transport_.send(req, cb);

            if (!is_retryable(last.status, last.http_status)) return last;
            if (attempt + 1 >= cfg_.max_attempts) return last;

            // Honour the server's retry-after over our own ladder when present.
            long delay = cfg_.base_backoff_ms << attempt;
            if (delay > cfg_.max_backoff_ms) delay = cfg_.max_backoff_ms;
            if (last.retry_after_s > 0) delay = last.retry_after_s * 1000;
            sleep_interruptibly(delay);
        }
        return last;
    }

    // Sleep in slices so stop() is not held up by a full backoff interval.
    void sleep_interruptibly(long ms) noexcept {
        constexpr long kSlice = 50;
        for (long e = 0; e < ms && running_.load(std::memory_order_acquire); e += kSlice) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kSlice));
        }
    }

    bridge::IntentCommitQueue& queue_;
    IIntentTransport&          transport_;
    ContextProvider            ctx_;
    Config                     cfg_;

    std::function<void(std::string_view)>            on_text_;
    std::function<void(const Usage&)>                on_usage_;
    std::function<void(const bridge::IntentRecord&)> on_dispatch_start_;
    CompletionFn                                     on_complete_;

    std::thread       thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> dispatched_{0};
    std::atomic<uint64_t> failed_{0};
};

}  // namespace blackwell::cloud
