#pragma once
// =============================================================================
// cloud/offline_transport.hpp — OfflineTransport: the simulated remote.
//
// A first-class production fallback, not a test double. It is what
// voice_assistant runs when no ANTHROPIC_API_KEY is present, what a developer
// without network access develops against, and what the app degrades to rather
// than failing outright. (Test doubles belong in tests/ and may be named
// accordingly; nothing in src/ is a mock.)
//
// It deliberately imitates the SHAPE of a real exchange, not just its result:
// the reply is streamed in small pieces with a delay between them, so the GUI's
// incremental-render path and its "remote is still talking" state are exercised
// offline. A fallback that returned one finished string would leave both
// untested until the first live request.
//
// No network, no libcurl, no API key -- includes only intent_transport.hpp and
// therefore cloud_types.hpp, which is what keeps BUILD_CLOUD_CLIENT=OFF a real,
// working configuration rather than a build-only one.
// =============================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <thread>

#include "intent_transport.hpp"

namespace blackwell::cloud {

class OfflineTransport final : public IIntentTransport {
public:
    struct Config {
        int  latency_ms = 1000;    // simulated round trip before the first token
        int  chunk_delay_ms = 25;  // between streamed pieces
        int  chunk_chars = 12;     // piece size
        bool simulate_usage = true;
    };

    explicit OfflineTransport(Config cfg = {}) : cfg_(cfg) {}

    [[nodiscard]] const char* name() const noexcept override { return "Offline (simulated)"; }
    [[nodiscard]] bool is_live() const noexcept override { return false; }

    [[nodiscard]] Result send(const TransportRequest& req, const Callbacks& cb) noexcept override {
        Result r;
        shutting_down_.store(false, std::memory_order_release);

        if (!sleep_interruptibly(cfg_.latency_ms)) {
            r.status = Status::ShuttingDown;
            return r;
        }

        std::string reply = "[Offline Assistant]: Received user intent: \"";
        reply.append(req.intent);
        reply.append("\"");

        // Stream it in pieces, exactly as an SSE text_delta sequence would.
        for (size_t i = 0; i < reply.size();) {
            if (shutting_down_.load(std::memory_order_acquire)) {
                r.status = Status::ShuttingDown;
                return r;
            }
            const size_t n = std::min(static_cast<size_t>(cfg_.chunk_chars), reply.size() - i);
            if (cb.on_text) cb.on_text(std::string_view(reply).substr(i, n));
            i += n;
            if (i < reply.size() && !sleep_interruptibly(cfg_.chunk_delay_ms)) {
                r.status = Status::ShuttingDown;
                return r;
            }
        }

        if (cfg_.simulate_usage) {
            // Plausible-looking numbers so the telemetry bar has something to
            // render offline. cache_read is non-zero on purpose: a UI that only
            // ever saw zeros here would never exercise its "cache is working"
            // rendering, which is the state operators most need to recognise.
            r.usage.input_tokens = static_cast<uint32_t>(req.body.size() / 4);
            r.usage.output_tokens = static_cast<uint32_t>(reply.size() / 4);
            r.usage.cache_read_input_tokens =
                r.usage.input_tokens > 32 ? r.usage.input_tokens - 32 : 0;
            if (cb.on_usage) cb.on_usage(r.usage);
        }

        r.status = Status::Ok;
        r.stop_reason = "end_turn";
        r.http_status = 200;
        return r;
    }

    void shutdown() noexcept override { shutting_down_.store(true, std::memory_order_release); }

private:
    // Sliced so shutdown() is not held up for a whole simulated round trip.
    [[nodiscard]] bool sleep_interruptibly(int ms) noexcept {
        constexpr int kSlice = 20;
        for (int e = 0; e < ms; e += kSlice) {
            if (shutting_down_.load(std::memory_order_acquire)) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(kSlice));
        }
        return !shutting_down_.load(std::memory_order_acquire);
    }

    Config cfg_;
    std::atomic<bool> shutting_down_{false};
};

}  // namespace blackwell::cloud
