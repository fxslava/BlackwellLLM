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
        // Latched, never cleared -- see abort(). Everything below compares
        // against this snapshot, so a Stop pressed between two turns lands ahead
        // of the next one instead of killing it.
        entry_epoch_ = cancel_epoch_.load(std::memory_order_acquire);

        if (!sleep_interruptibly(cfg_.latency_ms)) return stopped();

        std::string reply = "[Offline Assistant]: Received user intent: \"";
        reply.append(req.intent);
        reply.append("\"");

        // Stream it in pieces, exactly as an SSE text_delta sequence would.
        for (size_t i = 0; i < reply.size();) {
            if (shutting_down_.load(std::memory_order_acquire) || cancelled()) return stopped();
            const size_t n = chunk_len(reply, i, static_cast<size_t>(cfg_.chunk_chars));
            if (cb.on_text) cb.on_text(std::string_view(reply).substr(i, n));
            i += n;
            if (i < reply.size() && !sleep_interruptibly(cfg_.chunk_delay_ms)) return stopped();
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

    // The simulated leg is interruptible too, and that is not a courtesy: the
    // Stop button has to be exercisable on a laptop with no GPU, no key and no
    // network, or the only way to test it is to spend money.
    void abort() noexcept override { cancel_epoch_.fetch_add(1, std::memory_order_acq_rel); }

private:
    // Which of the two ended the exchange. Cancelled outranks ShuttingDown when
    // both are set: a user who pressed Stop and then closed the window asked for
    // the first thing, and the second is what always happens afterwards.
    [[nodiscard]] Result stopped() const noexcept {
        Result r;
        r.status = cancelled() ? Status::Cancelled : Status::ShuttingDown;
        return r;
    }

    [[nodiscard]] bool cancelled() const noexcept {
        return cancel_epoch_.load(std::memory_order_acquire) != entry_epoch_;
    }

    // How many bytes to take from `s` at `pos`, aiming for `want` but never
    // stopping INSIDE a UTF-8 sequence.
    //
    // WHY THIS IS NOT JUST cfg_.chunk_chars. The field is named chars and the
    // reply is bytes, and a Cyrillic or CJK character is two to four of them --
    // so a fixed byte stride cuts characters in half. That is not a cosmetic
    // difference for a class whose whole job is to imitate the real transport:
    // an SSE text_delta carries a complete JSON string, so a live reply NEVER
    // arrives half a character, and a fallback that emits something worse than
    // the thing it stands in for is a fallback that tests the wrong contract.
    // (It also shipped: the offline echo of a Russian utterance reached the UI
    // with a pair of U+FFFD wherever a 12-byte boundary landed mid-character.)
    //
    // The scan is open-coded rather than pulled from bridge/utf8_stream.hpp on
    // purpose -- this header's stated invariant is that it includes nothing but
    // intent_transport.hpp, which is what keeps BUILD_CLOUD_CLIENT=OFF a real
    // configuration rather than a build-only one.
    static size_t chunk_len(const std::string& s, size_t pos, size_t want) noexcept {
        const size_t remaining = s.size() - pos;
        if (want >= remaining) return remaining;
        // Walk FORWARD off any continuation byte (10xxxxxx): the next sequence's
        // lead byte is the nearest legal split point at or after `want`. Bounded
        // by the string end, so malformed input terminates instead of scanning on.
        size_t n = want;
        while (n < remaining && (static_cast<unsigned char>(s[pos + n]) & 0xC0) == 0x80) {
            ++n;
        }
        return n;
    }

    // Sliced so shutdown() -- and now Stop -- is not held up for a whole
    // simulated round trip. Waiting out a 1 s fake latency after the user asked
    // to stop would make the button feel broken on precisely the configuration
    // most people first try it on.
    [[nodiscard]] bool sleep_interruptibly(int ms) noexcept {
        constexpr int kSlice = 20;
        for (int e = 0; e < ms; e += kSlice) {
            if (shutting_down_.load(std::memory_order_acquire) || cancelled()) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(kSlice));
        }
        return !shutting_down_.load(std::memory_order_acquire) && !cancelled();
    }

    Config cfg_;
    std::atomic<bool> shutting_down_{false};
    // Bumped by abort() from any thread; `entry_epoch_` is touched only by
    // send(), which the dispatcher calls from one thread with one exchange in
    // flight (intent_dispatcher.hpp) -- so it needs no atomicity of its own.
    std::atomic<uint64_t> cancel_epoch_{0};
    uint64_t              entry_epoch_ = 0;
};

}  // namespace blackwell::cloud
