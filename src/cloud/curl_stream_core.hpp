#pragma once
// =============================================================================
// cloud/curl_stream_core.hpp — the HTTP half of every streaming client:
// connection reuse, the two watchdogs, and Result classification.
//
// INTERNAL header. Only the client .cpp files include it, so libcurl stays
// confined to exactly the TUs that need it (same discipline as claude_sse.hpp
// and simdjson).
//
// WHY IT IS SHARED RATHER THAN COPIED
//   Anthropic and OpenAI-compatible endpoints differ in three places: the URL,
//   the auth header, and what a `data:` payload means. They do NOT differ in the
//   part that is hard -- separate TTFT and inter-byte-stall budgets, draining
//   the completion message even when a watchdog fired, and the precedence rule
//   that lets an error delivered inside an HTTP 200 beat the transport verdict.
//   A second copy of that would be a second place for those rules to drift, and
//   only one copy would keep getting fixed.
//
// Synchronous by design. curl_multi is still used, not curl_easy_perform, for
// one reason: easy_perform cannot express separate TTFT and inter-byte-stall
// budgets, and CURLOPT_TIMEOUT is a wall-clock cap on the whole transfer --
// exactly wrong for a long generation. The multi poll loop below gives both
// budgets, and the connection cache still lives on the reused easy handle so
// keep-alive works across requests.
//
// Protocol contract (compile-time; the per-byte path stays devirtualised):
//     Protocol::Sink      constructible from SinkTarget*
//     Protocol::Decoder   constructible from Sink&, with
//                             bool feed(const char*, size_t)
//   The sink writes into the SinkTarget it was handed. That indirection is what
//   breaks the otherwise circular Transfer <-> Sink dependency, and it is also
//   what keeps the protocol headers free of any curl types.
// =============================================================================
#include <curl/curl.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cloud_types.hpp"

namespace blackwell::cloud::detail {

using Clock = std::chrono::steady_clock;

inline long ms_since(Clock::time_point t) noexcept {
    return static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count());
}

// curl_global_init() is not thread-safe and libcurl's lazy fallback init is a
// documented race. Every client used to inherit that as a "call it from main()"
// contract in its header -- which nothing in this tree actually honoured, and
// which got harder to honour the moment there were two clients that might each
// be the first one constructed.
//
// A function-local static is the whole fix: initialisation is guaranteed to run
// exactly once and to be thread-safe, and it is anchored to the only event that
// matters (a handle is about to be created). No curl_global_cleanup() to match
// it: it must not run while any handle is alive, and there is no point in the
// process where we know that AND still care.
inline void ensure_curl_global_init() noexcept {
    static const CURLcode once = curl_global_init(CURL_GLOBAL_DEFAULT);
    (void)once;
}

// The curl-level knobs. Protocol-agnostic on purpose: a client's own Config
// struct owns the vendor fields (api key, model, beta headers) and lowers them
// into this plus a header list.
struct StreamOptions {
    std::string url;
    long connect_timeout_ms = 2000;
    long ttft_timeout_ms = 15000;   // request sent -> first body byte
    long stall_timeout_ms = 10000;  // max gap between body bytes (pings count)
    bool enable_http2 = true;       // requires an nghttp2-enabled libcurl
    bool verbose = false;           // CURLOPT_VERBOSE; NEVER enable in Release
};

// Everything a protocol sink is allowed to touch. Deliberately NOT the Transfer
// itself: a sink that could see the curl state would be a sink that could be
// written to depend on it.
struct SinkTarget {
    const Callbacks* cb = nullptr;
    Result           result{};
};

// Per-transfer state. Not shared across threads: run() is synchronous, so this
// lives on the caller's stack frame for the transfer's duration.
//
// Declaration order IS construction order here (CLAUDE.md extension pattern #3):
// the sink points into `target`, and the decoder holds a reference to the sink.
template <class Protocol>
struct Transfer {
    SinkTarget                 target{};
    typename Protocol::Sink    sink{&target};
    typename Protocol::Decoder decoder{sink};

    bool              got_first_byte = false;
    Clock::time_point started{};
    Clock::time_point last_byte{};
    char              err_buf[CURL_ERROR_SIZE]{};
    std::atomic<bool>* shutting_down = nullptr;
};

template <class Protocol>
class CurlStreamCore {
public:
    // `headers` are whole "Name: value" lines, copied into a curl_slist that is
    // built ONCE and outlives every transfer. Throws std::runtime_error on init
    // failure (INIT tier, CLAUDE.md #4).
    CurlStreamCore(StreamOptions opts, const std::vector<std::string>& headers,
                   const char* who)
        : opts_(std::move(opts)) {
        ensure_curl_global_init();
        multi_ = curl_multi_init();
        easy_ = curl_easy_init();
        if (multi_ == nullptr || easy_ == nullptr) {
            cleanup();
            throw std::runtime_error(std::string(who) + ": curl init failed");
        }
        for (const std::string& h : headers) {
            headers_ = curl_slist_append(headers_, h.c_str());
            if (headers_ == nullptr) {
                cleanup();
                throw std::runtime_error(std::string(who) + ": header list allocation failed");
            }
        }
    }

    ~CurlStreamCore() { cleanup(); }

    CurlStreamCore(const CurlStreamCore&) = delete;
    CurlStreamCore& operator=(const CurlStreamCore&) = delete;

    // Transmit `body` and stream the response. Blocks until the transfer ends.
    // `body` must outlive the call (CURLOPT_POSTFIELDS does not copy).
    // `cb` may be null for a warm-up request whose response nobody wants.
    [[nodiscard]] Result run(const std::string& body, const Callbacks* cb) noexcept {
        if (shutting_down_.load(std::memory_order_acquire)) {
            Result r;
            r.status = Status::ShuttingDown;
            return r;
        }

        Transfer<Protocol> t;
        t.target.cb = cb;
        t.shutting_down = &shutting_down_;
        t.started = t.last_byte = Clock::now();

        configure(body, t);
        curl_multi_add_handle(multi_, easy_);

        Status timeout_status = Status::Ok;
        int running = 1;
        while (running > 0) {
            curl_multi_perform(multi_, &running);
            if (running == 0) break;

            // Two separate budgets. A keep-alive comment counts as a byte, so a
            // long server-side prefill reads as alive, not as a stall -- which
            // is why TTFT must be its own (longer) budget rather than reusing
            // the stall one.
            if (!t.got_first_byte && ms_since(t.started) > opts_.ttft_timeout_ms) {
                timeout_status = Status::TtftTimeout;
                break;
            }
            if (t.got_first_byte && ms_since(t.last_byte) > opts_.stall_timeout_ms) {
                timeout_status = Status::StallTimeout;
                break;
            }
            if (shutting_down_.load(std::memory_order_acquire)) {
                timeout_status = Status::ShuttingDown;
                break;
            }

            // Cap the sleep so the watchdogs stay responsive. curl_multi_poll
            // (not _wait) sleeps properly when there are no file descriptors
            // instead of busy-spinning.
            int numfds = 0;
            curl_multi_poll(multi_, nullptr, 0, 100, &numfds);
        }

        // Drain the completion message even if a watchdog fired -- otherwise the
        // handle stays in the multi's message queue and poisons the next call.
        CURLcode rc = CURLE_OK;
        bool have_rc = false;
        int msgs = 0;
        while (CURLMsg* m = curl_multi_info_read(multi_, &msgs)) {
            if (m->msg == CURLMSG_DONE) {
                rc = m->data.result;
                have_rc = true;
            }
        }

        long http = 0;
        curl_easy_getinfo(easy_, CURLINFO_RESPONSE_CODE, &http);
        // Honour the server's own backoff hint over any local heuristic.
        curl_off_t retry_after = 0;
        if (curl_easy_getinfo(easy_, CURLINFO_RETRY_AFTER, &retry_after) == CURLE_OK) {
            t.target.result.retry_after_s = static_cast<long>(retry_after);
        }
        curl_multi_remove_handle(multi_, easy_);

        Result r = t.target.result;
        r.http_status = http;

        // Precedence: a status parsed OUT OF THE STREAM (Refusal / ApiError /
        // MalformedStream) is strictly more specific than the transport verdict,
        // so it wins. This is what stops an `overloaded_error` delivered inside
        // an HTTP 200 from being reported as success.
        if (r.status == Status::Ok) {
            if (timeout_status != Status::Ok) {
                r.status = timeout_status;
                if (r.error_detail.empty()) r.error_detail = "watchdog fired";
            } else if (!have_rc) {
                r.status = Status::NetworkError;
                r.error_detail = "transfer ended without a completion message";
            } else if (rc == CURLE_OK) {
                r.status = (http >= 200 && http < 300) ? Status::Ok : Status::HttpError;
            } else if (rc == CURLE_OPERATION_TIMEDOUT || rc == CURLE_COULDNT_CONNECT ||
                       rc == CURLE_COULDNT_RESOLVE_HOST) {
                r.status = Status::ConnectTimeout;
                r.error_detail = t.err_buf;
            } else {
                r.status = Status::NetworkError;
                r.error_detail = t.err_buf;
            }
        }
        return r;
    }

    // Abort an in-flight run() so the process can exit. Any thread; idempotent.
    void shutdown() noexcept { shutting_down_.store(true, std::memory_order_release); }

private:
    // The ONLY place response body bytes are touched.
    static size_t on_write(char* ptr, size_t size, size_t nmemb, void* ud) {
        auto* t = static_cast<Transfer<Protocol>*>(ud);
        const size_t n = size * nmemb;

        t->last_byte = Clock::now();
        t->got_first_byte = true;

        // Teardown abort. NOT a barge-in: the only thing that reaches this is
        // shutdown(), because in-flight requests are never superseded.
        if (t->shutting_down->load(std::memory_order_acquire)) {
            t->target.result.status = Status::ShuttingDown;
            return 0;  // != n -> CURLE_WRITE_ERROR, aborts the transfer
        }
        if (!t->decoder.feed(ptr, n)) {
            t->target.result.status = Status::MalformedStream;
            return 0;
        }
        return n;
    }

    void configure(const std::string& body, Transfer<Protocol>& t) noexcept {
        curl_easy_reset(easy_);
        curl_easy_setopt(easy_, CURLOPT_URL, opts_.url.c_str());
        curl_easy_setopt(easy_, CURLOPT_HTTPHEADER, headers_);
        curl_easy_setopt(easy_, CURLOPT_POST, 1L);
        // POSTFIELDS is NOT copied. `body` is the caller's immutable, committed
        // intent payload and outlives this call by contract (see the header).
        curl_easy_setopt(easy_, CURLOPT_POSTFIELDS, body.data());
        curl_easy_setopt(easy_, CURLOPT_POSTFIELDSIZE_LARGE,
                         static_cast<curl_off_t>(body.size()));
        curl_easy_setopt(easy_, CURLOPT_WRITEFUNCTION, &CurlStreamCore::on_write);
        curl_easy_setopt(easy_, CURLOPT_WRITEDATA, &t);
        curl_easy_setopt(easy_, CURLOPT_ERRORBUFFER, t.err_buf);

        // Mandatory in a multithreaded process: without it libcurl uses SIGALRM
        // for DNS timeouts and corrupts other threads' signal handling.
        curl_easy_setopt(easy_, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(easy_, CURLOPT_CONNECTTIMEOUT_MS, opts_.connect_timeout_ms);
        // Deliberately NO CURLOPT_TIMEOUT: it is a wall-clock cap on the whole
        // transfer and would kill long generations. TTFT and stall are enforced
        // separately in the poll loop above.

        curl_easy_setopt(easy_, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(easy_, CURLOPT_TCP_KEEPIDLE, 30L);
        curl_easy_setopt(easy_, CURLOPT_TCP_KEEPINTVL, 15L);
        curl_easy_setopt(easy_, CURLOPT_TCP_NODELAY, 1L);

        // Deliberately NO CURLOPT_FOLLOWLOCATION. Every request here carries a
        // bearer credential in a header, and libcurl replays headers across a
        // redirect: a 30x to another host would hand the API key to whoever
        // sent it. A misconfigured base URL must fail loudly instead.

        if (opts_.enable_http2) {
            curl_easy_setopt(easy_, CURLOPT_HTTP_VERSION,
                             static_cast<long>(CURL_HTTP_VERSION_2TLS));
        }
        // No ACCEPT_ENCODING: transparent decompression on an SSE stream can
        // introduce buffering latency, and the deltas are tiny anyway.

        curl_easy_setopt(easy_, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(easy_, CURLOPT_SSL_VERIFYHOST, 2L);
        if (opts_.verbose) curl_easy_setopt(easy_, CURLOPT_VERBOSE, 1L);
    }

    void cleanup() noexcept {
        if (headers_) { curl_slist_free_all(headers_); headers_ = nullptr; }
        if (easy_)    { curl_easy_cleanup(easy_);      easy_ = nullptr; }
        if (multi_)   { curl_multi_cleanup(multi_);    multi_ = nullptr; }
    }

    StreamOptions opts_;
    CURLM*        multi_ = nullptr;
    CURL*         easy_ = nullptr;    // reused: this is what keeps the connection pooled
    curl_slist*   headers_ = nullptr; // built once; must outlive every transfer

    std::atomic<bool> shutting_down_{false};
};

}  // namespace blackwell::cloud::detail
