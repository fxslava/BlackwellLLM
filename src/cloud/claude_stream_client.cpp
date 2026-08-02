// =============================================================================
// cloud/claude_stream_client.cpp — the ONLY TU that sees libcurl and simdjson.
//
// Synchronous by design (see the header). curl_multi is still used, not
// curl_easy_perform, for one reason: easy_perform cannot express separate TTFT
// and inter-byte-stall budgets, and CURLOPT_TIMEOUT is a wall-clock cap on the
// whole transfer -- exactly wrong for a long generation. The multi poll loop
// below gives both budgets, and the connection cache still lives on the reused
// easy handle so keep-alive works across requests.
// =============================================================================
#include "claude_stream_client.hpp"

#include <curl/curl.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include "claude_sse.hpp"

namespace blackwell::cloud {
namespace {

using Clock = std::chrono::steady_clock;

long ms_since(Clock::time_point t) noexcept {
    return static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count());
}

}  // namespace

// NOTE: to_string() and is_retryable() moved to cloud_types.hpp as `inline`.
// Keeping them here would have made every Result-classifying call site link
// against this TU -- and therefore against libcurl -- which would have made
// BUILD_CLOUD_CLIENT=OFF unusable for the OfflineTransport path.

// ---------------------------------------------------------------------------
// Per-transfer state. Not shared across threads: send() is synchronous, so this
// lives on the caller's stack frame for the transfer's duration.
struct Transfer {
    const Callbacks* cb = nullptr;
    Result           result{};
    bool             got_first_byte = false;
    Clock::time_point started{};
    Clock::time_point last_byte{};
    char             err_buf[CURL_ERROR_SIZE]{};
    std::atomic<bool>* shutting_down = nullptr;

    // Sink for SseDecoder. Non-virtual: resolved at compile time.
    struct Sink {
        Transfer* t;
        void on_text_delta(std::string_view s) {
            if (t->cb && t->cb->on_text) t->cb->on_text(s);
        }
        void on_usage(uint32_t in, uint32_t out, uint32_t cc, uint32_t cr) {
            Usage& u = t->result.usage;
            if (in) u.input_tokens = in;
            if (out) u.output_tokens = out;
            if (cc) u.cache_creation_input_tokens = cc;
            if (cr) u.cache_read_input_tokens = cr;
            if (t->cb && t->cb->on_usage) t->cb->on_usage(u);
        }
        void on_stop(std::string_view r) {
            t->result.stop_reason.assign(r);
            if (r == "refusal") t->result.status = Status::Refusal;
        }
        void on_fallback(std::string_view from, std::string_view to) {
            if (t->cb && t->cb->on_fallback) t->cb->on_fallback(from, to);
        }
        void on_api_error(std::string_view type, std::string_view msg) {
            t->result.status = Status::ApiError;
            t->result.error_detail.assign(type).append(": ").append(msg);
        }
    };

    Sink sink{this};
    detail::SseDecoder<Sink> decoder{sink};
};

// ---------------------------------------------------------------------------

struct ClaudeStreamClient::Impl {
    Config      config;
    CURLM*      multi = nullptr;
    CURL*       easy = nullptr;    // reused: this is what keeps the connection pooled
    curl_slist* headers = nullptr; // built once; must outlive every transfer

    std::atomic<bool> shutting_down{false};

    explicit Impl(Config c) : config(std::move(c)) {
        multi = curl_multi_init();
        easy = curl_easy_init();
        if (multi == nullptr || easy == nullptr) {
            cleanup();
            throw std::runtime_error("claude: curl init failed");
        }

        headers = curl_slist_append(headers, "content-type: application/json");
        headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
        headers = curl_slist_append(headers, "accept: text/event-stream");
        // Kill curl's default "Expect: 100-continue" on large POSTs -- it costs a
        // full RTT before the body is even sent, and our bodies are large
        // (whole cached context prefixes).
        headers = curl_slist_append(headers, "Expect:");
        {
            const std::string k = "x-api-key: " + config.api_key;
            headers = curl_slist_append(headers, k.c_str());
        }
        if (!config.beta.empty()) {
            const std::string b = "anthropic-beta: " + config.beta;
            headers = curl_slist_append(headers, b.c_str());
        }
        if (headers == nullptr) {
            cleanup();
            throw std::runtime_error("claude: header list allocation failed");
        }
    }

    ~Impl() { cleanup(); }

    void cleanup() noexcept {
        if (headers) { curl_slist_free_all(headers); headers = nullptr; }
        if (easy)    { curl_easy_cleanup(easy);      easy = nullptr; }
        if (multi)   { curl_multi_cleanup(multi);    multi = nullptr; }
    }

    // The ONLY place response body bytes are touched.
    static size_t on_write(char* ptr, size_t size, size_t nmemb, void* ud) {
        auto* t = static_cast<Transfer*>(ud);
        const size_t n = size * nmemb;

        t->last_byte = Clock::now();
        t->got_first_byte = true;

        // Teardown abort. NOT a barge-in: the only thing that reaches this now
        // is shutdown(), because in-flight requests are never superseded.
        if (t->shutting_down->load(std::memory_order_acquire)) {
            t->result.status = Status::ShuttingDown;
            return 0;  // != n -> CURLE_WRITE_ERROR, aborts the transfer
        }
        if (!t->decoder.feed(ptr, n)) {
            t->result.status = Status::MalformedStream;
            return 0;
        }
        return n;
    }

    void configure(const std::string& body, Transfer& t) noexcept {
        curl_easy_reset(easy);
        curl_easy_setopt(easy, CURLOPT_URL, config.url.c_str());
        curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(easy, CURLOPT_POST, 1L);
        // POSTFIELDS is NOT copied. `body` is the caller's immutable, committed
        // intent payload and outlives this call by contract (see the header).
        curl_easy_setopt(easy, CURLOPT_POSTFIELDS, body.data());
        curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE_LARGE,
                         static_cast<curl_off_t>(body.size()));
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &Impl::on_write);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA, &t);
        curl_easy_setopt(easy, CURLOPT_ERRORBUFFER, t.err_buf);

        // Mandatory in a multithreaded process: without it libcurl uses SIGALRM
        // for DNS timeouts and corrupts other threads' signal handling.
        curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, config.connect_timeout_ms);
        // Deliberately NO CURLOPT_TIMEOUT: it is a wall-clock cap on the whole
        // transfer and would kill long generations. TTFT and stall are enforced
        // separately in the poll loop below.

        curl_easy_setopt(easy, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(easy, CURLOPT_TCP_KEEPIDLE, 30L);
        curl_easy_setopt(easy, CURLOPT_TCP_KEEPINTVL, 15L);
        curl_easy_setopt(easy, CURLOPT_TCP_NODELAY, 1L);

        if (config.enable_http2) {
            curl_easy_setopt(easy, CURLOPT_HTTP_VERSION,
                             static_cast<long>(CURL_HTTP_VERSION_2TLS));
        }
        // No ACCEPT_ENCODING: transparent decompression on an SSE stream can
        // introduce buffering latency, and the deltas are tiny anyway.

        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYHOST, 2L);
        if (config.verbose) curl_easy_setopt(easy, CURLOPT_VERBOSE, 1L);
    }

    Result run(const std::string& body, const Callbacks* cb) noexcept {
        if (shutting_down.load(std::memory_order_acquire)) {
            Result r;
            r.status = Status::ShuttingDown;
            return r;
        }

        Transfer t;
        t.cb = cb;
        t.shutting_down = &shutting_down;
        t.started = t.last_byte = Clock::now();

        configure(body, t);
        curl_multi_add_handle(multi, easy);

        Status timeout_status = Status::Ok;
        int running = 1;
        while (running > 0) {
            curl_multi_perform(multi, &running);
            if (running == 0) break;

            // Two separate budgets. A `ping` event counts as a byte, so a long
            // server-side prefill reads as alive, not as a stall -- which is why
            // TTFT must be its own (longer) budget rather than reusing the
            // stall one.
            if (!t.got_first_byte && ms_since(t.started) > config.ttft_timeout_ms) {
                timeout_status = Status::TtftTimeout;
                break;
            }
            if (t.got_first_byte && ms_since(t.last_byte) > config.stall_timeout_ms) {
                timeout_status = Status::StallTimeout;
                break;
            }
            if (shutting_down.load(std::memory_order_acquire)) {
                timeout_status = Status::ShuttingDown;
                break;
            }

            // Cap the sleep so the watchdogs stay responsive. curl_multi_poll
            // (not _wait) sleeps properly when there are no file descriptors
            // instead of busy-spinning.
            int numfds = 0;
            curl_multi_poll(multi, nullptr, 0, 100, &numfds);
        }

        // Drain the completion message even if a watchdog fired -- otherwise the
        // handle stays in the multi's message queue and poisons the next call.
        CURLcode rc = CURLE_OK;
        bool have_rc = false;
        int msgs = 0;
        while (CURLMsg* m = curl_multi_info_read(multi, &msgs)) {
            if (m->msg == CURLMSG_DONE) {
                rc = m->data.result;
                have_rc = true;
            }
        }

        long http = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http);
        // Honour the server's own backoff hint over any local heuristic.
        curl_off_t retry_after = 0;
        if (curl_easy_getinfo(easy, CURLINFO_RETRY_AFTER, &retry_after) == CURLE_OK) {
            t.result.retry_after_s = static_cast<long>(retry_after);
        }
        curl_multi_remove_handle(multi, easy);

        Result r = t.result;
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
};

// ---------------------------------------------------------------------------

ClaudeStreamClient::ClaudeStreamClient(Config cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg))) {}

ClaudeStreamClient::~ClaudeStreamClient() = default;

Result ClaudeStreamClient::send(const std::string& body, const Callbacks& cb) noexcept {
    return impl_->run(body, &cb);
}

Result ClaudeStreamClient::prewarm(const std::string& prewarm_body) noexcept {
    // No callbacks: a max_tokens:0 response has an empty content array. The
    // point is the side effects -- a warm TLS connection and a written cache.
    return impl_->run(prewarm_body, nullptr);
}

void ClaudeStreamClient::shutdown() noexcept {
    impl_->shutting_down.store(true, std::memory_order_release);
}

}  // namespace blackwell::cloud
