// =============================================================================
// cloud/claude_stream_client.cpp — the Anthropic protocol bound to the shared
// curl core.
//
// Everything HTTP (the multi poll loop, the two watchdogs, Result
// classification) moved to curl_stream_core.hpp when the OpenAI-compatible leg
// arrived; everything Anthropic-specific (headers, event semantics) stayed
// here. This TU and openai_stream_client.cpp are now the only two that see
// libcurl and simdjson.
// =============================================================================
#include "claude_stream_client.hpp"

#include <string>
#include <utility>
#include <vector>

#include "claude_sse.hpp"
#include "curl_stream_core.hpp"

namespace blackwell::cloud {
namespace {

// NOTE: to_string() and is_retryable() live in cloud_types.hpp as `inline`.
// Keeping them here would have made every Result-classifying call site link
// against this TU -- and therefore against libcurl -- which would have made
// BUILD_CLOUD_CLIENT=OFF unusable for the OfflineTransport path.

// Anthropic event -> shared Result/Callbacks. Non-virtual: resolved at compile
// time by the core's Protocol contract.
struct ClaudeSink {
    detail::SinkTarget* t;

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

struct ClaudeProtocol {
    using Sink = ClaudeSink;
    using Decoder = detail::SseDecoder<ClaudeSink>;
};

}  // namespace

struct ClaudeStreamClient::Impl {
    detail::CurlStreamCore<ClaudeProtocol> core;

    explicit Impl(Config c) : core(lower(c), build_headers(c), "claude") {}

    // The vendor Config -> the protocol-agnostic curl knobs.
    static detail::StreamOptions lower(const Config& c) {
        detail::StreamOptions o;
        o.url = c.url;
        o.connect_timeout_ms = c.connect_timeout_ms;
        o.ttft_timeout_ms = c.ttft_timeout_ms;
        o.stall_timeout_ms = c.stall_timeout_ms;
        o.enable_http2 = c.enable_http2;
        o.verbose = c.verbose;
        return o;
    }

    static std::vector<std::string> build_headers(const Config& c) {
        std::vector<std::string> h{
            "content-type: application/json",
            "anthropic-version: 2023-06-01",
            "accept: text/event-stream",
            // Kill curl's default "Expect: 100-continue" on large POSTs -- it
            // costs a full RTT before the body is even sent, and our bodies are
            // large (whole cached context prefixes).
            "Expect:",
            "x-api-key: " + c.api_key,
        };
        if (!c.beta.empty()) h.push_back("anthropic-beta: " + c.beta);
        return h;
    }
};

// ---------------------------------------------------------------------------

ClaudeStreamClient::ClaudeStreamClient(Config cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg))) {}

ClaudeStreamClient::~ClaudeStreamClient() = default;

Result ClaudeStreamClient::send(const std::string& body, const Callbacks& cb) noexcept {
    return impl_->core.run(body, &cb);
}

Result ClaudeStreamClient::prewarm(const std::string& prewarm_body) noexcept {
    // No callbacks: a max_tokens:0 response has an empty content array. The
    // point is the side effects -- a warm TLS connection and a written cache.
    return impl_->core.run(prewarm_body, nullptr);
}

void ClaudeStreamClient::shutdown() noexcept { impl_->core.shutdown(); }

}  // namespace blackwell::cloud
