// =============================================================================
// cloud/openai_stream_client.cpp — the OpenAI-compatible protocol bound to the
// shared curl core. One of exactly two TUs that see libcurl and simdjson.
//
// Everything HTTP lives in curl_stream_core.hpp; everything about what a chunk
// MEANS lives in openai_sse.hpp. What is left here is the glue: the auth
// header, the derived endpoint, and the mapping from this vendor's vocabulary
// onto the shared Result.
// =============================================================================
#include "openai_stream_client.hpp"

#include <string>
#include <utility>
#include <vector>

#include "curl_stream_core.hpp"
#include "openai_request.hpp"  // join_url
#include "openai_sse.hpp"

namespace blackwell::cloud {
namespace {

// OpenAI event -> shared Result/Callbacks. Non-virtual: resolved at compile
// time by the core's Protocol contract.
struct OpenAiSink {
    detail::SinkTarget* t;

    void on_text_delta(std::string_view s) {
        if (t->cb && t->cb->on_text) t->cb->on_text(s);
    }

    void on_usage(uint32_t prompt, uint32_t completion, uint32_t cached) {
        Usage& u = t->result.usage;
        if (prompt) u.input_tokens = prompt;
        if (completion) u.output_tokens = completion;
        // `cached` is a SUBSET of prompt_tokens on this endpoint, not an extra
        // charge -- so it maps onto cache_read, never onto cache_creation.
        // There is no observable equivalent of a cache WRITE here.
        if (cached) u.cache_read_input_tokens = cached;
        if (t->cb && t->cb->on_usage) t->cb->on_usage(u);
    }

    // finish_reason -> the vocabulary the rest of this tree already speaks.
    // Normalised rather than passed through so the UI and the local transport
    // agree on what "the model stopped because it ran out of room" is called;
    // anything unrecognised is forwarded verbatim, because inventing a mapping
    // for a value we have not seen would be worse than showing the raw one.
    void on_finish(std::string_view r) {
        if (r == "stop") {
            t->result.stop_reason = "end_turn";
        } else if (r == "length") {
            t->result.stop_reason = "max_tokens";
        } else if (r == "content_filter") {
            // The endpoint's refusal shape. Classified as Refusal so it is NOT
            // retried -- a byte-identical resend is deterministically refused
            // too, and would just be billed twice.
            t->result.stop_reason = "refusal";
            t->result.status = Status::Refusal;
        } else {
            t->result.stop_reason.assign(r);
        }
    }

    void on_api_error(std::string_view type, std::string_view msg) {
        t->result.status = Status::ApiError;
        t->result.error_detail.assign(type).append(": ").append(msg);
    }
};

struct OpenAiProtocol {
    using Sink = OpenAiSink;
    using Decoder = detail::OpenAiSseDecoder<OpenAiSink>;
};

}  // namespace

struct OpenAiStreamClient::Impl {
    std::string endpoint;
    detail::CurlStreamCore<OpenAiProtocol> core;

    explicit Impl(Config c)
        : endpoint(join_url(c.base_url, "chat/completions")),
          core(lower(c, endpoint), build_headers(c), "openai") {}

    // The vendor Config -> the protocol-agnostic curl knobs.
    static detail::StreamOptions lower(const Config& c, const std::string& url) {
        detail::StreamOptions o;
        o.url = url;
        o.connect_timeout_ms = c.connect_timeout_ms;
        o.ttft_timeout_ms = c.ttft_timeout_ms;
        o.stall_timeout_ms = c.stall_timeout_ms;
        o.enable_http2 = c.enable_http2;
        o.verbose = c.verbose;
        return o;
    }

    static std::vector<std::string> build_headers(const Config& c) {
        return {
            "Content-Type: application/json",
            "Accept: text/event-stream",
            // Kill curl's default "Expect: 100-continue" on large POSTs -- it
            // costs a full RTT before the body is even sent.
            "Expect:",
            // An EMPTY key would otherwise be sent as a well-formed "Bearer "
            // and come back as an opaque 401. The caller is responsible for not
            // constructing this client without a key (main.cpp checks), but a
            // header line that cannot be mistaken for valid auth is cheap.
            "Authorization: Bearer " + c.api_key,
        };
    }
};

// ---------------------------------------------------------------------------

OpenAiStreamClient::OpenAiStreamClient(Config cfg)
    : impl_(std::make_unique<Impl>(std::move(cfg))) {}

OpenAiStreamClient::~OpenAiStreamClient() = default;

Result OpenAiStreamClient::send(const std::string& body, const Callbacks& cb) noexcept {
    return impl_->core.run(body, &cb);
}

void OpenAiStreamClient::shutdown() noexcept { impl_->core.shutdown(); }

void OpenAiStreamClient::abort() noexcept { impl_->core.abort_current(); }

const std::string& OpenAiStreamClient::endpoint() const noexcept { return impl_->endpoint; }

}  // namespace blackwell::cloud
