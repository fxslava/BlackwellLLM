#pragma once
// =============================================================================
// cloud/cloud_types.hpp — the vocabulary shared by every transport.
//
// Split out of claude_stream_client.hpp so that OfflineTransport (and therefore
// the whole offline voice_assistant workflow) can be built with NO libcurl, NO
// simdjson, and NO link dependency on blackwell_cloud at all.
//
// That is why to_string() and is_retryable() are `inline` here rather than
// living in claude_stream_client.cpp: if they stayed in the .cpp, anything that
// merely classified a Result would drag in the one TU that needs libcurl, and
// `BUILD_CLOUD_CLIENT=OFF` would stop being a real configuration.
// =============================================================================
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace blackwell::cloud {

enum class Status : uint32_t {
    Ok = 0,
    ShuttingDown,     // aborted by shutdown(); not an error
    ConnectTimeout,
    TtftTimeout,      // no first body byte within budget
    StallTimeout,     // inter-byte gap exceeded budget
    NetworkError,     // transport-level failure
    HttpError,        // non-2xx; see Result::http_status
    ApiError,         // HTTP 200 + `event: error` inside the stream
    Refusal,          // stop_reason == "refusal"
    MalformedStream,  // framing or JSON we could not parse
};

[[nodiscard]] inline const char* to_string(Status s) noexcept {
    switch (s) {
        case Status::Ok:              return "Ok";
        case Status::ShuttingDown:    return "ShuttingDown";
        case Status::ConnectTimeout:  return "ConnectTimeout";
        case Status::TtftTimeout:     return "TtftTimeout";
        case Status::StallTimeout:    return "StallTimeout";
        case Status::NetworkError:    return "NetworkError";
        case Status::HttpError:       return "HttpError";
        case Status::ApiError:        return "ApiError";
        case Status::Refusal:         return "Refusal";
        case Status::MalformedStream: return "MalformedStream";
    }
    return "Unknown";
}

// True when re-sending the SAME body has a chance of succeeding. Safe to act on
// precisely because the body is immutable: the commit gate froze it, so a retry
// is byte-identical and hits the warm prompt cache.
[[nodiscard]] inline bool is_retryable(Status s, long http) noexcept {
    switch (s) {
        case Status::ConnectTimeout:
        case Status::TtftTimeout:
        case Status::StallTimeout:
        case Status::NetworkError:
        case Status::ApiError:  // overloaded_error / api_error
            return true;
        case Status::HttpError:
            return http == 408 || http == 409 || http == 429 || http >= 500;
        default:
            // Ok / ShuttingDown / Refusal / MalformedStream: a byte-identical
            // retry produces a byte-identical outcome. Refusal in particular is
            // deterministic -- that is what the `fallbacks` parameter is for.
            return false;
    }
}

struct Usage {
    uint32_t input_tokens = 0;
    uint32_t output_tokens = 0;
    uint32_t cache_creation_input_tokens = 0;  // written this request (1.25x / 2x for 1h ttl)
    uint32_t cache_read_input_tokens = 0;      // served from cache (~0.1x)
};

struct Result {
    Status      status = Status::Ok;
    long        http_status = 0;
    Usage       usage{};
    std::string stop_reason;   // "end_turn" | "max_tokens" | "refusal" | ...
    std::string error_detail;  // API error type/message, or curl's error buffer
    long        retry_after_s = 0;  // parsed from a 429 retry-after header, 0 if absent
};

struct Callbacks {
    // Per text delta. `utf8` is valid ONLY for the duration of the call -- its
    // backing store is the transport's parse buffer. Copy it or consume it.
    std::function<void(std::string_view utf8)> on_text;

    // Fires twice on the real transport: message_start (input/cache counters)
    // and message_delta (output counter). Watch cache_read_input_tokens -- a
    // persistent zero means a silent prompt-cache invalidator, which is the most
    // expensive bug in this pipeline and raises no error on its own.
    std::function<void(const Usage&)> on_usage;

    // Server-side fallback fired: the request was re-served by another model.
    std::function<void(std::string_view from_model, std::string_view to_model)> on_fallback;
};

}  // namespace blackwell::cloud
