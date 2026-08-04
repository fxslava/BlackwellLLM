#pragma once
// =============================================================================
// cloud/openai_stream_client.hpp — streaming client for any OpenAI-compatible
// /chat/completions endpoint (OpenAI itself, a router, a local vLLM/llama.cpp
// server -- the wire contract is the same).
//
// SAME DOCTRINE AS ClaudeStreamClient, and for the same reasons:
//
//   DUMB ON PURPOSE. Every request that reaches send() has already been ruled
//   final by the local arbiter (the backbone terminated on EOS and
//   IntentCommitQueue accepted it). No admission control, no supersede, no
//   latest-wins, no barge-in cancellation -- transmit an immutable body, stream
//   the answer back, classify the outcome.
//
//   THREADING. send() is synchronous and blocks the calling thread for the
//   whole transfer. ONE thread may call it -- IntentDispatcher's. It is NEVER
//   the engine thread (single-threaded engine doctrine, CLAUDE.md): a 2-second
//   network stall must not sit on the thread that owns the KV cache. Callbacks
//   fire on the calling thread, inside send(), and must only enqueue toward the
//   UI/TTS. shutdown() is the ONE method callable from another thread.
//
//   ERROR DOCTRINE (CLAUDE.md #4). Ctor throws (INIT tier); send() returns a
//   Result and never throws (RUNTIME tier). curl and simdjson are PIMPL'd out.
//
// WHAT IS DIFFERENT FROM THE ANTHROPIC CLIENT
//   * Auth is `Authorization: Bearer <key>` instead of `x-api-key`.
//   * The endpoint is derived: base_url + "/chat/completions". The setting the
//     user edits is the BASE ("https://router.cheap/v1"), because that is what
//     every provider documents and what they paste from.
//   * There is no prewarm. On Anthropic, prewarm is a max_tokens:0 request that
//     writes the prompt cache for free. This endpoint has no such shape -- the
//     nearest equivalent is a real, BILLED completion, and silently spending
//     money to shave a TLS handshake is not a trade this client gets to make on
//     the user's behalf. prewarm() therefore succeeds without sending anything.
// =============================================================================
#include <memory>
#include <string>

#include "cloud_types.hpp"  // Status / Usage / Result / Callbacks, transport-agnostic

namespace blackwell::cloud {

class OpenAiStreamClient {
public:
    struct Config {
        std::string api_key;  // never logged, never placed in an error string
        // The BASE url, as the provider documents it. "/chat/completions" is
        // appended by the client -- see the header preamble.
        std::string base_url = "https://api.openai.com/v1";

        long connect_timeout_ms = 2000;
        long ttft_timeout_ms = 20000;  // request sent -> first body byte
        long stall_timeout_ms = 10000; // max gap between body bytes

        // Ask for a usage-bearing final chunk. See OpenAiRequestOptions: off by
        // default because a gateway that does not know the field 400s instead
        // of ignoring it. This flag only governs the HEADER/knob side; the body
        // side is set by whoever builds the request.
        bool include_usage = false;

        bool enable_http2 = true;  // requires an nghttp2-enabled libcurl
        bool verbose = false;      // CURLOPT_VERBOSE; NEVER enable in Release
    };

    // Throws std::runtime_error on libcurl init failure (INIT tier).
    // curl_global_init() is handled by the shared core, once per process and
    // thread-safely -- callers have nothing to arrange.
    explicit OpenAiStreamClient(Config cfg);
    ~OpenAiStreamClient();

    OpenAiStreamClient(const OpenAiStreamClient&) = delete;
    OpenAiStreamClient& operator=(const OpenAiStreamClient&) = delete;

    // Transmit `body` and stream the response. Blocks until the transfer ends.
    // `body` must outlive the call (CURLOPT_POSTFIELDS does not copy) -- taking
    // it by const& makes that the caller's explicit responsibility.
    [[nodiscard]] Result send(const std::string& body, const Callbacks& cb) noexcept;

    // Abort an in-flight send() so the process can exit. Callable from any
    // thread; idempotent. After this, send() returns Status::ShuttingDown
    // immediately. This is teardown only -- it is NOT a barge-in mechanism.
    void shutdown() noexcept;

    // The full endpoint this client POSTs to, for the startup log line. Useful
    // enough to be worth exposing: a wrong base URL is the single most likely
    // misconfiguration, and it otherwise only shows up as a 404 at first use.
    [[nodiscard]] const std::string& endpoint() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace blackwell::cloud
