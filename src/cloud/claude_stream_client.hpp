#pragma once
// =============================================================================
// cloud/claude_stream_client.hpp — Anthropic Messages API streaming client.
//
// LOCAL ROUTER PATTERN: this client is DUMB ON PURPOSE.
//   Every request that reaches send() has already been ruled final by the local
//   arbiter (Llama/Ultravox terminated on EOS and IntentCommitQueue accepted
//   it). The client therefore performs no admission control, no supersede, no
//   latest-wins, and no barge-in cancellation. Its only job is: transmit an
//   immutable body, stream the answer back, and classify the outcome.
//
// WHAT WAS REMOVED, AND WHY (v1 -> v2)
//   v1 aborted in-flight requests when the user barged in. That was financially
//   backwards: input/prefill tokens are billed at ACCEPTANCE, so by the time
//   there is anything to abort the expensive part is already sunk. The fix is
//   to not send fragments in the first place, which is now the commit gate's
//   job. With cancellation gone, so are: the worker thread, the SPSC job queue,
//   the monotone cancel epoch, and curl_multi_wakeup(). ~80 lines of
//   concurrency deleted, and the class became synchronous.
//
//   Residual honesty: aborting mid-stream would still have saved OUTPUT tokens,
//   which are billed as they stream. That is a real but much smaller number,
//   and it is not worth reintroducing cross-thread cancellation for. If a
//   barged-in answer should be discarded, discard it at the consumer -- the
//   bytes are already paid for either way.
//
// THREADING
//   send() is synchronous and blocks the calling thread for the whole transfer.
//   ONE thread may call send() -- in production that is IntentDispatcher's
//   thread. It is NEVER the engine thread (single-threaded engine doctrine,
//   CLAUDE.md): a 2-second network stall must not sit on the thread that owns
//   the KV cache.
//
//   Callbacks fire on the calling thread, inside send(). They must only enqueue
//   toward the UI/TTS -- same rule as the bridge's TokenSink.
//
//   shutdown() is the ONE method callable from another thread: it aborts an
//   in-flight transfer at teardown so the process can exit promptly.
//
// ERROR DOCTRINE (CLAUDE.md #4)
//   Ctor throws (INIT tier). send() returns a Result and never throws
//   (RUNTIME tier). curl and simdjson are PIMPL'd out of this header.
// =============================================================================
#include <cstdint>
#include <memory>
#include <string>

#include "cloud_types.hpp"  // Status / Usage / Result / Callbacks, transport-agnostic

namespace blackwell::cloud {

class ClaudeStreamClient {
public:
    struct Config {
        std::string api_key;  // never logged, never placed in an error string
        std::string url = "https://api.anthropic.com/v1/messages";
        std::string beta;     // e.g. "server-side-fallback-2026-07-01"; empty = omit

        long connect_timeout_ms = 2000;
        long ttft_timeout_ms = 15000;  // request sent -> first body byte
        long stall_timeout_ms = 10000; // max gap between body bytes (pings count)

        bool enable_http2 = true;  // requires an nghttp2-enabled libcurl
        bool verbose = false;      // CURLOPT_VERBOSE; NEVER enable in Release
    };

    // Throws std::runtime_error on libcurl init failure (INIT tier).
    // curl_global_init() is handled by the shared core, once per process and
    // thread-safely -- callers have nothing to arrange.
    explicit ClaudeStreamClient(Config cfg);
    ~ClaudeStreamClient();

    ClaudeStreamClient(const ClaudeStreamClient&) = delete;
    ClaudeStreamClient& operator=(const ClaudeStreamClient&) = delete;

    // Transmit `body` and stream the response. Blocks until the transfer ends.
    // `body` must outlive the call (CURLOPT_POSTFIELDS does not copy) -- taking
    // it by const& makes that the caller's explicit responsibility.
    [[nodiscard]] Result send(const std::string& body, const Callbacks& cb) noexcept;

    // Establish the TLS connection and prime the prompt cache without paying for
    // a generation: a `max_tokens: 0` request runs prefill, writes the cache,
    // and returns immediately with content:[] and zero output tokens billed.
    // Call at startup and on hotkey-arm -- it removes a cold handshake AND a
    // cold prefill from the first real request's critical path.
    //
    // NOTE: max_tokens:0 is REJECTED with stream:true, so `prewarm_body` must be
    // a NON-streaming variant of the real body (same prefix, same cache_control
    // breakpoints, no "stream" field).
    [[nodiscard]] Result prewarm(const std::string& prewarm_body) noexcept;

    // Abort an in-flight send() so the process can exit. Callable from any
    // thread; idempotent. After this, send() returns Status::ShuttingDown
    // immediately. This is teardown only -- it is NOT a barge-in mechanism.
    void shutdown() noexcept;

    // Stop the request in flight RIGHT NOW and leave the client armed. Callable
    // from any thread; a call with nothing in flight does nothing.
    //
    // This is the user's Stop button, and it is the ONE thing docs/LOCAL_ROUTER.md
    // records as deliberately absent -- because cancellation cannot save the
    // input/prefill tokens, which are billed at acceptance. It is here for the
    // OUTPUT half, which bills as it streams: an answer the user stopped reading
    // three tokens in should not keep being paid for to the end. The gate's
    // argument is untouched by this -- nothing unfinished is ever SENT; this only
    // ends a send that the user, having seen the start of it, no longer wants.
    void abort() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace blackwell::cloud
