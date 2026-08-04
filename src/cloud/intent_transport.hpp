#pragma once
// =============================================================================
// cloud/intent_transport.hpp — the seam between "drive the commit queue" and
// "actually talk to a remote model".
//
// INTERFACE ONLY. The two implementations live beside it and are symmetric:
//     claude_transport.hpp   ClaudeTransport   live, needs libcurl
//     offline_transport.hpp  OfflineTransport  simulated, no dependencies
//
// WHY A TRANSPORT SEAM RATHER THAN A POLYMORPHIC DISPATCHER
//   The parts that differ between the offline and live paths are tiny: send a
//   body, get a response. Everything IntentDispatcher does around that -- drain
//   the gate, retry with backoff, honour retry-after, count outcomes, stay
//   interruptible on shutdown -- is identical, and duplicating it per transport
//   would mean the offline path exercised different control flow than
//   production, which defeats the point of having an offline path at all.
//
//   The decisive practical argument: OfflineTransport must be usable with
//   BUILD_CLOUD_CLIENT=OFF. This header and offline_transport.hpp include only
//   cloud_types.hpp and intent_request.hpp -- both header-only and
//   dependency-free -- so they build and link with no libcurl and no simdjson
//   anywhere in the picture. A fallback that derived from the concrete client
//   could not.
//
// THE WIRE BODY IS THE TRANSPORT'S CHOICE (build_body, below).
//   It used to be the dispatcher's: it called build_intent_request() itself and
//   handed every transport an Anthropic Messages payload. That held exactly as
//   long as there was one remote protocol. An OpenAI-compatible endpoint wants
//   the SAME RequestContext rendered as model/messages/stream, so the render
//   step moved down here -- one virtual, called once per attempt, on the
//   dispatcher thread. Everything above it (the commit rule, the retry ladder,
//   the callbacks) is untouched and stays shared, which is the whole point.
// =============================================================================
#include <cstdint>
#include <string>
#include <string_view>

#include "cloud_types.hpp"
#include "intent_request.hpp"  // RequestContext + the default (Anthropic) rendering

namespace blackwell::cloud {

// One dispatch attempt. `body` is the fully-built wire payload; `intent` is the
// raw committed text that produced it.
//
// The live transport ignores `intent` -- it is already inside `body`. The
// offline transport uses it so it does not have to re-parse JSON it just
// watched being built. Carrying both is cheaper and more honest than making
// every transport able to reverse-engineer the request it was handed.
struct TransportRequest {
    const std::string& body;
    std::string_view   intent;
    uint64_t           sequence = 0;  // monotone commit number, for display/logs
};

class IIntentTransport {
public:
    virtual ~IIntentTransport() = default;

    // For the UI badge: "Claude (live)" / "Offline (simulated)".
    [[nodiscard]] virtual const char* name() const noexcept = 0;

    // Is this transport actually talking to a paid remote? The UI uses it to
    // decide whether to show a cost warning; nothing branches on it internally.
    [[nodiscard]] virtual bool is_live() const noexcept = 0;

    // Render `ctx` into this transport's wire format. Called on the dispatcher
    // thread, once per attempt, and the result is what lands in
    // TransportRequest::body.
    //
    // The default is the Anthropic Messages shape, which is what every
    // transport that does not care about the body (the offline stand-in) and
    // the one it was designed for (ClaudeTransport) both want.
    [[nodiscard]] virtual std::string build_body(const RequestContext& ctx) const {
        return build_intent_request(ctx);
    }

    // The pre-warm variant of the same context. Only sent when prewarm() below
    // actually does something.
    [[nodiscard]] virtual std::string build_prewarm_body(const RequestContext& ctx) const {
        return build_prewarm_request(ctx);
    }

    // Blocks for the whole exchange. Callbacks fire on the calling thread.
    [[nodiscard]] virtual Result send(const TransportRequest& req,
                                      const Callbacks& cb) noexcept = 0;

    // Optional connection/cache warm-up. Default: nothing to warm.
    //
    // A transport that overrides this to a no-op should ALSO be the one
    // deciding not to spend anything -- the dispatcher builds the body for it
    // unconditionally, but building a string costs nothing and sending is what
    // this method controls.
    [[nodiscard]] virtual Result prewarm(const std::string& /*body*/) noexcept { return Result{}; }

    // Abort an in-flight send() at teardown. Any thread; idempotent.
    virtual void shutdown() noexcept {}
};

}  // namespace blackwell::cloud
