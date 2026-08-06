#pragma once
// =============================================================================
// cloud/claude_transport.hpp — the LIVE transport: IIntentTransport over
// ClaudeStreamClient.
//
// Only compiled into targets that link blackwell_cloud (BUILD_CLOUD_CLIENT=ON).
// voice_assistant includes this header only behind BLACKWELL_HAVE_CLOUD_CLIENT,
// so an offline build never names ClaudeStreamClient and never needs libcurl.
//
// Thin on purpose. Everything interesting -- watchdogs, SSE framing, status
// classification -- already lives in the client; this adapter exists so the
// dispatcher can hold the live and offline transports behind one reference.
// =============================================================================
#include <string>

#include "claude_stream_client.hpp"
#include "intent_transport.hpp"

namespace blackwell::cloud {

class ClaudeTransport final : public IIntentTransport {
public:
    explicit ClaudeTransport(ClaudeStreamClient& client) : client_(client) {}

    [[nodiscard]] const char* name() const noexcept override { return "Claude (live)"; }
    [[nodiscard]] bool is_live() const noexcept override { return true; }

    [[nodiscard]] Result send(const TransportRequest& req, const Callbacks& cb) noexcept override {
        // `req.intent` is intentionally unused: it is already embedded in
        // req.body by build_intent_request(). Only OfflineTransport, which never
        // parses the body, needs it separately.
        return client_.send(req.body, cb);
    }

    [[nodiscard]] Result prewarm(const std::string& body) noexcept override {
        return client_.prewarm(body);
    }

    void shutdown() noexcept override { client_.shutdown(); }

    void abort() noexcept override { client_.abort(); }

private:
    ClaudeStreamClient& client_;
};

}  // namespace blackwell::cloud
