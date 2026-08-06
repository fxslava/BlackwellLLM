#pragma once
// =============================================================================
// cloud/openai_transport.hpp — the OpenAI-compatible LIVE transport:
// IIntentTransport over OpenAiStreamClient.
//
// Only compiled into targets that link blackwell_cloud (BUILD_CLOUD_CLIENT=ON).
// voice_assistant includes this header only behind BLACKWELL_HAVE_CLOUD_CLIENT,
// so an offline build never names OpenAiStreamClient and never needs libcurl.
//
// Thin on purpose, like ClaudeTransport. It carries exactly two things the
// client cannot know:
//
//   1. THE MODEL. The dispatcher's ContextProvider supplies a RequestContext
//      whose `model` defaults to an Anthropic id, and overriding it there would
//      make the context provider -- shared by every leg -- depend on which leg
//      is selected. So the model is bound to the transport, which is the object
//      that already knows which endpoint it is talking to. `model()` is a plain
//      member set at construction: it is a RESTART-tier setting (see
//      settings_store.hpp), so there is nothing to synchronise.
//
//   2. THE BODY SHAPE. build_body() overrides the Anthropic default; that is
//      the whole reason the seam exists (intent_transport.hpp).
//
// PREWARM IS A DELIBERATE NO-OP. On Anthropic it is free (max_tokens:0 writes
// the prompt cache and bills nothing for output). Here the nearest equivalent
// is a real, billed completion. Spending money at startup to shave a TLS
// handshake is not a trade this class gets to make silently, so it reports Ok
// without sending anything -- and build_prewarm_body() returns nothing to send.
// =============================================================================
#include <string>
#include <utility>

#include "intent_transport.hpp"
#include "openai_request.hpp"
#include "openai_stream_client.hpp"

namespace blackwell::cloud {

class OpenAiTransport final : public IIntentTransport {
public:
    // `client` is BORROWED and must outlive this object. `model` is the id sent
    // in every request body; `label` is what the UI badge shows.
    OpenAiTransport(OpenAiStreamClient& client, std::string model,
                    OpenAiRequestOptions opts = {})
        : client_(client), model_(std::move(model)), opts_(opts) {
        label_ = "Remote: " + model_;
    }

    [[nodiscard]] const char* name() const noexcept override { return label_.c_str(); }
    // TRUE: this leg spends money. The badge is the only thing that tells a
    // user a commit was billed, so it must not be softened.
    [[nodiscard]] bool is_live() const noexcept override { return true; }

    [[nodiscard]] std::string build_body(const RequestContext& ctx) const override {
        // The context arrives with whatever model the provider defaulted to;
        // the transport's own model is authoritative for its own endpoint.
        RequestContext c = ctx;
        c.model = model_;
        return build_openai_request(c, opts_);
    }

    // Nothing is sent, so nothing needs rendering. See the header preamble.
    [[nodiscard]] std::string build_prewarm_body(const RequestContext&) const override {
        return {};
    }

    [[nodiscard]] Result send(const TransportRequest& req, const Callbacks& cb) noexcept override {
        // `req.intent` is intentionally unused: it is already embedded in
        // req.body by build_body(). Only OfflineTransport, which never parses
        // the body, needs it separately.
        return client_.send(req.body, cb);
    }

    [[nodiscard]] Result prewarm(const std::string&) noexcept override { return Result{}; }

    void shutdown() noexcept override { client_.shutdown(); }

    void abort() noexcept override { client_.abort(); }

private:
    OpenAiStreamClient&  client_;
    std::string          model_;
    std::string          label_;  // backs name(); must outlive the const char*
    OpenAiRequestOptions opts_;
};

}  // namespace blackwell::cloud
