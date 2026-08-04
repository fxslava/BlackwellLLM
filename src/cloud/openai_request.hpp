#pragma once
// =============================================================================
// cloud/openai_request.hpp — builds the /chat/completions request body for a
// committed intent. Header-only and dependency-free (no curl, no simdjson, no
// CUDA) so it can be unit-tested on its own, exactly like intent_request.hpp.
//
// SAME RequestContext, DIFFERENT WIRE SHAPE. The dispatcher hands every
// transport the same context; which body it becomes is the transport's choice
// (IIntentTransport::build_body). That is what lets "answer remotely" mean
// Anthropic on one machine and an OpenAI-compatible gateway on another without
// a second dispatcher, a second commit rule, or a branch anywhere above here.
//
// THE MAPPING, and why each half lands where it does:
//
//   ctx.instructions  ─┐
//   ctx.glossary      ─┴─>  messages[0] role "system"
//   ctx.committed_prefix ─┐
//   ctx.intent           ─┴─>  messages[1] role "user"
//
//   The two system halves are joined into ONE message rather than sent as two,
//   because multi-part system content is an Anthropic shape; the portable
//   reading of "system" here is a single string. The two user halves are joined
//   for the same reason -- and blank-line separated, so the model sees a
//   transcript followed by the new utterance rather than one run-on sentence.
//
// NO PROMPT-CACHE BREAKPOINTS, and that is not an omission. Caching on this
// endpoint is automatic and prefix-based where it exists at all (and absent on
// many compatible gateways); there is no cache_control to place. The stability
// discipline from intent_request.hpp still pays off -- a timestamp in
// `instructions` still defeats an automatic prefix cache -- it just has no knob
// to express it with.
// =============================================================================
#include <string>
#include <string_view>

#include "intent_request.hpp"  // RequestContext + append_json_string

namespace blackwell::cloud {

// The knobs that have no home in RequestContext because they are properties of
// the ENDPOINT, not of the turn.
struct OpenAiRequestOptions {
    // Ask the server to append a final usage-bearing chunk. OFF by default: it
    // is standard, but "standard" on this endpoint means "most gateways", and a
    // gateway that rejects the unknown field fails the whole request with a 400
    // rather than ignoring it. Token counts are not worth that trade by
    // default; turn it on for an endpoint you know supports it.
    bool include_usage = false;
    // 0 omits the field entirely, which is the right default for a server whose
    // own cap is unknown. Non-zero is a spend ceiling, not a style preference.
    int max_tokens = 0;
};

// Joins two halves with `sep` only when both are non-empty, so an empty
// glossary or an empty committed prefix does not produce a leading separator
// the model has to interpret.
namespace detail {

inline std::string join_nonempty(std::string_view a, std::string_view b, std::string_view sep) {
    if (a.empty()) return std::string(b);
    if (b.empty()) return std::string(a);
    std::string out;
    out.reserve(a.size() + sep.size() + b.size());
    out.append(a).append(sep).append(b);
    return out;
}

}  // namespace detail

[[nodiscard]] inline std::string build_openai_request(const RequestContext& ctx,
                                                     const OpenAiRequestOptions& opt = {}) {
    const std::string system =
        detail::join_nonempty(ctx.instructions, ctx.glossary, "\n\n");
    const std::string user =
        detail::join_nonempty(ctx.committed_prefix, ctx.intent, "\n\n");

    std::string j;
    j.reserve(system.size() + user.size() + 256);

    j += R"({"model":)";
    append_json_string(j, ctx.model);
    j += R"(,"messages":[{"role":"system","content":)";
    append_json_string(j, system);
    j += R"(},{"role":"user","content":)";
    append_json_string(j, user);
    j += R"(}],"stream":true)";

    if (opt.max_tokens > 0) {
        j += R"(,"max_tokens":)";
        j += std::to_string(opt.max_tokens);
    }
    if (opt.include_usage) {
        j += R"(,"stream_options":{"include_usage":true})";
    }
    j += '}';
    return j;
}

// Join a base URL and a path with exactly one slash between them. Users type
// "https://host/v1", "https://host/v1/" and (occasionally) "https://host/v1 "
// interchangeably, and a double slash is a 404 on enough gateways that it is
// worth normalising here rather than in the settings validator -- this is the
// only place that knows what a valid endpoint looks like.
[[nodiscard]] inline std::string join_url(std::string_view base, std::string_view path) {
    constexpr std::string_view kSpace = " \t\r\n";
    while (!base.empty() && kSpace.find(base.front()) != std::string_view::npos) {
        base.remove_prefix(1);
    }
    while (!base.empty() &&
           (base.back() == '/' || kSpace.find(base.back()) != std::string_view::npos)) {
        base.remove_suffix(1);
    }
    while (!path.empty() && path.front() == '/') path.remove_prefix(1);
    std::string out(base);
    out.push_back('/');
    out.append(path);
    return out;
}

}  // namespace blackwell::cloud
