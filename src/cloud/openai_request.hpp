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
//   ctx.glossary      ─┴─>  messages[0]   role "system"
//   ctx.history          ──>  messages[1..2n]  role "user"/"assistant", paired
//   ctx.committed_prefix ─┐
//   ctx.intent           ─┴─>  messages[last]  role "user"
//
//   The two system halves are joined into ONE message rather than sent as two,
//   because multi-part system content is an Anthropic shape; the portable
//   reading of "system" here is a single string. The two user halves are joined
//   for the same reason -- and blank-line separated, so the model sees a
//   transcript followed by the new utterance rather than one run-on sentence.
//
// THE HISTORY IS WHAT MAKES THE REMOTE LEG REMEMBER, and it is capped HERE.
//   /chat/completions is stateless; without replayed turns the model forgets
//   the user's name as soon as the turn ends. Replaying the WHOLE session
//   instead makes the input bill grow quadratically, so the renderer takes only
//   the LAST opt.max_history_pairs pairs -- the front of the window is what gets
//   dropped, because the recent turn is the one the current utterance refers
//   back to.
//
//   Strict alternation is a protocol requirement, not a style choice: several
//   gateways reject two consecutive `user` messages outright. Each history entry
//   is emitted as a COMPLETE user+assistant pair and a half-formed one is
//   skipped, so the sequence is alternating by construction and stays that way
//   whatever the store hands over.
//
// NO PROMPT-CACHE BREAKPOINTS, and that is not an omission. Caching on this
// endpoint is automatic and prefix-based where it exists at all (and absent on
// many compatible gateways); there is no cache_control to place. The stability
// discipline from intent_request.hpp still pays off -- a timestamp in
// `instructions` still defeats an automatic prefix cache -- it just has no knob
// to express it with.
// =============================================================================
#include <cstddef>
#include <string>
#include <string_view>

#include "intent_request.hpp"  // RequestContext + ChatTurn + append_json_string

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
    // How many past user/assistant PAIRS to replay ahead of the current turn --
    // at most 2*N extra messages on the wire. This is the COST knob (the store
    // in chat_history.hpp has its own, separate memory bound); 3 keeps a short
    // conversation coherent -- names, referents, the last correction -- while
    // bounding the replayed input to something a per-turn budget can absorb.
    // 0 restores the previous, memoryless behaviour.
    int max_history_pairs = 3;
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

    // The tail of the window: `first` is the index of the oldest pair that still
    // fits under the cap, so a store larger than the cap loses its FRONT.
    const std::size_t kept = ctx.history.size();
    const std::size_t want =
        opt.max_history_pairs > 0 ? static_cast<std::size_t>(opt.max_history_pairs) : 0;
    const std::size_t first = want >= kept ? 0 : kept - want;

    std::size_t history_bytes = 0;
    for (std::size_t i = first; i < kept; ++i) {
        history_bytes += ctx.history[i].user.size() + ctx.history[i].assistant.size() + 64;
    }

    std::string j;
    j.reserve(system.size() + user.size() + history_bytes + 256);

    j += R"({"model":)";
    append_json_string(j, ctx.model);
    j += R"(,"messages":[{"role":"system","content":)";
    append_json_string(j, system);
    j += '}';

    for (std::size_t i = first; i < kept; ++i) {
        const ChatTurn& t = ctx.history[i];
        // Both halves or neither: a lone `user` here would put two consecutive
        // user messages on the wire (this one and the current turn), which the
        // stricter gateways answer with a 400.
        if (t.user.empty() || t.assistant.empty()) continue;
        j += R"(,{"role":"user","content":)";
        append_json_string(j, t.user);
        j += R"(},{"role":"assistant","content":)";
        append_json_string(j, t.assistant);
        j += '}';
    }

    j += R"(,{"role":"user","content":)";
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
