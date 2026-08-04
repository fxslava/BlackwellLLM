#pragma once
// =============================================================================
// cloud/intent_request.hpp — builds the Anthropic Messages API request body for
// a committed intent. Header-only, and deliberately dependency-free (no curl,
// no simdjson, no CUDA) so it can be unit-tested on its own.
//
// PROMPT CACHING IS THE POINT OF THIS FILE.
//   Caching is a byte-exact PREFIX match, rendered tools -> system -> messages.
//   One changed byte anywhere in the prefix invalidates everything after it, so
//   the layout below puts strictly-stable content first and volatile content
//   last, with breakpoints on the boundaries:
//
//     system[0]   frozen instructions          -- no breakpoint
//     system[1]   glossary + session context   <- cache_control, ttl "1h"
//     msg[0][0]   committed transcript prefix  <- cache_control (5m default)
//     msg[0][1]   the live intent              -- NO cache_control (volatile)
//
//   Two of the four allowed breakpoints. The 1h TTL costs 2x on write instead
//   of 1.25x, but a hotkey-driven background app idles past the 5-minute
//   default between uses and would otherwise pay a COLD write every single
//   time -- which is strictly worse.
//
// THINGS THAT SILENTLY DESTROY THE CACHE (all of them raise no error):
//   * a timestamp, uptime, or session UUID interpolated into `instructions`
//   * rebuilding the system prompt when a mode toggles instead of appending
//   * non-deterministic serialisation (unsorted map iteration in the glossary)
//   * moving the breakpoint to the end of a growing transcript -- every request
//     then WRITES a new entry and never READS one
//   Watch Usage::cache_read_input_tokens. A persistent zero means one of these.
//
// MODEL SETTINGS
//   Claude Opus 5 has thinking ON by default (unlike Opus 4.8), and max_tokens
//   caps thinking + response text TOGETHER. For a latency-critical router leg
//   we disable it, which is legal only at effort <= "high" (xhigh/max + disabled
//   returns 400). The generic no-XML-tags line is the documented mitigation for
//   the tag-leak failure mode of disabled thinking -- and it is generic ON
//   PURPOSE: naming the tags explicitly, or adding a "do not reason"
//   instruction, measurably makes leakage worse.
// =============================================================================
#include <cstdio>
#include <string>
#include <string_view>

namespace blackwell::cloud {

// Appends `s` as a JSON string literal (quotes included).
//
// Assumes `s` is valid UTF-8. Invalid UTF-8 is an HTTP 400 from the API, and
// the place to catch it is the detokenizer that produced the text, not here --
// a partial multi-byte sequence at a token boundary is a local bug, and
// silently repairing it here would hide it.
inline void append_json_string(std::string& out, std::string_view s) {
    out.push_back('"');
    for (const unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));  // UTF-8 passes through
                }
        }
    }
    out.push_back('"');
}

// The stable/volatile split, made explicit in the type so a caller cannot put
// a timestamp in the "frozen" slot without noticing what they are doing.
//
// EVERY FIELD IS A BORROWED VIEW. That is deliberate -- the caller already owns
// these buffers (the persisted system prompt, the commit queue's payload) and
// copying them per attempt would be pure waste on a path that runs per turn.
// The cost is the usual one: `ctx.intent = std::string(...)` COMPILES and
// leaves a dangling view at the end of the full expression. Assign from
// something that outlives the send, and take the whole struct by value only to
// override a field, never to extend a lifetime.
struct RequestContext {
    std::string_view instructions;      // FROZEN. No timestamps, no session ids.
    std::string_view glossary;          // stable for the whole session
    std::string_view committed_prefix;  // transcript up to the commit pointer
    std::string_view intent;            // THE committed, EOS-terminated payload
    std::string_view model = "claude-opus-5";
    int              max_tokens = 2048;
};

namespace detail {

// Shared body writer. `streaming=false` + max_tokens 0 is the pre-warm shape.
inline std::string build(const RequestContext& ctx, bool streaming, int max_tokens) {
    std::string j;
    j.reserve(ctx.instructions.size() + ctx.glossary.size() + ctx.committed_prefix.size() +
              ctx.intent.size() + 1024);

    j += R"({"model":)";
    append_json_string(j, ctx.model);
    j += R"(,"max_tokens":)";
    j += std::to_string(max_tokens);
    if (streaming) j += R"(,"stream":true)";

    // Latency-critical leg: see the header preamble for why disabled + "low".
    j += R"(,"thinking":{"type":"disabled"},"output_config":{"effort":"low"})";

    // Opus 5 classifiers can decline (HTTP 200 + stop_reason "refusal").
    // "default" routes by refusal category server-side and needs the
    // server-side-fallback-2026-07-01 beta header on the client.
    j += R"(,"fallbacks":"default")";

    j += R"(,"system":[{"type":"text","text":)";
    {
        std::string sys(ctx.instructions);
        sys += " Do not include internal or system XML tags in your response.";
        append_json_string(j, sys);
    }
    j += R"(},{"type":"text","text":)";
    append_json_string(j, ctx.glossary);
    j += R"(,"cache_control":{"type":"ephemeral","ttl":"1h"}}])";

    j += R"(,"messages":[{"role":"user","content":[{"type":"text","text":)";
    append_json_string(j, ctx.committed_prefix);
    j += R"(,"cache_control":{"type":"ephemeral"}},{"type":"text","text":)";
    append_json_string(j, ctx.intent);
    j += R"(}]}]})";
    return j;
}

}  // namespace detail

// The real, streaming request for a committed intent.
[[nodiscard]] inline std::string build_intent_request(const RequestContext& ctx) {
    return detail::build(ctx, /*streaming=*/true, ctx.max_tokens);
}

// Cache + TLS pre-warm. NON-streaming and max_tokens 0 -- both are required:
// max_tokens:0 is rejected outright when combined with stream:true.
//
// Pass the SAME instructions/glossary/committed_prefix as the real request; the
// intent slot should be a short placeholder. Only the bytes BEFORE the last
// breakpoint have to match for the cache to hit, and the intent sits after it.
[[nodiscard]] inline std::string build_prewarm_request(const RequestContext& ctx) {
    return detail::build(ctx, /*streaming=*/false, /*max_tokens=*/0);
}

}  // namespace blackwell::cloud
