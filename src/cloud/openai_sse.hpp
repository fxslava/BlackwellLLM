#pragma once
// =============================================================================
// cloud/openai_sse.hpp — OpenAI-compatible /chat/completions event extraction.
//
// INTERNAL header: only openai_stream_client.cpp includes it, so simdjson stays
// out of every other TU. Framing is shared with the Anthropic decoder
// (sse_framer.hpp); this file is only what a `data:` payload MEANS.
//
// THE WIRE SHAPE, and the three ways it differs from Anthropic's:
//
//   data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}
//   data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}
//   data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
//   data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4}}
//   data: [DONE]
//
//   1. TERMINATOR. `[DONE]` is a sentinel, NOT JSON. Handing it to a parser is
//      a hard error, which would have been reported as MalformedStream on every
//      otherwise-perfect request -- so it is checked before anything else.
//   2. EVERYTHING IS ONE EVENT TYPE. There is no `type` discriminator; a chunk
//      carries whichever of delta/finish_reason/usage it happens to have, and
//      all three may arrive in the same chunk. So the parse is field probes on
//      one object rather than a dispatch on a tag.
//   3. THE FIELDS ARE OPTIONAL AND THE SERVERS VARY. This endpoint is a de
//      facto standard implemented by many gateways: `usage` may never arrive,
//      `finish_reason` may be null, `delta` may be absent on the last chunk.
//      Every probe below therefore fails soft -- a missing field is normal
//      traffic, not a protocol error. Only unparseable JSON aborts.
//
// WHY WE STILL RETURN false ON BAD JSON: a gateway that starts emitting HTML
// (a proxy error page) mid-stream must surface as MalformedStream rather than
// as silence, because silence is indistinguishable from a model that answered
// with nothing.
// =============================================================================
#include <cstdint>
#include <string>
#include <string_view>

#include "simdjson_external.hpp"  // <simdjson.h>, warning-quarantined
#include "sse_framer.hpp"

namespace blackwell::cloud::detail {

// Sink concept (compile-time; no vtable on the per-delta path):
//   void on_text_delta(std::string_view utf8);
//   void on_usage(uint32_t prompt, uint32_t completion, uint32_t cached);
//   void on_finish(std::string_view finish_reason);
//   void on_api_error(std::string_view type, std::string_view message);

template <class Sink>
class OpenAiEventParser {
public:
    explicit OpenAiEventParser(Sink& sink) : sink_(sink) {}

    [[nodiscard]] bool parse_event(std::string& ev) {
        namespace sj = simdjson;

        // The stream terminator is a bare sentinel, not a document. See (1).
        if (is_done(ev)) return true;

        // simdjson reads up to SIMDJSON_PADDING bytes past the document end.
        // Over-reserving is NOT optional -- an unpadded buffer is a heap
        // overread that passes every test and crashes in production.
        if (ev.capacity() - ev.size() < sj::SIMDJSON_PADDING) {
            ev.reserve(ev.size() + sj::SIMDJSON_PADDING);
        }
        sj::padded_string_view view(ev.data(), ev.size(), ev.capacity());

        sj::ondemand::document doc;
        if (parser_.iterate(view).get(doc) != sj::SUCCESS) return false;

        sj::ondemand::object root;
        if (doc.get_object().get(root) != sj::SUCCESS) return false;

        // A SINGLE FORWARD PASS over the top-level fields, rather than three
        // keyed probes. On-Demand is forward-only: a keyed lookup that misses
        // (and `error` misses on every healthy chunk) rescans the whole object
        // and leaves the cursor wrapped, which is both wasteful and the kind of
        // ordering assumption that breaks on the next gateway. Dispatching on
        // the key as it goes past is order-independent by construction.
        for (auto field : root) {
            std::string_view key;
            if (field.unescaped_key().get(key) != sj::SUCCESS) return false;

            if (key == "choices") {
                sj::ondemand::array choices;
                if (field.value().get_array().get(choices) != sj::SUCCESS) continue;
                if (!read_choices(choices)) return false;
            } else if (key == "usage") {
                sj::ondemand::object usage;
                if (field.value().get_object().get(usage) != sj::SUCCESS) continue;
                read_usage(usage);
            } else if (key == "error") {
                // An error object can arrive inside an HTTP 200 stream (rate
                // limits and upstream failures on gateways commonly do), so it
                // MUST be parsed out of the body -- the transport reports
                // success and nothing downstream would notice otherwise.
                sj::ondemand::object err;
                if (field.value().get_object().get(err) != sj::SUCCESS) {
                    sink_.on_api_error("unknown", "");
                    continue;
                }
                std::string_view etype, emsg;
                (void)err["type"].get_string().get(etype);
                (void)err["message"].get_string().get(emsg);
                sink_.on_api_error(etype, emsg);
            }
            // id / object / created / model / system_fingerprint: nothing to
            // extract. Their values are skipped by the iterator on the next
            // advance, which is why this loop must not `break` early.
        }
        return true;
    }

private:
    // The array is ALWAYS drained, even though this client never sets n > 1 and
    // therefore only cares about index 0: abandoning an On-Demand array halfway
    // leaves the parent iterator at the wrong depth for the fields that follow
    // it (`usage` is one of them). Draining is cheap -- the array has one
    // element -- and it is what makes the enclosing field loop safe.
    [[nodiscard]] bool read_choices(simdjson::ondemand::array& choices) {
        namespace sj = simdjson;
        bool taken = false;
        for (auto element : choices) {
            sj::ondemand::object choice;
            if (element.get_object().get(choice) != sj::SUCCESS) return false;
            if (taken) continue;
            taken = true;

            sj::ondemand::object delta;
            if (choice["delta"].get_object().get(delta) == sj::SUCCESS) {
                std::string_view text;
                // Non-string (null) content is normal on the role-only chunk
                // and on tool-call chunks; skipping it is the correct reading.
                if (delta["content"].get_string().get(text) == sj::SUCCESS && !text.empty()) {
                    sink_.on_text_delta(text);
                }
            }
            std::string_view finish;
            if (choice["finish_reason"].get_string().get(finish) == sj::SUCCESS) {
                sink_.on_finish(finish);
            }
        }
        return true;
    }

    // Only present when the request asked for it (stream_options.include_usage)
    // and only on servers that implement it. Absent is normal, not a fault.
    void read_usage(simdjson::ondemand::object& usage) {
        const uint32_t prompt = u32(usage, "prompt_tokens");
        const uint32_t completion = u32(usage, "completion_tokens");
        // OpenAI reports prompt-cache hits nested one level down. Mapping it
        // onto the shared Usage is what keeps the UI's "cache is working"
        // indicator meaningful on this leg too.
        uint32_t cached = 0;
        simdjson::ondemand::object details;
        if (usage["prompt_tokens_details"].get_object().get(details) == simdjson::SUCCESS) {
            cached = u32(details, "cached_tokens");
        }
        sink_.on_usage(prompt, completion, cached);
    }

    // `[DONE]`, tolerating the whitespace a gateway may or may not add.
    static bool is_done(const std::string& ev) noexcept {
        std::string_view v(ev);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.remove_prefix(1);
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r')) {
            v.remove_suffix(1);
        }
        return v == "[DONE]";
    }

    static uint32_t u32(simdjson::ondemand::object& obj, const char* key) {
        uint64_t v = 0;
        if (obj[key].get_uint64().get(v) != simdjson::SUCCESS) return 0;
        return static_cast<uint32_t>(v);
    }

    Sink& sink_;
    simdjson::ondemand::parser parser_;  // reused: owns the tape + string buffers
};

// Framing + OpenAI semantics, glued. Member order is the dependency order: the
// framer holds a reference to the parser, so the parser is declared first.
template <class Sink>
class OpenAiSseDecoder {
public:
    explicit OpenAiSseDecoder(Sink& sink) : parser_(sink), framer_(parser_) {}

    OpenAiSseDecoder(const OpenAiSseDecoder&) = delete;
    OpenAiSseDecoder& operator=(const OpenAiSseDecoder&) = delete;

    void reset() noexcept { framer_.reset(); }

    [[nodiscard]] bool feed(const char* p, size_t n) { return framer_.feed(p, n); }

private:
    OpenAiEventParser<Sink>            parser_;
    SseFramer<OpenAiEventParser<Sink>> framer_;
};

}  // namespace blackwell::cloud::detail
