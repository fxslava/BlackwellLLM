#pragma once
// =============================================================================
// cloud/claude_sse.hpp — incremental SSE framing + Anthropic event extraction.
//
// INTERNAL header: only claude_stream_client.cpp includes it, so simdjson stays
// out of every other TU (same discipline as src/vad/silero_vad.hpp and
// ONNXRuntime -- exactly one TU sees the third-party API).
//
// Header-only and templated on the sink so it is unit-testable WITHOUT libcurl:
// feed a captured fixture one byte at a time and whole, and assert the two
// produce identical event sequences. That is the regression test for the
// chunk-boundary class of bug, which is the one that survives to production.
//
// INVARIANT: feed() must tolerate arbitrary split points. TCP will split a
// `data:` line mid-JSON, mid-UTF-8, and between the \r and the \n.
// =============================================================================
#include <simdjson.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace blackwell::cloud::detail {

// Sink concept (compile-time; no vtable on the per-delta path):
//   void on_text_delta(std::string_view utf8);
//   void on_usage(uint32_t in, uint32_t out, uint32_t cache_create, uint32_t cache_read);
//   void on_stop(std::string_view stop_reason);
//   void on_fallback(std::string_view from_model, std::string_view to_model);
//   void on_api_error(std::string_view type, std::string_view message);

template <class Sink>
class SseDecoder {
public:
    explicit SseDecoder(Sink& sink) : sink_(sink) {
        line_.reserve(256);
        ev_.reserve(kEventReserve);
    }

    void reset() noexcept {
        line_.clear();
        ev_.clear();
    }

    // Returns false on an unrecoverable framing/JSON error -> caller aborts.
    [[nodiscard]] bool feed(const char* p, size_t n) {
        const char* const end = p + n;
        while (p < end) {
            const auto* nl =
                static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
            if (nl == nullptr) {
                // Partial line -- carry it over to the next chunk.
                line_.append(p, static_cast<size_t>(end - p));
                return true;
            }

            std::string_view line;
            if (line_.empty()) {
                // Fast path: the whole line arrived in this chunk. Zero copy.
                line = std::string_view(p, static_cast<size_t>(nl - p));
            } else {
                line_.append(p, static_cast<size_t>(nl - p));
                line = line_;
            }
            if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

            const bool ok = handle_line(line);
            line_.clear();
            if (!ok) return false;
            p = nl + 1;
        }
        return true;
    }

private:
    static constexpr size_t kEventReserve = 4096;

    [[nodiscard]] bool handle_line(std::string_view line) {
        if (line.empty()) return dispatch();   // blank line terminates the event
        if (line.front() == ':') return true;  // comment / keep-alive

        constexpr std::string_view kData = "data:";
        if (line.size() >= kData.size() && line.compare(0, kData.size(), kData) == 0) {
            std::string_view v = line.substr(kData.size());
            if (!v.empty() && v.front() == ' ') v.remove_prefix(1);
            if (!ev_.empty()) ev_.push_back('\n');  // multi-line data:, per spec
            ev_.append(v);
        }
        // `event:` / `id:` / `retry:` deliberately ignored -- the JSON "type"
        // field is authoritative and always present on this endpoint.
        return true;
    }

    [[nodiscard]] bool dispatch() {
        if (ev_.empty()) return true;
        const bool ok = parse_event();
        ev_.clear();
        return ok;
    }

    [[nodiscard]] bool parse_event() {
        namespace sj = simdjson;

        // simdjson reads up to SIMDJSON_PADDING bytes past the document end.
        // Over-reserving is NOT optional -- an unpadded buffer is a heap
        // overread that passes every test and crashes in production.
        if (ev_.capacity() - ev_.size() < sj::SIMDJSON_PADDING) {
            ev_.reserve(ev_.size() + sj::SIMDJSON_PADDING);
        }
        sj::padded_string_view view(ev_.data(), ev_.size(), ev_.capacity());

        sj::ondemand::document doc;
        if (parser_.iterate(view).get(doc) != sj::SUCCESS) return false;

        std::string_view type;
        if (doc["type"].get_string().get(type) != sj::SUCCESS) return false;

        // Ordered by frequency AND by document field order: On-Demand is
        // forward-only, so out-of-order field access costs a rewind.
        if (type == "content_block_delta") {
            sj::ondemand::object delta;
            if (doc["delta"].get_object().get(delta) != sj::SUCCESS) return true;
            std::string_view dtype;
            if (delta["type"].get_string().get(dtype) != sj::SUCCESS) return true;
            if (dtype != "text_delta") return true;  // thinking_delta, input_json_delta
            std::string_view text;
            if (delta["text"].get_string().get(text) == sj::SUCCESS) {
                sink_.on_text_delta(text);
            }
            return true;
        }

        if (type == "message_delta") {
            sj::ondemand::object delta;
            if (doc["delta"].get_object().get(delta) == sj::SUCCESS) {
                std::string_view stop;
                if (delta["stop_reason"].get_string().get(stop) == sj::SUCCESS) {
                    sink_.on_stop(stop);
                }
            }
            sj::ondemand::object usage;
            if (doc["usage"].get_object().get(usage) == sj::SUCCESS) {
                sink_.on_usage(0, u32(usage, "output_tokens"), 0, 0);
            }
            return true;
        }

        if (type == "message_start") {
            sj::ondemand::object msg;
            if (doc["message"].get_object().get(msg) != sj::SUCCESS) return true;
            sj::ondemand::object usage;
            if (msg["usage"].get_object().get(usage) == sj::SUCCESS) {
                sink_.on_usage(u32(usage, "input_tokens"), u32(usage, "output_tokens"),
                               u32(usage, "cache_creation_input_tokens"),
                               u32(usage, "cache_read_input_tokens"));
            }
            return true;
        }

        if (type == "content_block_start") {
            // A `fallback` block marks a server-side model switch. Content
            // already streamed is never invalidated -- this is an audit marker.
            sj::ondemand::object cb;
            if (doc["content_block"].get_object().get(cb) != sj::SUCCESS) return true;
            std::string_view ctype;
            if (cb["type"].get_string().get(ctype) != sj::SUCCESS) return true;
            if (ctype == "fallback") {
                sink_.on_fallback(nested_model(cb, "from"), nested_model(cb, "to"));
            }
            return true;
        }

        if (type == "error") {
            // HTTP 200 + an error event. overloaded_error / api_error land here,
            // long after the status line was read, so they MUST be parsed out of
            // the body -- the transport reports success.
            sj::ondemand::object err;
            if (doc["error"].get_object().get(err) != sj::SUCCESS) {
                sink_.on_api_error("unknown", "");
                return true;
            }
            std::string_view etype, emsg;
            (void)err["type"].get_string().get(etype);
            (void)err["message"].get_string().get(emsg);
            sink_.on_api_error(etype, emsg);
            return true;
        }

        // ping / content_block_stop / message_stop -- nothing to extract.
        return true;
    }

    static uint32_t u32(simdjson::ondemand::object& obj, const char* key) {
        uint64_t v = 0;
        if (obj[key].get_uint64().get(v) != simdjson::SUCCESS) return 0;
        return static_cast<uint32_t>(v);
    }

    static std::string_view nested_model(simdjson::ondemand::object& cb, const char* key) {
        simdjson::ondemand::object o;
        if (cb[key].get_object().get(o) != simdjson::SUCCESS) return {};
        std::string_view m;
        if (o["model"].get_string().get(m) != simdjson::SUCCESS) return {};
        return m;
    }

    Sink& sink_;
    simdjson::ondemand::parser parser_;  // reused: owns the tape + string buffers
    std::string line_;                   // partial line carried across chunks
    std::string ev_;                     // accumulated `data:` payload, padded
};

}  // namespace blackwell::cloud::detail
