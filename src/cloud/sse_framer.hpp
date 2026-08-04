#pragma once
// =============================================================================
// cloud/sse_framer.hpp — text/event-stream FRAMING, and nothing else.
//
// Extracted from claude_sse.hpp when a second protocol (OpenAI-compatible
// chat/completions) arrived. Framing is where the bugs live and it is identical
// for every SSE endpoint: strip `\r`, join multi-line `data:`, ignore comments,
// flush on a blank line. The part that differs between vendors is only what the
// accumulated JSON MEANS, which is the EventParser's job.
//
// Duplicating this per protocol would have meant two copies of the one function
// whose failure mode survives to production -- see the invariant below -- and
// only one of them would keep getting the byte-at-a-time regression test.
//
// NO simdjson HERE. This header is pure framing, so it costs nothing to include
// and can be tested without a JSON parser at all.
//
// EventParser contract (compile-time; no vtable on the per-event path):
//     bool parse_event(std::string& payload);
//   `payload` is the accumulated `data:` bytes for ONE event, with the trailing
//   newline already stripped. It is passed by NON-const reference because a
//   simdjson-based parser must be able to over-reserve it for SIMDJSON_PADDING.
//   Returning false is unrecoverable and aborts the transfer.
//
// INVARIANT: feed() must tolerate arbitrary split points. TCP will split a
// `data:` line mid-JSON, mid-UTF-8, and between the \r and the \n.
// =============================================================================
#include <cstring>
#include <string>
#include <string_view>

namespace blackwell::cloud::detail {

template <class EventParser>
class SseFramer {
public:
    explicit SseFramer(EventParser& parser) : parser_(parser) {
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
        // `event:` / `id:` / `retry:` deliberately ignored -- on both endpoints
        // we speak, the JSON payload is self-describing and always present.
        return true;
    }

    [[nodiscard]] bool dispatch() {
        if (ev_.empty()) return true;
        const bool ok = parser_.parse_event(ev_);
        ev_.clear();
        return ok;
    }

    EventParser& parser_;
    std::string  line_;  // partial line carried across chunks
    std::string  ev_;    // accumulated `data:` payload for the current event
};

}  // namespace blackwell::cloud::detail
