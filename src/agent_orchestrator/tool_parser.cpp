#include "tool_parser.h"

#include <array>
#include <cctype>
#include <optional>

namespace agent::orch {
namespace {

constexpr size_t kNpos = std::string_view::npos;

bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

// A valid element name must be followed by a delimiter so that a search for
// "<finish" does not latch onto "<finished_state". The delimiter is whitespace,
// the tag close '>' or a self-close '/'.
bool is_name_delim(char c) { return c == '>' || c == '/' || is_space(c); }

// Locate "<tag" as a whole element opener at or after `from`. Returns the index
// of '<', or kNpos. Bounded: only touches indices proven in-range.
size_t find_element(std::string_view text, std::string_view tag, size_t from) {
    std::string needle = "<";
    needle.append(tag);
    size_t pos = from;
    while ((pos = text.find(needle, pos)) != kNpos) {
        size_t after = pos + needle.size();
        if (after >= text.size() || is_name_delim(text[after])) return pos;
        pos = after;  // "<tagxxx" -- not our element, keep scanning
    }
    return kNpos;
}

// Pull an attribute value out of an open tag slice (from '<' through '>'),
// tolerating single or double quotes and arbitrary surrounding whitespace.
// Returns nullopt if the attribute is absent or its quoting is unterminated.
std::optional<std::string> extract_attr(std::string_view open_tag,
                                        std::string_view attr) {
    size_t p = 0;
    while ((p = open_tag.find(attr, p)) != kNpos) {
        size_t after = p + attr.size();
        // Left boundary: start-of-slice or preceded by whitespace, so "name"
        // does not match the tail of some other attribute like "xname".
        bool left_ok = (p == 0) || is_space(open_tag[p - 1]);
        size_t q = after;
        while (q < open_tag.size() && is_space(open_tag[q])) ++q;
        if (left_ok && q < open_tag.size() && open_tag[q] == '=') {
            ++q;
            while (q < open_tag.size() && is_space(open_tag[q])) ++q;
            if (q < open_tag.size() && (open_tag[q] == '"' || open_tag[q] == '\'')) {
                char quote = open_tag[q++];
                size_t end = open_tag.find(quote, q);
                if (end == kNpos) return std::nullopt;  // unterminated -> give up
                return std::string(open_tag.substr(q, end - q));
            }
        }
        p = after;
    }
    return std::nullopt;
}

// True if index `pos` lies inside a <think>...</think> span. Reasoning models
// (Qwen-3.5, DeepSeek-R1) emit a chain-of-thought there that frequently *mentions*
// or rehearses protocol tags ("I should emit <finish> once done", "maybe call
// <tool_call ...>"); none of those are real actions, so the scanner must skip
// anything a think block encloses. A <think> with no matching </think> (the model
// is still reasoning) swallows the entire tail, so we never act on half-formed
// thoughts. Bounded: every index touched is proven in-range before use.
bool in_think_block(std::string_view text, size_t pos) {
    size_t scan = 0;
    while (true) {
        size_t open = find_element(text, "think", scan);
        if (open == kNpos || open >= pos) return false;  // no think opener before pos
        size_t open_end = text.find('>', open);
        if (open_end == kNpos) return true;  // "<think" never closes its tag -> all tail is thought
        size_t close = text.find("</think>", open_end + 1);
        if (close == kNpos) return true;     // open-ended think -> swallows the tail (incl. pos)
        size_t close_end = close + 8;        // strlen("</think>")
        if (pos < close_end) return true;    // pos sits between this <think> and its close
        scan = close_end;                    // pos is past this block; look for a later one
    }
}

// Earliest opener of `tag` that is NOT buried inside a think block, or kNpos.
size_t find_action_element(std::string_view text, std::string_view tag) {
    size_t pos = 0;
    while ((pos = find_element(text, tag, pos)) != kNpos) {
        if (!in_think_block(text, pos)) return pos;
        pos += 1;  // this opener is part of the model's thoughts; resume past its '<'
    }
    return kNpos;
}

std::string trim(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) ++b;
    while (e > b && is_space(s[e - 1])) --e;
    return std::string(s.substr(b, e - b));
}

// Parse <arg name="k">v</arg> children out of a tool_call body. Lenient: stops
// cleanly at the first malformed child rather than throwing.
void parse_args(std::string_view body, std::map<std::string, std::string>& out) {
    size_t p = 0;
    while ((p = find_element(body, "arg", p)) != kNpos) {
        size_t open_end = body.find('>', p);
        if (open_end == kNpos) break;
        auto key = extract_attr(body.substr(p, open_end - p + 1), "name");
        size_t vstart = open_end + 1;
        size_t close = body.find("</arg>", vstart);
        if (close == kNpos) break;
        if (key) out[*key] = std::string(body.substr(vstart, close - vstart));
        p = close + 6;  // strlen("</arg>")
    }
}

// Attempt to read a <tool_call ...> ... </tool_call> starting at `pos`.
std::optional<ParsedAction> try_tool_call(std::string_view text, size_t pos) {
    size_t open_end = text.find('>', pos);
    if (open_end == kNpos) return std::nullopt;
    std::string_view open_tag = text.substr(pos, open_end - pos + 1);

    auto name = extract_attr(open_tag, "name");
    if (!name || name->empty()) return std::nullopt;  // a tool call needs a name

    ParsedAction action;
    action.kind = ActionKind::ToolCall;
    action.tool.name = *name;

    // Self-closing form: <tool_call name="x"/> -- valid, simply no body/args.
    if (open_end > pos && text[open_end - 1] == '/') return action;

    size_t close = text.find("</tool_call>", open_end + 1);
    if (close == kNpos) return std::nullopt;  // truncated -> not yet actionable

    std::string_view body = text.substr(open_end + 1, close - (open_end + 1));
    action.tool.body = std::string(body);
    parse_args(body, action.tool.args);
    return action;
}

// Attempt to read a <finish> ... </finish> starting at `pos`. Lenient about the
// closing tag: a model that forgets </finish> should still be allowed to stop.
std::optional<ParsedAction> try_finish(std::string_view text, size_t pos) {
    size_t open_end = text.find('>', pos);
    if (open_end == kNpos) return std::nullopt;

    ParsedAction action;
    action.kind = ActionKind::Finish;

    if (open_end > pos && text[open_end - 1] == '/') return action;  // <finish/>

    size_t close = text.find("</finish>", open_end + 1);
    size_t body_end = (close == kNpos) ? text.size() : close;
    action.finish_text = trim(text.substr(open_end + 1, body_end - (open_end + 1)));
    return action;
}

}  // namespace

ParsedAction ToolParser::parse(std::string_view text) {
    // Collect both candidate openers, then resolve in document order, trying the
    // earliest first and falling back to the other if the earliest is malformed.
    // This is what lets a half-written <tool_call> not shadow a valid <finish>.
    struct Candidate {
        size_t pos;
        std::optional<ParsedAction> (*fn)(std::string_view, size_t);
    };
    // find_action_element skips openers buried inside <think>...</think>, so the
    // model's chain-of-thought can rehearse <finish>/<tool_call> without tripping
    // the loop; only a tag in the *real* answer (after </think>) counts.
    std::array<Candidate, 2> cands{
        Candidate{find_action_element(text, "tool_call"), &try_tool_call},
        Candidate{find_action_element(text, "finish"), &try_finish},
    };

    // Order by position (earliest first); kNpos sinks to the back.
    if (cands[1].pos < cands[0].pos) std::swap(cands[0], cands[1]);

    for (const auto& c : cands) {
        if (c.pos == kNpos) continue;
        if (auto a = c.fn(text, c.pos)) return *a;
    }
    return ParsedAction{};  // ActionKind::None
}

}  // namespace agent::orch
