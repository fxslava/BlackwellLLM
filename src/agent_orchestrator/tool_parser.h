// ToolParser: turns a blob of raw LLM text into at most one structured action.
//
// We parse an XML-tag dialect rather than JSON on purpose: LLMs that are busy
// emitting C++ inside an argument routinely break JSON string escaping (stray
// quotes, backslashes, newlines), whereas an `<arg>...</arg>` body can hold
// arbitrary bytes -- including `"` and `\` -- without any escaping at all.
//
// The single hard requirement here is that parsing is COMPLETELY crash-proof:
//   * static, pure, operates on a string_view, and NEVER throws;
//   * every find() is checked against npos before any substr(), so no malformed
//     or truncated tag can ever drive an out-of-range read;
//   * surrounding conversational prose ("yapping") is ignored -- we lock onto the
//     earliest *valid* action and discard the rest;
//   * anything we cannot make sense of degrades to ActionKind::None instead of
//     an exception, so the ReAct loop always gets a well-formed result.
#ifndef BLACKWELL_AGENT_ORCH_TOOL_PARSER_H
#define BLACKWELL_AGENT_ORCH_TOOL_PARSER_H

#include <map>
#include <string>
#include <string_view>

namespace agent::orch {

// A single decoded tool request. `args` holds the <arg name="k">v</arg> pairs;
// `body` is the raw inner text of the <tool_call> element, exposed verbatim for
// tools that would rather slurp the whole payload than read named arguments.
struct ToolInvocation {
    std::string name;
    std::map<std::string, std::string> args;
    std::string body;
};

enum class ActionKind {
    None,      // no actionable tag found (pure prose, or unparseable)
    ToolCall,  // a <tool_call ...> the dispatcher should run
    Finish,    // a <finish> tag -- the agent is done
};

struct ParsedAction {
    ActionKind kind = ActionKind::None;
    ToolInvocation tool;       // valid iff kind == ToolCall
    std::string finish_text;   // valid iff kind == Finish (trimmed)
};

class ToolParser {
public:
    // Scan `text` and return the earliest valid action, ignoring everything
    // around it. Never throws.
    static ParsedAction parse(std::string_view text);
};

}  // namespace agent::orch

#endif  // BLACKWELL_AGENT_ORCH_TOOL_PARSER_H
