// ToolRegistry: the dispatcher half of the ReAct loop.
//
// Maps a string tool name to a C++ callback (std::function) and turns a parsed
// ToolInvocation into a textual observation to feed back to the model. The whole
// point is that dispatch() is TOTAL -- it always returns a string and never
// throws -- so neither a hallucinated tool name nor a buggy/throwing callback
// can ever unwind out of the agent loop. Both failure modes become an ordinary
// "ERROR: ..." observation that the LLM can read and recover from.
#ifndef BLACKWELL_AGENT_ORCH_TOOL_REGISTRY_H
#define BLACKWELL_AGENT_ORCH_TOOL_REGISTRY_H

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tool_parser.h"

namespace agent::orch {

// A tool is any callable that maps a decoded invocation to an observation. It
// SHOULD return errors as plain text, but is allowed to throw -- the registry
// contains the blast radius either way.
using ToolCallback = std::function<std::string(const ToolInvocation&)>;

class ToolRegistry {
public:
    // Register (or replace) a tool. `description` is surfaced to the model in the
    // system prompt / "tool not found" observations so it can self-correct.
    void register_tool(std::string name, std::string description, ToolCallback fn);

    bool has(const std::string& name) const;

    // Run the named tool and return its observation. Never throws:
    //   * unknown name        -> "ERROR: tool '<n>' not found. Available: ..."
    //   * callback throws      -> "ERROR: tool '<n>' threw: <what>"
    std::string dispatch(const ToolInvocation& call) const;

    // (name, description) for every registered tool, in name order. Handy for
    // building the tool manifest portion of a system prompt.
    std::vector<std::pair<std::string, std::string>> manifest() const;

private:
    struct Entry {
        std::string description;
        ToolCallback fn;
    };
    std::map<std::string, Entry> tools_;
};

}  // namespace agent::orch

#endif  // BLACKWELL_AGENT_ORCH_TOOL_REGISTRY_H
