#include "tool_registry.h"

#include <exception>

namespace agent::orch {

void ToolRegistry::register_tool(std::string name, std::string description,
                                 ToolCallback fn) {
    tools_[std::move(name)] = Entry{std::move(description), std::move(fn)};
}

bool ToolRegistry::has(const std::string& name) const {
    return tools_.find(name) != tools_.end();
}

std::string ToolRegistry::dispatch(const ToolInvocation& call) const {
    auto it = tools_.find(call.name);
    if (it == tools_.end()) {
        std::string msg = "ERROR: tool '" + call.name +
                          "' not found. Available tools:";
        if (tools_.empty()) {
            msg += " (none registered)";
        } else {
            for (const auto& [name, entry] : tools_) msg += " " + name;
        }
        return msg;
    }

    // The single choke point where third-party callback code runs. A tool is
    // supposed to report failure as text, but if it lets an exception escape we
    // convert it to an observation rather than letting it kill the agent loop.
    try {
        if (!it->second.fn) return "ERROR: tool '" + call.name + "' is null.";
        return it->second.fn(call);
    } catch (const std::exception& e) {
        return "ERROR: tool '" + call.name + "' threw: " + e.what();
    } catch (...) {
        return "ERROR: tool '" + call.name + "' threw an unknown exception.";
    }
}

std::vector<std::pair<std::string, std::string>> ToolRegistry::manifest() const {
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(tools_.size());
    for (const auto& [name, entry] : tools_) out.emplace_back(name, entry.description);
    return out;
}

}  // namespace agent::orch
