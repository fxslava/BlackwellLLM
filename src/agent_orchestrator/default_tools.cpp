#include "default_tools.h"

#include <string>

namespace agent::orch {
namespace {

// Pull a named arg, falling back to the raw element body when the model emitted
// content without wrapping it in <arg>. Returns "" when neither is present.
std::string arg_or_body(const ToolInvocation& c, const std::string& key) {
    auto it = c.args.find(key);
    if (it != c.args.end()) return it->second;
    return c.body;
}

std::string arg(const ToolInvocation& c, const std::string& key,
                const std::string& fallback = {}) {
    auto it = c.args.find(key);
    return it != c.args.end() ? it->second : fallback;
}

}  // namespace

void register_fs_tools(ToolRegistry& registry, env::SandboxFs& fs) {
    registry.register_tool(
        "read_file",
        "Read a whole file. arg: path (sandbox-relative).",
        [&fs](const ToolInvocation& c) -> std::string {
            const std::string path = arg(c, "path");
            if (path.empty()) return "ERROR: read_file requires a 'path' arg.";
            auto content = fs.read(path);
            if (!content) return "ERROR: cannot read '" + path + "'.";
            return *content;
        });

    registry.register_tool(
        "write_file",
        "Create/overwrite a file. args: path, content (or element body).",
        [&fs](const ToolInvocation& c) -> std::string {
            const std::string path = arg(c, "path");
            if (path.empty()) return "ERROR: write_file requires a 'path' arg.";
            const std::string content = arg_or_body(c, "content");
            if (!fs.write(path, content))
                return "ERROR: failed to write '" + path + "' (unsafe path?).";
            return "OK: wrote " + std::to_string(content.size()) +
                   " bytes to '" + path + "'.";
        });

    registry.register_tool(
        "patch_file",
        "Replace first occurrence of 'find' with 'replace'. args: path, find, replace.",
        [&fs](const ToolInvocation& c) -> std::string {
            const std::string path = arg(c, "path");
            if (path.empty()) return "ERROR: patch_file requires a 'path' arg.";
            if (c.args.find("find") == c.args.end())
                return "ERROR: patch_file requires a 'find' arg.";
            if (!fs.patch(path, arg(c, "find"), arg(c, "replace")))
                return "ERROR: patch of '" + path +
                       "' failed (missing file, unsafe path, or 'find' not present).";
            return "OK: patched '" + path + "'.";
        });

    registry.register_tool(
        "list_dir",
        "List a sandbox directory (non-recursive). arg: path (default '.').",
        [&fs](const ToolInvocation& c) -> std::string {
            const std::string path = arg(c, "path", ".");
            auto entries = fs.list(path);
            if (entries.empty()) return "(empty or not a directory): " + path;
            std::string out;
            for (const auto& e : entries) {
                out += e;
                out += '\n';
            }
            if (!out.empty()) out.pop_back();
            return out;
        });
}

void register_code_tools(ToolRegistry& registry, ISemanticProvider& provider) {
    registry.register_tool(
        "analyze_source",
        "(Re)build the code graph from C++ source. arg: source (or element body).",
        [&provider](const ToolInvocation& c) -> std::string {
            const std::string src = arg_or_body(c, "source");
            if (src.empty()) return "ERROR: analyze_source needs 'source'.";
            if (!provider.analyze(src)) return "ERROR: source failed to parse.";
            const auto syms = provider.symbols();
            std::string out = "OK: analyzed. " + std::to_string(syms.size()) +
                              " symbol(s):";
            for (const auto& s : syms) out += " " + s;
            return out;
        });

    registry.register_tool(
        "callers_of",
        "List functions that call a symbol. arg: symbol. Run analyze_source first.",
        [&provider](const ToolInvocation& c) -> std::string {
            const std::string sym = arg(c, "symbol");
            if (sym.empty()) return "ERROR: callers_of needs 'symbol'.";
            const auto callers = provider.callers_of(sym);
            if (callers.empty()) return "(no known callers of '" + sym + "')";
            std::string out = "callers of '" + sym + "':";
            for (const auto& s : callers) out += " " + s;
            return out;
        });

    registry.register_tool(
        "callees_of",
        "List functions a symbol calls. arg: symbol. Run analyze_source first.",
        [&provider](const ToolInvocation& c) -> std::string {
            const std::string sym = arg(c, "symbol");
            if (sym.empty()) return "ERROR: callees_of needs 'symbol'.";
            const auto callees = provider.callees_of(sym);
            if (callees.empty()) return "(no known callees of '" + sym + "')";
            std::string out = "callees of '" + sym + "':";
            for (const auto& s : callees) out += " " + s;
            return out;
        });
}

}  // namespace agent::orch
