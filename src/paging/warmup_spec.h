#pragma once
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

// ============================================================================
// Warmup DSL — the Text-to-Cache specification the AOT compiler consumes
// ============================================================================
// A warmup spec is a JSON tree of prompt fragments. Each node contributes a
// text segment; a node's FULL PROMPT is the concatenation of segments from
// its root down to it. The AOT warmer prefills every node's full prompt
// (parents first, so each child costs only its own segment — the compiler
// prefix-hits its own output) and ships the resulting KV as .bkv files.
//
//   {
//     "name": "support-agent",
//     "emit": "leaves",                       // or "all" (default)
//     "nodes": [
//       { "id": "base", "text": "You are the BlackwellEngine assistant...",
//         "children": [
//           { "id": "base.en", "text": "Respond in English.",
//             "children": [
//               { "id": "base.en.tools", "text_file": "tools_prompt.md" } ] },
//           { "id": "base.ru", "text": "Отвечай по-русски.", "emit": true } ] }
//     ]
//   }
//
// Node fields:
//   id        (required)  unique across the spec; becomes "<id>.bkv"
//   text | text_file      exactly one; text_file is resolved against the
//                         directory the spec was loaded from
//   children  (optional)  nested nodes
//   emit      (optional)  per-node override of the spec-level emit mode
//
// Host-pure (nlohmann::json only) — parseable and testable without the
// engine; the pipeline that drives prefill lives in aot_cache_warmer.h.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

struct WarmupNode {
    std::string             id;
    std::string             text;    // resolved: text_file content lands here too
    std::optional<bool>     emit;    // overrides WarmupSpec::emit_mode
    std::vector<WarmupNode> children;
};

struct WarmupSpec {
    enum class EmitMode { All, Leaves };
    std::string             name;
    EmitMode                emit_mode = EmitMode::All;
    std::vector<WarmupNode> roots;
};

namespace detail {

inline std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("warmup spec: cannot read text_file " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline WarmupNode parse_node(const nlohmann::json& j, const std::string& base_dir,
                             std::unordered_set<std::string>& seen_ids,
                             const std::string& where) {
    if (!j.is_object())
        throw std::runtime_error("warmup spec: node is not an object at " + where);
    WarmupNode n;
    if (!j.contains("id") || !j["id"].is_string() || j["id"].get<std::string>().empty())
        throw std::runtime_error("warmup spec: missing/empty \"id\" at " + where);
    n.id = j["id"].get<std::string>();
    if (!seen_ids.insert(n.id).second)
        throw std::runtime_error("warmup spec: duplicate id \"" + n.id + "\"");

    const bool has_text = j.contains("text");
    const bool has_file = j.contains("text_file");
    if (has_text == has_file)
        throw std::runtime_error("warmup spec: node \"" + n.id +
                                 "\" needs exactly one of text / text_file");
    n.text = has_text
        ? j["text"].get<std::string>()
        : read_text_file((base_dir.empty() ? std::string()
                                           : base_dir + "/") +
                         j["text_file"].get<std::string>());
    if (n.text.empty())
        throw std::runtime_error("warmup spec: node \"" + n.id + "\" has empty text");

    if (j.contains("emit")) n.emit = j["emit"].get<bool>();
    if (j.contains("children")) {
        if (!j["children"].is_array())
            throw std::runtime_error("warmup spec: \"children\" of \"" + n.id +
                                     "\" is not an array");
        for (size_t i = 0; i < j["children"].size(); ++i)
            n.children.push_back(parse_node(j["children"][i], base_dir, seen_ids,
                                            n.id + ".children[" +
                                            std::to_string(i) + "]"));
    }
    return n;
}

} // namespace detail

// Parse a spec from JSON text. base_dir resolves text_file references (pass
// the directory the spec file came from; empty = text_file paths as-is).
// Throws std::runtime_error with node-path context on any violation.
inline WarmupSpec parse_warmup_spec(const std::string& json_text,
                                    const std::string& base_dir = {}) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("warmup spec: JSON parse error: ") +
                                 e.what());
    }
    WarmupSpec spec;
    spec.name = j.value("name", std::string("warmup"));
    const std::string emit = j.value("emit", std::string("all"));
    if (emit == "all")         spec.emit_mode = WarmupSpec::EmitMode::All;
    else if (emit == "leaves") spec.emit_mode = WarmupSpec::EmitMode::Leaves;
    else throw std::runtime_error("warmup spec: \"emit\" must be all|leaves, got " +
                                  emit);
    if (!j.contains("nodes") || !j["nodes"].is_array() || j["nodes"].empty())
        throw std::runtime_error("warmup spec: \"nodes\" must be a non-empty array");
    std::unordered_set<std::string> ids;
    for (size_t i = 0; i < j["nodes"].size(); ++i)
        spec.roots.push_back(detail::parse_node(j["nodes"][i], base_dir, ids,
                                                "nodes[" + std::to_string(i) + "]"));
    return spec;
}

}} // namespace blackwell::paging
