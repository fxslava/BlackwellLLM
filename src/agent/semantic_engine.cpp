#include "semantic_engine.h"

#include <algorithm>

namespace agent {
namespace {

std::string node_text(TSNode node, const std::string& src) {
    const uint32_t start = ts_node_start_byte(node);
    const uint32_t end = ts_node_end_byte(node);
    if (end <= start || end > src.size()) return {};
    return src.substr(start, end - start);
}

bool type_is(TSNode node, const char* type) {
    const char* t = ts_node_type(node);
    return t && std::string(type) == t;
}

// Pre-order search for the first descendant (or self) of the given node type.
TSNode find_first_of_type(TSNode node, const char* type) {
    if (ts_node_is_null(node)) return node;
    if (type_is(node, type)) return node;
    const uint32_t n = ts_node_child_count(node);
    for (uint32_t i = 0; i < n; ++i) {
        TSNode found = find_first_of_type(ts_node_child(node, i), type);
        if (!ts_node_is_null(found)) return found;
    }
    return TSNode{};  // null node (context == nullptr)
}

void push_unique(std::vector<std::string>& v, const std::string& s) {
    if (!s.empty() && std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
}

// Collect callees (called function names) and referenced type names within the
// subtree rooted at `node`, appending into the given SemanticNode.
void collect_dependencies(TSNode node, const std::string& src, SemanticNode& out) {
    const char* t = ts_node_type(node);
    if (t) {
        if (std::string("call_expression") == t) {
            TSNode callee = ts_node_child_by_field_name(node, "function", 8);
            if (!ts_node_is_null(callee)) {
                // For a plain call the "function" child is an identifier; for a
                // member call (a.b()) take the trailing field_identifier.
                TSNode name = callee;
                if (type_is(callee, "field_expression")) {
                    TSNode f = ts_node_child_by_field_name(callee, "field", 5);
                    if (!ts_node_is_null(f)) name = f;
                }
                if (type_is(name, "identifier") || type_is(name, "field_identifier")) {
                    push_unique(out.callees, node_text(name, src));
                }
            }
        } else if (std::string("type_identifier") == t) {
            push_unique(out.type_dependencies, node_text(node, src));
        }
    }
    const uint32_t n = ts_node_child_count(node);
    for (uint32_t i = 0; i < n; ++i)
        collect_dependencies(ts_node_child(node, i), src, out);
}

std::string function_name(TSNode fn_def, const std::string& src) {
    TSNode decl = ts_node_child_by_field_name(fn_def, "declarator", 10);
    if (ts_node_is_null(decl)) return {};
    TSNode fdecl = type_is(decl, "function_declarator")
                       ? decl
                       : find_first_of_type(decl, "function_declarator");
    if (ts_node_is_null(fdecl)) return {};
    TSNode name = ts_node_child_by_field_name(fdecl, "declarator", 10);
    if (ts_node_is_null(name)) return {};
    // qualified_identifier (Foo::bar) -> take the trailing identifier.
    if (type_is(name, "qualified_identifier")) {
        TSNode last = find_first_of_type(name, "identifier");
        const uint32_t n = ts_node_named_child_count(name);
        for (uint32_t i = 0; i < n; ++i) {
            TSNode c = ts_node_named_child(name, i);
            if (type_is(c, "identifier")) last = c;
        }
        if (!ts_node_is_null(last)) name = last;
    }
    return node_text(name, src);
}

// Walk the whole tree, registering one SemanticNode per function_definition.
void collect_functions(TSNode node, const std::string& src,
                       std::unordered_map<std::string, SemanticNode>& out) {
    if (type_is(node, "function_definition")) {
        SemanticNode sn;
        sn.kind = SemanticKind::Function;
        sn.name = function_name(node, src);
        sn.cst_type = ts_node_type(node);
        TSPoint p = ts_node_start_point(node);
        sn.range = SourceRange{ts_node_start_byte(node), ts_node_end_byte(node),
                                p.row, p.column};
        if (!sn.name.empty()) {
            collect_dependencies(node, src, sn);
            // Drop a self-recursive callee edge onto the name itself but keep it
            // out of the "drop the definition name" duplicate handling below.
            out[sn.name] = std::move(sn);
        }
    }
    const uint32_t n = ts_node_child_count(node);
    for (uint32_t i = 0; i < n; ++i)
        collect_functions(ts_node_child(node, i), src, out);
}

}  // namespace

bool TreeSitterSemanticProvider::analyze(std::string source) {
    nodes_.clear();
    if (!cst_.parse(std::move(source))) return false;

    const std::string& src = cst_.source();
    collect_functions(cst_.root(), src, nodes_);

    // Second pass: wire callers from the callee edges. Only edges that resolve
    // to a known defined function become caller relationships.
    for (const auto& [caller, node] : nodes_) {
        for (const std::string& callee : node.callees) {
            auto it = nodes_.find(callee);
            if (it != nodes_.end()) push_unique(it->second.callers, caller);
        }
    }
    return true;
}

std::optional<SemanticNode> TreeSitterSemanticProvider::node_for(
    const std::string& symbol) const {
    auto it = nodes_.find(symbol);
    if (it == nodes_.end()) return std::nullopt;
    return it->second;
}

std::vector<std::string> TreeSitterSemanticProvider::callers_of(
    const std::string& symbol) const {
    auto it = nodes_.find(symbol);
    return it == nodes_.end() ? std::vector<std::string>{} : it->second.callers;
}

std::vector<std::string> TreeSitterSemanticProvider::callees_of(
    const std::string& symbol) const {
    auto it = nodes_.find(symbol);
    return it == nodes_.end() ? std::vector<std::string>{} : it->second.callees;
}

std::vector<std::string> TreeSitterSemanticProvider::symbols() const {
    std::vector<std::string> out;
    out.reserve(nodes_.size());
    for (const auto& [name, _] : nodes_) out.push_back(name);
    return out;
}

}  // namespace agent
