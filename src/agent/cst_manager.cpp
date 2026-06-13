#include "cst_manager.h"

// Provided by the compiled tree-sitter-cpp grammar (parser.c).
extern "C" const TSLanguage* tree_sitter_cpp(void);

namespace agent {

CstManager::CstManager() {
    parser_ = ts_parser_new();
    // If this returns false the grammar ABI is incompatible with the runtime;
    // callers detect it via valid() staying false after parse().
    ts_parser_set_language(parser_, tree_sitter_cpp());
}

CstManager::~CstManager() {
    if (tree_) ts_tree_delete(tree_);
    if (parser_) ts_parser_delete(parser_);
}

bool CstManager::parse(std::string source) {
    source_ = std::move(source);
    reparse();
    return tree_ != nullptr;
}

void CstManager::reparse() {
    if (tree_) {
        ts_tree_delete(tree_);
        tree_ = nullptr;
    }
    // Fresh parse against the current buffer. We intentionally do not reuse the
    // old tree for incremental reparsing: a full reparse is simple and exact,
    // and the byte-accurate splice in replace_node already guarantees perfect
    // text preservation, which is what actually matters for the agent.
    tree_ = ts_parser_parse_string(
        parser_, /*old_tree=*/nullptr, source_.c_str(),
        static_cast<uint32_t>(source_.size()));
}

TSNode CstManager::root() const { return ts_tree_root_node(tree_); }

std::string CstManager::node_text(TSNode node) const {
    const uint32_t start = ts_node_start_byte(node);
    const uint32_t end = ts_node_end_byte(node);
    if (end <= start || end > source_.size()) return {};
    return source_.substr(start, end - start);
}

bool CstManager::replace_node(TSNode node, const std::string& replacement) {
    if (!tree_) return false;
    const uint32_t start = ts_node_start_byte(node);
    const uint32_t end = ts_node_end_byte(node);
    if (start > end || end > source_.size()) return false;

    std::string edited;
    edited.reserve(source_.size() - (end - start) + replacement.size());
    edited.append(source_, 0, start);
    edited.append(replacement);
    edited.append(source_, end, source_.size() - end);

    source_ = std::move(edited);
    reparse();
    return tree_ != nullptr;
}

}  // namespace agent
