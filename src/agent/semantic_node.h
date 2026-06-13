// Shadow-graph data structures for the autonomous code agent.
//
// A SemanticNode unifies two views of a piece of source code:
//   * the *physical* syntax (a byte range + tree-sitter node kind), which is
//     what lets the agent perform formatting-preserving, source-to-source
//     edits via CstManager; and
//   * the *semantic* relationships (call graph + type dependencies) derived by
//     an ISemanticProvider.
//
// It also carries a free-form `scratchpad` the LLM uses to stash its own
// reasoning about the node between planning steps.
#ifndef BLACKWELL_AGENT_SEMANTIC_NODE_H
#define BLACKWELL_AGENT_SEMANTIC_NODE_H

#include <cstdint>
#include <string>
#include <vector>

namespace agent {

// Half-open byte range [start_byte, end_byte) into the owning source buffer,
// plus the 0-based start point for human-facing diagnostics. Byte offsets are
// what CstManager splices on, so edits are exact regardless of encoding width.
struct SourceRange {
    uint32_t start_byte = 0;
    uint32_t end_byte = 0;
    uint32_t start_row = 0;  // 0-based line
    uint32_t start_col = 0;  // 0-based column (in bytes)
};

enum class SemanticKind {
    Unknown,
    Function,
    Type,
    Variable,
};

// One vertex of the shadow graph. Edges are stored as symbol *names* rather
// than pointers so the structure survives reparses/edits that invalidate the
// underlying tree-sitter nodes.
struct SemanticNode {
    SemanticKind kind = SemanticKind::Unknown;
    std::string name;                          // resolved symbol name
    SourceRange range;                         // physical location in source
    std::string cst_type;                      // tree-sitter node type string

    std::vector<std::string> callers;          // functions that call this one
    std::vector<std::string> callees;          // functions this one calls
    std::vector<std::string> type_dependencies;// type names this node references

    std::string scratchpad;                    // LLM internal thoughts
};

}  // namespace agent

#endif  // BLACKWELL_AGENT_SEMANTIC_NODE_H
