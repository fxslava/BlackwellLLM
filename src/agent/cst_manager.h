// CstManager: the syntax engine.
//
// Thin RAII wrapper over tree-sitter + the tree-sitter-cpp grammar. It owns the
// source buffer and the parse tree together so that node byte-ranges always
// refer to the buffer currently held. Because edits are applied by splicing the
// raw byte range of a single node, *all* surrounding whitespace, comments and
// macros are preserved verbatim -- which is the whole point for safe
// source-to-source translation.
#ifndef BLACKWELL_AGENT_CST_MANAGER_H
#define BLACKWELL_AGENT_CST_MANAGER_H

#include <string>

#include <tree_sitter/api.h>

namespace agent {

class CstManager {
public:
    CstManager();
    ~CstManager();

    CstManager(const CstManager&) = delete;
    CstManager& operator=(const CstManager&) = delete;

    // Parse `source` from scratch. Returns false if tree-sitter could not be
    // initialised with the C++ grammar. Takes ownership of the buffer.
    bool parse(std::string source);

    bool valid() const { return tree_ != nullptr; }
    const std::string& source() const { return source_; }
    TSNode root() const;

    // Text covered by `node` in the current source buffer.
    std::string node_text(TSNode node) const;

    // Atomic, formatting-preserving replacement: swap the exact byte range of
    // `node` for `replacement`, then reparse. Returns false if there is no live
    // tree. Everything outside the node's range is left byte-for-byte intact.
    bool replace_node(TSNode node, const std::string& replacement);

private:
    void reparse();

    TSParser* parser_ = nullptr;
    TSTree* tree_ = nullptr;
    std::string source_;
};

}  // namespace agent

#endif  // BLACKWELL_AGENT_CST_MANAGER_H
