// Semantic engine: builds the shadow graph (SemanticNode set) from source.
//
// ISemanticProvider is the seam that keeps the backend swappable. The first
// implementation, TreeSitterSemanticProvider, derives the call graph and type
// dependencies purely from the tree-sitter CST -- a heuristic, name-based
// resolution that needs no external toolchain. A future LibclangSemanticProvider
// could implement the same interface to add full type/overload resolution
// without changing anything downstream.
#ifndef BLACKWELL_AGENT_SEMANTIC_ENGINE_H
#define BLACKWELL_AGENT_SEMANTIC_ENGINE_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cst_manager.h"
#include "semantic_node.h"

namespace agent {

class ISemanticProvider {
public:
    virtual ~ISemanticProvider() = default;

    // (Re)build the graph for a translation unit. Returns false on parse error.
    virtual bool analyze(std::string source) = 0;

    // Look up the node for a symbol (e.g. a function name). nullopt if unknown.
    virtual std::optional<SemanticNode> node_for(const std::string& symbol) const = 0;

    // Convenience graph queries.
    virtual std::vector<std::string> callers_of(const std::string& symbol) const = 0;
    virtual std::vector<std::string> callees_of(const std::string& symbol) const = 0;

    // Every symbol with a node in the graph.
    virtual std::vector<std::string> symbols() const = 0;
};

// CST-only provider: no compiler, no LSP, no network. Resolves the call graph
// by matching call-expression callee names against defined function names.
class TreeSitterSemanticProvider : public ISemanticProvider {
public:
    bool analyze(std::string source) override;

    std::optional<SemanticNode> node_for(const std::string& symbol) const override;
    std::vector<std::string> callers_of(const std::string& symbol) const override;
    std::vector<std::string> callees_of(const std::string& symbol) const override;
    std::vector<std::string> symbols() const override;

    // Exposed so callers can perform formatting-preserving edits on the same
    // buffer the graph was built from.
    CstManager& cst() { return cst_; }

private:
    CstManager cst_;
    std::unordered_map<std::string, SemanticNode> nodes_;
};

}  // namespace agent

#endif  // BLACKWELL_AGENT_SEMANTIC_ENGINE_H
