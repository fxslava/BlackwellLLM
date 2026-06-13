// default_tools: wires concrete agent_env / agent_core capabilities into a
// ToolRegistry, giving the LLM safe file + code-graph "hands".
//
// This is the one place the orchestrator depends on the two otherwise-isolated
// subsystems at once:
//   * agent_env::SandboxFs  -- path-confined read / write / patch / list, so a
//     hallucinated "../../Windows/System32" can never escape the sandbox.
//   * agent::ISemanticProvider -- the shadow-graph queries (callers/callees of a
//     symbol) built from the tree-sitter CST.
//
// Tools report failure as ordinary "ERROR: ..." text (the ReAct contract) rather
// than throwing; the dispatcher would contain a throw anyway, but keeping it
// textual lets the model read the reason and self-correct.
#ifndef BLACKWELL_AGENT_ORCH_DEFAULT_TOOLS_H
#define BLACKWELL_AGENT_ORCH_DEFAULT_TOOLS_H

#include "../agent/semantic_engine.h"
#include "../agent_env/sandbox_fs.h"
#include "tool_registry.h"

namespace agent::orch {

// Register filesystem tools (read_file, write_file, patch_file, list_dir)
// backed by `fs`. `fs` must outlive the registry / any run that uses it.
void register_fs_tools(ToolRegistry& registry, env::SandboxFs& fs);

// Register code-analysis tools (analyze_source, callers_of, callees_of) backed
// by `provider`. `provider` must outlive the registry / any run that uses it.
void register_code_tools(ToolRegistry& registry, ISemanticProvider& provider);

}  // namespace agent::orch

#endif  // BLACKWELL_AGENT_ORCH_DEFAULT_TOOLS_H
