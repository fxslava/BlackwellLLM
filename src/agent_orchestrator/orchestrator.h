// AgentOrchestrator: the central ReAct (Reason + Act) state machine.
//
// It owns the conversation history and drives the loop that ties the LLM
// (ILLMGenerator) to the tool layer (ToolRegistry -> agent_core / agent_env):
//
//     user prompt
//        |
//        v
//   [render transcript] --> ILLMGenerator.generate() --> raw text
//        ^                                                   |
//        |                                                   v
//   append observation <-- ToolRegistry.dispatch() <-- ToolParser.parse()
//                                                           |
//                                                  <finish> => return
//
// Design notes:
//   * Communication with the model is pure text + robust string parsing; we do
//     NOT do CUDA-level constrained decoding, so the agent stays model-agnostic.
//   * A MAX_ITERATIONS cap guarantees termination even if the model never emits
//     <finish> and never converges.
//   * The loop is exception-safe end to end: parsing never throws and dispatch
//     never throws, so a run can only end by <finish> or by hitting the cap.
#ifndef BLACKWELL_AGENT_ORCH_ORCHESTRATOR_H
#define BLACKWELL_AGENT_ORCH_ORCHESTRATOR_H

#include <string>
#include <vector>

#include "llm_generator.h"
#include "tool_registry.h"

namespace agent::orch {

// One entry in the running transcript. Observations are tool results fed back to
// the model; they are a distinct role so the model can tell its own words apart
// from the environment's replies.
struct Message {
    enum class Role { System, User, Assistant, Observation };
    Role role;
    std::string content;
};

struct OrchestratorConfig {
    int max_iterations = 12;     // hard safeguard against infinite loops
    std::string system_prompt;   // optional; injected once at the front
};

struct RunResult {
    bool finished = false;            // true iff the loop ended on a <finish> tag
    bool max_iterations_hit = false;  // true iff we stopped on the cap
    std::string answer;               // <finish> text, or last assistant turn
    int iterations = 0;               // LLM generations actually performed
};

class AgentOrchestrator {
public:
    AgentOrchestrator(ILLMGenerator& llm, ToolRegistry& tools,
                      OrchestratorConfig config = {});

    // Append `user_prompt`, then run the ReAct loop to a <finish> or the
    // iteration cap. History is preserved across calls so run() can be used for
    // multi-turn conversations; call reset() to start fresh.
    RunResult run(const std::string& user_prompt);

    const std::vector<Message>& history() const { return history_; }
    void reset() { history_.clear(); system_injected_ = false; }

    // Render the current transcript exactly as it is handed to the model. Exposed
    // for tests / debugging.
    std::string render_transcript() const;

private:
    ILLMGenerator& llm_;
    ToolRegistry& tools_;
    OrchestratorConfig config_;
    std::vector<Message> history_;
    bool system_injected_ = false;
};

}  // namespace agent::orch

#endif  // BLACKWELL_AGENT_ORCH_ORCHESTRATOR_H
