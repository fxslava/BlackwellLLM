// ILLMGenerator: the seam between the ReAct orchestrator and "the model".
//
// The orchestrator talks to the LLM purely as text-in / text-out, so the heavy
// CUDA inference engine never has to be linked (or loaded) just to exercise the
// state machine. A production adapter wraps the real engine; tests inject a
// MockLLM that replays pre-canned completions. This is also what keeps the whole
// orchestrator model-agnostic -- we can swap the underlying architecture without
// touching a line of the loop, the parser or the tools.
#ifndef BLACKWELL_AGENT_ORCH_LLM_GENERATOR_H
#define BLACKWELL_AGENT_ORCH_LLM_GENERATOR_H

#include <string>

namespace agent::orch {

class ILLMGenerator {
public:
    virtual ~ILLMGenerator() = default;

    // Produce the model's next completion given the full rendered transcript.
    // The transcript already carries the system prompt, the user goal and every
    // prior assistant turn + tool observation; the implementation just continues
    // it. Implementations should return plain text -- the orchestrator does all
    // tool-call extraction itself.
    virtual std::string generate(const std::string& transcript) = 0;
};

}  // namespace agent::orch

#endif  // BLACKWELL_AGENT_ORCH_LLM_GENERATOR_H
