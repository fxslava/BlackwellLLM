#include "orchestrator.h"

#include "tool_parser.h"

namespace agent::orch {
namespace {

const char* role_tag(Message::Role r) {
    switch (r) {
        case Message::Role::System:      return "SYSTEM";
        case Message::Role::User:        return "USER";
        case Message::Role::Assistant:   return "ASSISTANT";
        case Message::Role::Observation: return "OBSERVATION";
    }
    return "UNKNOWN";
}

}  // namespace

AgentOrchestrator::AgentOrchestrator(ILLMGenerator& llm, ToolRegistry& tools,
                                     OrchestratorConfig config)
    : llm_(llm), tools_(tools), config_(std::move(config)) {}

std::string AgentOrchestrator::render_transcript() const {
    std::string out;
    for (const auto& m : history_) {
        out += '[';
        out += role_tag(m.role);
        out += "]\n";
        out += m.content;
        out += "\n\n";
    }
    // Trailing cue so the model continues as the assistant rather than echoing a
    // role header of its own.
    out += "[ASSISTANT]\n";
    return out;
}

void AgentOrchestrator::seed_history(std::vector<Message> past_context) {
    history_ = std::move(past_context);
    // A seed that carries its own leading System turn wins over the config
    // prompt; anything else leaves injection to run(), which puts the config
    // prompt at the front so seeded turns always sit behind it.
    system_injected_ =
        !history_.empty() && history_.front().role == Message::Role::System;
}

RunResult AgentOrchestrator::run(const std::string& user_prompt) {
    if (!system_injected_ && !config_.system_prompt.empty()) {
        // Insert at the FRONT, not the back: seed_history() may already have
        // populated the transcript, and the system prompt must precede it.
        history_.insert(history_.begin(),
                        {Message::Role::System, config_.system_prompt});
    }
    // Mark as injected regardless, so an empty system prompt is not retried and
    // a non-empty one is never duplicated across multi-turn run() calls.
    system_injected_ = true;

    history_.push_back({Message::Role::User, user_prompt});

    RunResult result;
    const int max_iters = config_.max_iterations > 0 ? config_.max_iterations : 1;

    for (int i = 0; i < max_iters; ++i) {
        const std::string raw = llm_.generate(render_transcript());
        result.iterations = i + 1;
        history_.push_back({Message::Role::Assistant, raw});

        const ParsedAction action = ToolParser::parse(raw);

        if (action.kind == ActionKind::Finish) {
            result.finished = true;
            result.answer = action.finish_text;
            return result;
        }

        if (action.kind == ActionKind::ToolCall) {
            // dispatch() is total and never throws; the observation is always a
            // usable string (a tool result or an "ERROR: ..." the model can fix).
            const std::string observation = tools_.dispatch(action.tool);
            history_.push_back({Message::Role::Observation, observation});

            // Act-then-finish: honor a <finish> in the remainder of this SAME
            // completion, so "one tool call, then finish" costs one generation.
            // Only when the finish is the remainder's EARLIEST action, though:
            // a second <tool_call> ahead of it is a protocol violation under
            // the one-tool-per-iteration contract, so it is not dispatched and
            // the finish behind it is not trusted either -- the model gets the
            // observation and must produce a fresh turn. The re-parse is as
            // crash-proof as the first pass (same parser), and <think> spans in
            // the remainder still neutralize rehearsed tags.
            const ParsedAction tail =
                ToolParser::parse(std::string_view(raw).substr(action.end));
            if (tail.kind == ActionKind::Finish) {
                result.finished = true;
                result.answer = tail.finish_text;
                return result;
            }
            continue;
        }

        // No actionable tag: the model produced only prose. Nudge it back onto
        // the protocol and spend another iteration rather than silently stopping.
        history_.push_back(
            {Message::Role::Observation,
             "No <tool_call> or <finish> tag detected. Reply with exactly one "
             "<tool_call name=\"...\">...</tool_call> to act, or "
             "<finish>...</finish> when the task is complete."});
    }

    // Fell out of the loop without a <finish>: report the cap and hand back the
    // last thing the model actually said as a best-effort answer.
    result.max_iterations_hit = true;
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (it->role == Message::Role::Assistant) {
            result.answer = it->content;
            break;
        }
    }
    return result;
}

}  // namespace agent::orch
