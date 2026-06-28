// Verification suite for the agent_orchestrator ReAct state machine.
//
// Driven entirely by a MockLLM (an ILLMGenerator that replays pre-canned
// completions) so no GPU / model load is needed. Covers the three pillars:
//   * ToolParser   -- extracts <tool_call>/<finish> from prose, and is
//                     completely crash-proof on malformed / truncated input.
//   * ToolRegistry -- dispatch is total: unknown tool AND a throwing callback
//                     both come back as ordinary "ERROR: ..." observations.
//   * AgentOrchestrator -- the full loop parses a tool call, dispatches it,
//                     appends the observation, and terminates on <finish>;
//                     MAX_ITERATIONS caps a model that never finishes.
#include <gtest/gtest.h>

#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "default_tools.h"
#include "llm_generator.h"
#include "orchestrator.h"
#include "tool_parser.h"
#include "tool_registry.h"

using namespace agent::orch;

namespace {

// Replays a fixed script of completions, one per generate() call, and records
// every transcript it was handed so tests can assert on what the loop fed back.
class MockLLM : public ILLMGenerator {
public:
    explicit MockLLM(std::deque<std::string> script) : script_(std::move(script)) {}

    std::string generate(const std::string& transcript) override {
        prompts.push_back(transcript);
        if (script_.empty()) return "<finish>script exhausted</finish>";
        std::string next = std::move(script_.front());
        script_.pop_front();
        return next;
    }

    std::vector<std::string> prompts;  // transcripts seen, in order

private:
    std::deque<std::string> script_;
};

bool history_has(const std::vector<Message>& h, Message::Role role,
                 const std::string& needle) {
    for (const auto& m : h)
        if (m.role == role && m.content.find(needle) != std::string::npos) return true;
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// ToolParser
// ---------------------------------------------------------------------------

TEST(ToolParser, ExtractsToolCallAmidYapping) {
    const std::string text =
        "Sure! Let me take a look at that file for you.\n"
        "<tool_call name=\"read_file\"><arg name=\"path\">src/x.cpp</arg></tool_call>\n"
        "I'll report back once I've read it.";
    auto a = ToolParser::parse(text);
    ASSERT_EQ(a.kind, ActionKind::ToolCall);
    EXPECT_EQ(a.tool.name, "read_file");
    ASSERT_EQ(a.tool.args.count("path"), 1u);
    EXPECT_EQ(a.tool.args.at("path"), "src/x.cpp");
}

TEST(ToolParser, ArgBodyMayContainQuotesAndBraces) {
    // The whole reason we chose XML over JSON: C++ payloads with quotes/braces
    // that would shatter JSON escaping pass through verbatim.
    const std::string text =
        "<tool_call name=\"write_file\">"
        "<arg name=\"path\">a.cpp</arg>"
        "<arg name=\"content\">int f(){ return \"x\\\"y\"[0]; }</arg>"
        "</tool_call>";
    auto a = ToolParser::parse(text);
    ASSERT_EQ(a.kind, ActionKind::ToolCall);
    EXPECT_EQ(a.tool.args.at("content"), "int f(){ return \"x\\\"y\"[0]; }");
}

TEST(ToolParser, ParsesFinishAndTrims) {
    auto a = ToolParser::parse("All done.\n<finish>  The answer is 42.  </finish>");
    ASSERT_EQ(a.kind, ActionKind::Finish);
    EXPECT_EQ(a.finish_text, "The answer is 42.");
}

TEST(ToolParser, EarliestActionWins) {
    auto a = ToolParser::parse(
        "<tool_call name=\"a\"></tool_call> then <finish>x</finish>");
    EXPECT_EQ(a.kind, ActionKind::ToolCall);
    EXPECT_EQ(a.tool.name, "a");
}

TEST(ToolParser, BoundaryAwareTagMatch) {
    // "<finished_state>" must NOT be read as a <finish> tag.
    auto a = ToolParser::parse("<finished_state>nope</finished_state>");
    EXPECT_EQ(a.kind, ActionKind::None);
}

TEST(ToolParser, MalformedToolCallFallsBackToFinish) {
    // Truncated tool_call (no close tag) should not shadow a valid finish.
    auto a = ToolParser::parse(
        "<tool_call name=\"x\">never closed... <finish>done anyway</finish>");
    ASSERT_EQ(a.kind, ActionKind::Finish);
    EXPECT_EQ(a.finish_text, "done anyway");
}

// The headline requirement: the parser must never crash on garbage.
TEST(ToolParser, CrashProofOnGarbage) {
    const char* nasties[] = {
        "",
        "<",
        "<tool_call",
        "<tool_call name=",
        "<tool_call name=\"",
        "<tool_call name=\"oops",
        "<tool_call name=\"x\"",          // no '>' at all
        "<tool_call>no name</tool_call>",  // missing required name
        "<finish",
        "<finish>",
        "<<<<>>>><tool_call<<<",
        "<arg name=\"k\">orphan</arg>",
        "<tool_call name=\"x\"><arg name=\"k\">unclosed",
    };
    for (const char* s : nasties) {
        auto a = ToolParser::parse(s);  // must simply not throw / not crash
        // Any kind is acceptable; the contract is "no crash, well-formed result".
        EXPECT_TRUE(a.kind == ActionKind::None || a.kind == ActionKind::ToolCall ||
                    a.kind == ActionKind::Finish);
    }
    SUCCEED();
}

// ---------------------------------------------------------------------------
// ToolParser -- reasoning-model <think> chain-of-thought handling
// ---------------------------------------------------------------------------

// The headline reasoning-model case: a <think> block must NOT abort parsing; the
// <finish> in the real answer after </think> is found exactly as usual.
TEST(ToolParser, ThinkBlockThenFinishParsesCleanly) {
    auto a = ToolParser::parse(
        "<think>some thoughts</think>\nNow here is the answer <finish/>");
    EXPECT_EQ(a.kind, ActionKind::Finish);
}

TEST(ToolParser, ThinkBlockThenFinishWithText) {
    auto a = ToolParser::parse(
        "<think>Let me reason about this step by step. The user asked X, so Y.</think>\n"
        "<finish>The answer is 42.</finish>");
    ASSERT_EQ(a.kind, ActionKind::Finish);
    EXPECT_EQ(a.finish_text, "The answer is 42.");
}

// Tags the model merely *rehearses* inside its thoughts must be inert: only the
// real tool_call after </think> is acted on.
TEST(ToolParser, TagsRehearsedInsideThinkAreIgnored) {
    auto a = ToolParser::parse(
        "<think>I could just <finish>give up</finish>, but I should really call "
        "<tool_call name=\"wrong\"/> first... actually let me read the file.</think>\n"
        "<tool_call name=\"read_file\"><arg name=\"path\">src/x.cpp</arg></tool_call>");
    ASSERT_EQ(a.kind, ActionKind::ToolCall);
    EXPECT_EQ(a.tool.name, "read_file");
    ASSERT_EQ(a.tool.args.count("path"), 1u);
    EXPECT_EQ(a.tool.args.at("path"), "src/x.cpp");
}

// A premature <finish> rehearsed in the thoughts must not shadow a later, real
// tool_call -- the earliest action OUTSIDE the think block wins.
TEST(ToolParser, ThinkFinishDoesNotShadowRealToolCall) {
    auto a = ToolParser::parse(
        "<think>maybe I'm done: <finish>done</finish></think>\n"
        "<tool_call name=\"list_dir\"></tool_call>");
    ASSERT_EQ(a.kind, ActionKind::ToolCall);
    EXPECT_EQ(a.tool.name, "list_dir");
}

// An unterminated <think> (the model is still reasoning) swallows the tail: there
// is no *completed* real answer yet, so nothing is actionable.
TEST(ToolParser, UnterminatedThinkSwallowsTail) {
    auto a = ToolParser::parse(
        "<think>I will eventually emit <finish>x</finish> but I'm still thinking");
    EXPECT_EQ(a.kind, ActionKind::None);
}

// A think block whose content is pure prose, followed by a pure-prose answer,
// stays None (and the orchestrator will nudge) -- think changes nothing here.
TEST(ToolParser, ThinkThenProseIsStillNone) {
    auto a = ToolParser::parse("<think>hmm</think>\nThe answer is probably 42.");
    EXPECT_EQ(a.kind, ActionKind::None);
}

// ---------------------------------------------------------------------------
// ToolRegistry (dispatcher)
// ---------------------------------------------------------------------------

TEST(ToolRegistry, DispatchesRegisteredTool) {
    ToolRegistry reg;
    reg.register_tool("echo", "echoes its body", [](const ToolInvocation& c) {
        return "echoed:" + c.body;
    });
    ToolInvocation call{"echo", {}, "hello"};
    EXPECT_EQ(reg.dispatch(call), "echoed:hello");
}

TEST(ToolRegistry, UnknownToolBecomesObservation) {
    ToolRegistry reg;
    reg.register_tool("known", "", [](const ToolInvocation&) { return "ok"; });
    ToolInvocation call{"does_not_exist", {}, ""};
    const std::string obs = reg.dispatch(call);
    EXPECT_NE(obs.find("not found"), std::string::npos);
    EXPECT_NE(obs.find("known"), std::string::npos);  // lists what IS available
}

TEST(ToolRegistry, ThrowingToolIsContained) {
    ToolRegistry reg;
    reg.register_tool("boom", "always throws", [](const ToolInvocation&) -> std::string {
        throw std::runtime_error("kaboom");
    });
    const std::string obs = reg.dispatch(ToolInvocation{"boom", {}, ""});
    EXPECT_NE(obs.find("ERROR"), std::string::npos);
    EXPECT_NE(obs.find("kaboom"), std::string::npos);
}

// ---------------------------------------------------------------------------
// AgentOrchestrator (the ReAct loop) -- the core verification
// ---------------------------------------------------------------------------

TEST(AgentOrchestrator, ParsesDispatchesAndUpdatesHistory) {
    // The model first calls a dummy tool, then finishes using its observation.
    MockLLM llm({
        "Let me check.\n"
        "<tool_call name=\"get_secret\"><arg name=\"id\">42</arg></tool_call>",
        "<finish>The secret is THE-EAGLE-HAS-LANDED.</finish>",
    });

    ToolRegistry reg;
    bool tool_called = false;
    std::string seen_id;
    reg.register_tool("get_secret", "returns a canned secret",
                      [&](const ToolInvocation& c) {
                          tool_called = true;
                          seen_id = c.args.count("id") ? c.args.at("id") : "";
                          return "secret(42) = THE-EAGLE-HAS-LANDED";
                      });

    AgentOrchestrator orch(llm, reg);
    RunResult r = orch.run("What is the secret for id 42?");

    // The tool was actually dispatched with the parsed argument.
    EXPECT_TRUE(tool_called);
    EXPECT_EQ(seen_id, "42");

    // The loop terminated cleanly on <finish>.
    EXPECT_TRUE(r.finished);
    EXPECT_FALSE(r.max_iterations_hit);
    EXPECT_EQ(r.iterations, 2);
    EXPECT_EQ(r.answer, "The secret is THE-EAGLE-HAS-LANDED.");

    // History is correctly threaded: user -> assistant -> observation -> assistant.
    const auto& h = orch.history();
    EXPECT_TRUE(history_has(h, Message::Role::User, "id 42"));
    EXPECT_TRUE(history_has(h, Message::Role::Assistant, "get_secret"));
    EXPECT_TRUE(history_has(h, Message::Role::Observation, "THE-EAGLE-HAS-LANDED"));

    // The observation was fed back: the 2nd generate() saw the tool result.
    ASSERT_EQ(llm.prompts.size(), 2u);
    EXPECT_NE(llm.prompts[1].find("secret(42) = THE-EAGLE-HAS-LANDED"),
              std::string::npos);
}

TEST(AgentOrchestrator, UnknownToolDoesNotCrashAndModelRecovers) {
    MockLLM llm({
        "<tool_call name=\"nonexistent\"></tool_call>",  // hallucinated tool
        "<finish>recovered</finish>",
    });
    ToolRegistry reg;  // empty registry
    AgentOrchestrator orch(llm, reg);
    RunResult r = orch.run("go");

    EXPECT_TRUE(r.finished);
    EXPECT_EQ(r.answer, "recovered");
    EXPECT_TRUE(history_has(orch.history(), Message::Role::Observation, "not found"));
}

TEST(AgentOrchestrator, PureProseGetsNudgedNotStuck) {
    MockLLM llm({
        "I think the answer might be 42 but let me ponder.",  // no tags at all
        "<finish>42</finish>",
    });
    ToolRegistry reg;
    AgentOrchestrator orch(llm, reg);
    RunResult r = orch.run("go");

    EXPECT_TRUE(r.finished);
    EXPECT_EQ(r.iterations, 2);
    EXPECT_TRUE(history_has(orch.history(), Message::Role::Observation,
                            "No <tool_call> or <finish>"));
}

TEST(AgentOrchestrator, ReasoningModelFinishesInOneTurnDespiteThinkBlock) {
    // A Qwen-3.5 / DeepSeek-style turn: a <think> CoT that even rehearses a
    // <finish>, then the real <finish>. The loop must terminate on the REAL finish
    // in a single iteration -- not abort, not get nudged for "no actionable tag",
    // and not stop early on the thought's rehearsed finish.
    MockLLM llm({
        "<think>The user wants the capital of France. I might just "
        "<finish>Paris?</finish> but let me be sure. It is Paris.</think>\n"
        "<finish>The capital of France is Paris.</finish>",
    });
    ToolRegistry reg;
    AgentOrchestrator orch(llm, reg);
    RunResult r = orch.run("What is the capital of France?");

    EXPECT_TRUE(r.finished);
    EXPECT_FALSE(r.max_iterations_hit);
    EXPECT_EQ(r.iterations, 1);
    EXPECT_EQ(r.answer, "The capital of France is Paris.");
    // The think block was preserved verbatim in the assistant turn (it streams to
    // the client as part of the message), not stripped or treated as an error.
    EXPECT_TRUE(history_has(orch.history(), Message::Role::Assistant, "<think>"));
    // It was NOT nudged for a missing action tag.
    EXPECT_FALSE(history_has(orch.history(), Message::Role::Observation,
                             "No <tool_call> or <finish>"));
}

TEST(AgentOrchestrator, ReasoningModelToolCallAfterThink) {
    // A reasoning turn that ends in a real tool_call (the CoT rehearses a bogus
    // one), then finishes off the observation. The bogus in-think tool must be
    // ignored and the real read_file dispatched.
    MockLLM llm({
        "<think>Should I call <tool_call name=\"bogus\"/>? No -- read the file.</think>\n"
        "<tool_call name=\"read_file\"><arg name=\"path\">a.txt</arg></tool_call>",
        "<finish>file read</finish>",
    });
    ToolRegistry reg;
    std::string seen_path;
    reg.register_tool("read_file", "reads a file", [&](const ToolInvocation& c) {
        seen_path = c.args.count("path") ? c.args.at("path") : "";
        return "contents-of-a.txt";
    });
    AgentOrchestrator orch(llm, reg);
    RunResult r = orch.run("read a.txt");

    EXPECT_TRUE(r.finished);
    EXPECT_EQ(seen_path, "a.txt");  // the REAL tool call, not the rehearsed "bogus"
    EXPECT_EQ(r.answer, "file read");
}

TEST(AgentOrchestrator, MaxIterationsSafeguard) {
    // A model that loops forever, never emitting <finish>.
    MockLLM llm({});  // empty script -> never finishes in our control...
    // ...so override: a registry tool that always succeeds, and a model that
    // always calls it. Use a generator that ignores the script.
    struct LoopLLM : ILLMGenerator {
        std::string generate(const std::string&) override {
            return "<tool_call name=\"noop\"></tool_call>";
        }
    } loop_llm;

    ToolRegistry reg;
    reg.register_tool("noop", "", [](const ToolInvocation&) { return "ok"; });

    OrchestratorConfig cfg;
    cfg.max_iterations = 5;
    AgentOrchestrator orch(loop_llm, reg, cfg);
    RunResult r = orch.run("loop forever");

    EXPECT_FALSE(r.finished);
    EXPECT_TRUE(r.max_iterations_hit);
    EXPECT_EQ(r.iterations, 5);
}

TEST(AgentOrchestrator, SystemPromptInjectedOnce) {
    MockLLM llm({"<finish>a</finish>", "<finish>b</finish>"});
    ToolRegistry reg;
    OrchestratorConfig cfg;
    cfg.system_prompt = "You are a helpful agent.";
    AgentOrchestrator orch(llm, reg, cfg);

    orch.run("first");
    orch.run("second");  // multi-turn: system prompt must NOT be duplicated

    int system_count = 0;
    for (const auto& m : orch.history())
        if (m.role == Message::Role::System) ++system_count;
    EXPECT_EQ(system_count, 1);
}

// ---------------------------------------------------------------------------
// Integration: default tools backed by the real agent_core / agent_env layers.
// ---------------------------------------------------------------------------

TEST(DefaultTools, CodeToolsDriveTheSemanticProvider) {
    agent::TreeSitterSemanticProvider provider;
    ToolRegistry reg;
    register_code_tools(reg, provider);

    // analyze_source then callees_of, exactly as the LLM would issue them.
    const std::string src =
        "int helper() { return 1; }\n"
        "int caller() { return helper(); }\n";
    ToolInvocation analyze{"analyze_source", {{"source", src}}, ""};
    const std::string a = reg.dispatch(analyze);
    EXPECT_NE(a.find("caller"), std::string::npos);
    EXPECT_NE(a.find("helper"), std::string::npos);

    ToolInvocation callees{"callees_of", {{"symbol", "caller"}}, ""};
    EXPECT_NE(reg.dispatch(callees).find("helper"), std::string::npos);
}
