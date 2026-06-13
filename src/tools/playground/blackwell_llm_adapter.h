// BlackwellLLMAdapter: the production ILLMGenerator that drives the real CUDA
// BlackwellEngine from the playground.
//
// The orchestrator (and the playground's manual "generate step") speaks in the
// neutral role-tagged transcript the AgentOrchestrator emits:
//
//     [SYSTEM]\n...\n\n[USER]\n...\n\n[ASSISTANT]\n
//
// This adapter is the single place that knows the *model's* prompt dialect. It is
// model-agnostic: at load time a lightweight heuristic over the checkpoint's
// tokenizer_config.json chat_template picks a native C++ formatting routine --
// Qwen/ChatML (<|im_start|>...<|im_end|>) or Llama-3
// (<|start_header_id|>...<|eot_id|>). It then tokenizes (the tokenizer maps the
// literal special-token strings to their ids), prefills the KV cache and decodes
// greedily/at the requested temperature until a stop token OR a stop *string* is
// seen. Each call is stateless: it re-prefills the whole conversation from
// position 0, so the playground can edit history arbitrarily between turns
// without corrupting the cache.
#ifndef BLACKWELL_PLAYGROUND_LLM_ADAPTER_H
#define BLACKWELL_PLAYGROUND_LLM_ADAPTER_H

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"
#include "llm_generator.h"  // agent::orch::ILLMGenerator

namespace playground {

class BlackwellLLMAdapter : public agent::orch::ILLMGenerator {
public:
    struct Params {
        float temperature = 0.2f;  // 0 => greedy/deterministic decode
        float top_p = 0.9f;
        int max_new_tokens = 2048;  // generous cap: code completions rarely fit in 512
    };

    // Which native formatting routine a checkpoint maps to. Chosen at load time by
    // a lightweight heuristic over tokenizer_config.json's chat_template -- no
    // Jinja engine, no new dependencies.
    enum class ChatTemplate { ChatML, Llama3 };

    // Invoked once per freshly decoded token, with the *incremental* piece of
    // text (not the running total). Returning false is a cooperative "stop"
    // signal: the decode loop emits no further tokens and returns what it has so
    // far. Runs on the engine's worker thread, so an implementation that touches
    // shared state must do its own locking.
    using StreamCallback = std::function<bool(const std::string& new_token)>;

    // Loads tokenizer + engine from a HuggingFace-layout checkpoint directory,
    // and reads tokenizer_config.json to pick the chat template + stop strings.
    // This is the SLOW call (DirectStorage weight streaming, VRAM allocation);
    // ModelService runs it on a background thread. Throws on failure.
    explicit BlackwellLLMAdapter(const std::string& model_dir, size_t max_seq_len = 8192);
    ~BlackwellLLMAdapter() override;

    BlackwellLLMAdapter(const BlackwellLLMAdapter&) = delete;
    BlackwellLLMAdapter& operator=(const BlackwellLLMAdapter&) = delete;

    // ILLMGenerator: continue the role-tagged transcript as the assistant.
    std::string generate(const std::string& transcript) override;

    // Same, but with explicit sampling params, an optional capture of the exact
    // prompt string handed to the model (for the UI's debug panel), and an
    // optional per-token streaming callback (for live UI rendering / "Stop").
    std::string generate(const std::string& transcript, const Params& params,
                         std::string* prompt_out, StreamCallback stream_cb = nullptr);

    void set_params(const Params& p) { params_ = p; }
    const Params& params() const { return params_; }

    // Install a streaming callback that the ILLMGenerator entry point
    // (generate(transcript)) will forward tokens to. This is how the ReAct loop
    // streams: the orchestrator only ever calls the single-argument override, so
    // the callback has to live on the adapter rather than the call site. Pass
    // nullptr to detach.
    void set_stream_callback(StreamCallback cb) { stream_cb_ = std::move(cb); }
    const std::string& model_dir() const { return model_dir_; }
    size_t max_seq_len() const { return max_seq_len_; }

    // Detected chat dialect + the stop strings collected for this checkpoint.
    ChatTemplate chat_template() const { return template_kind_; }
    const std::vector<std::string>& stop_strings() const { return stop_strings_; }

    // Re-frame an AgentOrchestrator role transcript into the model's chat dialect.
    // Static and pure (the family is passed explicitly) so it stays unit-testable
    // and reusable; OBSERVATION turns map to the "user" role (tool results are fed
    // back as user content).
    static std::string apply_chat_template(const std::string& role_transcript, ChatTemplate tmpl);

private:
    // Read tokenizer_config.json: route the chat template and collect stop strings
    // (merged with a foolproof fallback). Tolerates a missing/garbled sidecar by
    // keeping the ChatML default + fallback stops.
    void load_chat_config();

    std::string model_dir_;
    size_t max_seq_len_;
    std::unique_ptr<blackwell::ITokenizer> tokenizer_;
    std::unique_ptr<BlackwellEngine> engine_;
    Params params_;
    StreamCallback stream_cb_;  // forwarded to by generate(transcript); may be null

    ChatTemplate template_kind_ = ChatTemplate::ChatML;
    std::vector<std::string> stop_strings_;  // literal end markers; never emitted to UI
    size_t max_stop_len_ = 0;                // longest stop string, for the bounded tail scan
};

}  // namespace playground

#endif  // BLACKWELL_PLAYGROUND_LLM_ADAPTER_H
