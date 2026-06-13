// BlackwellLLMAdapter: the production ILLMGenerator that drives the real CUDA
// BlackwellEngine from the playground.
//
// The orchestrator (and the playground's manual "generate step") speaks in the
// neutral role-tagged transcript the AgentOrchestrator emits:
//
//     [SYSTEM]\n...\n\n[USER]\n...\n\n[ASSISTANT]\n
//
// This adapter is the single place that knows the *model's* prompt dialect: it
// re-frames that transcript into Qwen-Coder ChatML
//
//     <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
//
// tokenizes it (the tokenizer maps the literal "<|im_start|>" / "<|im_end|>"
// strings to their special ids), prefills the KV cache and decodes greedily/at
// the requested temperature until a stop token. Each call is stateless: it
// re-prefills the whole conversation from position 0, so the playground can edit
// history arbitrarily between turns without corrupting the cache.
#ifndef BLACKWELL_PLAYGROUND_LLM_ADAPTER_H
#define BLACKWELL_PLAYGROUND_LLM_ADAPTER_H

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

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

    // Invoked once per freshly decoded token, with the *incremental* piece of
    // text (not the running total). Returning false is a cooperative "stop"
    // signal: the decode loop emits no further tokens and returns what it has so
    // far. Runs on the engine's worker thread, so an implementation that touches
    // shared state must do its own locking.
    using StreamCallback = std::function<bool(const std::string& new_token)>;

    // Loads tokenizer + engine from a HuggingFace-layout checkpoint directory.
    // This is the SLOW call (DirectStorage weight streaming, VRAM allocation);
    // ModelService runs it on a background thread. Throws on failure.
    explicit BlackwellLLMAdapter(const std::string& model_dir, size_t max_seq_len = 8192);
    ~BlackwellLLMAdapter() override;

    BlackwellLLMAdapter(const BlackwellLLMAdapter&) = delete;
    BlackwellLLMAdapter& operator=(const BlackwellLLMAdapter&) = delete;

    // ILLMGenerator: continue the role-tagged transcript as the assistant.
    std::string generate(const std::string& transcript) override;

    // Same, but with explicit sampling params, an optional capture of the exact
    // ChatML string handed to the model (for the UI's debug panel), and an
    // optional per-token streaming callback (for live UI rendering / "Stop").
    std::string generate(const std::string& transcript, const Params& params,
                         std::string* chatml_out, StreamCallback stream_cb = nullptr);

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

    // Re-frame an AgentOrchestrator role transcript into Qwen ChatML. Static and
    // pure so it is unit-testable and reusable; OBSERVATION turns map to the
    // ChatML "user" role (tool results are fed back as user content).
    static std::string to_chatml(const std::string& role_transcript);

private:
    std::string model_dir_;
    size_t max_seq_len_;
    std::unique_ptr<blackwell::ITokenizer> tokenizer_;
    std::unique_ptr<BlackwellEngine> engine_;
    Params params_;
    StreamCallback stream_cb_;  // forwarded to by generate(transcript); may be null
};

}  // namespace playground

#endif  // BLACKWELL_PLAYGROUND_LLM_ADAPTER_H
