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
// seen.
//
// PROMPT PHASE -- two paths, one contract (KV resident, first token sampled):
//
//  (a) Prefix-cache path (Paged dense models, main sequence). generate() hands
//      the full rendered token array to the engine's EnginePrefillCoordinator
//      (prefill_prompt): the radix tree serves every page it has ever seen --
//      including the JIT-compiled .bkv prompt library, since both sides
//      tokenize the chat-template-rendered text with add_special_tokens=false
//      -- and only the unseen suffix pays GPU time. Decode continues on the
//      coordinator-issued engine sequence, which is finished/released on every
//      exit path (RAII), while the committed prompt pages stay in the tree for
//      the next turn.
//  (b) Legacy path (Continuous mode, hybrid SSM checkpoints, forked branches):
//      the adapter's own append-or-reset incremental prefill. generate()
//      prefix-matches the new prompt against the cached token history and
//      prefills only the appended tail (O(new tokens)); any divergence -- e.g.
//      the playground edited or deleted a past turn -- transparently falls
//      back to a full reprefill from position 0, so editing history can never
//      corrupt the cache.
#ifndef BLACKWELL_PLAYGROUND_LLM_ADAPTER_H
#define BLACKWELL_PLAYGROUND_LLM_ADAPTER_H

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackwell/engine.h"
#include "blackwell/runtime_config.h"  // blackwell::RuntimeOverrides
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
    //
    // use_paged_attention: when true the engine is built with the bf16 Paged KV
    // cache (Copy-on-Write fork/rewind, branching support); the default keeps the
    // legacy FP32 continuous cache. This is an engine-construction choice fixed at
    // load time, not a per-generate sampling parameter.
    //
    // num_gpu_layers: how many leading transformer layers stay resident in VRAM;
    // the remainder are offloaded to pinned CPU RAM and streamed in on demand by
    // the engine. SIZE_MAX (the default) keeps every layer on the GPU. The engine
    // performs the weight distribution itself, quant-agnostically (AWQ / FP8 /
    // bf16), so no backend-specific handling is needed in the adapter.
    //
    // overrides: optional low-level RuntimeConfig knobs (tiered KV prefix-cache
    // sizing: kv_vram_cache_pages / kv_ram_slots / kv_disk_slots /
    // kv_spill_path). kv_mode and num_gpu_layers inside it are IGNORED -- the
    // explicit constructor arguments win, to keep one source of truth.
    explicit BlackwellLLMAdapter(const std::string& model_dir, size_t max_seq_len = 8192,
                                 bool use_paged_attention = false,
                                 size_t num_gpu_layers = static_cast<size_t>(-1),
                                 const blackwell::RuntimeOverrides& overrides = {});
    ~BlackwellLLMAdapter() override;

    BlackwellLLMAdapter(const BlackwellLLMAdapter&) = delete;
    BlackwellLLMAdapter& operator=(const BlackwellLLMAdapter&) = delete;

    // ILLMGenerator: continue the role-tagged transcript as the assistant.
    std::string generate(const std::string& transcript) override;

    // Same, but with explicit sampling params, an optional capture of the exact
    // prompt string handed to the model (for the UI's debug panel), and an
    // optional per-token streaming callback (for live UI rendering / "Stop").
    // seq_id selects which branch (engine sequence) to decode on; default 0 =
    // "main". Each branch keeps its own KV-reuse history (see fork_sequence).
    std::string generate(const std::string& transcript, const Params& params,
                         std::string* prompt_out, StreamCallback stream_cb = nullptr,
                         int seq_id = 0);

    // Branch the engine KV cache (Paged mode only): share the parent's pages via
    // CoW and seed the child's token-history mirror from the parent. Throws under
    // the continuous cache (no branching). Host-only and fast; the caller must
    // serialize it against generation (ModelService runs it only when idle).
    void fork_sequence(int parent_id, int child_id);

    void set_params(const Params& p) { params_ = p; }
    const Params& params() const { return params_; }

    // The owned engine, for capability probes and the prefix-cache substrate
    // (engine().has_prefix_cache() / prefix_cache() / prefill_driver()). The
    // adapter binds its tokenizer to the prefill driver at construction, so
    // AOT warm-start and prefill_prompt() callers need no extra setup. Same
    // worker-thread ownership rules as generate().
    BlackwellEngine& engine() { return *engine_; }
    const BlackwellEngine& engine() const { return *engine_; }

    // The owned tokenizer, for callers that drive the engine directly via
    // engine().prefill_driver() (bypassing generate() entirely -- e.g.
    // LiveTranslationTracker's speculative-prefill / decode loop) and need
    // decode()/is_stop() without duplicating tokenizer setup. Same
    // worker-thread ownership rules as generate().
    blackwell::ITokenizer& tokenizer() { return *tokenizer_; }
    const blackwell::ITokenizer& tokenizer() const { return *tokenizer_; }

    // Install a streaming callback that the ILLMGenerator entry point
    // (generate(transcript)) will forward tokens to. This is how the ReAct loop
    // streams: the orchestrator only ever calls the single-argument override, so
    // the callback has to live on the adapter rather than the call site. Pass
    // nullptr to detach.
    void set_stream_callback(StreamCallback cb) { stream_cb_ = std::move(cb); }
    const std::string& model_dir() const { return model_dir_; }
    size_t max_seq_len() const { return max_seq_len_; }
    bool use_paged_attention() const { return use_paged_attention_; }
    // SIZE_MAX => all layers resident on the GPU; otherwise the GPU-resident count.
    size_t num_gpu_layers() const { return num_gpu_layers_; }

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
    bool use_paged_attention_;  // engine built with the Paged (CoW) KV cache
    size_t num_gpu_layers_;     // leading layers kept in VRAM; rest offloaded to CPU RAM
    std::unique_ptr<blackwell::ITokenizer> tokenizer_;
    std::unique_ptr<BlackwellEngine> engine_;
    Params params_;
    StreamCallback stream_cb_;  // forwarded to by generate(transcript); may be null

    ChatTemplate template_kind_ = ChatTemplate::ChatML;
    std::vector<std::string> stop_strings_;  // literal end markers; never emitted to UI
    size_t max_stop_len_ = 0;                // longest stop string, for the bounded tail scan

    // LEGACY-PATH state only: token history resident in the engine's KV cache
    // from the previous call (the prior prompt plus everything we generated last
    // turn). generate() prefix-matches the next prompt against this -- a strict
    // extension prefills only the new tail (APPEND); any divergence forces a
    // full reprefill from pos 0 (RESET). Mirrors the KV cache by position.
    // Touched only on the worker thread, so no locking is needed (same as
    // engine_). Keyed by engine seq_id (branch): seq 0 is "main";
    // fork_sequence() copies the parent branch's history to the child.
    // The prefix-cache path never reads or writes this: cross-turn reuse lives
    // in the engine's radix tree there, so on such models this map stays empty
    // for seq 0 (and a fork from it starts cold -- correct, just unshared).
    std::unordered_map<int, std::vector<int>> cached_prompt_by_seq_;
};

}  // namespace playground

#endif  // BLACKWELL_PLAYGROUND_LLM_ADAPTER_H
