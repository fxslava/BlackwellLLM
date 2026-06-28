#pragma once
#include <cstddef>
#include <string>
#include <memory>

// Tier-2 request struct (blackwell/runtime_config.h). Forward-declared so the
// public engine header stays light; engine.cpp pulls in the full definition.
namespace blackwell { struct InferenceConfig; }

// Static description of what the *loaded* model supports, derived once from the
// parsed ModelConfig at construction. The agent / playground queries this before
// attempting tree-search branching: hybrid SSM checkpoints (Qwen3.5) evolve a
// recurrent linear-attention state that we deliberately do NOT snapshot, so they
// run linear ReAct only -- fork/rewind are rejected rather than silently
// corrupting state. See get_capabilities().
struct ModelCapabilities {
    bool supports_cow_branching = false; // fork()/rewind() permitted
    bool requires_ssm_subsystem = false; // model has >=1 AttnKind::Linear layer
    bool is_hybrid              = false;  // mixes Full and Linear attention layers
    int  num_full_attention_layers   = 0;
    int  num_linear_attention_layers = 0;
};

class BlackwellEngine {
public:
    struct Impl;

    // KV-cache strategy selected at construction:
    //   Continuous - legacy FP32 contiguous cache + layer offloading (default).
    //   Paged      - bf16 paged cache with Copy-on-Write fork / rewind.
    enum class KVCacheMode { Continuous, Paged };

    // num_gpu_layers: how many leading transformer layers keep their weights /
    // KV cache resident in VRAM; the rest are offloaded to pinned host RAM and
    // streamed in asynchronously. Default (SIZE_MAX) keeps everything resident
    // unless the BLACKWELL_GPU_LAYERS environment variable overrides it.
    explicit BlackwellEngine(const std::string& index_path, size_t max_seq_len = 2048,
                             size_t num_gpu_layers = static_cast<size_t>(-1),
                             KVCacheMode kv_mode = KVCacheMode::Continuous);

    // Tier-2 constructor: state WHAT you want (context length, branching) and let
    // build_and_validate_runtime() resolve the execution plan. Preferred entry
    // point for the C-API / chat loop; the legacy constructor above maps its loose
    // args onto an InferenceConfig + low-level overrides internally.
    BlackwellEngine(const std::string& index_path, const blackwell::InferenceConfig& request);
    ~BlackwellEngine();

    // seq_id selects which sequence to decode (Paged mode; default 0). It is the
    // LAST parameter so existing positional calls -- forward(tok, pos) and
    // forward(tok, pos, temp, top_p) -- keep binding temperature/top_p correctly;
    // putting an int before the float defaults would silently capture them.
    // Continuous mode supports only seq_id 0.
    int forward(int token_id, int pos, float temperature = 0.6f, float top_p = 0.9f,
                int seq_id = 0);
    float forward_eval(int token_id, int pos, int target_token_id, int seq_id = 0);

    // Sequence branching (Paged mode only; throws under Continuous). fork shares
    // the parent's KV pages via CoW; rewind rolls a sequence back to `pos` tokens.
    // A forked child is decoded by passing its id as seq_id to forward().
    void fork(int parent_id, int child_id);
    void rewind(int seq_id, int pos);

    // Capabilities of the loaded model. fork()/rewind() throw std::runtime_error
    // when supports_cow_branching is false (hybrid SSM models, or Continuous mode).
    ModelCapabilities get_capabilities() const;

    Impl* get_impl() const { return pImpl.get(); }

private:
    std::unique_ptr<Impl> pImpl;
};