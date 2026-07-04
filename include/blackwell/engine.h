#pragma once
#include <cstddef>
#include <string>
#include <memory>

// Tier-2 request / low-level override structs (blackwell/runtime_config.h).
// Forward-declared so the public engine header stays light; engine.cpp pulls in
// the full definitions.
namespace blackwell {
struct InferenceConfig;
struct RuntimeOverrides;
}

// Prefix-cache substrate (Paged mode only). Forward-declared for the same
// reason: callers of prefix_cache() / prefill_driver() include the real
// headers (src/paging/prefix_cache_manager.h, src/engine_prefill_coordinator.h)
// from the engine source tree, exactly as the AOT warmer tooling already does.
namespace blackwell {
class EnginePrefillCoordinator;
namespace paging { class PrefixCacheManager; }
}

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

    // Tier-2 + explicit low-level overrides (the `-x264-params` seam): poke
    // individual RuntimeConfig knobs -- kv_mode, num_gpu_layers, the tiered
    // KV prefix-cache sizing (kv_vram_cache_pages / kv_ram_slots /
    // kv_disk_slots / kv_spill_path) -- without restating the whole plan.
    BlackwellEngine(const std::string& index_path, const blackwell::InferenceConfig& request,
                    const blackwell::RuntimeOverrides& overrides);
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

    // Restart a sequence: zero the recurrent linear-attention (SSM) state so the
    // next decode begins from an empty history. The attention KV caches are
    // position-addressed (re-decoding from pos 0 overwrites stale slots), but the
    // SSM state accumulates with every forward() and has no rewind -- so a hybrid
    // model MUST call this before reprefilling a fresh/divergent prompt, or the new
    // sequence decodes on top of the previous conversation's recurrent state
    // (collapse / repetition). No-op for dense (non-SSM) models. Hybrid models are
    // single-sequence, so seq_id must be 0 for them.
    void reset_state(int seq_id = 0);

    // Capabilities of the loaded model. fork()/rewind() throw std::runtime_error
    // when supports_cow_branching is false (hybrid SSM models, or Continuous mode).
    ModelCapabilities get_capabilities() const;

    // ------------------------------------------------------------------------
    // Prefix cache + prefill driver (Paged mode on dense uniform full-attention
    // models only; hybrid SSM / gated full-attention state lives outside the
    // paged pools, so those models never get a prefix cache).
    //
    //   prefill_driver().prefill_prompt(tokens, n) — radix-tree prefix reuse +
    //     GPU forward pass over only the uncached suffix + commit back to the
    //     tree; returns the seq_id to keep decoding with forward().
    //   prefix_cache() — the raw substrate, for AOTCacheWarmer::warm_start()
    //     (.bkv prompt libraries), dump()/load(), and introspection.
    //
    // Both throw std::runtime_error when has_prefix_cache() is false.
    // ------------------------------------------------------------------------
    bool has_prefix_cache() const noexcept;
    blackwell::paging::PrefixCacheManager& prefix_cache();
    blackwell::EnginePrefillCoordinator&   prefill_driver();

    Impl* get_impl() const { return pImpl.get(); }

private:
    std::unique_ptr<Impl> pImpl;
};