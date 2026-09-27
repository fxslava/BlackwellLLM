#pragma once
#include <cstddef>
#include <string>
#include <memory>
#include <vector>

#include "blackwell/engine_status.h"

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

namespace blackwell {

// One entry of a TRUE-batched decode step: sequence `seq_id` consumes `token_id`
// at logical position `pos`. The sequences in a batch are INDEPENDENT (distinct
// block tables, distinct positions) — this is the primitive parallel code-agent
// orchestration (Tree-of-Thoughts) decodes N branches on, one forward pass per
// wavefront instead of N sequential ones. Paged mode + a branching-capable model
// only (fork() produces the sibling sequences); mirrors DecodeRequest 1:1 onto a
// DecodeResult by seq_id.
struct DecodeRequest {
    int seq_id;     // target sequence (Paged; 0 or a fork() child); unique per batch
    int token_id;   // token to consume this step
    int pos;        // logical position of token_id in the sequence
};

// The sampled continuation for one DecodeRequest, matched back by seq_id (the
// results vector preserves the request order, so index-parity holds too).
struct DecodeResult {
    int seq_id;         // echoes the request's seq_id
    int next_token_id;  // greedy (argmax) continuation — see forward_batch
};

} // namespace blackwell

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

    // seq_id selects which sequence to decode (Paged mode; default 0).
    // Continuous mode supports only seq_id 0.
    //
    // Hybrid error doctrine (blackwell/engine_status.h): these are the ONLY
    // runtime inference endpoints, and they are noexcept -- the decode hot loop
    // reports failure exclusively by EngineStatus return value (stray
    // subsystem exceptions are caught inside and converted). There is no
    // throwing forward()/forward_eval(): a caller that wants exceptions writes
    // its own wrapper at its own tier; the engine's runtime never unwinds.
    blackwell::EngineStatus forward_status(int token_id, int pos, float temperature,
                                           float top_p, int seq_id,
                                           int* next_token) noexcept;
    blackwell::EngineStatus forward_eval_status(int token_id, int pos,
                                                int target_token_id, int seq_id,
                                                float* log_prob) noexcept;

    // PROMPT PREFILL: consume `num_tokens` context tokens starting at `start_pos`
    // and sample only the continuation that follows the LAST of them.
    //
    // Semantically identical to a forward_status() loop over the same tokens --
    // same KV cache contents, same final logits, same sampled token -- but it does
    // not pay for logits it is going to throw away. A forward_status() loop runs
    // the final RMSNorm and the lm_head GEMV for EVERY prompt token; on a
    // 151552-vocab model that head is a 1.24 GiB read per token, and the caller
    // discards all but the last. This endpoint passes want_logits=false for every
    // interior token, and additionally tiles the prompt through
    // Impl::run_chunk when the resolved token capacity allows it (Paged mode on a
    // dense model -- see resolve_token_capacity), so the residual stream, the
    // norms and the paged-flash attention are widened to a 64-token tile.
    //
    // Deliberately NOT capability-gated: when the token capacity is 1 (Continuous
    // mode) it falls back to the single-token sweep, which is still strictly
    // cheaper than the caller's own loop. Skipping work is never an error, so a
    // harness or a UI can call this unconditionally for a prompt and get the best
    // path the loaded configuration supports.
    //
    // RUNTIME tier, like forward_status: noexcept, reports by EngineStatus. A
    // faulted prefill leaves the KV cache truncated at whatever it managed to
    // append, so the caller must re-prefill (or rewind) rather than decode on.
    blackwell::EngineStatus prefill_status(const int* token_ids, int num_tokens,
                                           int start_pos, float temperature, float top_p,
                                           int seq_id, int* next_token) noexcept;

    // TRUE batched decode: advance `requests.size()` INDEPENDENT sequences by one
    // token each in a SINGLE forward pass (batched embedding -> per-layer batched
    // RMSNorm + Tensor-Core projections + batched paged-flash attention over
    // per-sequence block tables -> batched lm_head), then argmax-sample each row.
    // Returns one DecodeResult per request, in request order (result[i].seq_id ==
    // requests[i].seq_id). This is the wavefront primitive for parallel code-agent
    // orchestration (Tree-of-Thoughts): N branches decode together rather than in
    // N sequential forward() calls, amortizing the weight reads over the batch via
    // the dispatcher's num_tokens>1 Tensor-Core GEMM path.
    //
    // Requirements (capability-gated; a violation throws std::runtime_error with
    // the remedy, like fork()): Paged KV mode on a branching-capable dense model,
    // every seq_id known and UNIQUE within the batch, batch size <= the resolved
    // token capacity. Sampling is deterministic greedy (argmax) — the parallel
    // branches of a search each want their top continuation, and determinism keeps
    // the batched path bit-checkable against the single-sequence forward() loop
    // (see the BatchedDecodeMatchesSequential parity test). A caller that needs
    // temperature/top-p samples from the per-row logits itself.
    //
    // Unlike forward_status this stays on the exception tier (like fork/rewind):
    // it is an orchestration/admin entry point, not part of the live single-stream
    // decode hot loop the noexcept status endpoints protect.
    std::vector<blackwell::DecodeResult> forward_batch(
        const std::vector<blackwell::DecodeRequest>& requests);

    // Softmax probability, in [0, 1], of `token_id` under the CURRENT logits --
    // the distribution the most recent forward()/prefill left in the device
    // logits buffer. Call right after sampling a token to record how confident
    // the model was in it (the Developer-Mode heatmap does exactly this). Does a
    // full-vocab device->host read, so it is a debug-path convenience, not a
    // per-token steady-state cost. Returns 0 for an out-of-range id.
    float last_token_probability(int token_id) const;

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

    // Destroy a forked branch and recycle its sequence id: return the branch's
    // KV pages to the allocator and drop its id mapping, so a later fork() may
    // reuse `seq_id`. Paged (branching) models only -- throws under Continuous /
    // hybrid-without-branching, like fork(). The hybrid physical state stores
    // (SSM recurrent/conv, gated full-attention KV) are statically allocated per
    // slot and are simply left for the next fork() to overwrite wholesale, so
    // this frees only the paged pool; it is the recycling primitive a snapshot
    // ring needs (fork() rejects an id that still exists). seq_id 0 may be
    // released (e.g. to restore a snapshot INTO the active head), but the caller
    // must re-establish it via fork() before decoding it again.
    void release_sequence(int seq_id);

    // Number of physically-addressable concurrent sequence slots [0,
    // branch_capacity): the shared budget of the paged CoW pool and the hybrid
    // physical state stores, resolved from RuntimeConfig::paged_branch_factor at
    // construction (1 when the model cannot branch). A snapshot ring sizes its
    // slot set from this.
    int branch_capacity() const noexcept;

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

    // ------------------------------------------------------------------------
    // Inactivity lifecycle (RAII-friendly: the engine object stays alive
    // through both stages; nothing is torn down or re-loaded from disk).
    //
    //   spill_kv_cache() — stage 1: bulk-demote every unpinned KV prefix-cache
    //     page down the tier waterfall (VRAM -> pinned RAM -> NVMe spill file),
    //     freeing KV VRAM. The radix-tree index survives; pages fault back in
    //     transparently on the next prefill. Returns the number of page
    //     demotions (0 when the model has no prefix-cache substrate).
    //
    //   hibernate() — stage 2: offload the resident model weights from GPU
    //     VRAM into a pinned host-RAM stash and free the device arena. The
    //     engine must be idle (no decode in flight); forward() is invalid until
    //     wakeup(). Idempotent.
    //
    //   wakeup() — DMA the weights back over PCIe into a fresh device arena
    //     and resume serving. Fast: the stash is pinned, so this is a single
    //     bulk H2D burst, not a disk re-load. Idempotent.
    //
    // All three follow the engine's single-threaded control-plane rule: call
    // them only from the thread that owns forward()/prefill.
    // ------------------------------------------------------------------------
    int  spill_kv_cache();
    void hibernate();
    void wakeup();
    bool hibernated() const noexcept;

    Impl* get_impl() const { return pImpl.get(); }

private:
    std::unique_ptr<Impl> pImpl;
};