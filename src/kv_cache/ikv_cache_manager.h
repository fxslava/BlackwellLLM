#pragma once
#include <cstdint>
#include <memory>

// ============================================================================
// IKVCacheManager — KV-cache strategy abstraction for BlackwellEngine.
// ============================================================================
// Decouples the decoder's attention stage from HOW the KV cache is physically
// stored and addressed, so the engine can run on either:
//   * ContinuousKVManager — the legacy FP32 contiguous slab + VRAMArena layer
//     offloading (unchanged math; the fallback / parity reference).
//   * PagedKVManager      — the bf16 paged cache + SequenceManager (CoW fork /
//     rewind, paged-flash-attention kernel) for ReAct agent tree-search.
//
// The strategy is chosen ONCE at engine construction and never changes, so every
// call site below is monomorphic (see the dispatch note at the bottom of this
// file): the virtual indirection is host-side control plane only and is dwarfed
// by CUDA launch latency. No virtual dispatch ever crosses into device code.
//
// CONTRACT — call ordering per token / per chunk:
//   1. prepare_decode_step(seq, pos)         // once, BEFORE the layer loop
//   2. for L in [0, num_layers):
//        attention_decode(L, pos, dQ,dK,dV, dO)   // once per layer
//   The prepare_*_step call LATCHES the active sequence context (resolved append
//   slot, uploaded block table) that the subsequent per-layer attention_* calls
//   consume. d_Q/d_K/d_V are the engine's per-layer post-projection buffers.
// ----------------------------------------------------------------------------

namespace blackwell {

// Engine-facing sequence handle. Caller-assigned (so fork takes both ids); the
// paged adapter maps these onto its internal SequenceManager ids. The continuous
// adapter is single-sequence and accepts only id 0.
using SeqId = int32_t;

enum class KVCacheKind {
    Continuous,   // legacy FP32 contiguous cache + offloading
    Paged,        // bf16 paged cache + CoW SequenceManager
};

class IKVCacheManager {
public:
    virtual ~IKVCacheManager() = default;

    // ---------------------------------------------------------------------
    // Per-step control plane (host-side, OUTSIDE the layer loop).
    // ---------------------------------------------------------------------
    // Decode: resolve the physical append slot for the next token of `seq`
    // (paged: CoW the target page if it is fork-shared) and stage the device
    // block table for the upcoming layer sweep. Continuous: validate/track pos.
    virtual void prepare_decode_step(SeqId seq, int pos) = 0;

    // Prefill / speculative-verification chunk: `num_tokens` query positions
    // starting at `start_pos`. Paged: ensure the chunk's pages exist (CoW the
    // boundary page if shared) and stage the block table. Continuous: track range.
    virtual void prepare_prefill_step(SeqId seq, int start_pos, int num_tokens) = 0;

    // ---------------------------------------------------------------------
    // Per-layer attention (INSIDE the layer loop). The single virtual call that
    // routes the actual CUDA work. Encapsulates, for the layer's resolved slot:
    //   RoPE(Q,K) -> append K/V to this layer's cache -> exact attention over the
    //   sequence prefix -> context to d_O. Hides BOTH the K/V write layout
    //   (contiguous vs scattered pages) AND the kernel (decoding vs paged-flash
    //   block-table), plus any layer staging/commit (offload) hooks.
    // ---------------------------------------------------------------------
    virtual void attention_decode(int layer_idx, int pos,
                                  float* d_Q, float* d_K, float* d_V,
                                  float* d_O) = 0;

    // Chunked variant: full Tensor-Core BLOCK_M tiles over the prefill range
    // latched by prepare_prefill_step. d_Q/d_O are [num_tokens, q_heads, head_dim].
    virtual void attention_prefill(int layer_idx, int start_pos, int num_tokens,
                                   float* d_Q, float* d_K, float* d_V,
                                   float* d_O) = 0;

    // ---------------------------------------------------------------------
    // Cache base pointers — diagnostics / tests / adapter-internal use ONLY.
    // Element type is strategy-specific (float for continuous, __nv_bfloat16 for
    // paged), hence void*. The hot path never dereferences these: all cache
    // access is owned by the attention_* methods above.
    // ---------------------------------------------------------------------
    virtual void* get_layer_k_ptr(int layer_idx) = 0;
    virtual void* get_layer_v_ptr(int layer_idx) = 0;

    // ---------------------------------------------------------------------
    // Branch lifecycle (ReAct tree-search).
    // ---------------------------------------------------------------------
    // Branch `parent` into the caller-assigned `child`. Paged: share pages via
    // CoW (O(blocks), no data copy until first write). Continuous: naive deep
    // copy of the FP32 slab, or throw std::runtime_error (unsupported).
    virtual void fork(SeqId parent, SeqId child) = 0;

    // Roll `seq` back to `target_pos` tokens; paged frees now-unreachable pages.
    virtual void rewind(SeqId seq, int target_pos) = 0;

    // ---------------------------------------------------------------------
    // Capability / identity.
    // ---------------------------------------------------------------------
    virtual const char* name() const = 0;          // "continuous" | "paged"
    virtual bool supports_branching() const = 0;   // false for legacy continuous
};

// Factory (implemented in the engine TU). Selects the strategy at construction
// from config; the returned pointer is stored once and reused for the engine's
// lifetime. Heavy ctor args (VRAMArena, ModelConfig, page-pool sizing) are bound
// inside the concrete adapters, not exposed here.
// std::unique_ptr<IKVCacheManager> make_kv_cache_manager(KVCacheKind kind, ...);

} // namespace blackwell
