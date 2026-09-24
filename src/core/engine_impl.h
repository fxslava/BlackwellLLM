#pragma once
#include "blackwell/engine.h"
#include "blackwell/config.h"
#include "blackwell/runtime_config.h"
#include "device_buffer.h"
#include "safetensors.h"
#include "memory_pool.h"
#include "ops_dispatcher.h"
#include "kv_cache/ikv_cache_manager.h"
#include "ssm/ssm_state_pool.h"
#include <memory>
#include <vector>

// Prefix-cache substrate (Phase 3 wiring). Header-only classes; engine.cpp
// includes the real headers — Impl only holds owning pointers, so forward
// declarations keep this header light.
namespace blackwell {
class EnginePrefillCoordinator;
namespace paging {
class SmVramPool;
class CudaTierBackend;
class TieredMemoryPager;
class PrefixCacheManager;
} // namespace paging
} // namespace blackwell

struct BlackwellEngine::Impl {
    // Declaration order is the construction order. The tier-1 config, the derived
    // capabilities, and the validated tier-3 runtime plan must all precede `arena`
    // and `kv_mgr`, which size themselves from m_runtime (max_seq_len, num_gpu_layers,
    // kv_mode, paged_branch_factor).
    ModelConfig m_config;
    ModelCapabilities m_caps;          // topology-derived; supports_cow_branching finalized in ctor body
    blackwell::RuntimeConfig m_runtime; // resolved + validated execution plan

    // Batched-forward width: how many token rows one forward pass may process at
    // once (batched prefill chunk / true-batch decode). 1 for the plain batch=1
    // decode path. Sizes the arena ping-pong buffers and every per-step scratch
    // buffer below, so it must precede `arena`. Computed once from the plan.
    size_t m_token_capacity;

    // Concurrent-branch capacity for the HYBRID per-sequence state stores (the SSM
    // recurrent/conv pool and the dedicated gated full-attention KV cache). These
    // stores are physically snapshotted on fork() (no CoW), so they must be sized
    // for every branch that can be live at once. Paged mode sizes this to the
    // branch factor; single-sequence (Continuous, no fork) keeps it 1. Distinct
    // from m_token_capacity (tokens-per-pass): a hybrid model batches ONE token
    // per forward but may hold several forked sequences.
    int m_branch_capacity = 1;

    SafetensorsLoader loader;
    VRAMArena arena;
    LinearDispatcher dispatcher;

    // KV-cache strategy (chosen once at construction; see IKVCacheManager).
    // Declared after `arena` so it is destroyed before it -- the adapter holds a
    // reference into the arena. Defaults to the legacy continuous FP32 cache.
    std::unique_ptr<blackwell::IKVCacheManager> kv_mgr;

    // Prefix-cache substrate over the paged pools (constructed ONLY for Paged
    // mode on dense uniform full-attention models; all null otherwise — see the
    // composition root in the ctor). Declaration order == dependency order, and
    // everything sits after kv_mgr so destruction unwinds coordinator ->
    // prefix cache -> pager -> backend/pool BEFORE the SequenceManager they all
    // reference dies with kv_mgr.
    std::unique_ptr<blackwell::paging::SmVramPool>        kv_vram_pool;
    std::unique_ptr<blackwell::paging::CudaTierBackend>   kv_tier_backend;
    std::unique_ptr<blackwell::paging::TieredMemoryPager> kv_pager;
    std::unique_ptr<blackwell::paging::PrefixCacheManager> prefix_cache;
    std::unique_ptr<blackwell::EnginePrefillCoordinator>   prefill;

    // Hybrid linear-attention (SSM) state. Allocated only for models with
    // AttnKind::Linear layers; null otherwise. m_linear_layer_index maps an
    // absolute layer index to its ordinal among the linear layers (-1 if the
    // layer is full attention), which is how SsmStatePool addresses per-layer state.
    std::unique_ptr<blackwell::ssm::SsmStatePool> ssm_state;
    std::vector<int> m_linear_layer_index;
    // Scratch for the linear-attention (GatedDeltaNet) decode step (hybrid only:
    // declared empty, allocated in the ctor's SSM branch). d_ssm_qkv: in_proj_qkv
    // out (conv_dim); d_ssm_qkv_conv: post-conv1d; q/k/v: per-value-head split
    // (H*head_dim); z: gate (v_dim); a/b: per-head dt/beta sources (H); core:
    // scan output; o: gated-normed output. The *_f32 buffers hold per-layer bf16
    // params cast to fp32 for the fp32 kernels.
    blackwell::DeviceBuffer<float> d_ssm_qkv;
    blackwell::DeviceBuffer<float> d_ssm_z;
    blackwell::DeviceBuffer<float> d_ssm_qkv_conv;
    blackwell::DeviceBuffer<float> d_ssm_q, d_ssm_k, d_ssm_v;
    blackwell::DeviceBuffer<float> d_ssm_a, d_ssm_b;
    blackwell::DeviceBuffer<float> d_ssm_core, d_ssm_o;
    blackwell::DeviceBuffer<float> d_dt_bias_f32, d_A_log_f32;
    blackwell::DeviceBuffer<float> d_norm_f32, d_conv_w_f32;
    // CHUNKED-prefill scratch (hybrid + m_token_capacity>1 only). The per-token
    // in_proj outputs are staged batched (d_ssm_qkv_batch/d_ssm_z_batch), the conv
    // + split are still swept per token into the single-token d_ssm_* above and
    // scattered into these token-major chunk buffers, then one chunked-delta kernel
    // launch evolves the recurrent state over the whole chunk. Widths scale with
    // m_token_capacity; d_ssm_u_scratch is the per-head [64][Dv] UT-solve scratch.
    blackwell::DeviceBuffer<float> d_ssm_qkv_batch, d_ssm_z_batch, d_ssm_o_batch;
    blackwell::DeviceBuffer<float> d_ssm_chunk_q, d_ssm_chunk_k, d_ssm_chunk_v, d_ssm_chunk_o;
    blackwell::DeviceBuffer<float> d_ssm_chunk_logdecay, d_ssm_chunk_beta;
    blackwell::DeviceBuffer<float> d_ssm_u_scratch;

    // NON-owning views into the arena's ping-pong activation pool (bound in the
    // ctor body; the arena frees them) -- the only raw device pointers left here.
    float *d_X_accum = nullptr;
    float *d_X_norm = nullptr;
    // Owned per-step compute scratch (allocated unconditionally in the ctor).
    blackwell::DeviceBuffer<float> d_Q, d_K, d_V, d_Attn_out;
    blackwell::DeviceBuffer<float> d_Gate, d_Up, d_Swiglu_out;
    // GLM-4 fused MLP in-projection: ONE [num_tokens, 2*intermediate_dim] buffer
    // holding [gate | up] per row, so the projection is a single GEMV/GEMM and the
    // SwiGLU reads both halves in place (no split kernel, no copy). Allocated only
    // when m_config.mlp_fused_gate_up; d_Gate / d_Up stay empty in that case,
    // because nothing else ever writes them.
    blackwell::DeviceBuffer<float> d_GateUp;
    // GLM-4-0414 sandwich norms: a sub-layer output has to survive long enough to
    // be normalized BEFORE it joins the residual, so o_proj / down_proj write here
    // instead of accumulating straight into d_X_accum. Allocated only when
    // m_config.has_sandwich_norms.
    blackwell::DeviceBuffer<float> d_sublayer_out;
    blackwell::DeviceBuffer<float> d_logits;
    blackwell::DeviceBuffer<int> d_next_token;

    // Qwen3.5 hybrid FULL-attention scratch + cache (allocated only when the model
    // uses gated head_dim-256 attention, i.e. m_config.attn_output_gate). These
    // layers cannot use the shared 128-wide attention/KV path, so they keep a
    // dedicated continuous FP32 KV cache indexed by m_full_layer_index. d_QG holds
    // the [num_heads, 2*head_dim] q_proj output (query|gate); d_gate is the split
    // gate half. Hybrid models never branch, so a single continuous cache suffices.
    std::vector<int> m_full_layer_index;   // absolute layer -> full-attn ordinal (-1 if linear)
    blackwell::DeviceBuffer<float> d_QG, d_gate;
    blackwell::DeviceBuffer<float> d_full_k_cache, d_full_v_cache;
    size_t m_full_kv_layer_stride = 0;     // floats per layer in each of K/V cache
    // Per-SEQUENCE stride: floats one branch owns across all full-attn layers
    // (num_full_layers * m_full_kv_layer_stride). The cache is laid out
    // [seq][full-layer][pos][kv]; forking a hybrid model D2D-copies one such slice.
    size_t m_full_kv_seq_stride = 0;

    Impl(const std::string& index_path, const blackwell::InferenceConfig& request,
         const blackwell::RuntimeOverrides& overrides);
    // Device buffers release themselves (DeviceBuffer RAII); the out-of-line
    // `= default` stays in engine.cpp so this header keeps its forward decls.
    ~Impl();

    // One full decoder pass for one token: embedding -> N transformer layers
    // (per-token KV control plane latched via kv_mgr) -> optionally final norm
    // + lm_head. want_logits=false is the prefill fast path: every non-final
    // prompt token skips the vocab-size GEMV, since only its KV append matters.
    //
    // RUNTIME error tier (Hybrid doctrine, blackwell/engine_status.h): the
    // decode chain reports failure by EngineStatus return, never by throw --
    // callers propagate with ENGINE_TRY (engine.cpp) or translate at the edge.
    blackwell::EngineStatus run_token(int token_id, int pos, int seq_id,
                                      bool want_logits = true);

    // Batched forward over a CHUNK of num_tokens prompt tokens at logical
    // positions [start_pos, start_pos + num_tokens), in ONE pass: batched
    // embedding -> per-layer batched RMSNorm + projections (Tensor-Core GEMM for
    // BF16, per-row GEMV sweep for quantized) + batched paged-flash prefill
    // attention -> batched MLP. want_logits runs the final norm + lm_head GEMV for
    // the LAST token only (its logits are what a prompt needs). num_tokens must be
    // <= m_token_capacity (the caller tiles longer deltas). Dense uniform
    // full-attention runs the batched paged pipeline here; SSM / gated-full hybrid
    // models delegate to run_chunk_hybrid (chunked delta rule + dedicated cache).
    // RUNTIME error tier: reports by EngineStatus (prepare_prefill_step may throw a
    // KV-gap at the caller/coordinator boundary, as before).
    blackwell::EngineStatus run_chunk(const int* token_ids, int start_pos,
                                      int num_tokens, int seq_id,
                                      bool want_logits = true);

    // TRUE-batch decode: advance batch_size INDEPENDENT sequences by one token
    // each in ONE forward pass. token_ids / seqs / positions are host arrays of
    // batch_size ints; the greedy (argmax) continuations land in out_next_tokens.
    // Mirrors run_chunk's batched widening but across SEQUENCES (each with its own
    // block table, length and position) rather than consecutive positions of one
    // sequence: batched embedding -> per-layer batched RMSNorm + Tensor-Core
    // projections + batched paged-flash decode (attention_decode_batch) -> batched
    // final norm -> per-row lm_head + argmax. Leaves [batch, vocab] in d_logits.
    // Generic dense paged full-attention only (returns InvalidConfig for SSM /
    // gated / continuous). RUNTIME error tier for CUDA (status by return); the KV
    // control plane may still throw (unknown seq id / KV gap), which the
    // forward_batch facade -- the exception-tier boundary -- catches.
    blackwell::EngineStatus run_decode_batch(const int* token_ids, const int* seqs,
                                             const int* positions, int batch_size,
                                             int* out_next_tokens);

    // Branch a sequence for Tree-of-Thoughts search (admin / exception tier).
    // Clones EVERY per-sequence state store parent owns into child, in one place:
    //   1. attention KV pages     -> kv_mgr->fork (paged CoW share)
    //   2. recurrent SSM state    -> SsmStatePool::fork_sequence (physical D2D)
    //   3. gated full-attn KV     -> D2D copy of the dedicated cache slice
    // (2) and (3) are what make hybrid models forkable; a dense model exercises
    // only (1). child_id must fit the branch capacity of the physical stores.
    void fork(int parent_id, int child_id);

    blackwell::EngineStatus step_embedding(int token_id);
    blackwell::EngineStatus step_attention_norm(int layer_idx);
    blackwell::EngineStatus step_attention_qkv_projections(int layer_idx);
    blackwell::EngineStatus step_attention_math(int layer_idx, int pos);
    blackwell::EngineStatus step_attention_out(int layer_idx);
    // Linear-attention (SSM) layer: bypasses the KV cache, evolves the recurrent
    // state in SsmStatePool via the conv1d + GatedDeltaNet kernels. seq_id selects
    // this sequence's recurrent/conv slice (branching: a forked child owns its own
    // slice; default 0 is the single-stream path, unchanged).
    blackwell::EngineStatus step_linear_attention(int layer_idx, int pos, int seq_id = 0);
    // Qwen3.5 hybrid gated full-attention layer (head_dim 256, q_proj query|gate,
    // q_norm/k_norm, partial RoPE). Uses the dedicated full-attn KV cache; seq_id
    // selects this sequence's per-branch slice of that cache.
    blackwell::EngineStatus step_full_attention(int layer_idx, int pos, int seq_id = 0);

    // Hybrid True Batched Prefill. run_chunk delegates here when the model is
    // SSM/gated (ssm_state or attn_output_gate); it widens the residual stream to
    // num_tokens rows and dispatches each layer to the batched linear/full-attn
    // chunk steps below. The two step_*_chunk helpers mirror their per-token
    // siblings but process the whole [start_pos, start_pos+num_tokens) window:
    //   - linear: batched norm/proj, per-token conv+split+alpha swept and scattered
    //     into token-major chunk buffers, ONE launch_gated_delta_chunked_prefill
    //     carries the recurrent state, then batched gated-RMSNorm + out_proj;
    //   - full:   batched proj, per-token qg_split/norm/RoPE/append/decode/gate over
    //     the dedicated cache (positions are per-token), then batched o_proj.
    // seq_id selects the branch's SSM / full-attn slice (fork branching). Every
    // per-token/per-head kernel does the identical math its decode sibling does, so
    // a batched prefill reproduces the run_token sweep to fp32 rounding.
    blackwell::EngineStatus run_chunk_hybrid(const int* token_ids, int start_pos,
                                             int num_tokens, int seq_id, bool want_logits);
    blackwell::EngineStatus step_linear_attention_chunk(int layer_idx, int start_pos,
                                                        int num_tokens, int seq_id);
    blackwell::EngineStatus step_full_attention_chunk(int layer_idx, int start_pos,
                                                      int num_tokens, int seq_id);

    // GLM-4-0414 sandwich-norm epilogue: RMSNorm `num_tokens` rows of
    // d_sublayer_out with the named weight and ADD the result into the residual
    // stream d_X_accum. Called by step_attention_out / step_mlp_out (and their
    // batched twins) in place of the projection's fused residual-accumulate.
    void sandwich_norm_accum(const std::string& weight_name, size_t num_tokens);

    blackwell::EngineStatus step_mlp_norm(int layer_idx);
    blackwell::EngineStatus step_mlp_projections(int layer_idx);
    blackwell::EngineStatus step_mlp_out(int layer_idx);
    blackwell::EngineStatus step_final_ops();
};