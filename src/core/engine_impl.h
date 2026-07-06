#pragma once
#include "blackwell/engine.h"
#include "blackwell/config.h"
#include "blackwell/runtime_config.h"
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
    // Scratch for the linear-attention (GatedDeltaNet) decode step (hybrid only).
    // d_ssm_qkv: in_proj_qkv out (conv_dim); d_ssm_qkv_conv: post-conv1d; q/k/v:
    // per-value-head split (H*head_dim); z: gate (v_dim); a/b: per-head dt/beta
    // sources (H); core: scan output; o: gated-normed output. The *_f32 buffers
    // hold per-layer bf16 params cast to fp32 for the fp32 kernels.
    float* d_ssm_qkv = nullptr;
    float* d_ssm_z   = nullptr;
    float* d_ssm_qkv_conv = nullptr;
    float* d_ssm_q = nullptr, *d_ssm_k = nullptr, *d_ssm_v = nullptr;
    float* d_ssm_a = nullptr, *d_ssm_b = nullptr;
    float* d_ssm_core = nullptr, *d_ssm_o = nullptr;
    float* d_dt_bias_f32 = nullptr, *d_A_log_f32 = nullptr;
    float* d_norm_f32 = nullptr, *d_conv_w_f32 = nullptr;

    float *d_X_accum = nullptr;
    float *d_X_norm = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Attn_out = nullptr;
    float *d_Gate = nullptr, *d_Up = nullptr, *d_Swiglu_out = nullptr;
    float *d_logits = nullptr;
    int *d_next_token = nullptr;

    // Qwen3.5 hybrid FULL-attention scratch + cache (allocated only when the model
    // uses gated head_dim-256 attention, i.e. m_config.attn_output_gate). These
    // layers cannot use the shared 128-wide attention/KV path, so they keep a
    // dedicated continuous FP32 KV cache indexed by m_full_layer_index. d_QG holds
    // the [num_heads, 2*head_dim] q_proj output (query|gate); d_gate is the split
    // gate half. Hybrid models never branch, so a single continuous cache suffices.
    std::vector<int> m_full_layer_index;   // absolute layer -> full-attn ordinal (-1 if linear)
    float *d_QG = nullptr, *d_gate = nullptr;
    float *d_full_k_cache = nullptr, *d_full_v_cache = nullptr;
    size_t m_full_kv_layer_stride = 0;     // floats per layer in each of K/V cache

    Impl(const std::string& index_path, const blackwell::InferenceConfig& request,
         const blackwell::RuntimeOverrides& overrides);
    ~Impl();

    // One full decoder pass for one token: embedding -> N transformer layers
    // (per-token KV control plane latched via kv_mgr) -> optionally final norm
    // + lm_head. want_logits=false is the prefill fast path: every non-final
    // prompt token skips the vocab-size GEMV, since only its KV append matters.
    void run_token(int token_id, int pos, int seq_id, bool want_logits = true);

    void step_embedding(int token_id);
    void step_attention_norm(int layer_idx);
    void step_attention_qkv_projections(int layer_idx);
    void step_attention_math(int layer_idx, int pos);
    void step_attention_out(int layer_idx);
    // Linear-attention (SSM) layer: bypasses the KV cache, evolves the recurrent
    // state in SsmStatePool via the conv1d + GatedDeltaNet kernels.
    void step_linear_attention(int layer_idx, int pos);
    // Qwen3.5 hybrid gated full-attention layer (head_dim 256, q_proj query|gate,
    // q_norm/k_norm, partial RoPE). Uses the dedicated full-attn KV cache.
    void step_full_attention(int layer_idx, int pos);
    void step_mlp_norm(int layer_idx);
    void step_mlp_projections(int layer_idx);
    void step_mlp_out(int layer_idx);
    void step_final_ops();
};