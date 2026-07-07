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

    // NON-owning views into the arena's ping-pong activation pool (bound in the
    // ctor body; the arena frees them) -- the only raw device pointers left here.
    float *d_X_accum = nullptr;
    float *d_X_norm = nullptr;
    // Owned per-step compute scratch (allocated unconditionally in the ctor).
    blackwell::DeviceBuffer<float> d_Q, d_K, d_V, d_Attn_out;
    blackwell::DeviceBuffer<float> d_Gate, d_Up, d_Swiglu_out;
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

    blackwell::EngineStatus step_embedding(int token_id);
    blackwell::EngineStatus step_attention_norm(int layer_idx);
    blackwell::EngineStatus step_attention_qkv_projections(int layer_idx);
    blackwell::EngineStatus step_attention_math(int layer_idx, int pos);
    blackwell::EngineStatus step_attention_out(int layer_idx);
    // Linear-attention (SSM) layer: bypasses the KV cache, evolves the recurrent
    // state in SsmStatePool via the conv1d + GatedDeltaNet kernels.
    blackwell::EngineStatus step_linear_attention(int layer_idx, int pos);
    // Qwen3.5 hybrid gated full-attention layer (head_dim 256, q_proj query|gate,
    // q_norm/k_norm, partial RoPE). Uses the dedicated full-attn KV cache.
    blackwell::EngineStatus step_full_attention(int layer_idx, int pos);
    blackwell::EngineStatus step_mlp_norm(int layer_idx);
    blackwell::EngineStatus step_mlp_projections(int layer_idx);
    blackwell::EngineStatus step_mlp_out(int layer_idx);
    blackwell::EngineStatus step_final_ops();
};