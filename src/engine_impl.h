#pragma once
#include "blackwell/engine.h"
#include "blackwell/config.h"
#include "safetensors.h"
#include "memory_pool.h"
#include "ops_dispatcher.h"
#include "kv_cache/ikv_cache_manager.h"
#include "ssm/ssm_state_pool.h"
#include <memory>
#include <vector>

struct BlackwellEngine::Impl {
    ModelConfig m_config;
    SafetensorsLoader loader;
    VRAMArena arena;
    LinearDispatcher dispatcher;

    // KV-cache strategy (chosen once at construction; see IKVCacheManager).
    // Declared after `arena` so it is destroyed before it -- the adapter holds a
    // reference into the arena. Defaults to the legacy continuous FP32 cache.
    std::unique_ptr<blackwell::IKVCacheManager> kv_mgr;

    // Derived once from m_config (+ kv_mode) at construction; gates fork/rewind.
    ModelCapabilities m_caps;

    // Hybrid linear-attention (SSM) state. Allocated only for models with
    // AttnKind::Linear layers; null otherwise. m_linear_layer_index maps an
    // absolute layer index to its ordinal among the linear layers (-1 if the
    // layer is full attention), which is how SsmStatePool addresses per-layer state.
    std::unique_ptr<blackwell::ssm::SsmStatePool> ssm_state;
    std::vector<int> m_linear_layer_index;
    // Scratch for the linear-attention projections (hybrid models only):
    // d_ssm_qkv holds in_proj_qkv output (conv_dim), d_ssm_z the gate (v_dim).
    float* d_ssm_qkv = nullptr;
    float* d_ssm_z   = nullptr;

    float *d_X_accum = nullptr;
    float *d_X_norm = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Attn_out = nullptr;
    float *d_Gate = nullptr, *d_Up = nullptr, *d_Swiglu_out = nullptr;
    float *d_logits = nullptr;
    int *d_next_token = nullptr;

    Impl(const std::string& index_path, size_t max_seq_len, size_t num_gpu_layers,
         BlackwellEngine::KVCacheMode kv_mode);
    ~Impl();

    void step_embedding(int token_id);
    void step_attention_norm(int layer_idx);
    void step_attention_qkv_projections(int layer_idx);
    void step_attention_math(int layer_idx, int pos);
    void step_attention_out(int layer_idx);
    // Linear-attention (SSM) layer: bypasses the KV cache, evolves the recurrent
    // state in SsmStatePool via the conv1d + GatedDeltaNet kernels.
    void step_linear_attention(int layer_idx, int pos);
    void step_mlp_norm(int layer_idx);
    void step_mlp_projections(int layer_idx);
    void step_mlp_out(int layer_idx);
    void step_final_ops();
};