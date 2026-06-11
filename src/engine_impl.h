#pragma once
#include "blackwell/engine.h"
#include "blackwell/config.h"
#include "safetensors.h"
#include "memory_pool.h"
#include "ops_dispatcher.h"

struct BlackwellEngine::Impl {
    ModelConfig m_config;
    SafetensorsLoader loader;
    VRAMArena arena;
    LinearDispatcher dispatcher;

    float *d_X_accum = nullptr;
    float *d_X_norm = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Attn_out = nullptr;
    float *d_Gate = nullptr, *d_Up = nullptr, *d_Swiglu_out = nullptr;
    float *d_logits = nullptr;
    int *d_next_token;

    Impl(const std::string& index_path, size_t max_seq_len);
    ~Impl();

    void step_embedding(int token_id);
    void step_attention_norm(int layer_idx);
    void step_attention_qkv_projections(int layer_idx);
    void step_attention_math(int layer_idx, int pos);
    void step_attention_out(int layer_idx);
    void step_mlp_norm(int layer_idx);
    void step_mlp_projections(int layer_idx);
    void step_mlp_out(int layer_idx);
    void step_final_ops();
};