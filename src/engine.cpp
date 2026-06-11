#include "engine_impl.h"
#include "blackwell/engine.h"
#include "common.h"
#include "kernels/embedding.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/bf16_linear.cuh"
#include "kernels/rope.cuh"
#include "kernels/attention.cuh"
#include "kernels/swiglu.cuh"
#include "kernels/sampling.cuh"
#include <filesystem>
#include <iomanip>

BlackwellEngine::Impl::Impl(const std::string& index_path, size_t max_seq_len)
    : loader(index_path),
      m_config(ConfigLoader::load_from_json(
          (std::filesystem::path(index_path).parent_path() / "config.json").string())),
      arena(index_path, loader, m_config, max_seq_len),
      dispatcher(arena, m_config)
{
    // 1. Bind core activation buffers from the arena
    d_X_accum = arena.get_activation_buffer_A();
    d_X_norm  = arena.get_activation_buffer_B();

    // 2. Allocate layer-scoped compute buffers once
    CUDA_CHECK(cudaMalloc(&d_Q,        m_config.num_attention_heads * m_config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,        m_config.num_key_value_heads * m_config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,        m_config.num_key_value_heads * m_config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Attn_out, m_config.hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Gate,       m_config.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Up,         m_config.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Swiglu_out, m_config.intermediate_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_logits, m_config.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));
    
    // 🎯 Выделяем память под буфер динамического скейла токена (scale и inv_scale)
    CUDA_CHECK(cudaMalloc(&d_token_scale, 2 * sizeof(float)));

    CUDA_CHECK(cudaMemset(arena.get_k_cache(), 0, arena.get_k_cache_size()));
    CUDA_CHECK(cudaMemset(arena.get_v_cache(), 0, arena.get_v_cache_size()));
}

BlackwellEngine::Impl::~Impl() {
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
    cudaFree(d_logits); cudaFree(d_next_token);
    cudaFree(d_token_scale); // 🎯 Не забываем освобождать память
}

// ============================================================================
// STAGE 1: Embedding
// ============================================================================
void BlackwellEngine::Impl::step_embedding(int token_id) {
    CudaVector<int> d_tokens(1);
    std::vector<int> h_tokens = {token_id};
    d_tokens.upload(h_tokens);

    CUDA_CHECK(cudaMemset(d_X_accum, 0, m_config.hidden_dim * sizeof(float)));

    const void* d_embed_table = arena.get_weight_ptr("model.embed_tokens.weight");
    launch_bf16_embedding_kernel(d_tokens, d_embed_table, d_X_accum, 1, m_config.hidden_dim);
}

// ============================================================================
// STAGE 2: Granular Attention
// ============================================================================
void BlackwellEngine::Impl::step_attention_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "input_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim);
}

void BlackwellEngine::Impl::step_attention_qkv_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    launch_quantize_per_token_kernel(d_X_norm, d_token_scale, m_config.hidden_dim);

    dispatcher.forward(prefix + "q_proj", d_X_norm, d_Q,
                       m_config.num_attention_heads * m_config.head_dim, m_config.hidden_dim, d_token_scale);
    dispatcher.forward(prefix + "k_proj", d_X_norm, d_K,
                       m_config.num_key_value_heads * m_config.head_dim, m_config.hidden_dim, d_token_scale);
    dispatcher.forward(prefix + "v_proj", d_X_norm, d_V,
                       m_config.num_key_value_heads * m_config.head_dim, m_config.hidden_dim, d_token_scale);
}

void BlackwellEngine::Impl::step_attention_math(int layer_idx, int pos) {
    size_t layer_cache_offset = layer_idx * (m_config.num_key_value_heads * arena.get_max_seq_len() * m_config.head_dim);

    float* d_layer_k_cache = arena.get_k_cache() + layer_cache_offset;
    float* d_layer_v_cache = arena.get_v_cache() + layer_cache_offset;

    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, d_layer_k_cache, d_layer_v_cache, pos,
        m_config.num_attention_heads, m_config.num_key_value_heads, m_config.head_dim,
        arena.get_max_seq_len(), m_config.rope_theta);
    launch_attention_decoding_kernel(d_Q, d_layer_k_cache, d_layer_v_cache, d_Attn_out, pos,
        m_config.num_attention_heads, m_config.num_key_value_heads, m_config.head_dim,
        arena.get_max_seq_len());
}

void BlackwellEngine::Impl::step_attention_out(int layer_idx) {
    std::string base = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj";

    launch_quantize_per_token_kernel(d_Attn_out, d_token_scale, m_config.hidden_dim);

    dispatcher.forward(base, d_Attn_out, nullptr,
                       m_config.hidden_dim, m_config.hidden_dim, d_token_scale, d_X_accum);
}

// ============================================================================
// STAGE 3: Granular MLP
// ============================================================================
void BlackwellEngine::Impl::step_mlp_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim);
}

void BlackwellEngine::Impl::step_mlp_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";

    launch_quantize_per_token_kernel(d_X_norm, d_token_scale, m_config.hidden_dim);

    dispatcher.forward(prefix + "gate_proj", d_X_norm, d_Gate,
                       m_config.intermediate_dim, m_config.hidden_dim, d_token_scale);
    dispatcher.forward(prefix + "up_proj",   d_X_norm, d_Up,
                       m_config.intermediate_dim, m_config.hidden_dim, d_token_scale);
}

void BlackwellEngine::Impl::step_mlp_out(int layer_idx) {
    std::string base = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj";

    launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, m_config.intermediate_dim);

    launch_quantize_per_token_kernel(d_Swiglu_out, d_token_scale, m_config.intermediate_dim);

    dispatcher.forward(base, d_Swiglu_out, nullptr,
                       m_config.hidden_dim, m_config.intermediate_dim, d_token_scale, d_X_accum);
}

// ============================================================================
// STAGE 4: Final Operations
// ============================================================================
void BlackwellEngine::Impl::step_final_ops() {
    const void* d_w = arena.get_weight_ptr("model.norm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim);

    const void* d_head_w = arena.get_weight_ptr("lm_head.weight");
    launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, m_config.hidden_dim);
}

// Реализация фасада BlackwellEngine
BlackwellEngine::BlackwellEngine(const std::string& index_path, size_t max_seq_len)
    : pImpl(std::make_unique<Impl>(index_path, max_seq_len)) {}

BlackwellEngine::~BlackwellEngine() = default;

// ============================================================================
// Full Engine Inference
// ============================================================================
// Добавляем параметры сэмплирования прямо в forward (со значениями по умолчанию)
int BlackwellEngine::forward(int token_id, int pos, float temperature, float top_p) {
    auto* impl = pImpl.get();
    impl->step_embedding(token_id);
    
    for (size_t i = 0; i < impl->m_config.num_layers; ++i) {
        impl->step_attention_norm(i);
        impl->step_attention_qkv_projections(i);
        impl->step_attention_math(i, pos);
        impl->step_attention_out(i);

        impl->step_mlp_norm(i);
        impl->step_mlp_projections(i);
        impl->step_mlp_out(i);
    }

    impl->step_final_ops();

    int next_id = sample_top_p(impl->d_logits, impl->m_config.vocab_size, temperature, top_p);
    
    return next_id;
}

// ============================================================================
// Evaluation Inference (Для расчета Перплексии)
// ============================================================================
float BlackwellEngine::forward_eval(int token_id, int pos, int target_token_id) {
    auto* impl = pImpl.get();
    impl->step_embedding(token_id);
    
    for (size_t i = 0; i < impl->m_config.num_layers; ++i) {
        impl->step_attention_norm(i);
        impl->step_attention_qkv_projections(i);
        impl->step_attention_math(i, pos);
        impl->step_attention_out(i);

        impl->step_mlp_norm(i);
        impl->step_mlp_projections(i);
        impl->step_mlp_out(i);
    }

    impl->step_final_ops();

    return compute_log_prob(impl->d_logits, impl->m_config.vocab_size, target_token_id);
}