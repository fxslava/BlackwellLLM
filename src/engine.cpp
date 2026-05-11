#include "engine.h"
#include "common.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "fp8_linear.cuh"
#include "bf16_linear.cuh"
#include "rope.cuh"
#include "attention.cuh"
#include "swiglu.cuh"
#include "sampling.cuh"

BlackwellEngine::BlackwellEngine(const std::string& index_path, size_t max_seq_len) 
    : loader(index_path), arena(loader, max_seq_len) 
{
    // 1. Привязываем главные буферы из арены
    d_X_accum = arena.get_activation_buffer_A();
    d_X_norm  = arena.get_activation_buffer_B();

    // 2. Выделяем временные буферы для слоев (один раз)
    CUDA_CHECK(cudaMalloc(&d_Q, 32 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Attn_out, hidden_dim * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_Gate, intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Up,   intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Swiglu_out, intermediate_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));
}

BlackwellEngine::~BlackwellEngine() {
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
    cudaFree(d_logits); cudaFree(d_next_token);
}

// ============================================================================
// ЭТАП 1: Эмбеддинг
// ============================================================================
void BlackwellEngine::step_embedding(int token_id) {
    CudaVector<int> d_tokens(1);
    std::vector<int> h_tokens = {token_id};
    d_tokens.upload(h_tokens);

    CUDA_CHECK(cudaMemset(d_X_accum, 0, hidden_dim * sizeof(float)));

    const void* d_embed_table = arena.get_weight_ptr("model.embed_tokens.weight");
    launch_bf16_embedding_kernel(d_tokens, d_embed_table, d_X_accum, 1, hidden_dim);
}

// ============================================================================
// ЭТАП 2: Гранулярный Attention
// ============================================================================
void BlackwellEngine::step_attention_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const float* d_w = (float*)arena.get_weight_ptr(prefix + "input_layernorm.weight");
    
    // Читает из d_X_accum, пишет нормализованный ответ в d_X_norm
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);
}

void BlackwellEngine::step_attention_qkv_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    auto proj = [&](const std::string& name, float* out, size_t M) {
        auto w = (uint8_t*)arena.get_weight_ptr(prefix + "self_attn." + name + ".weight");
        auto s = (float*)arena.get_weight_ptr(prefix + "self_attn." + name + ".weight_scale");
        launch_fp8_gemv_kernel(w, d_X_norm, s, out, M, hidden_dim);
    };
    proj("q_proj", d_Q, 4096); 
    proj("k_proj", d_K, 1024); 
    proj("v_proj", d_V, 1024);
}

void BlackwellEngine::step_attention_math(int layer_idx, int pos) {
    // RoPE и кэширование KV
    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, arena.get_k_cache(), arena.get_v_cache(), pos, 32, 8, 128, 2048);
    // Математика внимания (декодирование)
    launch_attention_decoding_kernel(d_Q, arena.get_k_cache(), arena.get_v_cache(), d_Attn_out, pos, 32, 8, 128, 2048);
}

void BlackwellEngine::step_attention_out(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    // Выходная проекция с остаточным сложением (Residual Add)
    auto o_w = (uint8_t*)arena.get_weight_ptr(prefix + "self_attn.o_proj.weight");
    auto o_s = (float*)arena.get_weight_ptr(prefix + "self_attn.o_proj.weight_scale");
    launch_fp8_gemv_residual_kernel(o_w, d_Attn_out, o_s, d_X_accum, hidden_dim, hidden_dim);
}

// ============================================================================
// ЭТАП 3: Гранулярный MLP
// ============================================================================
void BlackwellEngine::step_mlp_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const float* d_w = (float*)arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);
}

void BlackwellEngine::step_mlp_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // Нормализация перед MLP
    const float* d_w = (float*)arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);

    // Проекции Gate и Up
    auto gate_w = (uint8_t*)arena.get_weight_ptr(prefix + "mlp.gate_proj.weight");
    auto gate_s = (float*)arena.get_weight_ptr(prefix + "mlp.gate_proj.weight_scale");
    launch_fp8_gemv_kernel(gate_w, d_X_norm, gate_s, d_Gate, intermediate_dim, hidden_dim);

    auto up_w = (uint8_t*)arena.get_weight_ptr(prefix + "mlp.up_proj.weight");
    auto up_s = (float*)arena.get_weight_ptr(prefix + "mlp.up_proj.weight_scale");
    launch_fp8_gemv_kernel(up_w, d_X_norm, up_s, d_Up, intermediate_dim, hidden_dim);
}

void BlackwellEngine::step_mlp_out(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // Активация SwiGLU
    launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, intermediate_dim);

    // Проекция Down с остаточным сложением (Residual Add)
    auto down_w = (uint8_t*)arena.get_weight_ptr(prefix + "mlp.down_proj.weight");
    auto down_s = (float*)arena.get_weight_ptr(prefix + "mlp.down_proj.weight_scale");
    launch_fp8_gemv_residual_kernel(down_w, d_Swiglu_out, down_s, d_X_accum, hidden_dim, intermediate_dim);
}

// ============================================================================
// ЭТАП 4: Финальные операции
// ============================================================================
void BlackwellEngine::step_final_ops() {
    const float* d_w = (float*)arena.get_weight_ptr("model.norm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);

    const void* d_head_w = arena.get_weight_ptr("lm_head.weight");
    launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, vocab_size, hidden_dim);
    
    launch_argmax_kernel(d_logits, d_next_token, vocab_size);
}

// ============================================================================
// Полный проход (для инференса в продакшене)
// ============================================================================
int BlackwellEngine::forward(int token_id, int pos) {
    step_embedding(token_id);
    
    for (size_t i = 0; i < num_layers; ++i) {
        step_attention_norm(i);
        step_attention_qkv_projections(i);
        step_attention_math(i, pos);
        step_attention_out(i);
        
        step_mlp_norm(i);
        step_mlp_projections(i);
        step_mlp_out(i);
    }
    
    step_final_ops();

    int next_id;
    CUDA_CHECK(cudaMemcpy(&next_id, d_next_token, sizeof(int), cudaMemcpyDeviceToHost));
    return next_id;
}