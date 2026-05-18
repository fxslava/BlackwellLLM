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
#include <iomanip>

BlackwellEngine::BlackwellEngine(const std::string& index_path, size_t max_seq_len) 
    : loader(index_path), arena(index_path, loader, max_seq_len)
{
    // 1. Bind core activation buffers from the arena
    d_X_accum = arena.get_activation_buffer_A();
    d_X_norm  = arena.get_activation_buffer_B();

    // 2. Allocate layer-scoped compute buffers once
    CUDA_CHECK(cudaMalloc(&d_Q, 32 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Attn_out, hidden_dim * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_Gate, intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Up,   intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Swiglu_out, intermediate_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));
    
    // 🎯 Выделяем память под буфер динамического скейла токена (scale и inv_scale)
    CUDA_CHECK(cudaMalloc(&d_token_scale, 2 * sizeof(float)));

    CUDA_CHECK(cudaMemset(arena.get_k_cache(), 0, arena.get_k_cache_size()));
    CUDA_CHECK(cudaMemset(arena.get_v_cache(), 0, arena.get_v_cache_size()));
}

BlackwellEngine::~BlackwellEngine() {
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
    cudaFree(d_logits); cudaFree(d_next_token);
    cudaFree(d_token_scale); // 🎯 Не забываем освобождать память
}

// ============================================================================
// STAGE 1: Embedding
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
// STAGE 2: Granular Attention
// ============================================================================
void BlackwellEngine::step_attention_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "input_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);
}

void BlackwellEngine::step_attention_qkv_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    // 🎯 Перед проекциями рассчитываем динамический скейл текущего нормализованного потока
    launch_quantize_per_token_kernel(d_X_norm, d_token_scale, hidden_dim);
    
    auto proj = [&](const std::string& name, float* out, size_t M) {
        const void* w   = arena.get_weight_ptr(prefix + "self_attn." + name + ".weight");
        const void* w_s = arena.get_weight_ptr(prefix + "self_attn." + name + ".weight_scale");
        const void* i_s = arena.get_weight_ptr_optional(prefix + "self_attn." + name + ".input_scale");
        
        // 🎯 Строго соблюдаем порядок аргументов: W, X, W_scale, Input_scale, Token_scale
        launch_fp8_gemv_kernel(w, d_X_norm, w_s, i_s, d_token_scale, out, M, hidden_dim, 1);
    };
    proj("q_proj", d_Q, 4096); 
    proj("k_proj", d_K, 1024); 
    proj("v_proj", d_V, 1024);
}

void BlackwellEngine::step_attention_math(int layer_idx, int pos) {
    // Шаг смещения для одного слоя = (число_kv_голов * max_seq_len * head_dim) элементов.
    // Для архитектуры Llama 3 8B: 8 голов * max_seq_len * 128
    size_t layer_cache_offset = layer_idx * (8 * arena.get_max_seq_len() * 128);

    float* d_layer_k_cache = arena.get_k_cache() + layer_cache_offset;
    float* d_layer_v_cache = arena.get_v_cache() + layer_cache_offset;

    // Передаем смещенные указатели целевого слоя
    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, d_layer_k_cache, d_layer_v_cache, pos, 32, 8, 128, arena.get_max_seq_len(), 500000.0f);
    launch_attention_decoding_kernel(d_Q, d_layer_k_cache, d_layer_v_cache, d_Attn_out, pos, 32, 8, 128, arena.get_max_seq_len());
}

void BlackwellEngine::step_attention_out(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    // 🎯 Оцениваем скейл для выхода внимания перед сверткой остаточным ядром
    launch_quantize_per_token_kernel(d_Attn_out, d_token_scale, hidden_dim);

    const void* o_w   = arena.get_weight_ptr(prefix + "self_attn.o_proj.weight");
    const void* o_w_s = arena.get_weight_ptr(prefix + "self_attn.o_proj.weight_scale");
    const void* o_i_s = arena.get_weight_ptr_optional(prefix + "self_attn.o_proj.input_scale");
    
    launch_fp8_gemv_residual_kernel(o_w, d_Attn_out, o_w_s, o_i_s, d_token_scale, d_X_accum, hidden_dim, hidden_dim, 1);
}

// ============================================================================
// STAGE 3: Granular MLP
// ============================================================================
void BlackwellEngine::step_mlp_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);
}

void BlackwellEngine::step_mlp_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // 🎯 Оцениваем скейл перед входом в проекции перцептрона
    launch_quantize_per_token_kernel(d_X_norm, d_token_scale, hidden_dim);

    const void* gate_w   = arena.get_weight_ptr(prefix + "mlp.gate_proj.weight");
    const void* gate_w_s = arena.get_weight_ptr(prefix + "mlp.gate_proj.weight_scale");
    const void* gate_i_s = arena.get_weight_ptr_optional(prefix + "mlp.gate_proj.input_scale");
    launch_fp8_gemv_kernel(gate_w, d_X_norm, gate_w_s, gate_i_s, d_token_scale, d_Gate, intermediate_dim, hidden_dim, 1);

    const void* up_w   = arena.get_weight_ptr(prefix + "mlp.up_proj.weight");
    const void* up_w_s = arena.get_weight_ptr(prefix + "mlp.up_proj.weight_scale");
    const void* up_i_s = arena.get_weight_ptr_optional(prefix + "mlp.up_proj.input_scale");
    launch_fp8_gemv_kernel(up_w, d_X_norm, up_w_s, up_i_s, d_token_scale, d_Up, intermediate_dim, hidden_dim, 1);
}

void BlackwellEngine::step_mlp_out(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, intermediate_dim);

    // 🎯 Рассчитываем скейл для широкого промежуточного вектора (14336 элементов) перед down_proj
    launch_quantize_per_token_kernel(d_Swiglu_out, d_token_scale, intermediate_dim);

    const void* down_w   = arena.get_weight_ptr(prefix + "mlp.down_proj.weight");
    const void* down_w_s = arena.get_weight_ptr(prefix + "mlp.down_proj.weight_scale");
    const void* down_i_s = arena.get_weight_ptr_optional(prefix + "mlp.down_proj.input_scale");
    launch_fp8_gemv_residual_kernel(down_w, d_Swiglu_out, down_w_s, down_i_s, d_token_scale, d_X_accum, hidden_dim, intermediate_dim, 1);
}

// ============================================================================
// STAGE 4: Final Operations
// ============================================================================
void BlackwellEngine::step_final_ops() {
    const void* d_w = arena.get_weight_ptr("model.norm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);

    const void* d_head_w = arena.get_weight_ptr("lm_head.weight");
    // Вычисляем логиты. Результат остается лежать в массиве d_logits на GPU
    launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, vocab_size, hidden_dim);
}

// ============================================================================
// Full Engine Inference
// ============================================================================
// Добавляем параметры сэмплирования прямо в forward (со значениями по умолчанию)
int BlackwellEngine::forward(int token_id, int pos, float temperature, float top_p) {
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
    
    // Подготавливаем логиты
    step_final_ops();

    // 🎯 ЗАПУСКАЕМ НАШ НОВЫЙ CPU/GPU СЭМПЛЕР
    // Он сам скачает логиты, применит температуру, softmax, отсечение top-p и выберет токен
    int next_id = sample_top_p(d_logits, vocab_size, temperature, top_p);
    
    return next_id;
}

// ============================================================================
// Evaluation Inference (Для расчета Перплексии)
// ============================================================================
float BlackwellEngine::forward_eval(int token_id, int pos, int target_token_id) {
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
    
    step_final_ops(); // Готовим массив d_logits

    // Возвращаем логарифм вероятности целевого токена!
    return compute_log_prob(d_logits, vocab_size, target_token_id);
}