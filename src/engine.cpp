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
    : loader(index_path), arena(loader, max_seq_len) 
{
    d_X_accum = arena.get_activation_buffer_A();
    d_X_norm  = arena.get_activation_buffer_B();

    // Инициализация аппаратного контекста cuBLASLt
    cublasLtCreate(&cublaslt_handle);

    CUDA_CHECK(cudaMalloc(&d_Q, 32 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Attn_out, hidden_dim * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_Gate, intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Up,   intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Swiglu_out, intermediate_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_token_scale, 2 * sizeof(float)));

    // 🎯 Выделяем буферы под максимальную ширину слоя (intermediate_dim = 14336)
    CUDA_CHECK(cudaMalloc(&d_X_fp8, intermediate_dim * sizeof(char))); // 1 байт на FP8 элемент
    CUDA_CHECK(cudaMalloc(&d_Y_raw, intermediate_dim * sizeof(float)));
}

BlackwellEngine::~BlackwellEngine() {
    cublasLtDestroy(cublaslt_handle); // Уничтожаем контекст
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
    cudaFree(d_logits); cudaFree(d_next_token);
    cudaFree(d_token_scale);
    cudaFree(d_X_fp8);
    cudaFree(d_Y_raw);
}

void BlackwellEngine::step_embedding(int token_id) {
    CudaVector<int> d_tokens(1);
    std::vector<int> h_tokens = {token_id};
    d_tokens.upload(h_tokens);

    CUDA_CHECK(cudaMemset(d_X_accum, 0, hidden_dim * sizeof(float)));

    const void* d_embed_table = arena.get_weight_ptr("model.embed_tokens.weight");
    launch_bf16_embedding_kernel(d_tokens, d_embed_table, d_X_accum, 1, hidden_dim);
}

void BlackwellEngine::step_attention_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "input_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);
}

void BlackwellEngine::step_attention_qkv_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    launch_quantize_per_token_kernel(d_X_norm, d_token_scale, hidden_dim);
    
    auto proj = [&](const std::string& name, float* out, size_t M) {
        const void* w   = arena.get_weight_ptr(prefix + "self_attn." + name + ".weight");
        const void* w_s = arena.get_weight_ptr(prefix + "self_attn." + name + ".weight_scale");
        const void* i_s = arena.get_weight_ptr_optional(prefix + "self_attn." + name + ".input_scale");
        
        // Запуск cuBLASLt с перезаписью буфера (accumulate = false)
        launch_fp8_linear_cublaslt(cublaslt_handle, w, d_X_norm, d_X_fp8, d_Y_raw, w_s, i_s, d_token_scale, out, M, hidden_dim, 1, false);
    };
    proj("q_proj", d_Q, 4096); 
    proj("k_proj", d_K, 1024); 
    proj("v_proj", d_V, 1024);
}

void BlackwellEngine::step_attention_math(int layer_idx, int pos) {
    launch_fused_rope_kv_kernel(d_Q, d_K, d_V, arena.get_k_cache(), arena.get_v_cache(), pos, 32, 8, 128, 2048);
    launch_attention_decoding_kernel(d_Q, arena.get_k_cache(), arena.get_v_cache(), d_Attn_out, pos, 32, 8, 128, 2048);
}

void BlackwellEngine::step_attention_out(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    launch_quantize_per_token_kernel(d_Attn_out, d_token_scale, hidden_dim);

    const void* o_w   = arena.get_weight_ptr(prefix + "self_attn.o_proj.weight");
    const void* o_w_s = arena.get_weight_ptr(prefix + "self_attn.o_proj.weight_scale");
    const void* o_i_s = arena.get_weight_ptr_optional(prefix + "self_attn.o_proj.input_scale");
    
    // Запуск cuBLASLt с остаточным слиянием в d_X_accum (accumulate = true)
    launch_fp8_linear_cublaslt(cublaslt_handle, o_w, d_Attn_out, d_X_fp8, d_Y_raw, o_w_s, o_i_s, d_token_scale, d_X_accum, hidden_dim, hidden_dim, 1, true);
}

void BlackwellEngine::step_mlp_norm(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);
}

void BlackwellEngine::step_mlp_projections(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    launch_quantize_per_token_kernel(d_X_norm, d_token_scale, hidden_dim);

    const void* gate_w   = arena.get_weight_ptr(prefix + "mlp.gate_proj.weight");
    const void* gate_w_s = arena.get_weight_ptr(prefix + "mlp.gate_proj.weight_scale");
    const void* gate_i_s = arena.get_weight_ptr_optional(prefix + "mlp.gate_proj.input_scale");
    launch_fp8_linear_cublaslt(cublaslt_handle, gate_w, d_X_norm, d_X_fp8, d_Y_raw, gate_w_s, gate_i_s, d_token_scale, d_Gate, intermediate_dim, hidden_dim, 1, false);

    const void* up_w   = arena.get_weight_ptr(prefix + "mlp.up_proj.weight");
    const void* up_w_s = arena.get_weight_ptr(prefix + "mlp.up_proj.weight_scale");
    const void* up_i_s = arena.get_weight_ptr_optional(prefix + "mlp.up_proj.input_scale");
    launch_fp8_linear_cublaslt(cublaslt_handle, up_w, d_X_norm, d_X_fp8, d_Y_raw, up_w_s, up_i_s, d_token_scale, d_Up, intermediate_dim, hidden_dim, 1, false);
}

void BlackwellEngine::step_mlp_out(int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, intermediate_dim);
    launch_quantize_per_token_kernel(d_Swiglu_out, d_token_scale, intermediate_dim);

    const void* down_w   = arena.get_weight_ptr(prefix + "mlp.down_proj.weight");
    const void* down_w_s = arena.get_weight_ptr(prefix + "mlp.down_proj.weight_scale");
    const void* down_i_s = arena.get_weight_ptr_optional(prefix + "mlp.down_proj.input_scale");
    launch_fp8_linear_cublaslt(cublaslt_handle, down_w, d_Swiglu_out, d_X_fp8, d_Y_raw, down_w_s, down_i_s, d_token_scale, d_X_accum, hidden_dim, intermediate_dim, 1, true);
}

void BlackwellEngine::step_final_ops() {
    const void* d_w = arena.get_weight_ptr("model.norm.weight");
    launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, hidden_dim);

    const void* d_head_w = arena.get_weight_ptr("lm_head.weight");
    launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, vocab_size, hidden_dim);
    
    launch_argmax_kernel(d_logits, d_next_token, vocab_size);
}

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