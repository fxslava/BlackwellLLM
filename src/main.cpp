#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "common.h"
#include "safetensors.h"
#include "memory_pool.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "fp8_linear.cuh"
#include "bf16_linear.cuh"
#include "rope.cuh"
#include "attention.cuh"
#include "swiglu.cuh"
#include "sampling.cuh"

// Конфигурация Llama 3 / 3.1 8B
const size_t VOCAB_SIZE = 128256;
const size_t HIDDEN_DIM = 4096;
const size_t INTERMEDIATE_DIM = 14336;
const size_t Q_HEADS = 32;
const size_t KV_HEADS = 8;
const size_t HEAD_DIM = 128;
const size_t NUM_LAYERS = 32;
const size_t MAX_SEQ_LEN = 2048;
const float  NORM_EPS = 1e-5f;
const float  ROPE_THETA = 500000.0f;

int main() {
    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Full Prefill Pipeline (1 Token) \n";
    std::cout << "==================================================\n\n";

    try {
        // 1. Инициализация загрузчика и статической арены памяти
        std::string index_path = "llama3-8b-fp8/model.safetensors.index.json";
        SafetensorsLoader loader(index_path);
        VRAMArena arena(loader, MAX_SEQ_LEN);

        // Получаем указатели на статические рабочие буферы арены (FP32)
        float* d_X_accum = arena.get_activation_buffer_A(); // Главный буфер накопителя (Residual)
        float* d_X_norm  = arena.get_activation_buffer_B(); // Буфер для нормализованных данных

        // Локальные буферы для внимания и FFN (выделим их единоразово)
        float *d_Q, *d_K, *d_V, *d_Attn_out, *d_Gate, *d_Up, *d_Swiglu_out;
        CUDA_CHECK(cudaMalloc(&d_Q, Q_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, KV_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, KV_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Attn_out, HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Gate, INTERMEDIATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Up, INTERMEDIATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Swiglu_out, INTERMEDIATE_DIM * sizeof(float)));

        // Буфер для финальных логитов
        float* d_logits;
        CUDA_CHECK(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));

        // 🚨 ИСПРАВЛЕНИЕ: Выделяем память под скейл именно на GPU
        float* d_dummy_scale;
        CUDA_CHECK(cudaMalloc(&d_dummy_scale, sizeof(float)));
        float h_dummy_scale = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_dummy_scale, &h_dummy_scale, sizeof(float), cudaMemcpyHostToDevice));

        // 2. Входные данные
        int current_pos = 0; 
        std::vector<int> h_tokens = {42};
        int* d_tokens;
        CUDA_CHECK(cudaMalloc(&d_tokens, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), sizeof(int), cudaMemcpyHostToDevice));

        std::cout << "[Engine] Running Forward Pass with hardware FP8 scales...\n";

        // ШАГ 1: Входной эмбеддинг
        const void* d_embed_table = arena.get_weight_ptr("model.embed_tokens.weight");
        launch_bf16_embedding_kernel(d_tokens, d_embed_table, d_X_accum, 1, HIDDEN_DIM);

        // ШАГ 2: Цикл по слоям (32 слоя)
        for (size_t l = 0; l < NUM_LAYERS; ++l) {
            std::string layer_prefix = "model.layers." + std::to_string(l) + ".";

            // --- A. Блок Внимания (Attention) ---
            const float* d_attn_norm_w = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "input_layernorm.weight"));
            launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_attn_norm_w, 1, HIDDEN_DIM, NORM_EPS);

            // Извлекаем веса Q, K, V
            const uint8_t* d_q_w = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "self_attn.q_proj.weight"));
            const uint8_t* d_k_w = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "self_attn.k_proj.weight"));
            const uint8_t* d_v_w = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "self_attn.v_proj.weight"));
            
            // 🚨 ИЗВЛЕКАЕМ РЕАЛЬНЫЕ СКЕЙЛЫ КВАНТОВАНИЯ
            // В моделях Neural Magic это обычно векторы [M] или скаляры [1]
            const float* d_q_s = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "self_attn.q_proj.weight_scale"));
            const float* d_k_s = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "self_attn.k_proj.weight_scale"));
            const float* d_v_s = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "self_attn.v_proj.weight_scale"));

            launch_fp8_gemv_kernel(d_q_w, d_X_norm, d_q_s, d_Q, Q_HEADS * HEAD_DIM, HIDDEN_DIM);
            launch_fp8_gemv_kernel(d_k_w, d_X_norm, d_k_s, d_K, KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            launch_fp8_gemv_kernel(d_v_w, d_X_norm, d_v_s, d_V, KV_HEADS * HEAD_DIM, HIDDEN_DIM);

            // RoPE и Attention
            launch_fused_rope_kv_kernel(d_Q, d_K, d_V, arena.get_k_cache(), arena.get_v_cache(), current_pos, Q_HEADS, KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, ROPE_THETA);
            launch_attention_decoding_kernel(d_Q, arena.get_k_cache(), arena.get_v_cache(), d_Attn_out, current_pos, Q_HEADS, KV_HEADS, HEAD_DIM, MAX_SEQ_LEN);

            // Выходная проекция внимания + скейл
            const uint8_t* d_o_w = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "self_attn.o_proj.weight"));
            const float* d_o_s = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "self_attn.o_proj.weight_scale"));
            launch_fp8_gemv_residual_kernel(d_o_w, d_Attn_out, d_o_s, d_X_accum, HIDDEN_DIM, HIDDEN_DIM);

            // --- B. Блок MLP (SwiGLU) ---
            const float* d_ffn_norm_w = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "post_attention_layernorm.weight"));
            launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_ffn_norm_w, 1, HIDDEN_DIM, NORM_EPS);

            const uint8_t* d_gate_w = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "mlp.gate_proj.weight"));
            const uint8_t* d_up_w   = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "mlp.up_proj.weight"));
            
            // Скейлы для MLP
            const float* d_gate_s = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "mlp.gate_proj.weight_scale"));
            const float* d_up_s   = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "mlp.up_proj.weight_scale"));

            launch_fp8_gemv_kernel(d_gate_w, d_X_norm, d_gate_s, d_Gate, INTERMEDIATE_DIM, HIDDEN_DIM);
            launch_fp8_gemv_kernel(d_up_w,   d_X_norm, d_up_s,   d_Up,   INTERMEDIATE_DIM, HIDDEN_DIM);

            launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, INTERMEDIATE_DIM);

            const uint8_t* d_down_w = reinterpret_cast<const uint8_t*>(arena.get_weight_ptr(layer_prefix + "mlp.down_proj.weight"));
            const float* d_down_s = reinterpret_cast<const float*>(arena.get_weight_ptr(layer_prefix + "mlp.down_proj.weight_scale"));
            launch_fp8_gemv_residual_kernel(d_down_w, d_Swiglu_out, d_down_s, d_X_accum, HIDDEN_DIM, INTERMEDIATE_DIM);
        }

        // ====================================================================
        // ШАГ 3: Финальная нормализация и получение логитов
        // ====================================================================
        std::cout << "[Engine] Layers processed. Computing vocabulary logits...\n";

        const float* d_final_norm_w = reinterpret_cast<const float*>(arena.get_weight_ptr("model.norm.weight"));
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_final_norm_w, 1, HIDDEN_DIM, NORM_EPS);

        // Проекция на словарь (lm_head лежит в BF16, как мы выяснили ранее)
        // Для базовой валидации запустим дефолтное умножение BF16 матрицы на наш FP32 вектор d_X_norm
        const void* d_lm_head_w = arena.get_weight_ptr("lm_head.weight");
        
        // *Примечание: Для чистой компиляции MWV подразумевается, что у нас есть ядро BF16_GEMV, 
        // либо мы используем стандартный вызов. Реализуем псевдо-вызов для завершения графа:
        launch_bf16_gemv_kernel(d_lm_head_w, d_X_norm, d_logits, VOCAB_SIZE, HIDDEN_DIM);

        // Синхронизируем и проверяем ошибки всего тяжелого конвейера
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================================
        // ШАГ 4: Сэмплинг (Выбор следующего токена)
        // ====================================================================
        int next_token_id = -1;
        int* d_next_token;
        CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));

        launch_argmax_kernel(d_logits, d_next_token, VOCAB_SIZE);
        CUDA_CHECK(cudaMemcpy(&next_token_id, d_next_token, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "--------------------------------------------------\n";
        std::cout << "[SUCCESS] Pipeline execution complete!\n";
        std::cout << "          Input Token:  " << h_tokens[0] << "\n";
        std::cout << "          Output Token: " << next_token_id << "\n";
        std::cout << "--------------------------------------------------\n\n";

        // Очистка локальных буферов
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
        cudaFree(d_logits); cudaFree(d_tokens); cudaFree(d_next_token);
        cudaFree(d_dummy_scale);

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL PIPELINE ERROR]: " << e.what() << "\n";
    }

    return 0;
}