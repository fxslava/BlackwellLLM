#pragma once
#include <string>
#include <vector>
#include "gtest_prod.h"
#include "safetensors.h"
#include "memory_pool.h"

class BlackwellEngine {
public:
    BlackwellEngine(const std::string& index_path, size_t max_seq_len = 2048);
    ~BlackwellEngine();

    // Публичный метод для работы в чате
    int forward(int token_id, int pos, float temperature = 0.6f, float top_p = 0.9f);

private:
    // Даем тестам доступ к приватным методам и буферам
    FRIEND_TEST(EngineVerificationTest, LayerByLayerComparison);

    // --- Приватные методы прогонки по этапам ---
    void step_embedding(int token_id);
    
    // Гранулярный Attention
    void step_attention_norm(int layer_idx);
    void step_attention_qkv_projections(int layer_idx);
    void step_attention_math(int layer_idx, int pos);
    void step_attention_out(int layer_idx);

    // Гранулярный MLP
    void step_mlp_norm(int layer_idx);
    void step_mlp_projections(int layer_idx);
    void step_mlp_out(int layer_idx);

    void step_final_ops();

    // Ресурсы
    SafetensorsLoader loader;
    VRAMArena arena;

    // Главные рабочие буферы (проецируются из VRAMArena)
    float *d_X_accum; // Накопитель (Residual stream)
    float *d_X_norm;  // Буфер для нормализованных данных

    // Вспомогательные буферы для вычислений внутри слоев
    float *d_Q, *d_K, *d_V, *d_Attn_out;
    float *d_Gate, *d_Up, *d_Swiglu_out;
    float *d_logits, *d_token_scale;
    int *d_next_token;

    // Параметры модели (Llama 3 8B)
    const size_t num_layers = 32;
    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336;
    const size_t vocab_size = 128256;
};