#pragma once
#include <string>
#include <vector>
#include <cublasLt.h> // Подключаем заголовок cuBLASLt
#include "gtest_prod.h"
#include "safetensors.h"
#include "memory_pool.h"

class BlackwellEngine {
public:
    BlackwellEngine(const std::string& index_path, size_t max_seq_len = 2048);
    ~BlackwellEngine();

    int forward(int token_id, int pos);

private:
    FRIEND_TEST(EngineVerificationTest, LayerByLayerComparison);

    void step_embedding(int token_id);
    void step_attention_norm(int layer_idx);
    void step_attention_qkv_projections(int layer_idx);
    void step_attention_math(int layer_idx, int pos);
    void step_attention_out(int layer_idx);

    void step_mlp_norm(int layer_idx);
    void step_mlp_projections(int layer_idx);
    void step_mlp_out(int layer_idx);

    void step_final_ops();

    SafetensorsLoader loader;
    VRAMArena arena;

    // Аппаратный контекст cuBLASLt
    cublasLtHandle_t cublaslt_handle;

    float *d_X_accum; 
    float *d_X_norm;  

    float *d_Q, *d_K, *d_V, *d_Attn_out;
    float *d_Gate, *d_Up, *d_Swiglu_out;
    float *d_logits, *d_token_scale;
    
    // 🎯 Внутренние буферы слоя для конвейера cuBLASLt
    void  *d_X_fp8; // Буфер заквантованных активаций (FP8)
    float *d_Y_raw; // Буфер сырых выходов матричного перемножения (Float32)
    
    int *d_next_token;

    const size_t num_layers = 32;
    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336;
    const size_t vocab_size = 128256;
};