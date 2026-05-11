#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>

#include "common.h"
#include "engine.h"

// Helper function to load raw binary dumps from PyTorch directly into RAM
static std::vector<float> load_golden_dump(const std::string& filename, size_t num_elements) {
    std::string full_path = "D:/Projects/BlackwellLLM/dumps/" + filename;
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Dump file not found: " + full_path + ". Please run generate_golden_dumps.py first.");
    }
    
    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(float));
    
    if (!file) {
        throw std::runtime_error("Dump file size mismatch for requested dimensions: " + full_path);
    }
    return buffer;
}

TEST(EngineVerificationTest, LayerByLayerComparison) {
    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336; // Cleaned up magic number
    const size_t vocab_size = 128256;
    const int start_token = 128000; // Standard BOS token
    const int pos = 0;

    std::cout << "\n[Integration Test] Initializing BlackwellEngine and allocating VRAM...\n";
    //BlackwellEngine engine("llama3-8b-fp8/model.safetensors.index.json", 2048);
    BlackwellEngine engine("D:/Projects/BlackwellLLM/llama3-8b-fp8/model.safetensors.index.json", 2048);

    // Persistent host buffers allocated ONCE outside the loop to prevent overhead
    std::vector<float> h_gpu_buffer(hidden_dim);
    std::vector<float> h_gpu_gate(intermediate_dim);
    std::vector<float> h_gpu_kv(1024);

    // ========================================================================
    // STAGE 1: Embedding Lookup Validation
    // ========================================================================
    std::cout << "[Integration Test] Step 1: Verifying embedding lookup table...\n";
    engine.step_embedding(start_token);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download the accumulation buffer directly from the private d_X_accum pointer
    CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_accum, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::vector<float> golden_embed = load_golden_dump("embed_out.bin", hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        ASSERT_NEAR(golden_embed[i], h_gpu_buffer[i], 1e-4f) 
            << "Embedding mismatch at absolute index: " << i << "\n"
            << "  Expected (PyTorch Golden): " << golden_embed[i] << "\n"
            << "  Actual   (Blackwell GPU):  " << h_gpu_buffer[i] << "\n"
            << "  Tolerance (Epsilon):       1e-4";
    }
    std::cout << "  [OK] Input embeddings perfectly match the PyTorch golden reference!\n";

    // ========================================================================
    // STAGE 2: Layer-by-Layer Verification (All 32 Transformer Layers)
    // ========================================================================
    std::cout << "[Integration Test] Step 2: Step-by-step execution across 32 layers (Attention + MLP)...\n";
    
    for (int l = 0; l < 32; ++l) {
        std::string l_str = std::to_string(l);

        // --- 0. 🚨 ВЕРИФИКАЦИЯ ВХОДНОЙ НОРМАЛИЗАЦИИ (RMSNorm) ---
        engine.step_attention_norm(l);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Скачиваем буфер d_X_norm (куда RMSNorm записал результат)
        CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_norm, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> golden_input_norm = load_golden_dump("layer_" + l_str + "_input_norm.bin", hidden_dim);

        for (size_t i = 0; i < hidden_dim; ++i) {
            ASSERT_NEAR(golden_input_norm[i], h_gpu_buffer[i], 3.5e-2f) 
                << "RMSNorm (Input Layernorm) mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected (PyTorch Golden): " << golden_input_norm[i] << "\n"
                << "  Actual   (Blackwell GPU):  " << h_gpu_buffer[i] << "\n"
                << "  Absolute Difference:       " << std::abs(golden_input_norm[i] - h_gpu_buffer[i]) << "\n"
                << "  Tolerance (Epsilon):       3.5e-2";
        }

        // --- 1. Granular Attention Projections Check (Q, K, V) ---
        engine.step_attention_qkv_projections(l); // <-- Вызываем обновленный метод проекций
        CUDA_CHECK(cudaDeviceSynchronize());

        // 1a. Проверка Q_proj (4096 элементов)
        CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_Q, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> golden_q = load_golden_dump("layer_" + l_str + "_q_proj.bin", hidden_dim);
        for (size_t i = 0; i < hidden_dim; ++i) {
            ASSERT_NEAR(golden_q[i], h_gpu_buffer[i], 4.0e-2f) 
                << "Q_proj mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected: " << golden_q[i] << ", Actual: " << h_gpu_buffer[i];
        }

        // 1b. Проверка K_proj (1024 элемента)
        CUDA_CHECK(cudaMemcpy(h_gpu_kv.data(), engine.d_K, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> golden_k = load_golden_dump("layer_" + l_str + "_k_proj.bin", 1024);
        for (size_t i = 0; i < 1024; ++i) {
            ASSERT_NEAR(golden_k[i], h_gpu_kv[i], 4.0e-2f) 
                << "K_proj mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected: " << golden_k[i] << ", Actual: " << h_gpu_kv[i];
        }

        // 1c. Проверка V_proj (1024 элемента)
        CUDA_CHECK(cudaMemcpy(h_gpu_kv.data(), engine.d_V, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> golden_v = load_golden_dump("layer_" + l_str + "_v_proj.bin", 1024);
        for (size_t i = 0; i < 1024; ++i) {
            ASSERT_NEAR(golden_v[i], h_gpu_kv[i], 4.0e-2f) 
                << "V_proj mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected: " << golden_v[i] << ", Actual: " << h_gpu_kv[i];
        }

        // --- 2. 🚨 ВЕРИФИКАЦИЯ СРАЗУ ПОСЛЕ МАТЕМАТИКИ ВНИМАНИЯ (RoPE + SDPA) ---
        engine.step_attention_math(l, pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Скачиваем буфер d_Attn_out ДО применения выходной проекции o_proj
        CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_Attn_out, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> golden_attn_math = load_golden_dump("layer_" + l_str + "_attn_math.bin", hidden_dim);

        for (size_t i = 0; i < hidden_dim; ++i) {
            ASSERT_NEAR(golden_attn_math[i], h_gpu_buffer[i], 5e-3f) 
                << "Attention Math (RoPE + SDPA) mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected (PyTorch Golden): " << golden_attn_math[i] << "\n"
                << "  Actual   (Blackwell GPU):  " << h_gpu_buffer[i] << "\n"
                << "  Absolute Difference:       " << std::abs(golden_attn_math[i] - h_gpu_buffer[i]) << "\n"
                << "  Tolerance (Epsilon):       5e-3";
        }

        // --- 3. Output Projection ---
        engine.step_attention_out(l);

        engine.step_mlp_norm(l);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- 3. Granular MLP Projections Check (Gate_proj) ---
        engine.step_mlp_projections(l);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_gpu_gate.data(), engine.d_Gate, intermediate_dim * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> golden_gate = load_golden_dump("layer_" + l_str + "_gate_proj.bin", intermediate_dim);

        for (size_t i = 0; i < intermediate_dim; ++i) {
            ASSERT_NEAR(golden_gate[i], h_gpu_gate[i], 4.0e-2f) 
                << "Gate_proj mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected (PyTorch Golden): " << golden_gate[i] << "\n"
                << "  Actual   (Blackwell GPU):  " << h_gpu_gate[i] << "\n"
                << "  Absolute Difference:       " << std::abs(golden_gate[i] - h_gpu_gate[i]) << "\n"
                << "  Tolerance (Epsilon):       5e-3";
        }

        // --- 4. Finalize Layer & Check Residual Stream Accumulation ---
        engine.step_mlp_out(l);

        CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_accum, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

        std::string dump_name = "layer_" + l_str + "_accum_out.bin";
        std::vector<float> golden_layer = load_golden_dump(dump_name, hidden_dim);

        float layer_factor = static_cast<float>(l);
        float dynamic_epsilon = 5.0e-2f + (layer_factor * 1.0e-2f) + (layer_factor * layer_factor * 1.0e-3f);

        for (size_t i = 0; i < hidden_dim; ++i) {
            ASSERT_NEAR(golden_layer[i], h_gpu_buffer[i], dynamic_epsilon) 
                << "Integration mismatch detected at Layer " << l << ", dimension index: " << i << "\n"
                << "  Expected (PyTorch Golden): " << golden_layer[i] << "\n"
                << "  Actual   (Blackwell GPU):  " << h_gpu_buffer[i] << "\n"
                << "  Absolute Difference:       " << std::abs(golden_layer[i] - h_gpu_buffer[i]) << "\n"
                << "  Current Tolerance Limit:   " << dynamic_epsilon;
        }
    }
    std::cout << "  [OK] All 32 transformer layers successfully passed step-by-step numerical verification!\n";

    // ========================================================================
    // STAGE 3: Final Normalization, Logits Projection, and Sampling
    // ========================================================================
    std::cout << "[Integration Test] Step 3: Computing vocabulary projection and sampling...\n";
    engine.step_final_ops();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_gpu_logits.data(), engine.d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> golden_logits = load_golden_dump("logits_out.bin", vocab_size);
    
    for (size_t i = 0; i < vocab_size; ++i) {
        ASSERT_NEAR(golden_logits[i], h_gpu_logits[i], 5e-2f) 
            << "Final output logits mismatch at vocabulary token ID: " << i << "\n"
            << "  Expected Logit (PyTorch): " << golden_logits[i] << "\n"
            << "  Actual Logit   (GPU):     " << h_gpu_logits[i] << "\n"
            << "  Tolerance (Epsilon):      5e-2";
    }

    int next_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&next_token_id, engine.d_next_token, sizeof(int), cudaMemcpyDeviceToHost));
    
    int golden_max_idx = 0;
    float max_val = golden_logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (golden_logits[i] > max_val) {
            max_val = golden_logits[i];
            golden_max_idx = static_cast<int>(i);
        }
    }

    std::cout << "  [OK] Logits verified! Selected Token ID: " << next_token_id 
              << " (PyTorch Reference: " << golden_max_idx << ")\n";
    
    ASSERT_EQ(golden_max_idx, next_token_id) 
        << "Critical error: C++ Argmax kernel selected an incorrect final token!\n"
        << "  Expected Token ID: " << golden_max_idx << "\n"
        << "  Actual Token ID:   " << next_token_id;
}