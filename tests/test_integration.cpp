#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

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
    const size_t intermediate_dim = 14336; 
    const size_t vocab_size = 128256;
    const int start_token = 128000; // Standard BOS token
    const int pos = 0;

    std::cout << "\n[Integration Test] Initializing BlackwellEngine and allocating VRAM...\n";
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
    // STAGE 2: Core Execution Pipeline (Granular checks gated to Layer 0 only)
    // ========================================================================
    std::cout << "[Integration Test] Step 2: Executing 32 layers (Detailed dump verification on Layer 0 only)...\n";
    
    for (int l = 0; l < 32; ++l) {
        std::string l_str = std::to_string(l);

        // --- 0. RMSNorm ---
        engine.step_attention_norm(l);
        
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_norm, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_input_norm = load_golden_dump("layer_0_input_norm.bin", hidden_dim);

            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_input_norm[i], h_gpu_buffer[i], 3.5e-2f) 
                    << "RMSNorm mismatch detected at Layer 0, dimension index: " << i << "\n"
                    << "  Expected: " << golden_input_norm[i] << ", Actual: " << h_gpu_buffer[i];
            }
        }

        // --- 1. Attention Projections (Q, K, V) ---
        engine.step_attention_qkv_projections(l);
        
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());

            // 1a. Q_proj check
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_Q, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_q = load_golden_dump("layer_0_q_proj.bin", hidden_dim);
            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_q[i], h_gpu_buffer[i], 4.0e-2f) 
                    << "Q_proj mismatch detected at Layer 0, dimension index: " << i << "\n"
                    << "  Expected: " << golden_q[i] << ", Actual: " << h_gpu_buffer[i];
            }

            // 1b. K_proj check
            CUDA_CHECK(cudaMemcpy(h_gpu_kv.data(), engine.d_K, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_k = load_golden_dump("layer_0_k_proj.bin", 1024);
            for (size_t i = 0; i < 1024; ++i) {
                ASSERT_NEAR(golden_k[i], h_gpu_kv[i], 4.0e-2f) 
                    << "K_proj mismatch detected at Layer 0, dimension index: " << i;
            }

            // 1c. V_proj check
            CUDA_CHECK(cudaMemcpy(h_gpu_kv.data(), engine.d_V, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_v = load_golden_dump("layer_0_v_proj.bin", 1024);
            for (size_t i = 0; i < 1024; ++i) {
                ASSERT_NEAR(golden_v[i], h_gpu_kv[i], 4.0e-2f) 
                    << "V_proj mismatch detected at Layer 0, dimension index: " << i;
            }
        }

        // --- 2. Attention Math (RoPE + SDPA) ---
        engine.step_attention_math(l, pos);
        
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_Attn_out, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_attn_math = load_golden_dump("layer_0_attn_math.bin", hidden_dim);

            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_attn_math[i], h_gpu_buffer[i], 5e-3f) 
                    << "Attention Math mismatch detected at Layer 0, dimension index: " << i;
            }
        }

        // --- 3. Output Projection & MLP Normalization ---
        engine.step_attention_out(l);
        engine.step_mlp_norm(l);

        // --- 4. MLP Projections (Gate & Up) ---
        engine.step_mlp_projections(l);
        
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_gate.data(), engine.d_Gate, intermediate_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_gate = load_golden_dump("layer_0_gate_proj.bin", intermediate_dim);

            for (size_t i = 0; i < intermediate_dim; ++i) {
                ASSERT_NEAR(golden_gate[i], h_gpu_gate[i], 4.0e-2f) 
                    << "Gate_proj mismatch detected at Layer 0, dimension index: " << i << "\n"
                    << "  Expected: " << golden_gate[i] << ", Actual: " << h_gpu_gate[i];
            }
            std::cout << "  [OK] Layer 0 passed full structural kernel validation successfully!\n";
        }

        // --- 5. MLP Output & Residual Stream Accumulation ---
        engine.step_mlp_out(l);

        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_accum, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_layer = load_golden_dump("layer_0_accum_out.bin", hidden_dim);

            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_layer[i], h_gpu_buffer[i], 5.0e-2f) 
                    << "Accumulated residual stream mismatch at Layer 0, index: " << i;
            }
        }
    }
    std::cout << "  [OK] Execution completed across all 32 layers seamlessly.\n";

    // ========================================================================
    // STAGE 3: Final Normalization, Logits Projection, and Sampling
    // ========================================================================
    std::cout << "[Integration Test] Step 3: Computing vocabulary projection and sampling...\n";
    engine.step_final_ops();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_gpu_logits.data(), engine.d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> golden_logits = load_golden_dump("logits_out.bin", vocab_size);

    // Находим победителя по версии эталонного дампа PyTorch
    int golden_max_idx = 0;
    float golden_max_val = golden_logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (golden_logits[i] > golden_max_val) {
            golden_max_val = golden_logits[i];
            golden_max_idx = static_cast<int>(i);
        }
    }

    // Находим победителя непосредственно в выгруженных логитах GPU (для детального лога)
    int gpu_max_idx = 0;
    float gpu_max_val = h_gpu_logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (h_gpu_logits[i] > gpu_max_val) {
            gpu_max_val = h_gpu_logits[i];
            gpu_max_idx = static_cast<int>(i);
        }
    }

    // Скачиваем результат работы нашего аппаратного ядра Argmax
    int next_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&next_token_id, engine.d_next_token, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "  [INFO] PyTorch Top-1 Token: " << golden_max_idx << " (Logit: " << golden_max_val << ")\n";
    std::cout << "  [INFO] Custom GPU Top-1 Token: " << gpu_max_idx << " (Logit: " << gpu_max_val << ")\n";
    std::cout << "  [INFO] Hardware Argmax Result: " << next_token_id << "\n";

    // Допускаем системный сдвиг логита победителя в пределах 1.5 единиц (учитывая инвариантность Softmax)
    ASSERT_NEAR(golden_max_val, gpu_max_val, 1.5f)
        << "Top-1 logit amplitude baseline drifted beyond stable bounds!";

    // ========================================================================
    // ДИАГНОСТИЧЕСКИЙ ЭКСПОРТ ЛОГИТОВ В CSV
    // ========================================================================
    std::cout << "  [INFO] Экспорт сравнительной таблицы логитов в файл logits_comparison.csv...\n";
    std::ofstream csv_file("logits_comparison.csv");
    if (csv_file.is_open()) {
        // Записываем заголовок таблицы
        csv_file << "TokenID,GoldenLogit,GPULogit,AbsDifference\n";
        
        // Выгружаем полный словарь
        for (size_t i = 0; i < vocab_size; ++i) {
            float diff = std::abs(golden_logits[i] - h_gpu_logits[i]);
            csv_file << i << "," 
                     << golden_logits[i] << "," 
                     << h_gpu_logits[i] << "," 
                     << diff << "\n";
        }
        csv_file.close();
        std::cout << "  [OK] Таблица логитов успешно сохранена в logits_comparison.csv!\n";
    } else {
        std::cerr << "  [ERROR] Не удалось открыть файл для записи CSV-дампа!\n";
    }

    // ========================================================================
    // АНАЛИЗ ВЕРХУШКИ РАСПРЕДЕЛЕНИЯ (ТОП-5)
    // ========================================================================
    std::cout << "\n  --- Top-5 PyTorch Golden Logits ---\n";
    // Создаем вектор индексов и сортируем его по убыванию эталонных логитов
    std::vector<int> golden_indices(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) golden_indices[i] = static_cast<int>(i);
    std::sort(golden_indices.begin(), golden_indices.end(), [&](int a, int b) {
        return golden_logits[a] > golden_logits[b];
    });
    for (int k = 0; k < 5; ++k) {
        int idx = golden_indices[k];
        std::cout << "  Rank " << k+1 << ": Token " << idx 
                  << " | Golden: " << golden_logits[idx] 
                  << " | GPU actual: " << h_gpu_logits[idx] 
                  << " | Diff: " << std::abs(golden_logits[idx] - h_gpu_logits[idx]) << "\n";
    }

    std::cout << "\n  --- Top-5 Custom GPU Logits ---\n";
    std::vector<int> gpu_indices(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) gpu_indices[i] = static_cast<int>(i);
    std::sort(gpu_indices.begin(), gpu_indices.end(), [&](int a, int b) {
        return h_gpu_logits[a] > h_gpu_logits[b];
    });
    for (int k = 0; k < 5; ++k) {
        int idx = gpu_indices[k];
        std::cout << "  Rank " << k+1 << ": Token " << idx 
                  << " | GPU actual: " << h_gpu_logits[idx] 
                  << " | Golden: " << golden_logits[idx] 
                  << " | Diff: " << std::abs(golden_logits[idx] - h_gpu_logits[idx]) << "\n";
    }
    std::cout << "\n";

    // 🎯 ФИНАЛЬНАЯ ПРОВЕРКА: Совпал ли итоговый ответ сети?
    ASSERT_EQ(golden_max_idx, next_token_id) 
        << "Critical error: Network produced an incorrect output token distribution after 32 layers!\n"
        << "  Expected Token ID: " << golden_max_idx << "\n"
        << "  Actual Token ID:   " << next_token_id;

    std::cout << "\n  🎉 [SUCCESS] The FP8 custom engine correctly predicted the identical next token!\n";
}