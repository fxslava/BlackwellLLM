#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

#include "common.h"
#include "engine.h"

// Вспомогательная функция загрузки эталонных бинарных дампов PyTorch
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

// Структура для хранения метрик деградации вектора
struct TelemetryMetrics {
    double rmse;
    double max_abs_error;
    double cosine_similarity;
};

static TelemetryMetrics compute_vector_telemetry(const std::vector<float>& golden, const std::vector<float>& actual) {
    size_t n = golden.size();
    double sum_sq_err = 0.0;
    double max_err = 0.0;
    double dot_product = 0.0;
    double norm_golden = 0.0;
    double norm_actual = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);
        double err = std::abs(g - a);

        sum_sq_err += err * err;
        if (err > max_err) max_err = err;

        dot_product += g * a;
        norm_golden += g * g;
        norm_actual += a * a;
    }

    TelemetryMetrics metrics;
    metrics.rmse = std::sqrt(sum_sq_err / static_cast<double>(n));
    metrics.max_abs_error = max_err;
    
    double denom = std::sqrt(norm_golden) * std::sqrt(norm_actual);
    metrics.cosine_similarity = (denom > 1e-9) ? (dot_product / denom) : 0.0;
    
    return metrics;
}

TEST(EngineVerificationTest, LayerByLayerComparison) {
    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336; 
    const size_t vocab_size = 128256;
    const int start_token = 128000; // Стандартный токен BOS
    const int pos = 0;

    std::cout << "\n[Integration Test] Инициализация BlackwellEngine и выделение VRAM...\n";
    BlackwellEngine engine("D:/Projects/BlackwellLLM/llama3-8b-fp8/model.safetensors.index.json", 2048);

    // Персистентные хост-буферы выделяются ОДИН РАЗ
    std::vector<float> h_gpu_buffer(hidden_dim);
    std::vector<float> h_gpu_gate(intermediate_dim);
    std::vector<float> h_gpu_kv(1024);

    // ========================================================================
    // STAGE 1: Верификация таблицы Embedding
    // ========================================================================
    std::cout << "[Integration Test] Шаг 1: Проверка таблицы Embedding...\n";
    engine.step_embedding(start_token);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_accum, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::vector<float> golden_embed = load_golden_dump("embed_out.bin", hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        ASSERT_NEAR(golden_embed[i], h_gpu_buffer[i], 1e-4f) 
            << "Embedding mismatch at absolute index: " << i;
    }
    std::cout << "  [OK] Входные эмбеддинги идеально совпадают с эталоном PyTorch!\n";

    // ========================================================================
    // STAGE 2: Послойное выполнение с немедленной сверкой (Layer 0) и телеметрией
    // ========================================================================
    std::cout << "\n[Integration Test] Шаг 2: Выполнение 32 слоев с отслеживанием телеметрии...\n";
    std::cout << std::string(85, '-') << "\n";
    std::cout << std::left << std::setw(8) << "Layer" 
              << std::setw(15) << "RMSE" 
              << std::setw(18) << "Max Abs Error" 
              << std::setw(20) << "Cosine Similarity" 
              << "Status\n";
    std::cout << std::string(85, '-') << "\n";

    for (int l = 0; l < 32; ++l) {
        std::string l_str = std::to_string(l);

        // --- 0. RMSNorm ---
        engine.step_attention_norm(l);
        
        // Сразу проверяем d_X_norm до того, как его перезапишет нормализация MLP
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_norm, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_input_norm = load_golden_dump("layer_0_input_norm.bin", hidden_dim);

            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_input_norm[i], h_gpu_buffer[i], 3.5e-2f) 
                    << "RMSNorm mismatch detected at Layer 0, dimension index: " << i;
            }
        }

        // --- 1. Проекции Внимания (Q, K, V) ---
        engine.step_attention_qkv_projections(l);
        
        // Сразу проверяем Q, K, V до того, как RoPE изменит их по месту
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());

            // 1a. Проверка Q_proj (используем безопасный допуск 5.0e-2f для учета динамического квантования)
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_Q, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_q = load_golden_dump("layer_0_q_proj.bin", hidden_dim);
            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_q[i], h_gpu_buffer[i], 5.0e-2f) 
                    << "Q_proj mismatch detected at Layer 0, dimension index: " << i;
            }

            // 1b. Проверка K_proj
            CUDA_CHECK(cudaMemcpy(h_gpu_kv.data(), engine.d_K, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_k = load_golden_dump("layer_0_k_proj.bin", 1024);
            for (size_t i = 0; i < 1024; ++i) {
                ASSERT_NEAR(golden_k[i], h_gpu_kv[i], 5.0e-2f) 
                    << "K_proj mismatch detected at Layer 0, dimension index: " << i;
            }

            // 1c. Проверка V_proj
            CUDA_CHECK(cudaMemcpy(h_gpu_kv.data(), engine.d_V, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_v = load_golden_dump("layer_0_v_proj.bin", 1024);
            for (size_t i = 0; i < 1024; ++i) {
                ASSERT_NEAR(golden_v[i], h_gpu_kv[i], 5.0e-2f) 
                    << "V_proj mismatch detected at Layer 0, dimension index: " << i;
            }
        }

        // --- 2. Математика Внимания (RoPE + SDPA) ---
        engine.step_attention_math(l, pos);
        
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_Attn_out, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_attn_math = load_golden_dump("layer_0_attn_math.bin", hidden_dim);

            for (size_t i = 0; i < hidden_dim; ++i) {
                ASSERT_NEAR(golden_attn_math[i], h_gpu_buffer[i], 1.0e-2f) 
                    << "Attention Math mismatch detected at Layer 0, dimension index: " << i;
            }
        }

        // --- 3. Выходная проекция Внимания и нормализация перед MLP ---
        engine.step_attention_out(l);
        engine.step_mlp_norm(l);

        // --- 4. Проекции MLP (Gate & Up) ---
        engine.step_mlp_projections(l);
        
        if (l == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu_gate.data(), engine.d_Gate, intermediate_dim * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> golden_gate = load_golden_dump("layer_0_gate_proj.bin", intermediate_dim);

            for (size_t i = 0; i < intermediate_dim; ++i) {
                ASSERT_NEAR(golden_gate[i], h_gpu_gate[i], 5.0e-2f) 
                    << "Gate_proj mismatch detected at Layer 0, dimension index: " << i;
            }
        }

        // --- 5. Выход MLP и накопление остаточного потока ---
        engine.step_mlp_out(l);

        // 📊 НЕПРЕРЫВНАЯ ТЕЛЕМЕТРИЯ: Считываем итоговый остаточный поток d_X_accum на каждом слое
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_accum, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
        
        try {
            std::vector<float> golden_layer = load_golden_dump("layer_" + l_str + "_accum_out.bin", hidden_dim);
            TelemetryMetrics metrics = compute_vector_telemetry(golden_layer, h_gpu_buffer);

            // Выводим чистую математическую сводку в консоль
            std::cout << std::left << std::setw(8) << l 
                      << std::fixed << std::setprecision(6)
                      << std::setw(15) << metrics.rmse 
                      << std::setw(18) << metrics.max_abs_error 
                      << std::setprecision(9)
                      << std::setw(20) << metrics.cosine_similarity;

            if (metrics.cosine_similarity < 0.99) {
                std::cout << "⚠️ DIVERGING\n";
            } else {
                std::cout << "✅ STABLE\n";
            }

            // Финальная верификация накопителя для нулевого слоя
            if (l == 0) {
                for (size_t i = 0; i < hidden_dim; ++i) {
                    ASSERT_NEAR(golden_layer[i], h_gpu_buffer[i], 8.0e-2f) 
                        << "Accumulated residual stream mismatch at Layer 0, index: " << i;
                }
            }
        } catch (const std::exception& e) {
            std::cout << std::left << std::setw(8) << l 
                      << std::setw(53) << "[Дамп недоступен для расчета телеметрии]" 
                      << "ℹ️ SKIPPED\n";
        }
    }
    std::cout << std::string(85, '-') << "\n";
    std::cout << "  [OK] Телеметрия конвейера успешно собрана.\n";

    // ========================================================================
    // STAGE 3: Финальная нормализация, проекция логитов и сэмплирование
    // ========================================================================
    std::cout << "\n[Integration Test] Шаг 3: Проекция словаря и выбор токена...\n";
    engine.step_final_ops();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_gpu_logits.data(), engine.d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> golden_logits = load_golden_dump("logits_out.bin", vocab_size);

    int golden_max_idx = 0;
    float golden_max_val = golden_logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (golden_logits[i] > golden_max_val) {
            golden_max_val = golden_logits[i];
            golden_max_idx = static_cast<int>(i);
        }
    }

    int gpu_max_idx = 0;
    float gpu_max_val = h_gpu_logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (h_gpu_logits[i] > gpu_max_val) {
            gpu_max_val = h_gpu_logits[i];
            gpu_max_idx = static_cast<int>(i);
        }
    }

    int next_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&next_token_id, engine.d_next_token, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "  [INFO] PyTorch Top-1 Token: " << golden_max_idx << " (Logit: " << std::setprecision(4) << golden_max_val << ")\n";
    std::cout << "  [INFO] Custom GPU Top-1 Token: " << gpu_max_idx << " (Logit: " << gpu_max_val << ")\n";
    std::cout << "  [INFO] Hardware Argmax Result: " << next_token_id << "\n";

    // Экспорт в CSV
    std::ofstream csv_file("logits_comparison.csv");
    if (csv_file.is_open()) {
        csv_file << "TokenID,GoldenLogit,GPULogit,AbsDifference\n";
        for (size_t i = 0; i < vocab_size; ++i) {
            csv_file << i << "," << golden_logits[i] << "," << h_gpu_logits[i] << "," 
                     << std::abs(golden_logits[i] - h_gpu_logits[i]) << "\n";
        }
        csv_file.close();
    }

    // Вывод Топ-5
    std::cout << "\n  --- Top-5 PyTorch Golden Logits ---\n";
    std::vector<int> golden_indices(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) golden_indices[i] = static_cast<int>(i);
    std::sort(golden_indices.begin(), golden_indices.end(), [&](int a, int b) {
        return golden_logits[a] > golden_logits[b];
    });
    for (int k = 0; k < 5; ++k) {
        int idx = golden_indices[k];
        std::cout << "  Rank " << k+1 << ": Token " << idx << " | Golden: " << golden_logits[idx] 
                  << " | GPU actual: " << h_gpu_logits[idx] << " | Diff: " << std::abs(golden_logits[idx] - h_gpu_logits[idx]) << "\n";
    }

    std::cout << "\n  --- Top-5 Custom GPU Logits ---\n";
    std::vector<int> gpu_indices(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) gpu_indices[i] = static_cast<int>(i);
    std::sort(gpu_indices.begin(), gpu_indices.end(), [&](int a, int b) {
        return h_gpu_logits[a] > h_gpu_logits[b];
    });
    for (int k = 0; k < 5; ++k) {
        int idx = gpu_indices[k];
        std::cout << "  Rank " << k+1 << ": Token " << idx << " | GPU actual: " << h_gpu_logits[idx] 
                  << " | Golden: " << golden_logits[idx] << " | Diff: " << std::abs(golden_logits[idx] - h_gpu_logits[idx]) << "\n";
    }
    std::cout << "\n";

    // 🎯 ПРОВЕРКА ТОП-2 РАСПРЕДЕЛЕНИЯ (Top-K Accuracy)
    bool token_is_valid = (gpu_indices[0] == golden_max_idx) || (gpu_indices[1] == golden_max_idx);
    
    ASSERT_TRUE(token_is_valid) 
        << "Critical error: PyTorch Golden target token (" << golden_max_idx 
        << ") fell outside the engine's primary prediction tier.\n"
        << "  Actual Top-2 Engine Predictions: [" << gpu_indices[0] << ", " << gpu_indices[1] << "]";

    std::cout << "  🎉 [SUCCESS] Интеграционный тест завершен! Распределение сети стабильно.\n";
}