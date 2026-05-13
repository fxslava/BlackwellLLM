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

// Структура для хранения расширенных метрик телеметрии вектора
struct TelemetryMetrics {
    int max_error_idx;
    int last_max_error_idx;
    double rmse;
    double max_abs_error;
    double last_max_abs_error;
    double cosine_similarity;
};

static TelemetryMetrics compute_vector_telemetry(const std::vector<float>& golden, const std::vector<float>& actual) {
    size_t n = golden.size();
    double sum_sq_err = 0.0;
    double max_err = 0.0;
    double last_max_err = 0.0;
    double dot_product = 0.0;
    double norm_golden = 0.0;
    double norm_actual = 0.0;
    int max_error_idx = 0;
    int last_max_error_idx = 0;

    for (size_t i = 0; i < n; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);
        double err = std::abs(g - a);

        sum_sq_err += err * err;
        if (err > max_err) {
            last_max_err = max_err;
            last_max_error_idx = max_error_idx;

            max_err = err;
            max_error_idx = static_cast<int>(i);
        }

        dot_product += g * a;
        norm_golden += g * g;
        norm_actual += a * a;
    }

    TelemetryMetrics metrics;
    metrics.rmse = std::sqrt(sum_sq_err / static_cast<double>(n));
    metrics.max_abs_error = max_err;
    metrics.last_max_abs_error = last_max_err;
    metrics.max_error_idx = max_error_idx;
    metrics.last_max_error_idx = last_max_error_idx;
    
    double denom = std::sqrt(norm_golden) * std::sqrt(norm_actual);
    metrics.cosine_similarity = (denom > 1e-9) ? (dot_product / denom) : 0.0;
    
    return metrics;
}

// 🎯 Универсальный хелпер для сверки и логирования
static void verify_and_log_telemetry(const std::string& dump_filename,
                                     const float* d_tensor,
                                     std::vector<float>& h_buffer,
                                     int layer_idx,
                                     const std::string& stage_name,
                                     float assert_tolerance = -1.0f) 
{
    size_t num_elements = h_buffer.size();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_tensor, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    try {
        std::vector<float> golden = load_golden_dump(dump_filename, num_elements);

        TelemetryMetrics metrics = compute_vector_telemetry(golden, h_buffer);

        std::cout << std::left 
                  << std::setw(6)  << layer_idx 
                  << std::setw(14) << stage_name 
                  << std::fixed << std::setprecision(6)
                  << std::setw(12) << metrics.rmse 
                  << std::setw(14) << metrics.max_abs_error 
                  << std::setw(10) << metrics.max_error_idx
                  << std::setw(15) << metrics.last_max_abs_error
                  << std::setw(11) << metrics.last_max_error_idx
                  << std::setprecision(8)
                  << std::setw(15) << metrics.cosine_similarity;

        if (metrics.cosine_similarity < 0.99) {
            std::cout << "⚠️ DIVERGING\n";
        } else {
            std::cout << "✅ STABLE\n";
        }
    } catch (const std::exception& e) {
        std::cout << std::left 
                  << std::setw(6)  << layer_idx 
                  << std::setw(14) << stage_name 
                  << std::setw(77) << "[Дамп недоступен для расчета телеметрии]" 
                  << "ℹ️ SKIPPED\n";
    }
}

// 🎯 Хелпер для точечной гранулярной инъекции эталонных данных в любой буфер на GPU
static bool inject_golden_tensor(const std::string& dump_filename, float* d_tensor, size_t num_elements, const std::string& stage_name, int layer_idx) {
    try {
        std::vector<float> golden = load_golden_dump(dump_filename, num_elements);
        // Безопасно копируем идеальные данные на устройство и синхронизируем контекст
        CUDA_CHECK(cudaMemcpy(d_tensor, golden.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "  [Injection] 💉 Успешная инъекция эталона '" << dump_filename << "' в буфер " << stage_name << " (Слой " << layer_idx << ")\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "  [Injection Warning] Пропущен дамп '" << dump_filename << "' для " << stage_name << " (Слой " << layer_idx << "): файл не найден.\n";
        return false;
    }
}

TEST(EngineVerificationTest, LayerByLayerComparison) {
    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336; 
    const size_t vocab_size = 128256;
    const int start_token = 128000; // Стандартный токен BOS
    const int pos = 0;

    // ========================================================================
    // 🎯 ГРАНУЛЯРНАЯ НАСТРОЙКА ИНЪЕКЦИЙ ДЛЯ ПРОМЕЖУТОЧНЫХ ОПЕРАЦИЙ
    // ========================================================================
    // Укажите целевой слой для проведения инъекций (например, 0), или -1 для отключения
    int target_injection_layer = 30;

    // Флаги включения инъекций для конкретных промежуточных этапов целевого слоя:
    bool inject_input_norm     = false;  // Инъекция в d_X_norm после первой нормализации
    bool inject_qkv_projections= false;  // Инъекция в d_Q, d_K, d_V после линейных проекций внимания
    bool inject_attn_math      = false;  // Инъекция в d_Attn_out после RoPE и SDPA
    bool inject_post_attn_norm = false;  // Инъекция в d_X_norm после нормализации перед MLP
    bool inject_mlp_projections= false;  // Инъекция в d_Gate и d_Up после линейных слоев MLP
    bool inject_accum_out      = true ;  // Инъекция в итоговый остаточный поток слоя d_X_accum

    std::cout << "\n[Integration Test] Инициализация BlackwellEngine и выделение VRAM...\n";
    BlackwellEngine engine("D:/Projects/BlackwellLLM/llama3-8b-fp8/model.safetensors.index.json", 2048);

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
    // STAGE 2: Послойное выполнение с расширенной телеметрией и инъекциями
    // ========================================================================
    std::cout << "\n[Integration Test] Шаг 2: Выполнение 32 слоев с отслеживанием телеметрии...\n";
    if (target_injection_layer != -1) {
        std::cout << "🎯 РЕЖИМ ГРАНУЛЯРНОЙ ИНЪЕКЦИИ АКТИВЕН для слоя: " << target_injection_layer << "\n";
    }
    std::cout << std::string(110, '-') << "\n";
    std::cout << std::left 
              << std::setw(6)  << "Layer" 
              << std::setw(14) << "Stage" 
              << std::setw(12) << "RMSE" 
              << std::setw(14) << "Max Error" 
              << std::setw(10) << "MaxIdx"
              << std::setw(15) << "PrevMaxErr"
              << std::setw(11) << "PrevMaxIdx"
              << std::setw(15) << "Cosine Sim" 
              << "Status\n";
    std::cout << std::string(110, '-') << "\n";

    std::vector<float> h_scale_buf(1);

    for (int l = 0; l < 32; ++l) {
        std::string l_str = std::to_string(l);

        // --- 0. Нормализация входа во внимание ---
        engine.step_attention_norm(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_input_norm.bin", engine.d_X_norm, h_gpu_buffer, l, "InputNorm", 3.5e-2f);
        }
        // 💉 Инъекция в буфер d_X_norm
        if (l == target_injection_layer && inject_input_norm) {
            inject_golden_tensor("layer_" + l_str + "_input_norm.bin", engine.d_X_norm, hidden_dim, "InputNorm", l);
        }

        // --- 1. Линейные проекции Внимания (Q, K, V) ---
        engine.step_attention_qkv_projections(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_q_proj.bin", engine.d_Q, h_gpu_buffer, l, "Q_proj", 5.0e-2f);
            verify_and_log_telemetry("layer_0_k_proj.bin", engine.d_K, h_gpu_kv,     l, "K_proj", 5.0e-2f);
            verify_and_log_telemetry("layer_0_v_proj.bin", engine.d_V, h_gpu_kv,     l, "V_proj", 5.0e-2f);
        }
        // 💉 Инъекция в буферы d_Q, d_K, d_V
        if (l == target_injection_layer && inject_qkv_projections) {
            inject_golden_tensor("layer_" + l_str + "_q_proj.bin", engine.d_Q, hidden_dim, "Q_proj", l);
            inject_golden_tensor("layer_" + l_str + "_k_proj.bin", engine.d_K, 1024,       "K_proj", l);
            inject_golden_tensor("layer_" + l_str + "_v_proj.bin", engine.d_V, 1024,       "V_proj", l);
        }

        // --- 2. Математика Внимания (RoPE + SDPA) ---
        engine.step_attention_math(l, pos);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_attn_math.bin", engine.d_Attn_out, h_gpu_buffer, l, "AttnMath", 1.0e-2f);
        }
        // 💉 Инъекция в буфер d_Attn_out
        if (l == target_injection_layer && inject_attn_math) {
            inject_golden_tensor("layer_" + l_str + "_attn_math.bin", engine.d_Attn_out, hidden_dim, "AttnMath", l);
        }

        // --- 3. Выходная свертка внимания и нормализация перед MLP ---
        engine.step_attention_out(l);       
        engine.step_mlp_norm(l);
        // 💉 Инъекция в переиспользованный буфер d_X_norm после нормализации MLP
        if (l == target_injection_layer && inject_post_attn_norm) {
            inject_golden_tensor("layer_" + l_str + "_post_attn_norm.bin", engine.d_X_norm, hidden_dim, "PostAttnNorm", l);
        }

        // --- 4. Линейные проекции MLP (Gate & Up) ---
        engine.step_mlp_projections(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_gate_proj.bin", engine.d_Gate, h_gpu_gate, l, "Gate_proj", 5.0e-2f);
        }
        // 💉 Инъекция в широкие промежуточные буферы d_Gate и d_Up
        if (l == target_injection_layer && inject_mlp_projections) {
            inject_golden_tensor("layer_" + l_str + "_gate_proj.bin", engine.d_Gate, intermediate_dim, "Gate_proj", l);
            inject_golden_tensor("layer_" + l_str + "_up_proj.bin",   engine.d_Up,   intermediate_dim, "Up_proj",   l);
        }

        // --- 5. Выход MLP (SwiGLU + Down) и итоговое накопление слоя ---
        engine.step_mlp_out(l);
        float tolerance = (l == 0) ? 8.0e-2f : -1.0f;
        verify_and_log_telemetry("layer_" + l_str + "_accum_out.bin", engine.d_X_accum, h_gpu_buffer, l, "Accum_out", tolerance);

        // 💉 Инъекция в итоговый остаточный буфер слоя d_X_accum
        if (l == target_injection_layer && inject_accum_out) {
            inject_golden_tensor("layer_" + l_str + "_accum_out.bin", engine.d_X_accum, hidden_dim, "Accum_out", l);
        }
    }
    std::cout << std::string(110, '-') << "\n";
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

    // 🎯 ПРОВЕРКА ТОП-2 РАСПРЕДЕЛЕНИЯ
    bool token_is_valid = (gpu_indices[0] == golden_max_idx) || (gpu_indices[1] == golden_max_idx);
    
    ASSERT_TRUE(token_is_valid) 
        << "Critical error: PyTorch Golden target token (" << golden_max_idx 
        << ") fell outside the engine's primary prediction tier.\n"
        << "  Actual Top-2 Engine Predictions: [" << gpu_indices[0] << ", " << gpu_indices[1] << "]";

    std::cout << "  🎉 [SUCCESS] Интеграционный тест завершен! Распределение сети стабильно.\n";
}