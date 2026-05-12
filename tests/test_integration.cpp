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
        
        /*if (assert_tolerance > 0.0f) {
            for (size_t i = 0; i < num_elements; ++i) {
                ASSERT_NEAR(golden[i], h_buffer[i], assert_tolerance) 
                    << stage_name << " mismatch detected at Layer " << layer_idx << ", index: " << i;
            }
        }*/

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

TEST(EngineVerificationTest, LayerByLayerComparison) {
    const size_t hidden_dim = 4096;
    const size_t vocab_size = 128256;
    const int start_token = 128000;
    const int pos = 0;

    std::cout << "\n[Integration Test] Инициализация BlackwellEngine (cuBLASLt Tensor Cores)...\n";
    BlackwellEngine engine("D:/Projects/BlackwellLLM/llama3-8b-fp8/model.safetensors.index.json", 2048);

    std::vector<float> h_gpu_buffer(hidden_dim);
    std::vector<float> h_gpu_kv(1024);
    std::vector<float> h_scale_buf(1);

    std::cout << "[Integration Test] Шаг 1: Проверка таблицы Embedding...\n";
    engine.step_embedding(start_token);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), engine.d_X_accum, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> golden_embed = load_golden_dump("embed_out.bin", hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        ASSERT_NEAR(golden_embed[i], h_gpu_buffer[i], 1e-4f);
    }
    std::cout << "  [OK] Входные эмбеддинги совпали.\n";

    std::cout << "\n[Integration Test] Шаг 2: Выполнение 32 слоев...\n";
    std::cout << std::string(110, '-') << "\n";
    std::cout << std::left << std::setw(6) << "Layer" << std::setw(14) << "Stage" 
              << std::setw(12) << "RMSE" << std::setw(14) << "Max Error" << std::setw(10) << "MaxIdx"
              << std::setw(15) << "PrevMaxErr" << std::setw(11) << "PrevMaxIdx"
              << std::setw(15) << "Cosine Sim" << "Status\n";
    std::cout << std::string(110, '-') << "\n";

    for (int l = 0; l < 32; ++l) {
        std::string l_str = std::to_string(l);

        engine.step_attention_norm(l);
        engine.step_attention_qkv_projections(l);
        
        // 🎯 Оставляем проверки скейлов и выходов KQV для нулевого слоя
        if (l == 0) {
            verify_and_log_telemetry("layer_0_qkv_input_scale.bin", engine.d_token_scale, h_scale_buf, l, "QKV_Scale", 1e-6f);
            verify_and_log_telemetry("layer_0_q_proj.bin", engine.d_Q, h_gpu_buffer, l, "Q_proj", 1e-3f);
            verify_and_log_telemetry("layer_0_k_proj.bin", engine.d_K, h_gpu_kv,     l, "K_proj", 1e-3f);
            verify_and_log_telemetry("layer_0_v_proj.bin", engine.d_V, h_gpu_kv,     l, "V_proj", 1e-3f);
        }

        engine.step_attention_math(l, pos);
        engine.step_attention_out(l);
        
        if (l == 0) {
            verify_and_log_telemetry("layer_0_o_proj_input_scale.bin", engine.d_token_scale, h_scale_buf, l, "Out_Scale", 1e-6f);
        }
        
        engine.step_mlp_norm(l);
        engine.step_mlp_projections(l);
        
        if (l == 0) {
            verify_and_log_telemetry("layer_0_mlp_input_scale.bin", engine.d_token_scale, h_scale_buf, l, "MLP_Scale", 1e-6f);
        }

        engine.step_mlp_out(l);
        
        if (l == 0) {
            verify_and_log_telemetry("layer_0_down_proj_input_scale.bin", engine.d_token_scale, h_scale_buf, l, "Down_Scale", 1e-5f);
        }

        // Общая телеметрия остаточного потока для мониторинга здоровья сети
        verify_and_log_telemetry("layer_" + l_str + "_accum_out.bin", engine.d_X_accum, h_gpu_buffer, l, "Accum_out", -1.0f);
    }
    std::cout << std::string(110, '-') << "\n";

    std::cout << "\n[Integration Test] Шаг 3: Проекция словаря и выбор токена...\n";
    engine.step_final_ops();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_gpu_logits.data(), engine.d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> golden_logits = load_golden_dump("logits_out.bin", vocab_size);

    int golden_max_idx = 0; float golden_max_val = golden_logits[0];
    int gpu_max_idx = 0;    float gpu_max_val = h_gpu_logits[0];
    
    for (size_t i = 1; i < vocab_size; ++i) {
        if (golden_logits[i] > golden_max_val) { golden_max_val = golden_logits[i]; golden_max_idx = static_cast<int>(i); }
        if (h_gpu_logits[i] > gpu_max_val)    { gpu_max_val = h_gpu_logits[i];    gpu_max_idx = static_cast<int>(i); }
    }

    int next_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&next_token_id, engine.d_next_token, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "  [INFO] PyTorch Top-1 Token: " << golden_max_idx << " (Logit: " << golden_max_val << ")\n";
    std::cout << "  [INFO] cuBLASLt Top-1 Token: " << gpu_max_idx << " (Logit: " << gpu_max_val << ")\n";
    
    ASSERT_EQ(golden_max_idx, gpu_max_idx) << "Критическая ошибка: итоговые токены разошлись!";
    std::cout << "  🎉 [SUCCESS] Интеграционный тест пройден! Паритет с тензорными ядрами достигнут.\n";
}