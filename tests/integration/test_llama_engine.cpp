// End-to-end engine verification: runs one BOS-token decode step through the
// real BlackwellEngine (Llama 3 8B FP8 checkpoint) and compares every stage
// against golden PyTorch dumps, with per-layer telemetry (RMSE / max error /
// cosine similarity) and optional golden-tensor injection for fault isolation.
//
// External requirements (the test SKIPs if either is missing):
//   - model checkpoint index, default F:/AI/llama3-8b-fp8/model.safetensors.index.json
//     (override with env var BLACKWELL_LLAMA_INDEX)
//   - golden dumps directory produced by generate_golden_dumps.py, default
//     <repo>/dumps (override with env var BLACKWELL_DUMPS_DIR)
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "common.h"
#include "blackwell/engine.h"
#include "engine_impl.h"

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

std::string model_index_path() {
    return env_or("BLACKWELL_LLAMA_INDEX", "F:/AI/llama3-8b-fp8/model.safetensors.index.json");
}

std::string dumps_dir() {
    return env_or("BLACKWELL_DUMPS_DIR", "D:/Projects/BlackwellLLM/dumps");
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// Loads a golden binary dump produced by generate_golden_dumps.py.
std::vector<float> load_golden_dump(const std::string& filename, size_t num_elements) {
    const std::string full_path = dumps_dir() + "/" + filename;
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Dump file not found: " + full_path +
                                 ". Run generate_golden_dumps.py first.");
    }

    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(float));

    if (!file) {
        throw std::runtime_error("Dump file size mismatch for requested dimensions: " + full_path);
    }
    return buffer;
}

// Extended divergence metrics for one verified tensor.
struct TelemetryMetrics {
    int max_error_idx;
    int last_max_error_idx;
    double rmse;
    double max_abs_error;
    double last_max_abs_error;
    double cosine_similarity;
};

TelemetryMetrics compute_vector_telemetry(const std::vector<float>& golden,
                                          const std::vector<float>& actual) {
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

// Downloads a device tensor, compares it against a golden dump and prints one
// telemetry table row. Missing dumps are reported as SKIPPED, not failures.
void verify_and_log_telemetry(const std::string& dump_filename,
                              const float* d_tensor,
                              std::vector<float>& h_buffer,
                              int layer_idx,
                              const std::string& stage_name)
{
    size_t num_elements = h_buffer.size();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_tensor, num_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));

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
            std::cout << "DIVERGING\n";
        } else {
            std::cout << "STABLE\n";
        }
    } catch (const std::exception&) {
        std::cout << std::left
                  << std::setw(6)  << layer_idx
                  << std::setw(14) << stage_name
                  << std::setw(77) << "[dump unavailable for telemetry]"
                  << "SKIPPED\n";
    }
}

// Granular fault isolation: overwrite a device buffer with the golden tensor
// so divergence accumulated upstream of this stage is reset to zero.
bool inject_golden_tensor(const std::string& dump_filename, float* d_tensor,
                          size_t num_elements, const std::string& stage_name,
                          int layer_idx) {
    try {
        std::vector<float> golden = load_golden_dump(dump_filename, num_elements);
        CUDA_CHECK(cudaMemcpy(d_tensor, golden.data(), num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "  [Injection] injected golden '" << dump_filename
                  << "' into " << stage_name << " (layer " << layer_idx << ")\n";
        return true;
    } catch (const std::exception&) {
        std::cout << "  [Injection warning] dump '" << dump_filename
                  << "' for " << stage_name << " (layer " << layer_idx
                  << ") not found, skipped.\n";
        return false;
    }
}

}  // namespace

TEST(LlamaEngineIntegration, LayerByLayerComparison) {
    if (!file_exists(model_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << model_index_path()
                     << " (set BLACKWELL_LLAMA_INDEX to override)";
    }
    if (!file_exists(dumps_dir() + "/embed_out.bin")) {
        GTEST_SKIP() << "Golden dumps not found in " << dumps_dir()
                     << " (run generate_golden_dumps.py, or set BLACKWELL_DUMPS_DIR)";
    }

    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336;
    const size_t vocab_size = 128256;
    const int start_token = 128000;  // BOS
    const int pos = 0;

    // ========================================================================
    // Granular injection configuration for fault isolation.
    // Set target_injection_layer to a layer index (or -1 to disable) and flip
    // the per-stage flags to replace that stage's output with the golden dump.
    // ========================================================================
    int target_injection_layer = 31;

    bool inject_input_norm      = false;  // d_X_norm after the first normalization
    bool inject_qkv_projections = false;  // d_Q/d_K/d_V after the attention projections
    bool inject_attn_math       = true;   // d_Attn_out after RoPE and SDPA
    bool inject_post_attn_norm  = false;  // d_X_norm after the pre-MLP normalization
    bool inject_mlp_projections = false;  // d_Gate/d_Up after the MLP projections
    bool inject_accum_out       = false;  // d_X_accum, the layer's residual output

    std::cout << "\n[Integration] Initializing BlackwellEngine and allocating VRAM...\n";
    BlackwellEngine engine(model_index_path(), 2048);
    auto* core = engine.get_impl();

    std::vector<float> h_gpu_buffer(hidden_dim);
    std::vector<float> h_gpu_gate(intermediate_dim);
    std::vector<float> h_gpu_kv(1024);

    // ========================================================================
    // STAGE 1: embedding lookup
    // ========================================================================
    std::cout << "[Integration] Stage 1: verifying the embedding table...\n";
    core->step_embedding(start_token);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu_buffer.data(), core->d_X_accum,
                          hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> golden_embed = load_golden_dump("embed_out.bin", hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        ASSERT_NEAR(golden_embed[i], h_gpu_buffer[i], 1e-4f)
            << "Embedding mismatch at absolute index: " << i;
    }
    std::cout << "  [OK] Input embeddings match the PyTorch golden dump.\n";

    // ========================================================================
    // STAGE 2: layer-by-layer execution with telemetry and injections
    // ========================================================================
    std::cout << "\n[Integration] Stage 2: executing 32 layers with telemetry...\n";
    if (target_injection_layer != -1) {
        std::cout << "Granular injection ACTIVE for layer " << target_injection_layer << "\n";
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

    for (int l = 0; l < 32; ++l) {
        std::string l_str = std::to_string(l);

        // --- 0. Pre-attention normalization ---
        core->step_attention_norm(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_input_norm.bin", core->d_X_norm, h_gpu_buffer, l, "InputNorm");
        }
        if (l == target_injection_layer && inject_input_norm) {
            inject_golden_tensor("layer_" + l_str + "_input_norm.bin", core->d_X_norm, hidden_dim, "InputNorm", l);
        }

        // --- 1. Attention linear projections (Q, K, V) ---
        core->step_attention_qkv_projections(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_q_proj.bin", core->d_Q, h_gpu_buffer, l, "Q_proj");
            verify_and_log_telemetry("layer_0_k_proj.bin", core->d_K, h_gpu_kv,     l, "K_proj");
            verify_and_log_telemetry("layer_0_v_proj.bin", core->d_V, h_gpu_kv,     l, "V_proj");
        }
        if (l == target_injection_layer && inject_qkv_projections) {
            inject_golden_tensor("layer_" + l_str + "_q_proj.bin", core->d_Q, hidden_dim, "Q_proj", l);
            inject_golden_tensor("layer_" + l_str + "_k_proj.bin", core->d_K, 1024,       "K_proj", l);
            inject_golden_tensor("layer_" + l_str + "_v_proj.bin", core->d_V, 1024,       "V_proj", l);
        }

        // --- 2. Attention math (RoPE + SDPA) ---
        core->step_attention_math(l, pos);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_attn_math.bin", core->d_Attn_out, h_gpu_buffer, l, "AttnMath");
        }
        if (l == target_injection_layer && inject_attn_math) {
            inject_golden_tensor("layer_" + l_str + "_attn_math.bin", core->d_Attn_out, hidden_dim, "AttnMath", l);
        }

        // --- 3. Attention output projection and pre-MLP normalization ---
        core->step_attention_out(l);
        core->step_mlp_norm(l);
        if (l == target_injection_layer && inject_post_attn_norm) {
            inject_golden_tensor("layer_" + l_str + "_post_attn_norm.bin", core->d_X_norm, hidden_dim, "PostAttnNorm", l);
        }

        // --- 4. MLP linear projections (Gate & Up) ---
        core->step_mlp_projections(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_gate_proj.bin", core->d_Gate, h_gpu_gate, l, "Gate_proj");
        }
        if (l == target_injection_layer && inject_mlp_projections) {
            inject_golden_tensor("layer_" + l_str + "_gate_proj.bin", core->d_Gate, intermediate_dim, "Gate_proj", l);
            inject_golden_tensor("layer_" + l_str + "_up_proj.bin",   core->d_Up,   intermediate_dim, "Up_proj",   l);
        }

        // --- 5. MLP output (SwiGLU + Down) and the layer's residual output ---
        core->step_mlp_out(l);
        verify_and_log_telemetry("layer_" + l_str + "_accum_out.bin", core->d_X_accum, h_gpu_buffer, l, "Accum_out");

        if (l == target_injection_layer && inject_accum_out) {
            inject_golden_tensor("layer_" + l_str + "_accum_out.bin", core->d_X_accum, hidden_dim, "Accum_out", l);
        }
    }
    std::cout << std::string(110, '-') << "\n";
    std::cout << "  [OK] Pipeline telemetry collected.\n";

    // ========================================================================
    // STAGE 3: final norm, logits projection and sampling
    // ========================================================================
    std::cout << "\n[Integration] Stage 3: vocab projection and token selection...\n";
    core->step_final_ops();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_gpu_logits.data(), core->d_logits,
                          vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaMemcpy(&next_token_id, core->d_next_token, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "  [INFO] PyTorch top-1 token: " << golden_max_idx
              << " (logit: " << std::setprecision(4) << golden_max_val << ")\n";
    std::cout << "  [INFO] Engine top-1 token: " << gpu_max_idx
              << " (logit: " << gpu_max_val << ")\n";
    std::cout << "  [INFO] Hardware argmax result: " << next_token_id << "\n";

    // CSV export for offline analysis.
    std::ofstream csv_file("logits_comparison.csv");
    if (csv_file.is_open()) {
        csv_file << "TokenID,GoldenLogit,GPULogit,AbsDifference\n";
        for (size_t i = 0; i < vocab_size; ++i) {
            csv_file << i << "," << golden_logits[i] << "," << h_gpu_logits[i] << ","
                     << std::abs(golden_logits[i] - h_gpu_logits[i]) << "\n";
        }
        csv_file.close();
    }

    auto print_top5 = [&](const char* title, const std::vector<float>& primary) {
        std::cout << "\n  --- Top-5 by " << title << " ---\n";
        std::vector<int> indices(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) indices[i] = static_cast<int>(i);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return primary[a] > primary[b];
        });
        for (int k = 0; k < 5; ++k) {
            int idx = indices[k];
            std::cout << "  Rank " << k + 1 << ": Token " << idx
                      << " | Golden: " << golden_logits[idx]
                      << " | GPU: " << h_gpu_logits[idx]
                      << " | Diff: " << std::abs(golden_logits[idx] - h_gpu_logits[idx]) << "\n";
        }
        return indices;
    };

    print_top5("PyTorch golden logits", golden_logits);
    std::vector<int> gpu_indices = print_top5("engine logits", h_gpu_logits);
    std::cout << "\n";

    // The golden top-1 token must land in the engine's top-2 predictions.
    bool token_is_valid = (gpu_indices[0] == golden_max_idx) || (gpu_indices[1] == golden_max_idx);

    ASSERT_TRUE(token_is_valid)
        << "Critical error: PyTorch golden target token (" << golden_max_idx
        << ") fell outside the engine's primary prediction tier.\n"
        << "  Actual top-2 engine predictions: [" << gpu_indices[0] << ", " << gpu_indices[1] << "]";

    std::cout << "  [SUCCESS] Integration test finished; network distribution is stable.\n";
}
