// End-to-end engine verification for Qwen2.5-Coder-7B-Instruct-AWQ: runs one
// BOS-token decode step through the real BlackwellEngine (28 layers of int4
// AWQ weights) and compares per-layer telemetry plus the final logits against
// golden PyTorch dumps produced by generate_qwen_dumps.py.
//
// Convergence metric (quantization drift is expected on this stack):
//   cosine_similarity(final logits) >= 0.95
//   OR the PyTorch golden top-1 token appears in the engine's top-5.
//
// External requirements (the test SKIPs if either is missing):
//   - model checkpoint index, default
//     F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ/model.safetensors.index.json
//     (override with env var BLACKWELL_QWEN_INDEX)
//   - golden dumps directory produced by generate_qwen_dumps.py, default
//     <repo>/tests/integration/golden_dumps/qwen2.5_awq
//     (override with env var BLACKWELL_QWEN_DUMPS_DIR)
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
    return env_or("BLACKWELL_QWEN_INDEX",
                  "F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ/model.safetensors.index.json");
}

std::string dumps_dir() {
    return env_or("BLACKWELL_QWEN_DUMPS_DIR",
                  "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/qwen2.5_awq");
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

std::vector<float> load_golden_dump(const std::string& filename, size_t num_elements) {
    const std::string full_path = dumps_dir() + "/" + filename;
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Dump file not found: " + full_path +
                                 ". Run generate_qwen_dumps.py first.");
    }

    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(float));

    if (!file) {
        throw std::runtime_error("Dump file size mismatch for requested dimensions: " + full_path);
    }
    return buffer;
}

double cosine_similarity(const std::vector<float>& golden, const std::vector<float>& actual) {
    double dot = 0.0, norm_g = 0.0, norm_a = 0.0;
    for (size_t i = 0; i < golden.size(); ++i) {
        double g = golden[i], a = actual[i];
        dot += g * a;
        norm_g += g * g;
        norm_a += a * a;
    }
    double denom = std::sqrt(norm_g) * std::sqrt(norm_a);
    return (denom > 1e-9) ? (dot / denom) : 0.0;
}

// Downloads a device tensor and prints one telemetry row (RMSE / max error /
// cosine similarity) against the golden dump. Missing dumps are SKIPPED rows.
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

        double sum_sq = 0.0, max_err = 0.0;
        for (size_t i = 0; i < num_elements; ++i) {
            double err = std::abs(static_cast<double>(golden[i]) - h_buffer[i]);
            sum_sq += err * err;
            max_err = std::max(max_err, err);
        }
        double rmse = std::sqrt(sum_sq / static_cast<double>(num_elements));
        double cos = cosine_similarity(golden, h_buffer);

        std::cout << std::left
                  << std::setw(6)  << layer_idx
                  << std::setw(16) << stage_name
                  << std::fixed << std::setprecision(6)
                  << std::setw(12) << rmse
                  << std::setw(14) << max_err
                  << std::setprecision(8)
                  << std::setw(15) << cos
                  << (cos < 0.99 ? "DIVERGING" : "STABLE") << "\n";
    } catch (const std::exception&) {
        std::cout << std::left
                  << std::setw(6)  << layer_idx
                  << std::setw(16) << stage_name
                  << std::setw(41) << "[dump unavailable for telemetry]"
                  << "SKIPPED\n";
    }
}

std::vector<int> topk_indices(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

}  // namespace

TEST(QwenEngineIntegration, AwqBosDecodeConvergence) {
    if (!file_exists(model_index_path())) {
        GTEST_SKIP() << "Model checkpoint not found at " << model_index_path()
                     << " (set BLACKWELL_QWEN_INDEX to override)";
    }
    if (!file_exists(dumps_dir() + "/logits_out.bin")) {
        GTEST_SKIP() << "Golden dumps not found in " << dumps_dir()
                     << " (run generate_qwen_dumps.py, or set BLACKWELL_QWEN_DUMPS_DIR)";
    }

    // Qwen2.5-Coder-7B geometry (config.json at the checkpoint root).
    const size_t hidden_dim       = 3584;   // 28 heads x 128
    const size_t kv_dim           = 512;    // 4 KV heads x 128
    const size_t intermediate_dim = 18944;
    const size_t vocab_size       = 152064;
    const int    num_layers       = 28;
    const int    start_token      = 151643; // <|endoftext|> (BOS)
    const int    pos              = 0;

    std::cout << "\n[Integration] Initializing BlackwellEngine (Qwen2.5 AWQ)...\n";
    BlackwellEngine engine(model_index_path(), 2048);
    auto* core = engine.get_impl();

    std::vector<float> h_hidden(hidden_dim);
    std::vector<float> h_gate(intermediate_dim);
    std::vector<float> h_kv(kv_dim);

    // ========================================================================
    // STAGE 1: embedding lookup (telemetry only; the binding verdict is the
    // final-logits convergence metric below).
    // ========================================================================
    std::cout << "[Integration] Stage 2: executing " << num_layers
              << " layers with telemetry...\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left
              << std::setw(6)  << "Layer"
              << std::setw(16) << "Stage"
              << std::setw(12) << "RMSE"
              << std::setw(14) << "Max Error"
              << std::setw(15) << "Cosine Sim"
              << "Status\n";
    std::cout << std::string(80, '-') << "\n";

    core->step_embedding(start_token);
    verify_and_log_telemetry("embed_out.bin", core->d_X_accum, h_hidden, -1, "Embedding");

    // ========================================================================
    // STAGE 2: layer-by-layer execution with telemetry
    // ========================================================================
    for (int l = 0; l < num_layers; ++l) {
        std::string l_str = std::to_string(l);

        core->step_attention_norm(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_input_norm.bin", core->d_X_norm, h_hidden, l, "InputNorm");
        }

        core->step_attention_qkv_projections(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_q_proj.bin", core->d_Q, h_hidden, l, "Q_proj");
            verify_and_log_telemetry("layer_0_k_proj.bin", core->d_K, h_kv,     l, "K_proj");
            verify_and_log_telemetry("layer_0_v_proj.bin", core->d_V, h_kv,     l, "V_proj");
        }

        core->step_attention_math(l, pos);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_attn_math.bin", core->d_Attn_out, h_hidden, l, "AttnMath");
        }

        core->step_attention_out(l);
        core->step_mlp_norm(l);

        core->step_mlp_projections(l);
        if (l == 0) {
            verify_and_log_telemetry("layer_0_gate_proj.bin", core->d_Gate, h_gate, l, "Gate_proj");
            verify_and_log_telemetry("layer_0_up_proj.bin",   core->d_Up,   h_gate, l, "Up_proj");
        }

        core->step_mlp_out(l);
        verify_and_log_telemetry("layer_" + l_str + "_accum_out.bin", core->d_X_accum, h_hidden, l, "Accum_out");
    }
    std::cout << std::string(80, '-') << "\n";

    // ========================================================================
    // STAGE 3: final norm, vocab projection and the convergence verdict
    // ========================================================================
    std::cout << "\n[Integration] Stage 3: vocab projection and convergence metric...\n";
    core->step_final_ops();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits(vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), core->d_logits,
                          vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> golden_logits = load_golden_dump("logits_out.bin", vocab_size);

    const double logits_cosine = cosine_similarity(golden_logits, h_logits);
    std::vector<int> golden_top5 = topk_indices(golden_logits, 5);
    std::vector<int> engine_top5 = topk_indices(h_logits, 5);
    const int golden_top1 = golden_top5[0];

    std::cout << std::fixed << std::setprecision(8)
              << "  [METRIC] logits cosine similarity: " << logits_cosine << "\n";
    auto print_top5 = [&](const char* title, const std::vector<int>& idx) {
        std::cout << "  [METRIC] " << title << " top-5:";
        for (int t : idx) {
            std::cout << " " << t << " (g=" << std::setprecision(3) << golden_logits[t]
                      << "/e=" << h_logits[t] << ")";
        }
        std::cout << "\n";
    };
    print_top5("PyTorch golden", golden_top5);
    print_top5("engine        ", engine_top5);

    const bool top1_in_engine_top5 =
        std::find(engine_top5.begin(), engine_top5.end(), golden_top1) != engine_top5.end();

    // Convergence target: cosine >= 0.95 over the full logits vector, OR the
    // golden top-1 token within the engine's top-5 predictions.
    ASSERT_TRUE(logits_cosine >= 0.95 || top1_in_engine_top5)
        << "Convergence FAILED: cosine " << logits_cosine
        << " < 0.95 and golden top-1 token " << golden_top1
        << " is absent from the engine's top-5";

    std::cout << "  [SUCCESS] Convergence metric met (cosine "
              << std::setprecision(8) << logits_cosine
              << ", golden top-1 in engine top-5: "
              << (top1_in_engine_top5 ? "yes" : "no") << ").\n";
}
