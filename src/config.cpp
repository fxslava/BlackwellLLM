#include "blackwell/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

// Map the raw JSON quantization fields to an internal QuantStrategy.
// All ambiguity is resolved here so no other code ever inspects strings or bit widths.
static QuantStrategy resolve_quant_strategy(const std::string& method, int bits) {
    // Weight-only packed int4: AWQ and GPTQ both use the same packed-weight path.
    if (method == "awq" || method == "gptq") {
        if (bits != 4)
            throw std::runtime_error(
                "ConfigLoader: quant_method \"" + method + "\" requires bits=4, got " +
                std::to_string(bits));
        return QuantStrategy::WEIGHT_ONLY_PACKED;
    }

    // Per-row FP8 weights with per-token activation scaling.
    if (method == "fp8") {
        return QuantStrategy::ROWWISE_FP8;
    }

    // 🎯 HUGGINGFACE/NEURAL MAGIC MUTANT CHECKPOINT WORKAROUND:
    // Some official vLLM/Neural Magic export scripts for FP8 models generate a corrupted 
    // quantization_config block where "quant_method" is set to "compressed-tensors", but 
    // "bits" mistakenly retains the baseline model's value (16) instead of 8. 
    // Since "compressed-tensors" with bits=4 is handled as AWQ/Packed, any other bit width 
    // (8 or mislabelled 16) in the Neural Magic ecosystem signifies a row-wise FP8 execution path.
    if (method == "compressed-tensors") {
        if (bits == 4) {
            return QuantStrategy::WEIGHT_ONLY_PACKED;
        } else {
            // Fallback for both proper 8-bit and mislabelled 16-bit FP8 manifests
            return QuantStrategy::ROWWISE_FP8;
        }
    }

    // Explicit unquantized path.
    if (method == "none") {
        return QuantStrategy::NONE;
    }

    throw std::runtime_error(
        "ConfigLoader: unrecognised quantization configuration "
        "(quant_method=\"" + method + "\", bits=" + std::to_string(bits) + "). "
        "Supported combinations: awq/4, gptq/4, fp8/8, compressed-tensors/8, none.");
}

ModelConfig ConfigLoader::load_from_json(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("ConfigLoader: cannot open " + json_path);

    nlohmann::json j;
    f >> j;

    ModelConfig cfg{};

    cfg.hidden_dim          = j.at("hidden_size").get<size_t>();
    cfg.intermediate_dim    = j.at("intermediate_size").get<size_t>();
    cfg.num_layers          = j.at("num_hidden_layers").get<size_t>();
    cfg.num_attention_heads = j.at("num_attention_heads").get<size_t>();
    cfg.num_key_value_heads = j.at("num_key_value_heads").get<size_t>();
    cfg.vocab_size          = j.at("vocab_size").get<size_t>();
    cfg.head_dim            = cfg.hidden_dim / cfg.num_attention_heads;

    cfg.rope_theta   = j.value("rope_theta", 10000.0f);
    cfg.has_qkv_bias = j.value("attention_bias", false);

    if (j.contains("quantization_config")) {
        const auto& qc   = j.at("quantization_config");
        std::string method = qc.value("quant_type", qc.value("quant_method", "none"));
        cfg.quant_bits       = qc.value("bits", 16);
        cfg.quant_group_size = qc.value("group_size", 128);
        cfg.quant_strategy   = resolve_quant_strategy(method, cfg.quant_bits);
        cfg.quant_method     = method; // kept for VRAMArena weight-routing (memory_pool.cpp)
    } else {
        cfg.quant_bits       = 16;
        cfg.quant_group_size = 128;
        cfg.quant_strategy   = QuantStrategy::NONE;
        cfg.quant_method     = "none";
    }

    return cfg;
}
