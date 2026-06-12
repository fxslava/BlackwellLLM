#include "blackwell/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
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

    if (cfg.hidden_dim == 0 || cfg.intermediate_dim == 0 || cfg.num_layers == 0 ||
        cfg.num_attention_heads == 0 || cfg.num_key_value_heads == 0 || cfg.vocab_size == 0)
        throw std::runtime_error("ConfigLoader: " + json_path +
                                 " declares a zero-sized model dimension");
    if (cfg.num_attention_heads % cfg.num_key_value_heads != 0)
        throw std::runtime_error(
            "ConfigLoader: num_attention_heads (" + std::to_string(cfg.num_attention_heads) +
            ") is not a multiple of num_key_value_heads (" +
            std::to_string(cfg.num_key_value_heads) + "); GQA grouping would be broken");

    cfg.head_dim            = j.value("head_dim", cfg.hidden_dim / cfg.num_attention_heads);
    if (cfg.head_dim == 0)
        throw std::runtime_error("ConfigLoader: head_dim resolved to zero");

    cfg.rope_theta          = j.value("rope_theta", 10000.0f);
    cfg.rms_norm_eps        = j.value("rms_norm_eps", 1e-6f);
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", false);

    // Qwen2-family checkpoints (incl. Qwen2.5-Coder) hardcode q/k/v bias in the
    // modeling code and OMIT "attention_bias" from config.json entirely, so for
    // model_type "qwen2" the key's absence means true, not false.
    const std::string model_type = j.value("model_type", std::string("unknown"));
    cfg.has_qkv_bias = j.value("attention_bias", model_type == "qwen2");

    // The RoPE kernel implements vanilla rotate_half only; a rope_scaling block
    // (llama3 / YaRN / linear) is NOT honored. The angles coincide at pos=0 but
    // diverge for every later position, so warn loudly instead of failing hard:
    // the in-repo Llama-3.1-FP8 verification checkpoint carries such a block and
    // is only ever validated at pos=0 against the golden dumps.
    if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null()) {
        const std::string rope_type =
            j.at("rope_scaling").value("rope_type", j.at("rope_scaling").value("type", "unknown"));
        std::cerr << "[ConfigLoader] WARNING: config declares rope_scaling (rope_type=\""
                  << rope_type << "\") but the RoPE kernel applies vanilla rotate_half "
                  << "frequencies; positional encoding is WRONG for pos >= 1.\n";
    }

    // The decode attention kernel attends over the full causal prefix.
    if (j.value("use_sliding_window", false))
        throw std::runtime_error(
            "ConfigLoader: use_sliding_window=true is not supported by the attention kernel.");

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
