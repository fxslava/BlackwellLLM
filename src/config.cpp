#include "blackwell/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

ModelConfig ConfigLoader::load_from_json(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("ConfigLoader: cannot open " + json_path);

    nlohmann::json j;
    f >> j;

    ModelConfig cfg{};

    cfg.hidden_dim           = j.at("hidden_size").get<size_t>();
    cfg.intermediate_dim     = j.at("intermediate_size").get<size_t>();
    cfg.num_layers           = j.at("num_hidden_layers").get<size_t>();
    cfg.num_attention_heads  = j.at("num_attention_heads").get<size_t>();
    cfg.num_key_value_heads  = j.at("num_key_value_heads").get<size_t>();
    cfg.vocab_size           = j.at("vocab_size").get<size_t>();
    cfg.head_dim             = cfg.hidden_dim / cfg.num_attention_heads;

    cfg.rope_theta   = j.value("rope_theta", 10000.0f);
    cfg.has_qkv_bias = j.value("attention_bias", false);

    if (j.contains("quantization_config")) {
        const auto& qc   = j.at("quantization_config");
        cfg.quant_method     = qc.value("quant_type", qc.value("quant_method", "none"));
        cfg.quant_bits       = qc.value("bits", 16);
        cfg.quant_group_size = qc.value("group_size", 128);
    } else {
        cfg.quant_bits       = 16;
        cfg.quant_group_size = 128;
        cfg.quant_method     = "none";
    }

    return cfg;
}
