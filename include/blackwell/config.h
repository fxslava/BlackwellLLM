#pragma once

#include <string>

struct ModelConfig {
    size_t hidden_dim;
    size_t intermediate_dim;
    size_t num_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t vocab_size;
    size_t head_dim;

    float rope_theta;

    bool has_qkv_bias;

    int quant_bits;
    int quant_group_size;
    std::string quant_method;
};

class ConfigLoader {
public:
    static ModelConfig load_from_json(const std::string& json_path);
};
