#pragma once
// -----------------------------------------------------------------------------
// translator/audio_head_loader.hpp — build a ready AudioEmbeddingPipeline from an
// Ultravox "audio head" checkpoint directory.
//
// The audio head (--audio-head, default F:/AI/ultravox-v0_5-llama-3_1-8b) bundles
// BOTH halves of the audio frontend in one model.safetensors:
//   audio_tower.*            -> the Whisper large-v3-turbo ENCODER weights
//   multi_modal_projector.*  -> the Ultravox projector weights
// Both are BF16 in the checkpoint; we de-quantise to fp32 (a left shift by 16 is
// bit-exact) to feed the fp32 pipeline, then hand them to the encoder/projector.
//
// This is INIT-tier setup (throws on a missing file / tensor). It uses the engine
// white-box SafetensorsLoader (src/core) + blackwell_audio; the loaded pipeline
// runs entirely on its own audio stream (single engine-owning thread).
// -----------------------------------------------------------------------------
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "safetensors.h"                    // SafetensorsLoader, TensorEntry
#include "whisper_encoder.h"                // WhisperEncoderConfig, WhisperWeights
#include "ultravox_projector_pipeline.cuh"  // ProjectorConfig
#include "audio_embedding_pipeline.h"       // AudioEmbeddingPipeline

namespace rt {

namespace detail {

// Read one BF16 tensor from the checkpoint and widen it to fp32.
inline std::vector<float> load_bf16_tensor(const SafetensorsLoader& st,
                                           const std::string& name) {
    const TensorEntry& e = st.get_tensor(name);
    const std::size_t n = e.byte_size / sizeof(uint16_t);
    std::ifstream f(e.file_path, std::ios::binary);
    if (!f) throw std::runtime_error("audio head: cannot open shard " + e.file_path);
    f.seekg(static_cast<std::streamoff>(e.file_offset));
    std::vector<uint16_t> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(e.byte_size));
    if (!f) throw std::runtime_error("audio head: short read for " + name);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        out[i] = v;
    }
    return out;
}

}  // namespace detail

// Construct + fully load an AudioEmbeddingPipeline from <audio_head>/model.safetensors.
// `text_hidden` is the projector output width; pass the backbone hidden_size so the
// audio soft-tokens land in the text embedding space. Throws (INIT tier) on any
// missing tensor or file.
inline std::unique_ptr<blackwell::audio::AudioEmbeddingPipeline>
load_audio_pipeline(const std::string& audio_head, int text_hidden) {
    using blackwell::audio::AudioEmbeddingPipeline;
    using blackwell::audio::ProjectorConfig;
    using blackwell::audio::WhisperEncoderConfig;
    using blackwell::audio::WhisperLayerWeights;
    using blackwell::audio::WhisperWeights;

    SafetensorsLoader st(audio_head + "/model.safetensors");
    const auto A = [&](const std::string& n) {
        return detail::load_bf16_tensor(st, "audio_tower." + n);
    };
    const auto P = [&](const std::string& n) {
        return detail::load_bf16_tensor(st, "multi_modal_projector." + n);
    };

    WhisperEncoderConfig enc_cfg;  // whisper-large-v3-turbo defaults
    WhisperWeights ew;
    ew.conv1_w = A("conv1.weight");
    ew.conv1_b = A("conv1.bias");
    ew.conv2_w = A("conv2.weight");
    ew.conv2_b = A("conv2.bias");
    ew.embed_positions = A("embed_positions.weight");
    ew.layer_norm_w = A("layer_norm.weight");
    ew.layer_norm_b = A("layer_norm.bias");
    ew.layers.resize(static_cast<std::size_t>(enc_cfg.num_layers));
    for (int l = 0; l < enc_cfg.num_layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        WhisperLayerWeights& s = ew.layers[static_cast<std::size_t>(l)];
        s.self_attn_layer_norm_w = A(p + "self_attn_layer_norm.weight");
        s.self_attn_layer_norm_b = A(p + "self_attn_layer_norm.bias");
        s.q_w = A(p + "self_attn.q_proj.weight");
        s.q_b = A(p + "self_attn.q_proj.bias");
        s.k_w = A(p + "self_attn.k_proj.weight");
        s.v_w = A(p + "self_attn.v_proj.weight");
        s.v_b = A(p + "self_attn.v_proj.bias");
        s.out_w = A(p + "self_attn.out_proj.weight");
        s.out_b = A(p + "self_attn.out_proj.bias");
        s.final_layer_norm_w = A(p + "final_layer_norm.weight");
        s.final_layer_norm_b = A(p + "final_layer_norm.bias");
        s.fc1_w = A(p + "fc1.weight");
        s.fc1_b = A(p + "fc1.bias");
        s.fc2_w = A(p + "fc2.weight");
        s.fc2_b = A(p + "fc2.bias");
    }

    ProjectorConfig proj_cfg;
    proj_cfg.text_hidden = text_hidden;

    auto pipe = std::make_unique<AudioEmbeddingPipeline>(enc_cfg, proj_cfg);
    pipe->load_weights(ew, P("ln_pre.weight"), P("linear_1.weight"),
                       P("ln_mid.weight"), P("linear_2.weight"));
    return pipe;
}

}  // namespace rt
