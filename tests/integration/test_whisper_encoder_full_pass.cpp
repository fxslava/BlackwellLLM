// =============================================================================
// Whisper encoder full-pass parity test (Ultravox audio_tower, Phase 3).
//
// Runs the CUDA WhisperEncoder (src/audio/whisper_encoder.*) end to end on the
// GPU and matches the FP32 PyTorch golden reference produced by
// scripts/generate_ultravox_audio_dumps.py:
//
//     input_features [1,128,3000]  (log-mel spectrogram, real librispeech clip)
//        -> conv1+GELU -> conv2+GELU -> +pos -> 32x EncoderLayer -> final LN
//        -> encoder_last_hidden [1,1500,1280]                      (PASS/FAIL)
//
// The encoder WEIGHTS are the BF16 audio_tower.* tensors in the local Ultravox
// checkpoint (F:/AI/ultravox-v0_5-llama-3_2-1b/model.safetensors); the test
// de-quantises them to fp32 exactly as the reference does (.float()) so the deep
// 32-layer stack lands within the parity bar.
//
// Pass bar (requirement): cosine similarity > 0.995 OR MSE < 1e-4 on the final
// output. conv_out.bin is checked too, as a diagnostic that localises a
// regression to the conv feature extractor vs. the transformer stack.
//
// Skips (does not fail) when the local checkpoint or the golden dumps are
// absent, matching the rest of the integration suite.
//
// Env overrides:
//   BLACKWELL_ULTRAVOX_DUMPS  golden-dump dir (default <repo>/tests/integration/golden_dumps/ultravox)
//   BLACKWELL_ULTRAVOX_CKPT   Ultravox checkpoint .safetensors (default under F:/AI)
// =============================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"            // CUDA_CHECK_THROW
#include "device_buffer.h"     // blackwell::DeviceBuffer
#include "audio_test_utils.h"  // load_bin_file, compute_cosine_similarity
#include "safetensors.h"       // SafetensorsLoader
#include "whisper_encoder.h"   // blackwell::audio::WhisperEncoder

using audio_test::compute_cosine_similarity;
using audio_test::load_bin_file;
using blackwell::DeviceBuffer;
using blackwell::audio::WhisperEncoder;
using blackwell::audio::WhisperEncoderConfig;
using blackwell::audio::WhisperLayerWeights;
using blackwell::audio::WhisperWeights;

namespace {

// --- Geometry (whisper-large-v3-turbo encoder, cross-checked vs. dump sizes) --
constexpr int kMel        = 128;
constexpr int kConvFrames = 3000;    // 30 s log-mel
constexpr int kSeq        = 1500;    // encoder output frames
constexpr int kHidden     = 1280;    // d_model
constexpr int kLayers     = 32;

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}
std::string dumps_dir() {
    return env_or("BLACKWELL_ULTRAVOX_DUMPS",
                  "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/ultravox");
}
std::string ckpt_path() {
    return env_or("BLACKWELL_ULTRAVOX_CKPT",
                  "F:/AI/ultravox-v0_5-llama-3_2-1b/model.safetensors");
}
std::string dump(const std::string& name) { return dumps_dir() + "/" + name; }
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

// Reads a BF16 tensor from the checkpoint and de-quantises to fp32 (bit-exact:
// BF16 is the high 16 bits of the fp32 significand/exponent, so widening is a
// left shift by 16). Throws on a missing tensor or dtype surprise.
std::vector<float> load_bf16(const SafetensorsLoader& st, const std::string& name) {
    const TensorEntry& e = st.get_tensor(name);
    if (e.dtype != "BF16")
        throw std::runtime_error("expected BF16 for " + name + ", got " + e.dtype);
    const size_t n = e.byte_size / sizeof(uint16_t);

    std::ifstream f(e.file_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open shard " + e.file_path);
    f.seekg(static_cast<std::streamoff>(e.file_offset));
    std::vector<uint16_t> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(e.byte_size));
    if (!f) throw std::runtime_error("short read for " + name);

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        out[i] = v;
    }
    return out;
}

double compute_mse(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return acc / static_cast<double>(a.size());
}

std::vector<float> from_device(const float* d, size_t count) {
    std::vector<float> h(count);
    CUDA_CHECK_THROW(cudaMemcpy(h.data(), d, count * sizeof(float),
                                cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace

TEST(WhisperEncoderFullPass, EncoderLastHiddenParity) {
    if (!file_exists(dump("input_features.bin")) ||
        !file_exists(dump("encoder_last_hidden.bin"))) {
        GTEST_SKIP() << "golden dumps absent in " << dumps_dir()
                     << " (run scripts/generate_ultravox_audio_dumps.py, or set "
                        "BLACKWELL_ULTRAVOX_DUMPS)";
    }
    if (!file_exists(ckpt_path())) {
        GTEST_SKIP() << "Ultravox checkpoint absent: " << ckpt_path()
                     << " (set BLACKWELL_ULTRAVOX_CKPT)";
    }

    // ---- Load the log-mel input and the encoder-output reference -------------
    const std::vector<float> mel = load_bin_file(dump("input_features.bin"));
    const std::vector<float> ref = load_bin_file(dump("encoder_last_hidden.bin"));
    ASSERT_EQ(mel.size(), static_cast<size_t>(kMel) * kConvFrames);
    ASSERT_EQ(ref.size(), static_cast<size_t>(kSeq) * kHidden);

    // ---- Load audio_tower.* weights (BF16 -> fp32) ---------------------------
    SafetensorsLoader st(ckpt_path());
    const auto W = [&](const std::string& n) { return load_bf16(st, "audio_tower." + n); };

    WhisperWeights w;
    w.conv1_w = W("conv1.weight");
    w.conv1_b = W("conv1.bias");
    w.conv2_w = W("conv2.weight");
    w.conv2_b = W("conv2.bias");
    w.embed_positions = W("embed_positions.weight");
    w.layer_norm_w = W("layer_norm.weight");
    w.layer_norm_b = W("layer_norm.bias");
    w.layers.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        WhisperLayerWeights& s = w.layers[static_cast<size_t>(l)];
        s.self_attn_layer_norm_w = W(p + "self_attn_layer_norm.weight");
        s.self_attn_layer_norm_b = W(p + "self_attn_layer_norm.bias");
        s.q_w = W(p + "self_attn.q_proj.weight");
        s.q_b = W(p + "self_attn.q_proj.bias");
        s.k_w = W(p + "self_attn.k_proj.weight");   // no bias in Whisper
        s.v_w = W(p + "self_attn.v_proj.weight");
        s.v_b = W(p + "self_attn.v_proj.bias");
        s.out_w = W(p + "self_attn.out_proj.weight");
        s.out_b = W(p + "self_attn.out_proj.bias");
        s.final_layer_norm_w = W(p + "final_layer_norm.weight");
        s.final_layer_norm_b = W(p + "final_layer_norm.bias");
        s.fc1_w = W(p + "fc1.weight");
        s.fc1_b = W(p + "fc1.bias");
        s.fc2_w = W(p + "fc2.weight");
        s.fc2_b = W(p + "fc2.bias");
    }

    // ---- Build the encoder, report VRAM footprint ----------------------------
    size_t free_before = 0, total_vram = 0;
    CUDA_CHECK_THROW(cudaMemGetInfo(&free_before, &total_vram));

    WhisperEncoderConfig cfg;  // turbo defaults match the geometry above
    ASSERT_EQ(cfg.d_model, kHidden);
    ASSERT_EQ(cfg.num_layers, kLayers);

    WhisperEncoder encoder(cfg);
    encoder.load_weights(w);

    DeviceBuffer<float> d_mel(mel.size());
    CUDA_CHECK_THROW(cudaMemcpy(d_mel.get(), mel.data(), mel.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    size_t free_after = 0;
    CUDA_CHECK_THROW(cudaMemGetInfo(&free_after, &total_vram));
    const double used_mb = static_cast<double>(free_before - free_after) / (1024.0 * 1024.0);

    // ---- Run + time the full forward pass ------------------------------------
    // forward() enqueues on encoder.stream() (input copy + one cudaGraphLaunch)
    // and does not sync, so events are bracketed on THAT stream. The first call
    // builds the CUDA graph (warm-up + capture) and must not be timed.
    const cudaStream_t s = encoder.stream();
    const float* d_out = encoder.forward(d_mel.get());  // warm-up: build graph
    CUDA_CHECK_THROW(cudaStreamSynchronize(s));
    CUDA_CHECK_THROW(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK_THROW(cudaEventCreate(&start));
    CUDA_CHECK_THROW(cudaEventCreate(&stop));
    CUDA_CHECK_THROW(cudaEventRecord(start, s));
    d_out = encoder.forward(d_mel.get());  // steady-state: single graph launch
    CUDA_CHECK_THROW(cudaEventRecord(stop, s));
    CUDA_CHECK_THROW(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK_THROW(cudaEventDestroy(start));
    CUDA_CHECK_THROW(cudaEventDestroy(stop));
    CUDA_CHECK_THROW(cudaStreamSynchronize(s));  // output ready before readback
    CUDA_CHECK_THROW(cudaGetLastError());

    // ---- Diagnostic: conv feature extractor vs. reference --------------------
    if (file_exists(dump("conv_out.bin"))) {
        const std::vector<float> conv_ref = load_bin_file(dump("conv_out.bin"));
        const std::vector<float> conv_got =
            from_device(encoder.conv_out(), static_cast<size_t>(kSeq) * kHidden);
        const double conv_cos = compute_cosine_similarity(conv_got, conv_ref);
        std::printf("[whisper] conv feature-extractor cosine = %.8f\n", conv_cos);
        EXPECT_GT(conv_cos, 0.995);
    }

    // ---- Final encoder-output parity (the pass/fail assertion) ---------------
    const std::vector<float> got = from_device(d_out, ref.size());
    const double cos = compute_cosine_similarity(got, ref);
    const double mse = compute_mse(got, ref);

    std::printf("[whisper] encoder forward pass       = %.3f ms\n", static_cast<double>(ms));
    std::printf("[whisper] weights + workspace VRAM   = %.1f MB\n", used_mb);
    std::printf("[whisper] encoder_last_hidden cosine = %.8f\n", cos);
    std::printf("[whisper] encoder_last_hidden MSE    = %.3e\n", mse);

    EXPECT_TRUE(cos > 0.995 || mse < 1e-4)
        << "encoder output parity failed: cosine=" << cos << " MSE=" << mse;
}
