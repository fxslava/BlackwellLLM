// =============================================================================
// Async double-buffered (ping-pong) audio pipeline tests.
//
// Verifies the Stream-1 (audio: Whisper encoder -> Ultravox projector) /
// Stream-2 (LLM consumer) overlap handoff built on PingPongAudioBuffer +
// AudioEmbeddingPipeline (src/audio):
//
//   A) PingPongOverlapNoRaces  -- no checkpoint needed, always runs.
//      Producer memsets a per-frame byte pattern into ping-pong slots on a
//      producer stream; a consumer stream samples the PREVIOUS frame's slot
//      concurrently. The whole run uses ONLY cudaEventRecord/cudaStreamWaitEvent
//      (no cudaDeviceSynchronize on the hot path), then a single drain sync.
//      Every consumed sample must equal its frame's pattern -- a missing ready
//      (read-after-write) or consumed (write-after-read, slot reuse two frames
//      later) event would corrupt this, so exact readback over many overlapped
//      frames is the race/corruption proof.
//
//   B) RealPipelineParity  -- skips unless the local checkpoint + golden dumps
//      exist. Runs the REAL WhisperEncoder -> UltravoxProjector on the audio
//      stream, stages into the ping-pong, and the consumer stream reads the slot
//      (event-gated). The result matches the golden audio_embeds.bin, proving the
//      encoder->projector wiring is correct end-to-end through the double buffer.
//
// Env overrides (same as the sibling audio tests):
//   BLACKWELL_ULTRAVOX_DUMPS  golden-dump dir
//   BLACKWELL_ULTRAVOX_CKPT   Ultravox checkpoint .safetensors
// =============================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"                         // CUDA_CHECK_THROW
#include "device_buffer.h"                  // blackwell::DeviceBuffer
#include "audio_test_utils.h"               // load_bin_file, compute_cosine_similarity
#include "safetensors.h"                    // SafetensorsLoader
#include "whisper_encoder.h"                // WhisperEncoder{,Config}, WhisperWeights
#include "ultravox_projector_pipeline.cuh"  // ProjectorConfig
#include "ping_pong_audio_buffer.h"         // PingPongAudioBuffer
#include "audio_embedding_pipeline.h"       // AudioEmbeddingPipeline

using audio_test::compute_cosine_similarity;
using audio_test::load_bin_file;
using blackwell::DeviceBuffer;
using blackwell::audio::AudioEmbeddingPipeline;
using blackwell::audio::PingPongAudioBuffer;
using blackwell::audio::ProjectorConfig;
using blackwell::audio::WhisperEncoderConfig;
using blackwell::audio::WhisperLayerWeights;
using blackwell::audio::WhisperWeights;

namespace {

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

std::vector<float> load_bf16(const SafetensorsLoader& st, const std::string& name) {
    const TensorEntry& e = st.get_tensor(name);
    const size_t n = e.byte_size / sizeof(uint16_t);
    std::ifstream f(e.file_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open shard " + e.file_path);
    f.seekg(static_cast<std::streamoff>(e.file_offset));
    std::vector<uint16_t> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(e.byte_size));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        out[i] = v;
    }
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// A) Pure double-buffer / event machinery: overlap two streams, verify no races.
// -----------------------------------------------------------------------------
TEST(AudioPingPong, PingPongOverlapNoRaces) {
    constexpr std::size_t kSlotElems = 1u << 20;  // 1M floats = 4 MB (real overlap)
    constexpr int kFrames = 200;
    constexpr int kSamples = 3;
    const std::size_t offs[kSamples] = {0, kSlotElems / 2, kSlotElems - 1};

    PingPongAudioBuffer pp(kSlotElems);

    cudaStream_t prod = nullptr, cons = nullptr;
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&prod, cudaStreamNonBlocking));
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&cons, cudaStreamNonBlocking));

    // Pinned host sink; sentinel != any expected pattern so an un-copied sample
    // is caught too.
    uint32_t* h_out = nullptr;
    const size_t n_out = static_cast<size_t>(kFrames) * kSamples;
    CUDA_CHECK_THROW(cudaHostAlloc(reinterpret_cast<void**>(&h_out),
                                   n_out * sizeof(uint32_t), cudaHostAllocDefault));
    for (size_t i = 0; i < n_out; ++i) h_out[i] = 0xDEADBEEFu;

    auto consume = [&](int c) {
        pp.consumer_acquire(c, cons);  // wait producer done writing slot(c)
        for (int s = 0; s < kSamples; ++s) {
            const void* src = reinterpret_cast<const void*>(pp.slot(c) + offs[s]);
            CUDA_CHECK_THROW(cudaMemcpyAsync(&h_out[(size_t)c * kSamples + s], src,
                                             sizeof(uint32_t), cudaMemcpyDeviceToHost, cons));
        }
        pp.consumer_release(c, cons);  // slot free for producer to reuse (frame c+2)
    };

    // Producer writes frame N while the consumer reads frame N-1 -- overlapped,
    // event-gated, no device sync inside the loop.
    for (int i = 0; i < kFrames; ++i) {
        const int b = i & 0xFF;
        pp.producer_acquire(i, prod);  // slot free (frame i-2 consumed)?
        CUDA_CHECK_THROW(cudaMemsetAsync(pp.slot(i), b, pp.bytes(), prod));
        pp.producer_publish(i, prod);
        if (i > 0) consume(i - 1);
    }
    consume(kFrames - 1);  // drain the last produced frame

    CUDA_CHECK_THROW(cudaStreamSynchronize(cons));  // single end-of-run sync
    CUDA_CHECK_THROW(cudaStreamSynchronize(prod));

    int mismatches = 0;
    for (int i = 0; i < kFrames; ++i) {
        const uint32_t expect = 0x01010101u * static_cast<uint32_t>(i & 0xFF);
        for (int s = 0; s < kSamples; ++s)
            if (h_out[(size_t)i * kSamples + s] != expect) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0)
        << mismatches << " corrupted samples across " << kFrames
        << " overlapped frames -- the double-buffer event handoff has a race";

    CUDA_CHECK_THROW(cudaFreeHost(h_out));
    CUDA_CHECK_THROW(cudaStreamDestroy(prod));
    CUDA_CHECK_THROW(cudaStreamDestroy(cons));
    std::printf("[pingpong] %d frames x %d samples overlapped, 0 races\n", kFrames, kSamples);
}

// -----------------------------------------------------------------------------
// B) Real audio pipeline through the ping-pong, consumed cross-stream.
// -----------------------------------------------------------------------------
TEST(AudioPingPong, RealPipelineParity) {
    if (!file_exists(dump("input_features.bin")) || !file_exists(dump("audio_embeds.bin")))
        GTEST_SKIP() << "golden dumps absent in " << dumps_dir();
    if (!file_exists(ckpt_path()))
        GTEST_SKIP() << "Ultravox checkpoint absent: " << ckpt_path();

    constexpr int kLayers = 32;
    constexpr int kTextHidden = 2048;  // v0_5-llama-3_2-1b projector width

    const std::vector<float> mel = load_bin_file(dump("input_features.bin"));
    const std::vector<float> embeds_ref = load_bin_file(dump("audio_embeds.bin"));

    // Encoder weights (BF16 -> fp32) from the checkpoint.
    SafetensorsLoader st(ckpt_path());
    const auto W = [&](const std::string& n) { return load_bf16(st, "audio_tower." + n); };
    WhisperWeights ew;
    ew.conv1_w = W("conv1.weight");
    ew.conv1_b = W("conv1.bias");
    ew.conv2_w = W("conv2.weight");
    ew.conv2_b = W("conv2.bias");
    ew.embed_positions = W("embed_positions.weight");
    ew.layer_norm_w = W("layer_norm.weight");
    ew.layer_norm_b = W("layer_norm.bias");
    ew.layers.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        WhisperLayerWeights& s = ew.layers[static_cast<size_t>(l)];
        s.self_attn_layer_norm_w = W(p + "self_attn_layer_norm.weight");
        s.self_attn_layer_norm_b = W(p + "self_attn_layer_norm.bias");
        s.q_w = W(p + "self_attn.q_proj.weight");
        s.q_b = W(p + "self_attn.q_proj.bias");
        s.k_w = W(p + "self_attn.k_proj.weight");
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

    // Projector weights from the golden dumps.
    const std::vector<float> w_ln_pre = load_bin_file(dump("w_ln_pre.bin"));
    const std::vector<float> w_linear_1 = load_bin_file(dump("w_linear_1.bin"));
    const std::vector<float> w_ln_mid = load_bin_file(dump("w_ln_mid.bin"));
    const std::vector<float> w_linear_2 = load_bin_file(dump("w_linear_2.bin"));

    WhisperEncoderConfig enc_cfg;   // turbo defaults
    ProjectorConfig proj_cfg;
    proj_cfg.text_hidden = kTextHidden;

    AudioEmbeddingPipeline pipe(enc_cfg, proj_cfg);
    pipe.load_weights(ew, w_ln_pre, w_linear_1, w_ln_mid, w_linear_2);

    const size_t embed_elems = pipe.embed_elems();
    ASSERT_EQ(embed_elems, embeds_ref.size());

    PingPongAudioBuffer pp(embed_elems);

    DeviceBuffer<float> d_mel(mel.size());
    CUDA_CHECK_THROW(cudaMemcpy(d_mel.get(), mel.data(), mel.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    cudaStream_t cons = nullptr;
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&cons, cudaStreamNonBlocking));
    std::vector<float> got(embed_elems);

    // Two overlapped frames of the SAME audio: produce frame N (audio stream)
    // while consuming frame N-1 (consumer stream), event-gated. Both must match
    // the golden embeddings; the second is read back for the parity check.
    pipe.process_frame(d_mel.get(), pp, /*frame=*/0);
    pipe.process_frame(d_mel.get(), pp, /*frame=*/1);
    pp.consumer_acquire(0, cons);   // gate on the audio stream's ready event
    pp.consumer_release(0, cons);
    pp.consumer_acquire(1, cons);
    CUDA_CHECK_THROW(cudaMemcpyAsync(got.data(), pp.slot(1), pp.bytes(),
                                     cudaMemcpyDeviceToHost, cons));
    pp.consumer_release(1, cons);
    CUDA_CHECK_THROW(cudaStreamSynchronize(cons));
    CUDA_CHECK_THROW(cudaGetLastError());

    const double cos = compute_cosine_similarity(got, embeds_ref);
    std::printf("[pingpong] real audio pipeline -> audio_embeds cosine = %.8f\n", cos);
    EXPECT_GT(cos, 0.99);

    CUDA_CHECK_THROW(cudaStreamDestroy(cons));
}
