// =============================================================================
// The Ultimate Field Test — Dynamic Overlap Reconciliation on REAL audio.
//
// Runs two real-audio sources (a real-speech mel window and its time-reversed
// twin) through the REAL Whisper encoder + Ultravox projector, then SPLICES their
// projected soft-tokens at a KNOWN token index k: rows [0,k) are byte-identical to
// the injected history (cosine 1.0), rows [k,N) come from the divergent twin. This
// pins the divergence coordinate to exactly k, so coordinate detection and the
// rewind cap can be asserted INDEPENDENTLY:
//
//   1. Uncapped coordinate: cap large -> refill_from == k, rewind_depth == overlap-k.
//   2. Cap enforcement (same splice): cap 2 -> rewind_depth == 2, refill_from == overlap-2.
//
// The streaming parameters come from the REAL config pipeline
// (build_and_validate_runtime), not literals — exercising tier-2 -> tier-3
// resolution and the capability gate.
//
// SKIPs (does not fail) when the audio head or the golden log-mel are absent:
//   BLACKWELL_AUDIO_HEAD      Ultravox head dir (default F:/AI/ultravox-v0_5-llama-3_1-8b)
//   BLACKWELL_ULTRAVOX_DUMPS  golden-dump dir   (default tests/integration/golden_dumps/ultravox)
// =============================================================================
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"                    // CUDA_CHECK
#include "device_buffer.h"             // blackwell::DeviceBuffer
#include "blackwell/config.h"          // ModelConfig
#include "blackwell/runtime_config.h"  // InferenceConfig / build_and_validate_runtime
#include "audio_embedding_pipeline.h"  // AudioEmbeddingPipeline, ReconciliationPlan
#include "sliding_audio_window.h"      // blackwell::audio::kMelFramesPerSoftToken
#include "audio_head_loader.hpp"       // rt::load_audio_pipeline (real encoder+projector)

using blackwell::DeviceBuffer;
using blackwell::audio::AudioEmbeddingPipeline;

namespace {

std::string env_or(const char* n, const std::string& f) {
    const char* v = std::getenv(n);
    return (v && *v) ? std::string(v) : f;
}
std::string audio_head_dir() {
    return env_or("BLACKWELL_AUDIO_HEAD", "F:/AI/ultravox-v0_5-llama-3_1-8b");
}
std::string dumps_dir() {
    return env_or("BLACKWELL_ULTRAVOX_DUMPS",
                  "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/ultravox");
}
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

constexpr int kNumMel     = 128;
constexpr int kMelFrames  = 3000;   // input_features.bin is [128, 3000]
constexpr int kTextHidden = 4096;   // 8B Ultravox head projector output width

std::vector<float> load_input_features() {
    const std::string path = dumps_dir() + "/input_features.bin";
    std::ifstream f(path, std::ios::binary);
    std::vector<float> v(static_cast<size_t>(kNumMel) * kMelFrames);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return v;
}

// Copy a [kNumMel, win_frames] window out of the mel-major [128,3000] features,
// optionally reversing the time axis within the window (a distinct "second WAV").
DeviceBuffer<float> make_window(const std::vector<float>& mel, int win_frames, bool reversed) {
    std::vector<float> host(static_cast<size_t>(kNumMel) * win_frames);
    for (int r = 0; r < kNumMel; ++r)
        for (int c = 0; c < win_frames; ++c) {
            const int src = reversed ? (win_frames - 1 - c) : c;
            host[static_cast<size_t>(r) * win_frames + c] =
                mel[static_cast<size_t>(r) * kMelFrames + src];
        }
    DeviceBuffer<float> d(host.size());
    CUDA_CHECK(cudaMemcpy(d.get(), host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return d;
}

// Encode+project a window and copy the projector output into a PERSISTENT device
// buffer (the projector reuses its workspace across calls). Returns num soft-tokens.
int project_persistent(AudioEmbeddingPipeline& pipe, const float* d_mel, int win_frames,
                       DeviceBuffer<float>& out) {
    const auto w = pipe.encode_project_window(d_mel, win_frames);
    CUDA_CHECK(cudaStreamSynchronize(pipe.audio_stream()));
    out.allocate(static_cast<size_t>(w.num_tokens) * kTextHidden);
    CUDA_CHECK(cudaMemcpy(out.get(), w.embeds, out.count() * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return w.num_tokens;
}

// Splice [num_tokens, H]: rows [0,k) from dA, rows [k,num_tokens) from dB.
DeviceBuffer<float> splice_at(const DeviceBuffer<float>& dA, const DeviceBuffer<float>& dB,
                              int k, int num_tokens, int H) {
    DeviceBuffer<float> dS(static_cast<size_t>(num_tokens) * H);
    const size_t head = static_cast<size_t>(k) * H;
    const size_t tail = static_cast<size_t>(num_tokens - k) * H;
    CUDA_CHECK(cudaMemcpy(dS.get(), dA.get(), head * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dS.get() + head, dB.get() + head, tail * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return dS;
}

std::vector<float> download_row(const DeviceBuffer<float>& d, int row, int H) {
    std::vector<float> h(H);
    CUDA_CHECK(cudaMemcpy(h.data(), d.get() + static_cast<size_t>(row) * H,
                          H * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// Resolve the streaming plan through the REAL config pipeline for a given rewind cap
// (dense model -> reconciliation enabled; window 2240 ms / hop 320 ms).
blackwell::AudioStreamingPlan plan_for_cap(int cap) {
    ModelConfig m{};
    m.hidden_dim = kTextHidden;
    m.num_layers = 32;
    m.head_dim = 128;
    m.max_position_embeddings = 4096;
    blackwell::InferenceConfig req;
    req.audio_streaming.enable = true;
    req.audio_streaming.max_reconciliation_rewind_tokens = cap;
    return build_and_validate_runtime(m, blackwell::derive_capabilities(m), req).audio_streaming;
}

}  // namespace

TEST(AudioOverlapReconciliation, ExactDivergenceCoordinateAndCapEnforcement) {
    if (!file_exists(audio_head_dir() + "/model.safetensors"))
        GTEST_SKIP() << "audio head absent: " << audio_head_dir() << " (set BLACKWELL_AUDIO_HEAD)";
    if (!file_exists(dumps_dir() + "/input_features.bin"))
        GTEST_SKIP() << "golden log-mel absent in " << dumps_dir()
                     << " (set BLACKWELL_ULTRAVOX_DUMPS)";

    const auto geom = plan_for_cap(8);   // geometry (cap irrelevant to window/hop/overlap)
    ASSERT_TRUE(geom.enabled);
    ASSERT_EQ(geom.window_tokens, 14);
    ASSERT_EQ(geom.overlap_tokens, 12);
    const int N = geom.window_tokens;        // 14
    const int overlap = geom.overlap_tokens; // 12
    const int H = kTextHidden;
    const float thr = geom.reconciliation_threshold;

    // --- real encoder + projector over two real-audio windows -------------------
    auto pipe = rt::load_audio_pipeline(audio_head_dir(), kTextHidden);
    const std::vector<float> mel = load_input_features();
    const int win_frames = N * blackwell::audio::kMelFramesPerSoftToken;   // 224
    const DeviceBuffer<float> melA = make_window(mel, win_frames, /*reversed=*/false);
    const DeviceBuffer<float> melB = make_window(mel, win_frames, /*reversed=*/true);

    DeviceBuffer<float> dA, dB;   // persistent projected soft-tokens [N, H]
    ASSERT_EQ(project_persistent(*pipe, melA.get(), win_frames, dA), N);
    ASSERT_EQ(project_persistent(*pipe, melB.get(), win_frames, dB), N);

    // Splice the NEW window at a KNOWN coordinate: [0,k) == history (audio A),
    // [k,N) from the divergent twin (audio B).
    const int k = 4;
    const DeviceBuffer<float> dS = splice_at(dA, dB, k, N, H);
    const int kBasePos = 100;
    const int cur_pos = kBasePos + overlap;   // 112 (next free KV slot after history)

    // --- prove the divergence coordinate is EXACTLY k (independent of reconcile) --
    // reconcile compares dS[i] against history[i] == dA[i] (base 0). By construction
    // dS[k-1] == dA[k-1] (cosine 1.0) and dS[k] == dB[k] != dA[k].
    const double cos_before =
        AudioEmbeddingPipeline::cosine_similarity(download_row(dS, k - 1, H).data(),
                                                  download_row(dA, k - 1, H).data(), H);
    const double cos_at =
        AudioEmbeddingPipeline::cosine_similarity(download_row(dS, k, H).data(),
                                                  download_row(dA, k, H).data(), H);
    EXPECT_GE(cos_before, thr) << "token k-1 must match history (identical prefix)";
    EXPECT_LT(cos_at, thr)     << "token k must diverge (spliced twin); cos=" << cos_at;

    auto run = [&](int cap) {
        pipe->configure_streaming(plan_for_cap(cap));
        pipe->reset_history();
        pipe->record_injected(dA.get(), kBasePos, overlap);   // history = dA[0..overlap)
        return pipe->reconcile(dS.get(), N);
    };

    // === 1. UNCAPPED COORDINATE (cap 16 -> clamped to overlap 12; want_rewind 8 < 12) ===
    const auto uncapped = run(16);
    EXPECT_TRUE(uncapped.diverged);
    EXPECT_EQ(uncapped.refill_from, k);                             // exact coordinate
    EXPECT_EQ(cur_pos - uncapped.rewind_to_pos, overlap - k);       // depth == overlap-k == 8

    // === 2. CAP ENFORCEMENT (same splice, cap 2) ===
    const auto capped = run(2);
    EXPECT_TRUE(capped.diverged);
    EXPECT_EQ(cur_pos - capped.rewind_to_pos, 2);                   // depth clamped to cap
    EXPECT_EQ(capped.refill_from, overlap - 2);                     // 10

    std::printf("[reconcile-field] k=%d cos(k-1)=%.5f cos(k)=%.5f | "
                "uncapped: refill=%d depth=%d | capped: refill=%d depth=%d\n",
                k, cos_before, cos_at, uncapped.refill_from, cur_pos - uncapped.rewind_to_pos,
                capped.refill_from, cur_pos - capped.rewind_to_pos);
}
