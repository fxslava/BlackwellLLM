// =============================================================================
// The Ultimate Field Test — Dynamic Overlap Reconciliation on REAL audio.
//
// Splices two real-audio sources (a real-speech mel window and its time-reversed
// twin), runs them through the REAL Whisper encoder + Ultravox projector, and
// asserts that AudioEmbeddingPipeline::reconcile:
//   * does NOT false-trigger when the overlap is unchanged (identical projection
//     -> cosine 1.0 -> no divergence), and
//   * DOES detect divergence when the overlapping audio changes, and caps the
//     rewind depth at RuntimeConfig::AudioStreamingPlan::max_rewind_tokens.
//
// The streaming parameters come from the REAL config pipeline
// (build_and_validate_runtime), not literals — this exercises the tier-2 -> tier-3
// resolution and the capability gate end to end.
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

}  // namespace

TEST(AudioOverlapReconciliation, DetectsDivergenceAndCapsRewindOnRealAudio) {
    if (!file_exists(audio_head_dir() + "/model.safetensors"))
        GTEST_SKIP() << "audio head absent: " << audio_head_dir() << " (set BLACKWELL_AUDIO_HEAD)";
    if (!file_exists(dumps_dir() + "/input_features.bin"))
        GTEST_SKIP() << "golden log-mel absent in " << dumps_dir()
                     << " (set BLACKWELL_ULTRAVOX_DUMPS)";

    // --- streaming plan from the REAL config pipeline (no literals) --------------
    ModelConfig m{};
    m.hidden_dim = kTextHidden;
    m.num_layers = 32;
    m.head_dim = 128;
    m.max_position_embeddings = 4096;               // dense full-attention (no SSM gate)
    blackwell::InferenceConfig req;
    req.audio_streaming.enable = true;              // 2240 ms window / 320 ms hop
    req.audio_streaming.max_reconciliation_rewind_tokens = 2;   // tight cap under test
    const auto rt = build_and_validate_runtime(m, blackwell::derive_capabilities(m), req);
    const blackwell::AudioStreamingPlan& plan = rt.audio_streaming;
    ASSERT_TRUE(plan.enabled);
    ASSERT_EQ(plan.window_tokens, 14);
    ASSERT_EQ(plan.hop_tokens, 2);
    ASSERT_EQ(plan.overlap_tokens, 12);
    ASSERT_EQ(plan.max_rewind_tokens, 2);

    // --- real encoder + projector ------------------------------------------------
    auto pipe = rt::load_audio_pipeline(audio_head_dir(), kTextHidden);
    pipe->configure_streaming(plan);

    const std::vector<float> mel = load_input_features();
    const int win_frames = plan.window_tokens * blackwell::audio::kMelFramesPerSoftToken;  // 224
    DeviceBuffer<float> melA = make_window(mel, win_frames, /*reversed=*/false);
    DeviceBuffer<float> melB = make_window(mel, win_frames, /*reversed=*/true);   // "second WAV"

    // History = the first `overlap` projected soft-tokens of window A, injected at a
    // known base position. (record_injected copies to host, so melA's projector
    // output may be overwritten by the next encode.)
    const int kBasePos = 100;
    auto wa = pipe->encode_project_window(melA.get(), win_frames);
    CUDA_CHECK(cudaStreamSynchronize(pipe->audio_stream()));
    ASSERT_EQ(wa.num_tokens, plan.window_tokens);
    pipe->record_injected(wa.embeds, kBasePos, plan.overlap_tokens);   // [100, 112)
    ASSERT_EQ(pipe->history_size(), plan.overlap_tokens);

    // --- STABLE: re-encoding the SAME audio must NOT trigger a rewrite -----------
    auto wa2 = pipe->encode_project_window(melA.get(), win_frames);
    CUDA_CHECK(cudaStreamSynchronize(pipe->audio_stream()));
    const auto stable = pipe->reconcile(wa2.embeds, wa2.num_tokens);
    EXPECT_FALSE(stable.diverged) << "identical overlap must not diverge (false positive)";

    // --- DIVERGENT + CAP: reversed audio must diverge, rewind capped -------------
    auto wb = pipe->encode_project_window(melB.get(), win_frames);
    CUDA_CHECK(cudaStreamSynchronize(pipe->audio_stream()));
    const auto div = pipe->reconcile(wb.embeds, wb.num_tokens);

    const int cur_pos = kBasePos + plan.overlap_tokens;   // 112 (next free KV slot)
    EXPECT_TRUE(div.diverged) << "reversed-audio overlap must diverge below the 0.999 bar";
    EXPECT_EQ(cur_pos - div.rewind_to_pos, plan.max_rewind_tokens)   // rewind depth == cap
        << "rewind depth must be capped at max_reconciliation_rewind_tokens";
    EXPECT_EQ(div.rewind_to_pos, cur_pos - plan.max_rewind_tokens);   // 110
    EXPECT_EQ(div.refill_from, plan.overlap_tokens - plan.max_rewind_tokens);  // 10

    std::printf("[reconcile-field] stable.diverged=%d  div.diverged=%d "
                "rewind_to=%d (depth %d, cap %d) refill_from=%d\n",
                stable.diverged, div.diverged, div.rewind_to_pos,
                cur_pos - div.rewind_to_pos, plan.max_rewind_tokens, div.refill_from);
}
