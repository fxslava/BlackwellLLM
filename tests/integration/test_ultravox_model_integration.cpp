// =============================================================================
// Ultravox model-integration parity test (end-to-end audio frontend, 1B).
//
// Chains the REAL CUDA stages the engine runs at inference time and matches the
// PyTorch golden reference at every boundary (cosine similarity > 0.999):
//
//     encoder_last_hidden [1500,1280]            (audio embeddings := Whisper out)
//       -> UltravoxProjector (GPU)  ---> audio_embeds  [188, 2048]   [Stage 1]
//       -> PromptInjector    (GPU)  ---> spliced_embeds [194, 2048]  [Stage 2]
//       -> Llama-3.2-1B prefill     ---> logits [194, vocab]         [Stage 3]
//
// Unlike the per-kernel sibling tests (test_ultravox_projector.cpp,
// test_prompt_injector.cpp), Stage 2 splices the projector's OWN GPU output (not
// the golden audio_embeds.bin) into the text sequence, so a regression anywhere
// in the projector graph surfaces here as a spliced-parity failure. Stage 2 is
// the last boundary reproducible from the LOCAL Ultravox checkpoint, which ships
// only audio_tower.* + multi_modal_projector.* (no Llama backbone).
//
// GEOMETRY: the golden dumps in tests/integration/golden_dumps/ultravox are the
// v0_5-llama-3_2-1b variant (linear_2 [2048,2048], audio_embeds [188,2048]).
// ProjectorConfig now DEFAULTS to the 8B backbone (text_hidden=4096), so this
// test MUST pin text_hidden=2048 to match its 1B weights -- otherwise linear_2
// reads out of bounds on the 1B weight buffer (illegal device access). The 8B
// full-pipeline path lives in test_ultravox8b_full_pipeline.cpp.
//
// Stage 3 (Llama prefill -> logits) is the DROP-IN SEAM: a separate test that
// SKIPS until the text backbone weights exist locally. See the LlamaLogitsSeam
// test below.
//
// Env overrides:
//   BLACKWELL_ULTRAVOX_DUMPS   dumps dir (default <repo>/tests/integration/golden_dumps/ultravox)
//   BLACKWELL_LLAMA32_1B_DIR   Llama-3.2-1B-Instruct dir (default F:/AI/Llama-3.2-1B-Instruct)
// =============================================================================

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"             // CUDA_CHECK_THROW
#include "device_buffer.h"      // blackwell::DeviceBuffer
#include "audio_test_utils.h"   // load_bin_file, compute_cosine_similarity
#include "ultravox_projector.cuh"
#include "ultravox_projector_pipeline.cuh"
#include "prompt_injector.cuh"

using blackwell::DeviceBuffer;
using blackwell::audio::ProjectorConfig;
using blackwell::audio::UltravoxProjector;
using audio_test::compute_cosine_similarity;
using audio_test::load_bin_file;

namespace {

// --- Geometry (Ultravox v0_5-llama-3_2-1b; cross-checked against dump sizes) --
constexpr int kNumFrames  = 1500;   // whisper encoder frames (30 s)
constexpr int kHidden     = 1280;   // whisper d_model
constexpr int kStack      = 8;      // stack_factor / compression
constexpr int kOutFrames  = (kNumFrames + kStack - 1) / kStack;  // 188 audio soft-tokens
constexpr int kTextHidden = 2048;   // projector out == Llama-3.2-1B hidden
constexpr double kThreshold = 0.999;

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

std::string dumps_dir() {
    return env_or("BLACKWELL_ULTRAVOX_DUMPS",
                  "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/ultravox");
}
std::string llama_dir() { return env_or("BLACKWELL_LLAMA32_1B_DIR", "F:/AI/Llama-3.2-1B-Instruct"); }

bool file_exists(const std::string& p) { return std::ifstream(p).good(); }
std::string path(const std::string& name) { return dumps_dir() + "/" + name; }

DeviceBuffer<float> to_device(const std::vector<float>& h) {
    DeviceBuffer<float> d(h.size());
    CUDA_CHECK_THROW(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    return d;
}

std::vector<float> from_device(const float* d, size_t count) {
    std::vector<float> h(count);
    CUDA_CHECK_THROW(cudaMemcpy(h.data(), d, count * sizeof(float),
                                cudaMemcpyDeviceToHost));
    return h;
}

// The injector splice params are read from the generator's sidecar so the test
// never hard-codes what the prompt tokenizer decided (see dump_injector_tensors).
struct SpliceMeta {
    int seq_len = 0, audio_pos = 0, num_audio = 0, hidden = 0;
    int out_rows() const { return seq_len - 1 + num_audio; }
};

}  // namespace

// =============================================================================
// Stages 1 + 2: the end-to-end audio frontend reproducible from the LOCAL
// checkpoint. Each stage consumes the GPU output of the previous one, so a
// projector regression surfaces at the spliced-parity boundary.
// =============================================================================
TEST(UltravoxModelIntegration, ProjectorInjectorParity) {
    if (!file_exists(path("audio_embeds.bin")))
        GTEST_SKIP() << "dumps absent in " << dumps_dir()
                     << " (run scripts/generate_ultravox_audio_dumps.py, or set "
                        "BLACKWELL_ULTRAVOX_DUMPS)";

    // Injector splice sidecar: "seq_len audio_pos num_audio hidden".
    SpliceMeta sp;
    {
        std::ifstream m(path("injector_meta.txt"));
        ASSERT_TRUE(bool(m >> sp.seq_len >> sp.audio_pos >> sp.num_audio >> sp.hidden))
            << "cannot read injector_meta.txt in " << dumps_dir();
    }
    ASSERT_EQ(sp.hidden, kTextHidden)
        << "splice meta hidden disagrees with 1B geometry";
    ASSERT_EQ(sp.num_audio, kOutFrames)
        << "splice meta num_audio disagrees with 1B geometry";

    // ---- Stage 1: UltravoxProjector (audio embeddings -> soft tokens) --------
    auto whisper    = load_bin_file(path("encoder_last_hidden.bin"));
    auto proj_ref   = load_bin_file(path("audio_embeds.bin"));
    auto w_ln_pre   = load_bin_file(path("w_ln_pre.bin"));
    auto w_linear_1 = load_bin_file(path("w_linear_1.bin"));
    auto w_ln_mid   = load_bin_file(path("w_ln_mid.bin"));
    auto w_linear_2 = load_bin_file(path("w_linear_2.bin"));

    ProjectorConfig cfg;
    cfg.text_hidden = kTextHidden;   // 1B: linear_2 out 4096(default 8B) -> 2048
    ASSERT_EQ(cfg.hidden_dim, kHidden);
    ASSERT_EQ(cfg.stack_factor, kStack);

    UltravoxProjector projector(cfg, /*max_input_frames=*/kNumFrames);
    projector.load_weights(w_ln_pre, w_linear_1, w_ln_mid, w_linear_2);

    DeviceBuffer<float> d_in = to_device(whisper);
    const float* d_out = projector.forward(d_in, kNumFrames);
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    const size_t n_audio = static_cast<size_t>(kOutFrames) * kTextHidden;
    const double cos_s1 = compute_cosine_similarity(from_device(d_out, n_audio), proj_ref);
    std::printf("[uv1b] Stage 1 projector cosine     = %.8f\n", cos_s1);
    EXPECT_GT(cos_s1, kThreshold);

    // Own a persistent copy so it survives past the projector's workspace.
    DeviceBuffer<float> d_audio(n_audio);
    CUDA_CHECK_THROW(cudaMemcpy(d_audio.get(), d_out, n_audio * sizeof(float),
                                cudaMemcpyDeviceToDevice));

    // ---- Stage 2: PromptInjector (splice the LIVE Stage-1 output) ------------
    // Splices the projector's GPU output -- not the golden dump -- so this is the
    // true end-to-end audio-frontend parity boundary.
    auto text        = load_bin_file(path("text_embeds.bin"));
    auto spliced_ref = load_bin_file(path("spliced_embeds_ref.bin"));

    DeviceBuffer<float> d_text = to_device(text);
    const size_t n_spliced = static_cast<size_t>(sp.out_rows()) * sp.hidden;
    DeviceBuffer<float> d_spliced(n_spliced);

    inject_audio_embeddings(d_text, d_audio, d_spliced, sp.seq_len, sp.audio_pos,
                            sp.num_audio, sp.hidden);
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    const double cos_s2 = compute_cosine_similarity(
        from_device(d_spliced.get(), n_spliced), spliced_ref);
    std::printf("[uv1b] Stage 2 projector+injector   = %.8f\n", cos_s2);
    EXPECT_GT(cos_s2, kThreshold);
}

// =============================================================================
// Stage 3: Llama-3.2-1B prefill -> logits (DROP-IN SEAM).
//
// The spliced embeddings from Stage 2 are [out_rows, 2048] INPUT embeddings ready
// to feed a Llama prefill that starts from embeddings (bypassing embed_tokens for
// the audio soft-token rows). When the backbone weights land at llama_dir(), wire
// this up exactly like test_ultravox8b_full_pipeline.cpp's Stage 3 (granular
// step_* sweep injecting each spliced row into d_X_accum), NO change to Stage 1-2:
//
//   1. Generate the reference (07_final_logits.bin [out_rows, vocab] FP32) via a
//      run_full_model() pass in scripts/generate_ultravox_audio_dumps.py (needs the
//      gated meta-llama/Llama-3.2-1B-Instruct backbone).
//   2. Load Llama-3.2-1B via BlackwellEngine (KVCacheMode::Continuous), prefill
//      seeded with the spliced rows, read per-row logits back to host.
//   3. EXPECT_GT(cosine(logits, 07_final_logits.bin), bar) + top-1 argmax match.
// =============================================================================
TEST(UltravoxModelIntegration, LlamaLogitsSeam) {
    if (!file_exists(llama_dir() + "/config.json"))
        GTEST_SKIP() << "backbone absent: " << llama_dir()
                     << " (set BLACKWELL_LLAMA32_1B_DIR)";
    if (!file_exists(path("07_final_logits.bin")))
        GTEST_SKIP() << "reference absent: 07_final_logits.bin "
                        "(regenerate dumps with the backbone)";
    // Weights + reference present but prefill-from-embeddings wiring is not yet
    // implemented; skip loudly so activation is a deliberate step, not an accident.
    GTEST_SKIP() << "TODO: implement Llama prefill-from-embeddings (see header)";
}
