// Standalone parity test for the Ultravox projector CUDA kernels
// (launch_stack_audio_frames, launch_swiglu) against the FP32 PyTorch golden
// dumps, using cosine similarity > 0.999.
//
// Build (from a VS/vcvars64 shell so nvcc finds cl.exe):
//   nvcc -std=c++17 -arch=sm_120 \
//        tests/test_ultravox_projector.cpp src/audio/ultravox_projector.cu \
//        -I src -I src/core -I src/audio -o test_ultravox_projector.exe
//   ./test_ultravox_projector.exe [golden_dumps/ultravox dir]
//
// RECONCILIATION WITH THE BRIEF:
//   * The dumps are FP32, not FP16 (see scripts/generate_ultravox_audio_dumps.py
//     and the .bin sizes), so load_bin_file reads float32; loading them as half
//     would misread the bytes. Cosine is still accumulated in double.
//   * File names map to what the generator actually wrote:
//       00_whisper_out  -> encoder_last_hidden.bin   (stack input)
//       01_stacked_out  -> proj_stacked.bin          (stack expected)
//       03_linear_1     -> proj_linear_1.bin         (swiglu input)
//       04_swiglu_out   -> proj_swiglu.bin           (swiglu expected)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"             // CUDA_CHECK_THROW
#include "device_buffer.h"      // blackwell::DeviceBuffer
#include "ultravox_projector.cuh"
#include "ultravox_projector_pipeline.cuh"

using blackwell::DeviceBuffer;
using blackwell::audio::ProjectorConfig;
using blackwell::audio::UltravoxProjector;

// --- Geometry (Ultravox v0_5-llama-3_2-1b; cross-checked against dump sizes) --
static constexpr int kNumFrames = 1500;   // whisper encoder frames (30 s)
static constexpr int kHidden    = 1280;   // whisper d_model
static constexpr int kStack     = 8;      // stack_factor / compression
static constexpr int kOutFrames = (kNumFrames + kStack - 1) / kStack;  // 188
static constexpr int kStackedDim = kHidden * kStack;                   // 10240
static constexpr int kTokens    = kOutFrames;                          // 188
static constexpr int kLin1Dim   = 4096;   // projector hidden (linear_1 out)
static constexpr int kSwigluDim = kLin1Dim / 2;                        // 2048

// --- FP32 raw dump loader -----------------------------------------------------
std::vector<float> load_bin_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "FATAL: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// Cosine similarity, accumulated in double to avoid overflow / precision loss.
double compute_cosine_similarity(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "FATAL: size mismatch %zu vs %zu\n", a.size(), b.size());
        std::exit(2);
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

static DeviceBuffer<float> to_device(const std::vector<float>& h) {
    DeviceBuffer<float> d(h.size());
    CUDA_CHECK_THROW(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    return d;  // moved out
}

static std::vector<float> from_device(const DeviceBuffer<float>& d) {
    std::vector<float> h(d.count());
    CUDA_CHECK_THROW(cudaMemcpy(h.data(), d.get(), d.size_bytes(),
                                cudaMemcpyDeviceToHost));
    return h;
}

static void expect_size(const char* name, const std::vector<float>& v, size_t want) {
    if (v.size() != want) {
        std::fprintf(stderr, "FATAL: %s has %zu floats, expected %zu\n",
                     name, v.size(), want);
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    const std::string base = (argc > 1)
        ? argv[1]
        : std::string("tests/integration/golden_dumps/ultravox");
    const std::string sep = "/";

    constexpr double kThreshold = 0.999;
    int failures = 0;

    // ---- Test 1: StackAudioFrames ------------------------------------------
    {
        auto whisper  = load_bin_file(base + sep + "encoder_last_hidden.bin");
        auto expected = load_bin_file(base + sep + "proj_stacked.bin");
        expect_size("encoder_last_hidden", whisper, (size_t)kNumFrames * kHidden);
        expect_size("proj_stacked", expected, (size_t)kOutFrames * kStackedDim);

        DeviceBuffer<float> d_in = to_device(whisper);
        DeviceBuffer<float> d_out((size_t)kOutFrames * kStackedDim);

        launch_stack_audio_frames(d_in, d_out, kNumFrames, kHidden, kStack);
        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        const double cos = compute_cosine_similarity(from_device(d_out), expected);
        const bool ok = cos > kThreshold;
        std::printf("[%s] StackAudioFrames  cosine = %.8f  (> %.3f)\n",
                    ok ? "PASS" : "FAIL", cos, kThreshold);
        failures += !ok;
    }

    // ---- Test 2: SwiGLU -----------------------------------------------------
    {
        auto lin1     = load_bin_file(base + sep + "proj_linear_1.bin");
        auto expected = load_bin_file(base + sep + "proj_swiglu.bin");
        expect_size("proj_linear_1", lin1, (size_t)kTokens * kLin1Dim);
        expect_size("proj_swiglu", expected, (size_t)kTokens * kSwigluDim);

        DeviceBuffer<float> d_in = to_device(lin1);
        DeviceBuffer<float> d_out((size_t)kTokens * kSwigluDim);

        launch_swiglu(d_in, d_out, kTokens, kLin1Dim);
        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        const double cos = compute_cosine_similarity(from_device(d_out), expected);
        const bool ok = cos > kThreshold;
        std::printf("[%s] SwiGLU            cosine = %.8f  (> %.3f)\n",
                    ok ? "PASS" : "FAIL", cos, kThreshold);
        failures += !ok;
    }

    // ---- Test 3: full UltravoxProjector forward (end-to-end) ---------------
    {
        auto whisper  = load_bin_file(base + sep + "encoder_last_hidden.bin");
        auto expected = load_bin_file(base + sep + "audio_embeds.bin");
        auto w_ln_pre   = load_bin_file(base + sep + "w_ln_pre.bin");
        auto w_linear_1 = load_bin_file(base + sep + "w_linear_1.bin");
        auto w_ln_mid   = load_bin_file(base + sep + "w_ln_mid.bin");
        auto w_linear_2 = load_bin_file(base + sep + "w_linear_2.bin");
        expect_size("encoder_last_hidden", whisper, (size_t)kNumFrames * kHidden);
        expect_size("audio_embeds", expected, (size_t)kTokens * kSwigluDim);
        expect_size("w_ln_pre", w_ln_pre, (size_t)kStackedDim);
        expect_size("w_linear_1", w_linear_1, (size_t)kLin1Dim * kStackedDim);
        expect_size("w_ln_mid", w_ln_mid, (size_t)kSwigluDim);
        expect_size("w_linear_2", w_linear_2, (size_t)kSwigluDim * kSwigluDim);

        ProjectorConfig cfg;  // defaults match the v0_5-llama-3_2-1b geometry
        UltravoxProjector projector(cfg, /*max_input_frames=*/kNumFrames);
        projector.load_weights(w_ln_pre, w_linear_1, w_ln_mid, w_linear_2);

        DeviceBuffer<float> d_in = to_device(whisper);
        const float* d_out = projector.forward(d_in, kNumFrames);
        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        std::vector<float> got((size_t)kTokens * kSwigluDim);
        CUDA_CHECK_THROW(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        const double cos = compute_cosine_similarity(got, expected);
        const bool ok = cos > kThreshold;
        std::printf("[%s] UltravoxProjector end-to-end  cosine = %.8f  (> %.3f)\n",
                    ok ? "PASS" : "FAIL", cos, kThreshold);
        failures += !ok;
    }

    constexpr int kNumTests = 3;
    std::printf("\n%s: %d/%d checks passed the cosine>%.3f parity bar.\n",
                failures == 0 ? "SUCCESS" : "FAILURE",
                kNumTests - failures, kNumTests, kThreshold);
    return failures == 0 ? 0 : 1;
}
