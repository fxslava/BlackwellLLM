// -----------------------------------------------------------------------------
// main.cpp — sandbox driver: WAV -> log-mel -> parity check vs Python golden.
//
//   audio_sandbox <data_dir>
//     <data_dir>/test_audio.wav        (dr_wav -> mono f32, 16 kHz)
//     <data_dir>/mel_filters.bin       (loaded by WhisperDSP)
//     <data_dir>/expected_log_mel.bin  (golden, float32 [n_mels, n_frames])
//
// Exit 0 iff cosine similarity vs the golden exceeds the parity bar.
// -----------------------------------------------------------------------------
#define DR_WAV_IMPLEMENTATION
// dr_wav.h is third-party (SYSTEM include), but C4701 "potentially uninitialized
// local" is emitted late in codegen and slips past /external:W0, so /WX would
// flag it. Quarantine only this third-party #include (skill compiler-hygiene).
#pragma warning(push)
#pragma warning(disable: 4701)  // potentially uninitialized local variable
#include "dr_wav.h"
#pragma warning(pop)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "whisper_dsp.h"

namespace {

constexpr double kCosineBar = 0.9999;  // parity threshold (project rule: > 0.999)

// Load a mono 16 kHz float clip via dr_wav. Downmixes multi-channel by average
// and hard-fails on a sample-rate mismatch (the DSP geometry assumes 16 kHz).
std::vector<float> load_wav_mono_16k(const std::string& path, int expect_sr) {
    unsigned int channels = 0, sample_rate = 0;
    drwav_uint64 total_frames = 0;
    float* raw = drwav_open_file_and_read_pcm_frames_f32(
        path.c_str(), &channels, &sample_rate, &total_frames, nullptr);
    if (!raw) throw std::runtime_error("dr_wav failed to open: " + path);

    if (static_cast<int>(sample_rate) != expect_sr) {
        drwav_free(raw, nullptr);
        throw std::runtime_error("wav sample rate " + std::to_string(sample_rate) +
                                 " != expected " + std::to_string(expect_sr));
    }

    std::vector<float> pcm(static_cast<size_t>(total_frames));
    if (channels == 1) {
        std::copy(raw, raw + total_frames, pcm.begin());
    } else {
        for (drwav_uint64 i = 0; i < total_frames; ++i) {
            float acc = 0.0f;
            for (unsigned int c = 0; c < channels; ++c)
                acc += raw[i * channels + c];
            pcm[static_cast<size_t>(i)] = acc / static_cast<float>(channels);
        }
    }
    drwav_free(raw, nullptr);
    return pcm;
}

std::vector<float> load_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open: " + path);
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

struct Metrics {
    double cosine;
    double max_abs_err;
    double mean_abs_err;
};

Metrics compare(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        const double d = std::fabs(x - y);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    return {denom > 0.0 ? dot / denom : 0.0, max_abs, sum_abs / a.size()};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";
    const std::string wav_path = data_dir + "/test_audio.wav";
    const std::string mel_path = data_dir + "/mel_filters.bin";
    const std::string exp_path = data_dir + "/expected_log_mel.bin";

    try {
        whisper::DspConfig cfg;  // Whisper large-v3-turbo defaults
        whisper::WhisperDSP dsp(cfg, mel_path);

        const std::vector<float> pcm = load_wav_mono_16k(wav_path, cfg.sample_rate);
        std::printf("loaded %zu samples (%.3f s @ %d Hz)\n", pcm.size(),
                    static_cast<double>(pcm.size()) / cfg.sample_rate, cfg.sample_rate);

        const whisper::LogMel out = dsp.process(pcm);
        std::printf("computed log-mel: [%d mels x %d frames] = %zu values\n",
                    out.n_mels, out.n_frames, out.size());

        const std::vector<float> golden = load_f32(exp_path);
        if (golden.size() != out.size()) {
            std::fprintf(stderr,
                         "FAIL: shape mismatch — golden has %zu values, C++ produced %zu\n"
                         "      (expected %d frames; check the DSP framing/drop-last logic)\n",
                         golden.size(), out.size(), out.n_frames);
            return 2;
        }

        const Metrics m = compare(out.data, golden);
        std::printf("\nparity vs golden:\n");
        std::printf("  cosine similarity = %.8f  (bar > %.4f)\n", m.cosine, kCosineBar);
        std::printf("  max  abs error    = %.6e\n", m.max_abs_err);
        std::printf("  mean abs error    = %.6e\n", m.mean_abs_err);

        if (m.cosine > kCosineBar) {
            std::printf("\nPASS ✓\n");
            return 0;
        }
        std::printf("\nFAIL ✗ (cosine below parity bar)\n");
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 3;
    }
}
