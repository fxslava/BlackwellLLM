#pragma once
// -----------------------------------------------------------------------------
// whisper_dsp.h — CPU log-mel front-end for Whisper large-v3-turbo (the encoder
// used by fixie-ai/ultravox-v0_5-llama-3_2-1b). Standalone sandbox: no CUDA, no
// engine deps. Mirrors transformers' WhisperFeatureExtractor DSP exactly so it
// can be validated against a Python golden dump before porting to CUDA.
//
// Algorithm (must match scripts/export_whisper_dsp.py bit-for-bit):
//   reflect-pad by n_fft/2 (center=True) -> periodic Hann(n_fft) framing at hop
//   -> real FFT -> power spectrum -> drop last frame -> mel_filters^T @ power
//   -> log10(clamp 1e-10) -> floor at (global_max - 8) -> (x + 4) / 4.
//
// The mel filterbank is LOADED from mel_filters.bin (dumped by the Python side),
// never recomputed here — that is the single biggest source of float mismatch.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <string>
#include <vector>

namespace whisper {

// Whisper large-v3-turbo DSP geometry. Defaults match the checkpoint; the loaded
// mel_filters.bin size is cross-checked against these at load time.
struct DspConfig {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 128;
    int n_freqs() const { return n_fft / 2 + 1; }  // 201
};

// A log-mel spectrogram, row-major [n_mels, n_frames] (mel-major, matching the
// Python golden layout).
struct LogMel {
    std::vector<float> data;  // size == n_mels * n_frames
    int n_mels = 0;
    int n_frames = 0;
    size_t size() const { return data.size(); }
};

class WhisperDSP {
public:
    // Loads the filterbank from `mel_filters_path` (float32, [n_freqs, n_mels],
    // row-major). Throws std::runtime_error on missing file / size mismatch.
    explicit WhisperDSP(const DspConfig& cfg, const std::string& mel_filters_path);

    // Number of output time frames for `n_samples` input PCM samples, i.e.
    // torch.stft(center=True) frame count minus the dropped last frame.
    int predict_num_frames(size_t n_samples) const;

    // Full OFFLINE pipeline on one mono clip (16 kHz float PCM, nominally in
    // [-1, 1]) — includes the global (max-8) dynamic-range compression. This is
    // the PyTorch-verified path; do not change its numerics.
    LogMel process(const std::vector<float>& pcm) const;

    // STREAMING per-frame core, for real-time capture. Takes exactly n_fft raw
    // (un-windowed) samples and returns n_mels RAW log10-mel values. Shares the
    // exact inner math of process() (Hann -> rFFT -> power -> mel -> log10), so
    // it preserves parity; it OMITS only the global (max-8) compression, which
    // depends on the whole clip and therefore cannot be computed per frame. The
    // caller (viz layer) applies a rolling normalisation instead. Throws if
    // frame.size() != n_fft.
    std::vector<float> compute_log_mel_column(const std::vector<float>& frame) const;

    const DspConfig& config() const { return cfg_; }

private:
    // The one copy of the per-frame core math (raw log10-mel, [n_mels] doubles).
    void log_mel_frame_(const float* samples, double* mel_out) const;

    DspConfig cfg_;
    std::vector<float> mel_filters_;  // [n_freqs * n_mels], row-major (freq-major)
    std::vector<float> hann_;         // [n_fft], periodic Hann
};

}  // namespace whisper
