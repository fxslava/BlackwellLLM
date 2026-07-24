// -----------------------------------------------------------------------------
// whisper_dsp.cpp — see whisper_dsp.h for the parity contract.
//
// The per-frame core (Hann -> rFFT -> power -> mel -> log10(clamp 1e-10)) lives
// in log_mel_frame_(); BOTH the verified offline process() and the streaming
// compute_log_mel_column() call it, so there is exactly one copy of the core
// math. process() adds the offline-only, non-streamable global (max-8) dynamic-
// range compression on top; the streaming path deliberately does NOT (see header).
// -----------------------------------------------------------------------------
#include "whisper_dsp.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "pocketfft_hdronly.h"

namespace whisper {
namespace {

constexpr double kPi = 3.14159265358979323846;

// numpy 'reflect' / torch pad_mode='reflect': mirror WITHOUT repeating the edge
// sample. Maps an index into the padded signal back to the source range [0, n).
// Assumes n >= 2 (true for any real clip; the DSP is meaningless otherwise).
inline size_t reflect_index(long idx, long n) {
    if (n == 1) return 0;
    const long period = 2 * (n - 1);
    long m = idx % period;
    if (m < 0) m += period;
    return static_cast<size_t>(m < n ? m : period - m);
}

std::vector<float> load_binary_f32(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open: " + path);
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || static_cast<size_t>(bytes) != expected_count * sizeof(float)) {
        throw std::runtime_error(
            "size mismatch for " + path + ": got " + std::to_string(bytes) +
            " bytes, expected " + std::to_string(expected_count * sizeof(float)));
    }
    f.seekg(0);
    std::vector<float> out(expected_count);
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    if (!f) throw std::runtime_error("short read: " + path);
    return out;
}

}  // namespace

WhisperDSP::WhisperDSP(const DspConfig& cfg, const std::string& mel_filters_path)
    : cfg_(cfg) {
    const size_t n_freqs = static_cast<size_t>(cfg_.n_freqs());
    const size_t n_mels = static_cast<size_t>(cfg_.n_mels);
    // [n_freqs, n_mels], row-major (freq-major) — matches the Python dump.
    mel_filters_ = load_binary_f32(mel_filters_path, n_freqs * n_mels);

    // Periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n / N), n in [0, N). Matches
    // torch.hann_window(N) default (periodic=True), NOT the symmetric /(N-1) form.
    hann_.resize(static_cast<size_t>(cfg_.n_fft));
    for (int n = 0; n < cfg_.n_fft; ++n) {
        hann_[static_cast<size_t>(n)] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * n / static_cast<double>(cfg_.n_fft)));
    }
}

// The verified inner path, for ONE window of exactly n_fft raw samples. Writes
// n_mels raw log10-mel values (double). No global normalisation here — that step
// is not per-frame and so cannot live in a streamable core.
void WhisperDSP::log_mel_frame_(const float* samples, double* mel_out) const {
    const int n_fft = cfg_.n_fft;
    const int n_freqs = cfg_.n_freqs();
    const int n_mels = cfg_.n_mels;

    // Reused across calls on the (single) DSP thread to avoid per-frame allocs.
    thread_local std::vector<double> frame;
    thread_local std::vector<std::complex<double>> spec;
    frame.resize(static_cast<size_t>(n_fft));
    spec.resize(static_cast<size_t>(n_freqs));

    for (int k = 0; k < n_fft; ++k) {
        frame[static_cast<size_t>(k)] =
            static_cast<double>(samples[k]) * static_cast<double>(hann_[static_cast<size_t>(k)]);
    }

    const pocketfft::shape_t shape{static_cast<size_t>(n_fft)};
    const pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(sizeof(double))};
    const pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(sizeof(std::complex<double>))};
    const pocketfft::shape_t axes{0};
    pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
                   frame.data(), spec.data(), 1.0);

    for (int m = 0; m < n_mels; ++m) mel_out[m] = 0.0;
    // mel[m] = sum_f mel_filters[f, m] * |stft[f]|^2, freq-major (f ascending) —
    // identical summation order to the batched offline matmul.
    for (int f = 0; f < n_freqs; ++f) {
        const std::complex<double>& c = spec[static_cast<size_t>(f)];
        const double p = c.real() * c.real() + c.imag() * c.imag();
        const float* mf_row = &mel_filters_[static_cast<size_t>(f) * n_mels];
        for (int m = 0; m < n_mels; ++m) {
            const double w = static_cast<double>(mf_row[m]);
            if (w != 0.0) mel_out[m] += w * p;  // filterbank is triangular/sparse
        }
    }
    for (int m = 0; m < n_mels; ++m) mel_out[m] = std::log10(std::max(mel_out[m], 1e-10));
}

int WhisperDSP::predict_num_frames(size_t n_samples) const {
    // center=True reflect-pads by n_fft/2 each side -> padded length + n_fft.
    // torch.stft frame count = 1 + (padded_len - n_fft) / hop = 1 + n_samples/hop;
    // transformers drops the last frame (stft[..., :-1]).
    const long padded_len = static_cast<long>(n_samples) + cfg_.n_fft;
    const long full = 1 + (padded_len - cfg_.n_fft) / cfg_.hop_length;
    return static_cast<int>(std::max<long>(full - 1, 0));
}

std::vector<float> WhisperDSP::compute_log_mel_column(const std::vector<float>& frame) const {
    if (static_cast<int>(frame.size()) != cfg_.n_fft) {
        throw std::runtime_error("compute_log_mel_column: frame has " +
                                 std::to_string(frame.size()) + " samples, expected " +
                                 std::to_string(cfg_.n_fft));
    }
    const int n_mels = cfg_.n_mels;
    std::vector<double> col(static_cast<size_t>(n_mels));
    log_mel_frame_(frame.data(), col.data());
    std::vector<float> out(static_cast<size_t>(n_mels));
    for (int m = 0; m < n_mels; ++m) out[static_cast<size_t>(m)] = static_cast<float>(col[m]);
    return out;
}

LogMel WhisperDSP::process(const std::vector<float>& pcm) const {
    const int n_fft = cfg_.n_fft;
    const int hop = cfg_.hop_length;
    const int n_mels = cfg_.n_mels;
    const long n = static_cast<long>(pcm.size());
    const int n_frames = predict_num_frames(pcm.size());

    LogMel out;
    out.n_mels = n_mels;
    out.n_frames = n_frames;
    if (n_frames <= 0) return out;
    out.data.assign(static_cast<size_t>(n_mels) * n_frames, 0.0f);

    // Raw log10-mel, mel-major [n_mels, n_frames]; then the offline-only global
    // (max-8) compression. The per-frame values come from the SAME core the
    // streaming path uses, so the verified parity is preserved by construction.
    std::vector<double> logmel(static_cast<size_t>(n_mels) * n_frames);
    std::vector<float> frame_samples(static_cast<size_t>(n_fft));
    std::vector<double> col(static_cast<size_t>(n_mels));

    const long pad = n_fft / 2;  // center padding offset (offline path only)
    double log_max = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < n_frames; ++t) {
        const long base = static_cast<long>(t) * hop - pad;  // index into UNpadded pcm
        for (int k = 0; k < n_fft; ++k)
            frame_samples[static_cast<size_t>(k)] = pcm[reflect_index(base + k, n)];
        log_mel_frame_(frame_samples.data(), col.data());
        for (int m = 0; m < n_mels; ++m) {
            const double v = col[static_cast<size_t>(m)];
            logmel[static_cast<size_t>(m) * n_frames + t] = v;
            if (v > log_max) log_max = v;
        }
    }

    const double floor = log_max - 8.0;
    for (size_t i = 0; i < logmel.size(); ++i)
        out.data[i] = static_cast<float>((std::max(logmel[i], floor) + 4.0) / 4.0);
    return out;
}

}  // namespace whisper
