// -----------------------------------------------------------------------------
// F5MelExtractor implementation -- see f5_mel_extractor.hpp for the parity
// contract, which is the part that matters. Everything here is a transcription
// of torchaudio's own construction, and the comments name the torchaudio symbol
// each block reproduces so the two can be diffed by a reader who has the Python
// open next to this file.
// -----------------------------------------------------------------------------
#include "f5_mel_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace blackwell::tts {
namespace {

constexpr double kPi = 3.14159265358979323846;

// HTK mel scale -- torchaudio's _hz_to_mel/_mel_to_hz with mel_scale="htk",
// which is the default F5 does not override. NOT the Slaney scale librosa
// defaults to (piecewise: linear below 1 kHz, log above). The two disagree by
// enough to move every band edge.
inline double hz_to_mel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
inline double mel_to_hz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

// numpy 'reflect' / torch pad_mode='reflect': mirror WITHOUT repeating the edge
// sample. Maps an index in the padded signal back into [0, n).
// (Same helper as audio_sandbox/src/whisper_dsp.cpp -- duplicated rather than
// shared because that file is a sandbox target this leaf must not depend on.)
inline std::size_t reflect_index(long long idx, long long n) {
    if (n == 1) return 0;
    const long long period = 2 * (n - 1);
    long long m = idx % period;
    if (m < 0) m += period;
    return static_cast<std::size_t>(m < n ? m : period - m);
}

bool is_power_of_two(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace

F5MelExtractor::F5MelExtractor(const F5MelConfig& cfg) : cfg_(cfg) {
    if (cfg_.n_fft <= 0 || cfg_.hop_length <= 0 || cfg_.n_mels <= 0 ||
        cfg_.win_length <= 0 || cfg_.sample_rate <= 0) {
        throw std::invalid_argument("F5MelExtractor: geometry must be positive");
    }
    if (!is_power_of_two(cfg_.n_fft)) {
        // The radix-2 transform below is the reason. F5 is 1024 and this is not
        // a knob, so refusing is better than silently needing a mixed-radix FFT.
        throw std::invalid_argument("F5MelExtractor: n_fft must be a power of two (got " +
                                    std::to_string(cfg_.n_fft) + ")");
    }
    if (cfg_.win_length > cfg_.n_fft) {
        throw std::invalid_argument("F5MelExtractor: win_length must not exceed n_fft");
    }
    if (cfg_.f_max > cfg_.sample_rate / 2.0 || cfg_.f_min < 0.0 || cfg_.f_min >= cfg_.f_max) {
        throw std::invalid_argument("F5MelExtractor: require 0 <= f_min < f_max <= Nyquist");
    }

    const int n_fft   = cfg_.n_fft;
    const int n_freqs = cfg_.n_freqs();
    const int n_mels  = cfg_.n_mels;

    // --- periodic Hann ------------------------------------------------------
    // torch.hann_window(N) is PERIODIC by default: 0.5 - 0.5*cos(2*pi*n/N).
    // The symmetric form divides by (N-1) instead and is subtly wrong for STFT.
    hann_.resize(static_cast<std::size_t>(cfg_.win_length));
    for (int n = 0; n < cfg_.win_length; ++n) {
        hann_[static_cast<std::size_t>(n)] =
            0.5 - 0.5 * std::cos(2.0 * kPi * n / static_cast<double>(cfg_.win_length));
    }

    // --- FFT tables ---------------------------------------------------------
    bitrev_.resize(static_cast<std::size_t>(n_fft));
    int log2n = 0;
    while ((1 << log2n) < n_fft) ++log2n;
    for (int i = 0; i < n_fft; ++i) {
        int r = 0;
        for (int b = 0; b < log2n; ++b) {
            if (i & (1 << b)) r |= 1 << (log2n - 1 - b);
        }
        bitrev_[static_cast<std::size_t>(i)] = r;
    }
    tw_re_.resize(static_cast<std::size_t>(n_fft / 2));
    tw_im_.resize(static_cast<std::size_t>(n_fft / 2));
    for (int k = 0; k < n_fft / 2; ++k) {
        const double a = -2.0 * kPi * k / static_cast<double>(n_fft);
        tw_re_[static_cast<std::size_t>(k)] = std::cos(a);
        tw_im_[static_cast<std::size_t>(k)] = std::sin(a);
    }

    // --- mel filterbank -----------------------------------------------------
    // torchaudio.functional.melscale_fbanks + _create_triangular_filterbank,
    // transcribed. all_freqs = linspace(0, sample_rate//2, n_freqs); the mel
    // axis is linspace over n_mels+2 points so each filter gets a left edge,
    // a peak and a right edge.
    std::vector<double> all_freqs(static_cast<std::size_t>(n_freqs));
    const double nyquist = static_cast<double>(cfg_.sample_rate / 2);
    for (int i = 0; i < n_freqs; ++i) {
        all_freqs[static_cast<std::size_t>(i)] =
            nyquist * i / static_cast<double>(n_freqs - 1);
    }

    const double m_min = hz_to_mel(cfg_.f_min);
    const double m_max = hz_to_mel(cfg_.f_max);
    std::vector<double> f_pts(static_cast<std::size_t>(n_mels) + 2);
    for (int j = 0; j < n_mels + 2; ++j) {
        const double m = m_min + (m_max - m_min) * j / static_cast<double>(n_mels + 1);
        f_pts[static_cast<std::size_t>(j)] = mel_to_hz(m);
    }

    band_lo_.assign(static_cast<std::size_t>(n_mels), 0);
    band_hi_.assign(static_cast<std::size_t>(n_mels), 0);
    band_off_.assign(static_cast<std::size_t>(n_mels) + 1, 0);
    weights_.clear();
    weights_.reserve(static_cast<std::size_t>(n_freqs) * 2);

    // Dense row scratch, then compressed. Building the full row first and
    // slicing it afterwards is deliberate: it makes the compression obviously
    // equivalent to the dense filterbank rather than relying on the support
    // being contiguous while walking it.
    std::vector<double> row(static_cast<std::size_t>(n_freqs));
    for (int m = 0; m < n_mels; ++m) {
        const double f_left  = f_pts[static_cast<std::size_t>(m)];
        const double f_ctr   = f_pts[static_cast<std::size_t>(m) + 1];
        const double f_right = f_pts[static_cast<std::size_t>(m) + 2];
        // torchaudio's f_diff[m] and f_diff[m+1].
        const double d_left  = f_ctr - f_left;
        const double d_right = f_right - f_ctr;

        int lo = n_freqs, hi = 0;
        for (int i = 0; i < n_freqs; ++i) {
            const double f = all_freqs[static_cast<std::size_t>(i)];
            // torchaudio: down_slopes = (f_pts[m] - f)/-f_diff[m] = (f - f_left)/d_left
            //             up_slopes   = (f_pts[m+2] - f)/f_diff[m+1] = (f_right - f)/d_right
            //             fb = max(0, min(down, up))
            const double down = (d_left  > 0.0) ? (f - f_left)  / d_left  : 0.0;
            const double up   = (d_right > 0.0) ? (f_right - f) / d_right : 0.0;
            const double w = std::max(0.0, std::min(down, up));
            row[static_cast<std::size_t>(i)] = w;
            if (w > 0.0) {
                lo = std::min(lo, i);
                hi = std::max(hi, i + 1);
            }
        }
        // An empty filter is legitimate, not a bug: with 100 bands over a 12 kHz
        // HTK axis the narrowest low-frequency triangles can fall entirely
        // between two 23.4 Hz bins. torch produces an all-zero row there too,
        // and the mel value then floors to log(1e-5).
        if (hi <= lo) { lo = 0; hi = 0; }

        band_off_[static_cast<std::size_t>(m)] = static_cast<int>(weights_.size());
        band_lo_[static_cast<std::size_t>(m)] = lo;
        band_hi_[static_cast<std::size_t>(m)] = hi;
        for (int i = lo; i < hi; ++i) {
            weights_.push_back(static_cast<float>(row[static_cast<std::size_t>(i)]));
        }
    }
    band_off_[static_cast<std::size_t>(n_mels)] = static_cast<int>(weights_.size());

    // --- scratch ------------------------------------------------------------
    re_.assign(static_cast<std::size_t>(n_fft), 0.0);
    im_.assign(static_cast<std::size_t>(n_fft), 0.0);
    mag_.assign(static_cast<std::size_t>(n_freqs), 0.0);
}

// Iterative radix-2 decimation-in-time FFT, forward sign convention
// (exp(-2*pi*i*k*n/N)) so bin k matches numpy.fft/torch.fft exactly.
//
// A real-input transform could do this in a half-length complex FFT for ~2x.
// Deliberately not done: at n_fft=1024 and one call per REFERENCE CLIP (not per
// utterance, and certainly not per solver step) the whole extraction of a 10 s
// clip is a few milliseconds, and the packing/unpacking of the real-FFT trick is
// where sign errors hide. Cost that buys nothing is not worth the risk here.
void F5MelExtractor::Fft() {
    const int n = cfg_.n_fft;

    for (int i = 0; i < n; ++i) {
        const int j = bitrev_[static_cast<std::size_t>(i)];
        if (i < j) {
            std::swap(re_[static_cast<std::size_t>(i)], re_[static_cast<std::size_t>(j)]);
            std::swap(im_[static_cast<std::size_t>(i)], im_[static_cast<std::size_t>(j)]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;               // stride into the twiddle table
        for (int base = 0; base < n; base += len) {
            for (int k = 0; k < half; ++k) {
                const std::size_t tw = static_cast<std::size_t>(k * step);
                const double wr = tw_re_[tw];
                const double wi = tw_im_[tw];
                const std::size_t a = static_cast<std::size_t>(base + k);
                const std::size_t b = a + static_cast<std::size_t>(half);
                const double xr = re_[b] * wr - im_[b] * wi;
                const double xi = re_[b] * wi + im_[b] * wr;
                re_[b] = re_[a] - xr;
                im_[b] = im_[a] - xi;
                re_[a] += xr;
                im_[a] += xi;
            }
        }
    }
}

void F5MelExtractor::MelFrame(float* out) {
    const int n_freqs = cfg_.n_freqs();
    const int n_mels  = cfg_.n_mels;

    Fft();

    // power=1 -> MAGNITUDE. See the header: the Whisper front end squares here
    // and that difference is invisible until the cloned voice is wrong.
    for (int f = 0; f < n_freqs; ++f) {
        const double r = re_[static_cast<std::size_t>(f)];
        const double i = im_[static_cast<std::size_t>(f)];
        mag_[static_cast<std::size_t>(f)] = std::sqrt(r * r + i * i);
    }

    for (int m = 0; m < n_mels; ++m) {
        const int lo  = band_lo_[static_cast<std::size_t>(m)];
        const int hi  = band_hi_[static_cast<std::size_t>(m)];
        const int off = band_off_[static_cast<std::size_t>(m)];
        // double accumulator: the filterbank matmul is done in float32 by
        // torch, but accumulating in double makes our summation ORDER irrelevant
        // to the comparison instead of a source of drift.
        double acc = 0.0;
        for (int f = lo; f < hi; ++f) {
            acc += static_cast<double>(weights_[static_cast<std::size_t>(off + f - lo)]) *
                   mag_[static_cast<std::size_t>(f)];
        }
        // clamp(min=1e-5).log() -- natural log, not log10.
        out[m] = static_cast<float>(std::log(std::max(acc, cfg_.log_floor)));
    }
}

std::size_t F5MelExtractor::FrameCount(std::size_t n_samples) const {
    // torch.stft(center=True): padded length is n + 2*(n_fft/2) = n + n_fft, so
    // 1 + (padded - n_fft)/hop = 1 + n/hop. torchaudio keeps the last frame.
    return 1 + n_samples / static_cast<std::size_t>(cfg_.hop_length);
}

void F5MelExtractor::ComputeInto(const float* pcm, std::size_t n_samples,
                                 std::vector<float>& out) {
    const int n_fft  = cfg_.n_fft;
    const int hop    = cfg_.hop_length;
    const int n_mels = cfg_.n_mels;
    const int pad    = n_fft / 2;

    if (pcm == nullptr) {
        throw std::invalid_argument("F5MelExtractor: null pcm");
    }
    // torch's reflect pad requires pad <= n-1; below that the mirror would run
    // off the far end of the signal. Fabricating zeros instead would invent a
    // spectrum for audio that does not exist, so refuse.
    if (n_samples < static_cast<std::size_t>(pad) + 1) {
        throw std::invalid_argument(
            "F5MelExtractor: clip has " + std::to_string(n_samples) +
            " samples but reflect padding needs at least " + std::to_string(pad + 1) +
            " (n_fft/2 + 1). A reference clip this short (<" +
            std::to_string((pad + 1) * 1000 / cfg_.sample_rate) + " ms) cannot be used.");
    }

    const std::size_t frames = FrameCount(n_samples);
    out.resize(frames * static_cast<std::size_t>(n_mels));

    // Window offset for the win_length < n_fft case: torch centres a short
    // window inside the n_fft frame and zero-pads the rest. Equal here (both
    // 1024), so this is 0 -- kept so a test may shrink the window honestly.
    const int woff = (n_fft - cfg_.win_length) / 2;
    const long long n = static_cast<long long>(n_samples);

    for (std::size_t t = 0; t < frames; ++t) {
        // Frame t is CENTRED on sample t*hop, hence the -pad.
        const long long base = static_cast<long long>(t) * hop - pad;

        // Rebuild the complex input in place: zero everywhere, windowed signal
        // in the window's span. No allocation -- re_/im_ are ctor-sized.
        std::fill(re_.begin(), re_.end(), 0.0);
        std::fill(im_.begin(), im_.end(), 0.0);
        for (int k = 0; k < cfg_.win_length; ++k) {
            const std::size_t src = reflect_index(base + woff + k, n);
            re_[static_cast<std::size_t>(woff + k)] =
                static_cast<double>(pcm[src]) * hann_[static_cast<std::size_t>(k)];
        }

        MelFrame(out.data() + t * static_cast<std::size_t>(n_mels));
    }
}

std::vector<float> F5MelExtractor::Compute(const std::vector<float>& pcm) {
    std::vector<float> out;
    ComputeInto(pcm.data(), pcm.size(), out);
    return out;
}

}  // namespace blackwell::tts
