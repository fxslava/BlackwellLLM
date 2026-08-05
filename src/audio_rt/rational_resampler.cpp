// -----------------------------------------------------------------------------
// RationalResampler implementation — see the header for why the AEC needs it and
// what the group delay costs.
//
// THE POLYPHASE IDENTITY, stated once so the loop below is readable. The naive
// form is "insert L-1 zeros, low-pass, keep every M-th sample":
//
//     y[n] = sum_k h[k] * u[n*M - k],   u[j] = x[j/L] if L divides j else 0
//
// u is zero except where L divides the index, so only the taps with
// k = (n*M) mod L (mod L) contribute. Writing p = (n*M) mod L and k = p + L*t:
//
//     y[n] = sum_t h[p + L*t] * x[ floor(n*M/L) - t ]
//
// which is an ordinary T-tap FIR over the INPUT stream, with the coefficient set
// selected by the output phase p. The zeros are never stored and never
// multiplied -- that is the whole point.
// -----------------------------------------------------------------------------
#include "rational_resampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace blackwell::audio_rt {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Largest number of inputs consumed in one internal pass. The history buffer is
// sized to T + this, which is what makes every tap a phase asks for resident.
constexpr std::size_t kMaxPass = 4096;

int gcd_int(int a, int b) noexcept {
    while (b != 0) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

std::size_t round_up_pow2(std::size_t v) noexcept {
    std::size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

double sinc(double x) noexcept {
    if (std::abs(x) < 1e-12) return 1.0;
    return std::sin(kPi * x) / (kPi * x);
}

}  // namespace

RationalResampler::RationalResampler(int in_rate, int out_rate, int taps_per_phase) {
    if (in_rate <= 0 || out_rate <= 0 || taps_per_phase <= 0) {
        throw std::invalid_argument("RationalResampler: rates and taps must be positive");
    }
    const int g = gcd_int(out_rate, in_rate);
    l_ = out_rate / g;      // interpolate by the OUTPUT side of the ratio
    m_ = in_rate / g;
    taps_ = static_cast<std::size_t>(taps_per_phase);

    const std::size_t k = static_cast<std::size_t>(l_) * taps_;
    h_.assign(k, 0.0f);

    // Windowed sinc at the intermediate (L * in_rate) rate. The cutoff is the
    // LOWER of the two Nyquist limits -- protecting against imaging on the way up
    // and aliasing on the way down is the same constraint, and taking the min is
    // what makes one filter serve both.
    const double cutoff = 0.5 / static_cast<double>(std::max(l_, m_));
    const double centre = (static_cast<double>(k) - 1.0) * 0.5;
    for (std::size_t i = 0; i < k; ++i) {
        const double t = static_cast<double>(i) - centre;
        // Blackman: -74 dB sidelobes, which is the term that actually sets the
        // stopband floor here (the sinc truncation alone would be ~-21 dB).
        const double wn = static_cast<double>(i) / static_cast<double>(k - 1);
        const double w = 0.42 - 0.5 * std::cos(2.0 * kPi * wn) + 0.08 * std::cos(4.0 * kPi * wn);
        h_[i] = static_cast<float>(2.0 * cutoff * sinc(2.0 * cutoff * t) * w *
                                   static_cast<double>(l_));
    }

    // Normalise EACH polyphase branch to unit DC gain. Without this the branches
    // have slightly different sums and the output carries a periodic amplitude
    // ripple at the phase rate -- which on a reference signal is a tone the
    // canceller cannot explain and therefore cannot cancel.
    for (int p = 0; p < l_; ++p) {
        double sum = 0.0;
        for (std::size_t t = 0; t < taps_; ++t) {
            sum += static_cast<double>(h_[static_cast<std::size_t>(p) + static_cast<std::size_t>(l_) * t]);
        }
        if (std::abs(sum) < 1e-12) continue;
        for (std::size_t t = 0; t < taps_; ++t) {
            float& c = h_[static_cast<std::size_t>(p) + static_cast<std::size_t>(l_) * t];
            c = static_cast<float>(static_cast<double>(c) / sum);
        }
    }

    const std::size_t cap = round_up_pow2(taps_ + kMaxPass);
    hist_.assign(cap, 0.0f);
    mask_ = cap - 1;
}

void RationalResampler::Reset() noexcept {
    std::fill(hist_.begin(), hist_.end(), 0.0f);
    total_in_ = 0;
    out_n_ = 0;
}

double RationalResampler::group_delay_in_samples() const noexcept {
    const double k = static_cast<double>(l_) * static_cast<double>(taps_);
    return (k - 1.0) / (2.0 * static_cast<double>(l_));
}

std::size_t RationalResampler::OutputCountFor(std::size_t in_count) const noexcept {
    // Output n exists once floor(n*M/L) < total inputs, i.e. n < total*L/M. The
    // count is therefore ceil(total*L/M) minus the outputs already emitted.
    const std::uint64_t total = total_in_ + static_cast<std::uint64_t>(in_count);
    const std::uint64_t lm = static_cast<std::uint64_t>(l_);
    const std::uint64_t mm = static_cast<std::uint64_t>(m_);
    const std::uint64_t reachable = (total * lm + mm - 1) / mm;   // ceil
    if (reachable <= out_n_) return 0;
    return static_cast<std::size_t>(reachable - out_n_);
}

std::size_t RationalResampler::Process(const float* in, std::size_t in_count,
                                       float* out, std::size_t out_capacity) noexcept {
    if (out == nullptr || out_capacity == 0) return 0;
    if (in == nullptr) in_count = 0;

    std::size_t written = 0;
    std::size_t consumed = 0;
    const std::uint64_t lm = static_cast<std::uint64_t>(l_);
    const std::uint64_t mm = static_cast<std::uint64_t>(m_);

    // EVERY input is consumed even if the destination fills, and the phase
    // counter advances over the outputs that did not fit. Stopping early would
    // leave the resampler holding samples the caller believes it delivered, and
    // the reference stream would silently slip a block behind the microphone --
    // which the canceller cannot distinguish from a room that grew longer.
    do {
        const std::size_t pass = std::min(kMaxPass, in_count - consumed);
        for (std::size_t i = 0; i < pass; ++i) {
            hist_[static_cast<std::size_t>(total_in_ + i) & mask_] = in[consumed + i];
        }
        total_in_ += pass;
        consumed += pass;

        for (;;) {
            const std::uint64_t nm = out_n_ * mm;
            const std::uint64_t q = nm / lm;               // newest input this output needs
            if (q >= total_in_) break;                     // not enough input yet
            const std::size_t p = static_cast<std::size_t>(nm - q * lm);   // phase = (n*M) mod L

            if (written < out_capacity) {
                double acc = 0.0;
                for (std::size_t t = 0; t < taps_; ++t) {
                    if (static_cast<std::uint64_t>(t) > q) break;   // start-up: nothing before 0
                    const std::size_t idx = static_cast<std::size_t>(q - t) & mask_;
                    acc += static_cast<double>(h_[p + static_cast<std::size_t>(l_) * t]) *
                           static_cast<double>(hist_[idx]);
                }
                out[written++] = static_cast<float>(acc);
            }
            ++out_n_;
        }
    } while (consumed < in_count);

    return written;
}

}  // namespace blackwell::audio_rt
