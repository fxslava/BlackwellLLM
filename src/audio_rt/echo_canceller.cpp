// -----------------------------------------------------------------------------
// BlockFdafEchoCanceller — see echo_canceller.hpp for the design argument (two
// stages, why the step size IS the double-talk defence, and what the latency
// buys). This file is the arithmetic.
//
// THE FFT IS OURS ON PURPOSE. src/audio_rt/CMakeLists.txt declares this target
// "DEPENDENCY-FREE BY CONSTRUCTION -- STL only", because it is linked both from
// the audio sandbox (which must stay CUDA-free standalone) and from the
// engine-linked apps. Reaching for pocketfft via blackwell::sandbox_headers
// would put a fetched third-party include path on that link line for a
// 256-point radix-2 transform. The same trade is already made, for the same
// reason, in src/tts/f5_mel_extractor.cpp -- and it is written down in that
// target's CMakeLists so nobody "fixes" it later.
//
// OVERLAP-SAVE, PARTITIONED. The filter tail is cut into P partitions of B taps.
// Each block: one FFT of the 2B far-end window, P complex multiply-accumulates
// against the stored weight spectra, one inverse to get the predicted echo. The
// gradient constraint (projecting each weight partition back onto its B-tap
// causal support) is applied to ONE partition per block, round robin: applying
// it to all P would cost 2P transforms per block for a correction that is
// second-order, and staggering it is what every production MDF does.
// -----------------------------------------------------------------------------
#include "echo_canceller.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace blackwell::audio_rt {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Largest span processed in one internal pass; the FIFOs are sized from it.
constexpr std::size_t kMaxChunk = 4096;

// Mean-square floor below which the far end counts as silent. -80 dBFS: quiet
// enough that dither and denormals do not look like playback, loud enough that
// a real utterance never falls under it.
constexpr double kFarActiveMeanSq = 1e-8;

// Denominator guard for the NLMS normalisation. The far-end power it sits next
// to is an N-point spectrum magnitude summed over P partitions, so it is many
// orders of magnitude larger whenever adaptation is enabled at all.
constexpr float kPowerEps = 1e-6f;

bool is_pow2(std::size_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }

// Iterative radix-2 Cooley-Tukey, in place, with precomputed twiddles and a
// precomputed bit-reversal permutation. Real input is handled by zeroing the
// imaginary half -- at N = 256 the 2x waste is ~10 kFLOP per block and buying it
// back would mean maintaining a real-FFT packing convention across five call
// sites, which is a worse trade in a file this delicate.
class Fft {
public:
    explicit Fft(std::size_t n) : n_(n) {
        std::size_t bits = 0;
        while ((static_cast<std::size_t>(1) << bits) < n) ++bits;
        rev_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t r = 0;
            for (std::size_t b = 0; b < bits; ++b) {
                if ((i >> b) & 1u) r |= static_cast<std::size_t>(1) << (bits - 1 - b);
            }
            rev_[i] = r;
        }
        cos_.resize(n / 2);
        sin_.resize(n / 2);
        for (std::size_t i = 0; i < n / 2; ++i) {
            const double a = -2.0 * kPi * static_cast<double>(i) / static_cast<double>(n);
            cos_[i] = static_cast<float>(std::cos(a));
            sin_[i] = static_cast<float>(std::sin(a));
        }
    }

    void forward(float* re, float* im) const noexcept { run(re, im, false); }

    // Includes the 1/N so that inverse(forward(x)) == x.
    void inverse(float* re, float* im) const noexcept { run(re, im, true); }

private:
    void run(float* re, float* im, bool inv) const noexcept {
        for (std::size_t i = 0; i < n_; ++i) {
            const std::size_t j = rev_[i];
            if (j > i) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
        for (std::size_t len = 2; len <= n_; len <<= 1) {
            const std::size_t half = len >> 1;
            const std::size_t step = n_ / len;
            for (std::size_t base = 0; base < n_; base += len) {
                for (std::size_t j = 0; j < half; ++j) {
                    const std::size_t t = j * step;
                    const float wr = cos_[t];
                    const float wi = inv ? -sin_[t] : sin_[t];
                    const std::size_t a = base + j;
                    const std::size_t b = a + half;
                    const float xr = re[b] * wr - im[b] * wi;
                    const float xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
        if (inv) {
            const float s = 1.0f / static_cast<float>(n_);
            for (std::size_t i = 0; i < n_; ++i) {
                re[i] *= s;
                im[i] *= s;
            }
        }
    }

    std::size_t n_;
    std::vector<std::size_t> rev_;
    std::vector<float> cos_, sin_;
};

// Fixed-capacity sample FIFO. Exists so Process() can accept any block length
// while the filter runs on its own fixed frame, WITHOUT allocating on the
// capture-drain thread.
class Fifo {
public:
    void init(std::size_t capacity) {
        buf_.assign(capacity, 0.0f);
        head_ = 0;
        size_ = 0;
    }
    std::size_t size() const noexcept { return size_; }
    void clear() noexcept {
        std::fill(buf_.begin(), buf_.end(), 0.0f);
        head_ = 0;
        size_ = 0;
    }
    // Callers below never exceed capacity; a would-be overflow drops the excess
    // rather than corrupting the ring.
    void push(const float* src, std::size_t n) noexcept {
        const std::size_t cap = buf_.size();
        if (n > cap - size_) n = cap - size_;
        std::size_t tail = (head_ + size_) % cap;
        for (std::size_t i = 0; i < n; ++i) {
            buf_[tail] = src[i];
            tail = (tail + 1 == cap) ? 0 : tail + 1;
        }
        size_ += n;
    }
    void push_zeros(std::size_t n) noexcept {
        const std::size_t cap = buf_.size();
        if (n > cap - size_) n = cap - size_;
        std::size_t tail = (head_ + size_) % cap;
        for (std::size_t i = 0; i < n; ++i) {
            buf_[tail] = 0.0f;
            tail = (tail + 1 == cap) ? 0 : tail + 1;
        }
        size_ += n;
    }
    void pop(float* dst, std::size_t n) noexcept {
        const std::size_t cap = buf_.size();
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = (i < size_) ? buf_[(head_ + i) % cap] : 0.0f;
        }
        const std::size_t taken = std::min(n, size_);
        head_ = (head_ + taken) % cap;
        size_ -= taken;
    }

private:
    std::vector<float> buf_;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
};

}  // namespace

struct BlockFdafEchoCanceller::Impl {
    explicit Impl(const EchoCancellerConfig& c)
        : cfg(c),
          b(c.block_samples),
          n(c.block_samples * 2),
          p(std::max<std::size_t>(1, (c.filter_tail_samples + c.block_samples - 1) / c.block_samples)),
          fft(c.block_samples * 2) {
        far_win.assign(n, 0.0f);
        x_re.assign(p * n, 0.0f);
        x_im.assign(p * n, 0.0f);
        w_re.assign(p * n, 0.0f);
        w_im.assign(p * n, 0.0f);
        pow_bin.assign(n, 0.0f);
        s_re.assign(n, 0.0f);
        s_im.assign(n, 0.0f);
        t_re.assign(n, 0.0f);
        t_im.assign(n, 0.0f);
        e_re.assign(n, 0.0f);
        e_im.assign(n, 0.0f);
        gain.assign(n, 1.0f);
        yhat.assign(b, 0.0f);
        err.assign(b, 0.0f);
        prev_e.assign(b, 0.0f);
        prev_y.assign(b, 0.0f);
        ola.assign(b, 0.0f);
        d_blk.assign(b, 0.0f);
        x_blk.assign(b, 0.0f);
        o_blk.assign(b, 0.0f);

        // sqrt of a PERIODIC Hann. Squared and summed at 50 % overlap it is
        // exactly 1, so analysis * synthesis reconstructs the input unchanged
        // whenever the gain is unity -- the property that lets this stage sit in
        // the ASR path permanently.
        win.assign(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const double h = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                                                  static_cast<double>(n));
            win[i] = static_cast<float>(std::sqrt(h));
        }

        const std::size_t cap = kMaxChunk + 2 * b + 8;
        near_fifo.init(cap);
        far_fifo.init(cap);
        out_fifo.init(cap);
        // THE LATENCY, made explicit. One block of framing debt is pre-paid here
        // so Process() can always return as many samples as it was given; the
        // second block of delay comes from the overlap-add frame itself.
        out_fifo.push_zeros(b);
    }

    void ProcessBlock(const float* d, const float* x, float* out) noexcept;
    void Adapt(double ey, double ee) noexcept;
    void Suppress(float* out) noexcept;

    EchoCancellerConfig cfg;
    std::size_t b;      // block (adaptation frame hop)
    std::size_t n;      // FFT size = 2b
    std::size_t p;      // partitions
    Fft fft;

    std::vector<float> far_win;              // [previous block, current block]
    std::vector<float> x_re, x_im;           // p * n far-end spectra (newest at head)
    std::vector<float> w_re, w_im;           // p * n weight spectra
    std::vector<float> pow_bin;              // per-bin far-end power, summed over p
    std::vector<float> s_re, s_im;           // scratch: filter output / gradient
    std::vector<float> t_re, t_im;           // scratch: constraint / echo spectrum
    std::vector<float> e_re, e_im;           // suppressor: error spectrum
    std::vector<float> gain;                 // smoothed per-bin suppressor gain
    std::vector<float> win;                  // sqrt-Hann, analysis and synthesis
    std::vector<float> yhat, err;            // predicted echo, linear residual
    std::vector<float> prev_e, prev_y;       // previous block of each, for the WOLA frame
    std::vector<float> ola;                  // overlap-add tail
    std::vector<float> d_blk, x_blk, o_blk;  // block staging

    Fifo near_fifo, far_fifo, out_fifo;

    std::size_t head = 0;        // newest partition slot
    std::size_t constrain = 0;   // round-robin gradient-constraint cursor
    float leak = 1.0f;           // residual-to-echo ratio; 1 = nothing cancelled yet
    double erle_lin = 1.0;
    // Best ERLE reached since the last reset. Monotone by construction: it is
    // the CONFIDENCE the step-size rule is faded in against, and confidence that
    // fell during double-talk would re-open the filter to the very corruption
    // the rule exists to prevent.
    double peak_erle = 1.0;
    std::uint32_t diverged_blocks = 0;
    std::uint64_t resets = 0;
};

// -----------------------------------------------------------------------------

void BlockFdafEchoCanceller::Impl::ProcessBlock(const float* d, const float* x,
                                                float* out) noexcept {
    // ---- far-end analysis window (overlap-save) ------------------------------
    std::memmove(far_win.data(), far_win.data() + b, b * sizeof(float));
    std::memcpy(far_win.data() + b, x, b * sizeof(float));

    head = (head + 1) % p;
    float* xr = x_re.data() + head * n;
    float* xi = x_im.data() + head * n;
    std::memcpy(xr, far_win.data(), n * sizeof(float));
    std::memset(xi, 0, n * sizeof(float));
    fft.forward(xr, xi);

    // ---- predicted echo: sum over partitions in the frequency domain --------
    std::fill(s_re.begin(), s_re.end(), 0.0f);
    std::fill(s_im.begin(), s_im.end(), 0.0f);
    for (std::size_t q = 0; q < p; ++q) {
        const std::size_t slot = (head + p - q) % p;
        const float* wr = w_re.data() + q * n;
        const float* wi = w_im.data() + q * n;
        const float* cr = x_re.data() + slot * n;
        const float* ci = x_im.data() + slot * n;
        for (std::size_t k = 0; k < n; ++k) {
            s_re[k] += wr[k] * cr[k] - wi[k] * ci[k];
            s_im[k] += wr[k] * ci[k] + wi[k] * cr[k];
        }
    }
    fft.inverse(s_re.data(), s_im.data());
    // Overlap-save: only the second half is the true linear convolution; the
    // first half is the circular wrap-around and is discarded.
    for (std::size_t i = 0; i < b; ++i) yhat[i] = s_re[b + i];
    for (std::size_t i = 0; i < b; ++i) err[i] = d[i] - yhat[i];

    double ex = 0.0, ed = 0.0, ey = 0.0, ee = 0.0;
    for (std::size_t i = 0; i < b; ++i) {
        ex += static_cast<double>(x[i]) * x[i];
        ed += static_cast<double>(d[i]) * d[i];
        ey += static_cast<double>(yhat[i]) * yhat[i];
        ee += static_cast<double>(err[i]) * err[i];
    }
    const bool far_active = ex > kFarActiveMeanSq * static_cast<double>(b);

    // ---- divergence guard ---------------------------------------------------
    // A filter that makes the microphone LOUDER is not converging slowly, it is
    // wrong -- almost always because the reference stopped being aligned with
    // the microphone. Half a second of that and the learned response is worth
    // less than nothing, so it goes.
    if (far_active && ee > 1.5 * ed) {
        ++diverged_blocks;
        if (diverged_blocks > 60) {
            std::fill(w_re.begin(), w_re.end(), 0.0f);
            std::fill(w_im.begin(), w_im.end(), 0.0f);
            leak = 1.0f;
            erle_lin = 1.0;
            peak_erle = 1.0;   // the proof is gone with the weights that earned it
            diverged_blocks = 0;
            ++resets;
        }
    } else {
        diverged_blocks = 0;
    }

    if (far_active) {
        Adapt(ey, ee);
        if (ed > 1e-9 * static_cast<double>(b)) {
            const double inst = std::min(ed / (ee + 1e-20), 1e6);
            erle_lin = 0.98 * erle_lin + 0.02 * inst;
            peak_erle = std::max(peak_erle, erle_lin);
        }
        // The leak estimate is only meaningful when the echo is what the
        // microphone mostly CONTAINS: measured during double-talk it would
        // read the near-end talker as un-cancelled echo and the suppressor
        // would go on to remove the user's voice, which is the exact failure
        // this whole file exists to avoid.
        if (ey > 0.5 * ed) {
            const float inst = static_cast<float>(std::min(ee / (ey + 1e-20), 1.0));
            leak = 0.95f * leak + 0.05f * inst;
            leak = std::clamp(leak, 1e-3f, 1.0f);
        }
    }

    if (cfg.residual_suppression) {
        Suppress(out);
    } else {
        // Still delayed by one block, so latency_samples() is the same answer
        // whichever way this is configured.
        std::memcpy(out, prev_e.data(), b * sizeof(float));
    }

    std::memcpy(prev_e.data(), err.data(), b * sizeof(float));
    std::memcpy(prev_y.data(), yhat.data(), b * sizeof(float));
}

void BlockFdafEchoCanceller::Impl::Adapt(double ey, double ee) noexcept {
    // Per-bin far-end power, summed across the partitions the update touches.
    // Recomputed rather than smoothed: a smoothed denominator lags the onset of
    // speech, and lagging it is exactly when the step is too large.
    std::fill(pow_bin.begin(), pow_bin.end(), 0.0f);
    for (std::size_t q = 0; q < p; ++q) {
        const float* cr = x_re.data() + q * n;
        const float* ci = x_im.data() + q * n;
        for (std::size_t k = 0; k < n; ++k) {
            pow_bin[k] += cr[k] * cr[k] + ci[k] * ci[k];
        }
    }

    // THE DOUBLE-TALK DEFENCE (see the header), plus its bootstrap.
    //
    // ey/(ey+ee) is the right rule ONLY once the filter predicts something: with
    // zero weights ey is zero, the step is zero, and the filter can never leave
    // the state it starts in. So the rule is faded in against `conv`, a
    // confidence that the filter has PROVEN it can cancel -- driven by the best
    // ERLE reached since the last reset, and monotone on purpose. Falling ERLE
    // during double-talk must not be read as "the filter got worse and should
    // adapt harder"; that is the failure mode the residual rule exists to stop.
    // Before any evidence exists there is no evidence of a near-end talker
    // either, so the prior is "this is echo" and the step runs at full rate.
    const double conv = std::clamp((peak_erle - 1.0) / 3.0, 0.0, 1.0);   // 1 at ~6 dB
    const double residual = ey / (ey + ee + 1e-20);
    const double scale = (1.0 - conv) + conv * residual;
    const float mu_eff = cfg.mu * static_cast<float>(scale);
    if (mu_eff <= 0.0f) return;

    // Gradient spectrum: the error zero-padded into the FIRST half, which is the
    // overlap-save dual of taking the second half of the output above.
    std::memset(s_re.data(), 0, b * sizeof(float));
    std::memcpy(s_re.data() + b, err.data(), b * sizeof(float));
    std::memset(s_im.data(), 0, n * sizeof(float));
    fft.forward(s_re.data(), s_im.data());

    for (std::size_t q = 0; q < p; ++q) {
        const std::size_t slot = (head + p - q) % p;
        const float* cr = x_re.data() + slot * n;
        const float* ci = x_im.data() + slot * n;
        float* wr = w_re.data() + q * n;
        float* wi = w_im.data() + q * n;
        for (std::size_t k = 0; k < n; ++k) {
            const float nrm = mu_eff / (pow_bin[k] + kPowerEps);
            // conj(X) * E
            wr[k] += nrm * (cr[k] * s_re[k] + ci[k] * s_im[k]);
            wi[k] += nrm * (cr[k] * s_im[k] - ci[k] * s_re[k]);
        }
    }

    // ---- gradient constraint, one partition per block ------------------------
    // Each partition models B taps. Left unconstrained, the update leaks energy
    // into the second half of its 2B-point response -- taps that alias back onto
    // the first half through the circular convolution and slowly poison the
    // estimate. Projecting one partition per block back onto its causal support
    // costs two transforms instead of 2P and reaches every partition within
    // P blocks (256 ms here), which is fast against the room it is tracking.
    constrain = (constrain + 1) % p;
    float* wr = w_re.data() + constrain * n;
    float* wi = w_im.data() + constrain * n;
    std::memcpy(t_re.data(), wr, n * sizeof(float));
    std::memcpy(t_im.data(), wi, n * sizeof(float));
    fft.inverse(t_re.data(), t_im.data());
    std::memset(t_re.data() + b, 0, b * sizeof(float));
    std::memset(t_im.data(), 0, n * sizeof(float));   // re-impose a real response
    fft.forward(t_re.data(), t_im.data());
    std::memcpy(wr, t_re.data(), n * sizeof(float));
    std::memcpy(wi, t_im.data(), n * sizeof(float));
}

void BlockFdafEchoCanceller::Impl::Suppress(float* out) noexcept {
    // Analysis frames: the residual and the predicted echo over the SAME
    // [previous block, current block] span, sqrt-Hann windowed.
    for (std::size_t i = 0; i < b; ++i) {
        e_re[i] = prev_e[i] * win[i];
        e_re[b + i] = err[i] * win[b + i];
    }
    std::memset(e_im.data(), 0, n * sizeof(float));
    fft.forward(e_re.data(), e_im.data());

    for (std::size_t i = 0; i < b; ++i) {
        t_re[i] = prev_y[i] * win[i];
        t_re[b + i] = yhat[i] * win[b + i];
    }
    std::memset(t_im.data(), 0, n * sizeof(float));
    fft.forward(t_re.data(), t_im.data());

    const float over = cfg.over_subtraction;
    const float floor_g = cfg.suppression_floor;
    for (std::size_t k = 0; k < n; ++k) {
        const float se = e_re[k] * e_re[k] + e_im[k] * e_im[k];
        const float sy = t_re[k] * t_re[k] + t_im[k] * t_im[k];
        const float residual = leak * sy;
        float g = 1.0f;
        if (se > 1e-20f) g = (se - over * residual) / se;   // Wiener, in power
        g = std::sqrt(std::max(g, 0.0f));                   // ...to amplitude
        g = std::clamp(g, floor_g, 1.0f);
        // Smoothed over time so a single noisy frame cannot punch a hole in the
        // spectrum. Symmetric and fast: 50 % per 8 ms block reaches unity within
        // ~25 ms of the echo stopping, well inside the VAD's own decision time.
        gain[k] = 0.5f * gain[k] + 0.5f * g;
        e_re[k] *= gain[k];
        e_im[k] *= gain[k];
    }

    fft.inverse(e_re.data(), e_im.data());

    // Weighted overlap-add. The frame's FIRST half is now complete; its second
    // half waits for the next frame -- which is the second block of latency.
    for (std::size_t i = 0; i < b; ++i) out[i] = ola[i] + e_re[i] * win[i];
    for (std::size_t i = 0; i < b; ++i) ola[i] = e_re[b + i] * win[b + i];
}

// -----------------------------------------------------------------------------

BlockFdafEchoCanceller::BlockFdafEchoCanceller(const EchoCancellerConfig& cfg) {
    if (!is_pow2(cfg.block_samples)) {
        throw std::invalid_argument(
            "BlockFdafEchoCanceller: block_samples must be a power of two (the "
            "radix-2 transform is sized 2 * block_samples)");
    }
    if (cfg.filter_tail_samples == 0) {
        throw std::invalid_argument(
            "BlockFdafEchoCanceller: filter_tail_samples must cover the whole "
            "speaker-to-microphone delay; zero cancels nothing");
    }
    impl_ = std::make_unique<Impl>(cfg);
}

BlockFdafEchoCanceller::~BlockFdafEchoCanceller() = default;

void BlockFdafEchoCanceller::Process(const float* near_end, const float* far_end,
                                     float* out, std::size_t count) noexcept {
    if (out == nullptr || count == 0) return;
    if (near_end == nullptr) return;

    Impl& s = *impl_;
    std::size_t done = 0;
    while (done < count) {
        const std::size_t chunk = std::min(kMaxChunk, count - done);
        s.near_fifo.push(near_end + done, chunk);
        // A null reference is "the far end is silent", not an error: it is what
        // a caller passes when playback is not running at all.
        if (far_end != nullptr) {
            s.far_fifo.push(far_end + done, chunk);
        } else {
            s.far_fifo.push_zeros(chunk);
        }

        while (s.near_fifo.size() >= s.b) {
            s.near_fifo.pop(s.d_blk.data(), s.b);
            s.far_fifo.pop(s.x_blk.data(), s.b);
            s.ProcessBlock(s.d_blk.data(), s.x_blk.data(), s.o_blk.data());
            s.out_fifo.push(s.o_blk.data(), s.b);
        }

        // Always satisfiable: the constructor pre-paid one block of framing
        // debt, so the output FIFO never holds fewer samples than were pushed.
        s.out_fifo.pop(out + done, chunk);
        done += chunk;
    }
}

void BlockFdafEchoCanceller::Reset() noexcept {
    Impl& s = *impl_;
    std::fill(s.far_win.begin(), s.far_win.end(), 0.0f);
    std::fill(s.x_re.begin(), s.x_re.end(), 0.0f);
    std::fill(s.x_im.begin(), s.x_im.end(), 0.0f);
    std::fill(s.w_re.begin(), s.w_re.end(), 0.0f);
    std::fill(s.w_im.begin(), s.w_im.end(), 0.0f);
    std::fill(s.gain.begin(), s.gain.end(), 1.0f);
    std::fill(s.yhat.begin(), s.yhat.end(), 0.0f);
    std::fill(s.err.begin(), s.err.end(), 0.0f);
    std::fill(s.prev_e.begin(), s.prev_e.end(), 0.0f);
    std::fill(s.prev_y.begin(), s.prev_y.end(), 0.0f);
    std::fill(s.ola.begin(), s.ola.end(), 0.0f);
    s.near_fifo.clear();
    s.far_fifo.clear();
    s.out_fifo.clear();
    s.out_fifo.push_zeros(s.b);
    s.head = 0;
    s.constrain = 0;
    s.leak = 1.0f;
    s.erle_lin = 1.0;
    s.peak_erle = 1.0;
    s.diverged_blocks = 0;
}

float BlockFdafEchoCanceller::erle_db() const noexcept {
    const double v = impl_->erle_lin;
    if (v <= 1.0) return 0.0f;
    return static_cast<float>(10.0 * std::log10(v));
}

std::size_t BlockFdafEchoCanceller::latency_samples() const noexcept {
    return 2 * impl_->b;
}

std::uint64_t BlockFdafEchoCanceller::divergence_resets() const noexcept {
    return impl_->resets;
}

float BlockFdafEchoCanceller::leak_estimate() const noexcept { return impl_->leak; }

}  // namespace blackwell::audio_rt
