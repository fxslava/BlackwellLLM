// -----------------------------------------------------------------------------
// AecCaptureFilter implementation — see the header for the alignment argument.
// -----------------------------------------------------------------------------
#include "aec_capture_filter.hpp"

#include <algorithm>
#include <cstring>

#include "rational_resampler.hpp"

namespace blackwell::audio_rt {
namespace {

// Largest span handled in one pass. The capture worker pops at most 4096
// samples per call (audio_sandbox/src/realtime_dsp.cpp), so this is a ceiling
// rather than a limit anything hits.
constexpr std::size_t kMaxChunk = 4096;

}  // namespace

struct AecCaptureFilter::Impl {
    // `backend` may be null, in which case the built-in partitioned-block filter
    // is constructed from cfg.aec. INJECTED rather than selected by an enum here
    // because this translation unit must not know the set of backends: naming
    // AEC3 would put WebRTC's include path on blackwell_audio_rt, whose whole
    // premise is that it has none (see its CMakeLists).
    Impl(SpscRing<float>& ring, const AecCaptureFilterConfig& c,
         std::unique_ptr<IEchoCanceller> backend)
        : backend_was_injected(backend != nullptr),
          cfg(c),
          far_ring(ring),
          resampler(c.far_rate, c.near_rate),
          aec(backend ? std::move(backend)
                      : std::unique_ptr<IEchoCanceller>(
                            std::make_unique<BlockFdafEchoCanceller>(c.aec))) {
        // Only non-null when the DEFAULT backend is in use. The two observers
        // below describe an NLMS filter's internal health and have no meaning
        // for a backend that is not one -- so they report a neutral value
        // rather than inventing a number a caller might act on.
        fdaf = backend_was_injected ? nullptr
                                    : static_cast<BlockFdafEchoCanceller*>(aec.get());
        const double sr = static_cast<double>(c.near_rate) / 1000.0;
        target_lag = static_cast<std::size_t>(std::max(0.0, sr * c.target_lag_ms));
        max_lag = static_cast<std::size_t>(std::max(0.0, sr * c.max_lag_ms));
        if (max_lag <= target_lag) max_lag = target_lag + 1;

        // The far ring carries the PLAYBACK rate, so a near-end chunk needs
        // far_rate/near_rate as many samples -- rounded up, plus slack for the
        // backlog we deliberately keep.
        const std::size_t far_chunk =
            (kMaxChunk * static_cast<std::size_t>(c.far_rate)) /
                static_cast<std::size_t>(c.near_rate) + 8;
        raw_far.assign(far_chunk, 0.0f);
        // Worst case in one pass: everything already queued plus a full chunk.
        fifo.assign(kMaxChunk + max_lag + kMaxChunk + 16, 0.0f);
        blk.assign(kMaxChunk, 0.0f);
    }

    // Moves everything the ring holds through the resampler into `fifo`.
    void DrainReference() noexcept {
        for (;;) {
            const std::size_t want = std::min(raw_far.size(), far_ring.available());
            if (want == 0) break;
            const std::size_t got = far_ring.read(raw_far.data(), want);
            if (got == 0) break;
            const std::size_t need = resampler.OutputCountFor(got);
            const std::size_t room = fifo.size() - fill;
            if (need > room) {
                // Cannot happen with the sizes above unless the capture worker
                // stalled for seconds. Make room the same way the ceiling policy
                // does -- from the OLDEST end -- rather than dropping the newest
                // reference, which is the part still worth having.
                const std::size_t drop = need - room;
                DropOldest(std::min(drop, fill));
                ++resyncs;
            }
            fill += resampler.Process(raw_far.data(), got, fifo.data() + fill,
                                      fifo.size() - fill);
        }
    }

    void DropOldest(std::size_t n) noexcept {
        if (n == 0) return;
        if (n >= fill) {
            fill = 0;
            return;
        }
        std::memmove(fifo.data(), fifo.data() + n, (fill - n) * sizeof(float));
        fill -= n;
    }

    // Hands back exactly `count` reference samples aligned with the microphone.
    void TakeReference(std::size_t count) noexcept {
        DrainReference();

        // THE CEILING. Past it the reference is running behind the microphone,
        // which is the one condition the filter cannot model at all.
        if (fill > count + max_lag) {
            DropOldest(fill - count - target_lag);
            ++resyncs;
        }

        const std::size_t take = std::min(count, fill);
        if (ref_gain == 1.0f) {
            std::memcpy(blk.data(), fifo.data(), take * sizeof(float));
        } else {
            // Applied on the way OUT rather than on the way in, so a gain change
            // takes effect on the samples about to be used rather than on
            // whatever happened to be mid-queue when it arrived.
            for (std::size_t i = 0; i < take; ++i) blk[i] = fifo[i] * ref_gain;
        }
        if (take < count) {
            std::memset(blk.data() + take, 0, (count - take) * sizeof(float));
            ref_underruns += count - take;
        }
        DropOldest(take);
    }

    const bool backend_was_injected;
    AecCaptureFilterConfig cfg;
    SpscRing<float>& far_ring;
    RationalResampler resampler;
    // The canceller, owned through the seam so the backend is a caller's
    // choice (BlockFdaf by default, AEC3 when the app supplies one).
    std::unique_ptr<IEchoCanceller> aec;
    BlockFdafEchoCanceller* fdaf = nullptr;   // non-owning; see the ctor

    std::vector<float> raw_far;   // staging, playback rate
    std::vector<float> fifo;      // resampled reference, near-end rate
    std::size_t fill = 0;
    std::vector<float> blk;       // one aligned reference block

    std::size_t target_lag = 0;
    std::size_t max_lag = 0;
    bool enabled = true;
    // Plain float, not atomic: it is written by the settings handler and read by
    // the capture thread, and a torn read is not possible for a 4-byte aligned
    // float on any target this builds for. Worst case one block uses the old
    // gain, which is 20 ms of slightly-wrong reference -- far below what the
    // adaptive filter tracks out on its own.
    float ref_gain = 1.0f;

    std::uint64_t resyncs = 0;
    std::uint64_t ref_underruns = 0;
};

AecCaptureFilter::AecCaptureFilter(SpscRing<float>& far_ring,
                                   const AecCaptureFilterConfig& cfg)
    : impl_(std::make_unique<Impl>(far_ring, cfg, nullptr)) {}

AecCaptureFilter::AecCaptureFilter(SpscRing<float>& far_ring,
                                   const AecCaptureFilterConfig& cfg,
                                   std::unique_ptr<IEchoCanceller> backend)
    : impl_(std::make_unique<Impl>(far_ring, cfg, std::move(backend))) {}

AecCaptureFilter::~AecCaptureFilter() = default;

void AecCaptureFilter::Process(const float* near_end, std::size_t count,
                               float* out) noexcept {
    if (near_end == nullptr || out == nullptr || count == 0) return;
    Impl& s = *impl_;

    std::size_t done = 0;
    while (done < count) {
        const std::size_t chunk = std::min(kMaxChunk, count - done);
        s.TakeReference(chunk);
        if (s.enabled) {
            s.aec->Process(near_end + done, s.blk.data(), out + done, chunk);
        } else if (out != near_end) {
            std::memcpy(out + done, near_end + done, chunk * sizeof(float));
        }
        done += chunk;
    }
}

void AecCaptureFilter::SetEnabled(bool on) noexcept {
    Impl& s = *impl_;
    if (s.enabled == on) return;
    // Re-entering from bypass starts from a filter that learned nothing during
    // the gap and a reference queue that may have been resynced under it. Both
    // are stale; keeping them would mean the first seconds after re-enabling are
    // spent UN-learning rather than learning.
    if (on) s.aec->Reset();
    s.enabled = on;
}

bool AecCaptureFilter::enabled() const noexcept { return impl_->enabled; }

void AecCaptureFilter::Reset() noexcept {
    // Exactly what re-entering from bypass does above, for exactly the same
    // reason: both cases leave coefficients describing a path that is gone.
    impl_->aec->Reset();
}

void AecCaptureFilter::SetReferenceGain(float g) noexcept {
    // The NaN-safe form: `!(g >= 0)` is true for NaN, where `g < 0` is false.
    // A NaN gain would propagate into the weights and never leave them.
    if (!(g >= 0.0f)) g = 0.0f;
    impl_->ref_gain = std::min(g, 4.0f);
}

float AecCaptureFilter::reference_gain() const noexcept { return impl_->ref_gain; }

float AecCaptureFilter::erle_db() const noexcept { return impl_->aec->erle_db(); }

std::size_t AecCaptureFilter::latency_samples() const noexcept {
    return impl_->aec->latency_samples();
}

std::uint64_t AecCaptureFilter::resyncs() const noexcept { return impl_->resyncs; }

std::uint64_t AecCaptureFilter::reference_underruns() const noexcept {
    return impl_->ref_underruns;
}

std::uint64_t AecCaptureFilter::divergence_resets() const noexcept {
    return impl_->fdaf != nullptr ? impl_->fdaf->divergence_resets() : 0;
}

float AecCaptureFilter::leak_estimate() const noexcept {
    // 1.0 is the "nothing removed yet" end of the scale, which is the honest
    // reading for a backend that does not expose a leak estimate at all.
    return impl_->fdaf != nullptr ? impl_->fdaf->leak_estimate() : 1.0f;
}

}  // namespace blackwell::audio_rt
