#pragma once
// -----------------------------------------------------------------------------
// Aec3EchoCanceller — WebRTC's EchoCanceller3 behind this repo's IEchoCanceller
// seam. The seam already existed and already named this as its intended second
// implementation (echo_canceller.hpp); this is that implementation.
//
// =============================================================================
// WHAT AEC3 BRINGS THAT THE BUILT-IN FILTER DOES NOT
// =============================================================================
// BlockFdafEchoCanceller is a good partitioned-block NLMS filter with a Wiener
// residual stage, and against a LINEAR synthetic room it measures well (see
// tests/tts/echo_canceller_test.cpp). Two things it cannot do:
//
//   1. DELAY TRACKING. It assumes the reference it is handed is already aligned
//      with the microphone and models only what fits in its tail. AecCaptureFilter
//      keeps that true by bounding the backlog, and every time it has to discard
//      reference to catch up, the filter re-converges from nothing. AEC3 carries
//      its own delay estimator and render-buffer alignment, so a drifting or
//      re-synced reference is absorbed rather than re-learned.
//
//   2. NON-LINEAR ECHO. A real loudspeaker distorts -- amplifier and cone
//      non-linearity produce echo components that are NOT any linear function of
//      the reference, so no FIR of the reference can generate them and no linear
//      subtractor can remove them. This is the ceiling that made the linear stage
//      plateau in a real room while measuring 20+ dB in a simulated one. AEC3's
//      suppressor works on the residual's spectrum and is not bound by that.
//
// =============================================================================
// TWO THINGS THE SPEC FOR THIS WORK ASKED FOR THAT DO NOT EXIST AS SWITCHES
// =============================================================================
// There is no "drift compensator" to enable and no "NLP" to turn on. The
// explicit drift API (set_stream_drift_samples, and the DPLL-style resampling
// behind it) belonged to the PREVIOUS generation of this canceller, which AEC3
// replaced; AEC3 handles drift inside its delay estimator with no caller-facing
// control. Likewise the non-linear/residual suppressor is an integral stage of
// AEC3's pipeline, not an optional one -- it is tuned through
// EchoCanceller3Config::suppressor, not enabled. Constructing this class is what
// "turns them on".
//
// =============================================================================
// FRAMING, AND THE ONE COST THIS ADDS
// =============================================================================
// AEC3 works in fixed 10 ms frames. IEchoCanceller::Process accepts any block
// length, so this class buffers to the frame boundary, which imposes a FIXED
// latency of one frame (160 samples at 16 kHz) on the near-end path. That delay
// is on the path to the VAD and the ASR, not to the speaker -- the same trade
// the built-in filter already makes with its 2-block overlap-add, and a
// comparable amount.
//
// PIMPL, and strictly: WebRTC's headers are large, carry their own abseil
// dependency and their own compile definitions. This header must stay includable
// by code that links none of that, exactly as silero_vad.hpp is the only file in
// the repo that sees <onnxruntime_cxx_api.h>.
//
// Not thread-safe; one instance belongs to one capture stream on one thread.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <memory>

#include "echo_canceller.hpp"

namespace blackwell::audio_rt {

struct Aec3Config {
    // Must be one of AEC3's supported rates: 16000, 32000 or 48000. The capture
    // path here runs at 16 kHz and so does the loopback reference, which is why
    // this defaults to it and why there is no resampler on either side.
    int sample_rate_hz = 16000;

    // Mono in, mono out. Present so the field is not a magic 1 at the call site.
    int num_render_channels = 1;
    int num_capture_channels = 1;

    // Initial hint for the speaker->microphone delay, in milliseconds. AEC3
    // estimates this itself and will correct a wrong hint, so it is a
    // convergence accelerator rather than a setting: it saves the first second
    // or so of searching. 0 means "no hint".
    //
    // A loopback reference is tapped close to the endpoint, so the true delay is
    // small -- the render buffer plus the capture buffer, not a whole room's
    // worth.
    int initial_delay_ms = 0;
};

class Aec3EchoCanceller final : public IEchoCanceller {
public:
    // INIT tier: throws std::invalid_argument on an unsupported rate or channel
    // count, and std::runtime_error if AEC3 itself declines to construct.
    explicit Aec3EchoCanceller(const Aec3Config& cfg);
    ~Aec3EchoCanceller() override;

    void Process(const float* near_end, const float* far_end,
                 float* out, std::size_t count) noexcept override;
    void Reset() noexcept override;

    // AEC3's own echo-return-loss-enhancement metric, in dB. Sampled from
    // GetMetrics(), which AEC3 updates on a slower cadence than the frame rate --
    // so this is a smoothed number by construction and a fresh instance reports
    // 0 until it has something to say.
    float erle_db() const noexcept override;
    std::size_t latency_samples() const noexcept override;

    // AEC3's estimate of the current speaker->microphone delay, in milliseconds,
    // or -1 while it has not converged on one. THE diagnostic for a reference
    // that is plumbed to the wrong endpoint: a delay that never settles means
    // the two streams are not describing the same sound.
    int estimated_delay_ms() const noexcept;

    // Frames APM refused, which are passed through UNCANCELLED. Nonzero looks
    // exactly like a filter that will not converge, and the two want different
    // fixes -- so the counter is readable rather than merely kept.
    std::uint64_t process_errors() const noexcept;

private:
    struct Impl;                     // hides every WebRTC type
    std::unique_ptr<Impl> impl_;
};

}  // namespace blackwell::audio_rt
