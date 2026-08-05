#pragma once
// -----------------------------------------------------------------------------
// AecCaptureFilter — drops into a capture path and hands back echo-cancelled
// audio. It owns everything that stands between "there is a reference ring" and
// "the canceller can actually use it": rate conversion, and the far harder half,
// TIME ALIGNMENT.
//
// =============================================================================
// THE ALIGNMENT PROBLEM IS THE WHOLE JOB
// =============================================================================
// The canceller needs its two inputs block-synchronous: `count` microphone
// samples and `count` reference samples covering the same span of wall clock.
// Neither side offers that. The reference is written by the PLAYBACK device
// callback, the microphone is drained by the capture worker, they are different
// devices on different clocks with independent buffering, and the reference ring
// between them has whatever backlog history left in it.
//
// What makes it tractable is that a backlog is not symmetric in its
// consequences:
//
//   Reference AHEAD of the microphone (small backlog) is FINE. The samples we
//   pull describe sound the speaker has not emitted yet, so the echo shows up in
//   the microphone LATER -- a positive delay, which is exactly what an adaptive
//   FIR models. The filter simply learns a longer impulse response.
//
//   Reference BEHIND the microphone (large backlog) is FATAL. The echo would
//   have to appear BEFORE its reference, a negative delay no causal filter can
//   represent. The canceller cancels nothing, and it looks identical to a
//   broken filter.
//
// So the policy is deliberately asymmetric: keep a small cushion, and when the
// backlog grows past the ceiling, DROP THE OLDEST reference samples to pull the
// stream back towards the present. Each drop is a resync -- the filter re-learns
// over the next few hundred milliseconds -- so it is counted, and a resync count
// that keeps climbing means the two devices are drifting rather than that the
// policy is working.
//
// The one thing that would break this is a reference ring the writer does not
// keep continuously fed. The shipping feeder is a WASAPI loopback capture of the
// render endpoint, which runs continuously and delivers SILENCE rather than
// nothing while the speaker is idle -- so the stream has no gaps to mis-align
// on. Any other writer must hold that property too.
//
// =============================================================================
// WHY THIS IS NOT IN THE BRIDGE, AND NOT IN THE APP
// =============================================================================
// Not in TTSDuplexBridge, because the bridge is the speech-output stack and this
// is capture-side: it has no business knowing the microphone's rate or block
// size. Not in the app, because "resample, bound the backlog, cancel" is
// exactly the part that is worth a unit test and impossible to test through a
// GUI. The app is left with one call in its PCM tap.
//
// THREADING. One instance, one capture thread -- it is the SOLE consumer of the
// reference ring, which is what the ring's SPSC contract requires. All
// allocation is in the constructor; Process() is allocation-free and never
// blocks, so it is legal inline in the capture-drain loop.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "echo_canceller.hpp"
#include "spsc_ring.hpp"

namespace blackwell::audio_rt {

struct AecCaptureFilterConfig {
    int near_rate = 16000;   // microphone / VAD / ASR rate
    int far_rate = 24000;    // playback rate (F5-TTS emits 24 kHz)

    EchoCancellerConfig aec{};

    // Reference backlog the synchroniser aims to hold, in near-end milliseconds.
    // Small but nonzero: at zero, ordinary scheduling jitter empties the queue
    // and each empty call pushes the reference one block further ahead of the
    // microphone, which walks the echo out of the filter's tail.
    int target_lag_ms = 8;

    // Backlog above which the oldest reference is discarded. Must stay well
    // under the filter tail, because the backlog is subtracted from the delay
    // budget the filter has to work with.
    int max_lag_ms = 32;
};

class AecCaptureFilter {
public:
    // `far_ring` is owned by the caller and must outlive this object. INIT tier:
    // throws std::invalid_argument on a config the canceller cannot honour.
    //
    // This overload builds the DEFAULT backend (BlockFdafEchoCanceller) from
    // cfg.aec.
    AecCaptureFilter(SpscRing<float>& far_ring, const AecCaptureFilterConfig& cfg);

    // BRING YOUR OWN CANCELLER. Everything this class does -- rate conversion,
    // backlog bounding, alignment -- is backend-independent, so the subtractor
    // is a caller's choice: WebRTC AEC3 where it is available, the built-in
    // partitioned-block filter otherwise.
    //
    // The choice is INJECTED rather than named by an enum because this target is
    // dependency-free by construction (see its CMakeLists). An enum would put
    // WebRTC's include path on every consumer of a header that currently needs
    // nothing but the STL.
    //
    // A null `backend` is the same as the overload above. Note that
    // divergence_resets() and leak_estimate() describe an NLMS filter's internal
    // health and report neutral values for any other backend.
    AecCaptureFilter(SpscRing<float>& far_ring, const AecCaptureFilterConfig& cfg,
                     std::unique_ptr<IEchoCanceller> backend);
    ~AecCaptureFilter();

    AecCaptureFilter(const AecCaptureFilter&) = delete;
    AecCaptureFilter& operator=(const AecCaptureFilter&) = delete;

    // `count` microphone samples in, `count` echo-cancelled samples out, delayed
    // by latency_samples(). `out` may alias `near_end`. noexcept, allocation-free.
    void Process(const float* near_end, std::size_t count, float* out) noexcept;

    // Bypass. The reference ring is STILL drained while disabled -- stopping
    // would let it fill, the writer would start counting overruns on the audio
    // callback, and re-enabling would begin from a backlog measured in seconds.
    void SetEnabled(bool on) noexcept;
    bool enabled() const noexcept;

    // Forgets the learned impulse response and starts converging again.
    //
    // WHEN IT IS REQUIRED, and it is not optional there: after the PLAYBACK or
    // CAPTURE endpoint changes underneath us. The filter's coefficients describe
    // one specific acoustic path -- that speaker, that microphone, the latency
    // between them -- and after a device swap they describe a path that no
    // longer exists. Keeping them does not decay gracefully: the canceller
    // actively subtracts a signal that is not there, which is audibly worse than
    // not cancelling at all until it re-converges.
    //
    // Costs a few hundred milliseconds of reduced ERLE. Nothing covers that
    // window -- the VAD-side guards this replaced are gone and were removed on
    // purpose (see vad_score in main.cpp), so a device swap while the assistant
    // is mid-sentence can still self-trigger. It is a rare, user-initiated
    // moment, which is the trade being made.
    //
    // Safe from any thread; the next Process() sees the cleared state.
    void Reset() noexcept;

    // Scales the reference to match what the SPEAKER actually emits.
    //
    // NOT NEEDED BY THE SHIPPING PATH, and left here for the ones that are not.
    // A loopback reference is tapped downstream of the software volume, so it
    // already carries it and this stays at 1.
    //
    // WHY IT EXISTS. A reference tapped BEFORE the playback gain (the bridge's
    // AecTap::Playback) describes the pre-gain signal while the room hears the
    // post-gain one. Left alone, the canceller absorbs the difference into its
    // learned impulse response -- which works, and then breaks the moment the
    // volume MOVES: the response is instantly wrong by exactly that ratio, and
    // it spends a few hundred milliseconds re-converging while the assistant is
    // audibly speaking. That is precisely the window in which it must not be
    // blind. Applying the same gain here makes the learned response invariant to
    // volume. Such a caller must keep this equal to the playback gain; a
    // mismatch does not fault, it just costs ERLE.
    //
    // Clamped to [0, 4]. Negative is rejected because inverting the reference
    // would make the canceller ADD the echo rather than remove it.
    void SetReferenceGain(float g) noexcept;
    float reference_gain() const noexcept;

    // ---- observers -----------------------------------------------------------
    float erle_db() const noexcept;
    std::size_t latency_samples() const noexcept;
    // Times the backlog ceiling was hit and reference was discarded to catch up.
    std::uint64_t resyncs() const noexcept;
    // Reference samples the synchroniser had to invent (as silence) because the
    // ring was dry. A steady trickle is jitter; a flood means playback stopped.
    std::uint64_t reference_underruns() const noexcept;
    std::uint64_t divergence_resets() const noexcept;
    float leak_estimate() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace blackwell::audio_rt
