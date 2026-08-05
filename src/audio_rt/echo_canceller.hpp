#pragma once
// -----------------------------------------------------------------------------
// echo_canceller.hpp — the AEC seam (docs/TTS_INTEGRATION_AUDIT.md Phase 4) and
// the partitioned-block frequency-domain implementation behind it.
//
// =============================================================================
// WHAT PROBLEM THIS SOLVES, AND WHY MUTING THE MIC WAS NOT A SOLUTION
// =============================================================================
// The loudspeaker feeds the microphone. Silero scores the assistant's own voice
// as speech -- correctly, because it IS speech -- which fires speech-onset,
// which is barge-in, which cancels the generation currently being spoken. The
// system's own correctness works against it.
//
// The cheap answer was to mute the microphone whenever the speaker was live.
// That works, and it costs the entire feature: a user cannot interrupt by
// talking over the assistant, because during exactly the window in which they
// would talk, the assistant is deaf. Full duplex means the microphone NEVER
// stops, so the only admissible fix is to remove the assistant's voice from the
// microphone signal rather than to remove the microphone.
//
// =============================================================================
// TWO STAGES, AND WHY ONE IS NOT ENOUGH
// =============================================================================
// 1. LINEAR ECHO CANCELLATION. An adaptive FIR learns the speaker->room->mic
//    impulse response and subtracts the predicted echo. This is the stage that
//    preserves the near-end talker perfectly: it removes only what the reference
//    explains, so a user talking over the assistant comes through untouched.
//    Against a real loudspeaker it plateaus around 15-25 dB, because the path is
//    not linear -- amplifier and cone distortion produce echo components no FIR
//    of the reference can generate.
//
// 2. RESIDUAL ECHO SUPPRESSION. What stage 1 leaves is still audible speech, and
//    a neural VAD will happily score it. So a per-bin Wiener gain removes the
//    part of the remaining spectrum whose power is explained by the (measured,
//    not assumed) leakage of the echo estimate. It acts per FREQUENCY BIN, which
//    is what keeps it from being a gate: during double-talk the user's speech
//    occupies bins the echo does not, and those bins pass at unity.
//
// The suppressor runs in a weighted-overlap-add frame with sqrt-Hann analysis
// and synthesis, so when its gain is unity -- which is what happens whenever
// nothing is playing -- the reconstruction is the input, sample for sample, to
// within float rounding. That property is load-bearing: it is what lets the AEC
// sit permanently in the capture path without the ASR ever paying for it.
//
// =============================================================================
// DOUBLE-TALK: THE STEP SIZE IS THE DEFENCE, NOT A DETECTOR
// =============================================================================
// The classic failure is adapting the filter while the user is speaking: the
// near-end voice is not correlated with the reference, so it drives the weights
// away from the true room response and the canceller comes out of double-talk
// worse than it went in. Rather than a threshold-based double-talk detector
// (which has to be tuned per room and fails silently when it is wrong), the step
// size is scaled every block by
//
//     mu_eff = mu * Ey / (Ey + Ee)
//
// where Ey is the energy of the predicted echo and Ee the energy of what is left
// after subtracting it. In echo-only stretches Ee << Ey and adaptation runs at
// full speed; the moment a near-end talker dominates, Ee swamps Ey and the step
// collapses towards zero. It is continuous, has no cliff to mistune, and it
// degrades towards "stop learning" rather than towards "learn the wrong thing".
//
// =============================================================================
// LATENCY AND THREADING
// =============================================================================
// Process() accepts any block length and imposes a FIXED delay of
// 2 * block_samples (16 ms at the defaults) on the near-end path: one block of
// framing, one of overlap-add. That delay is on the path to the VAD and the ASR,
// not on the path to the speaker, so it costs barge-in reaction time and nothing
// else -- 16 ms against a VAD that needs 45-110 ms to decide.
//
// Not thread-safe; one instance belongs to one capture stream on one thread. All
// allocation happens in the constructor, so Process() is safe to run inline in
// the capture-drain loop.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace blackwell::audio_rt {

struct EchoCancellerConfig {
    // Adaptation block, in samples. Must be a power of two; 128 = 8 ms at
    // 16 kHz. Smaller tracks a moving talker sooner and costs more FFTs.
    std::size_t block_samples = 128;

    // How much room the filter can model. Must cover the WHOLE speaker->mic
    // delay (playback device buffer + flight time + capture buffer) plus the
    // reverberation tail, because everything past it is uncancellable by
    // construction. 256 ms is generous for a desktop; the cost is linear.
    std::size_t filter_tail_samples = 4096;

    // NLMS step ceiling, before the double-talk scaling above.
    //
    // MEASURED, not guessed, and the curve is not monotone -- a bigger step
    // converges sooner and settles WORSE, because the steady-state misadjustment
    // grows with it. Against the synthetic room in tests/tts/echo_canceller_test
    // the converged ERLE runs 9 dB at 0.05, 22 dB at 0.2, 25 dB at 0.35, then
    // collapses to 16 dB at 0.5 and 5 dB at 0.8. 0.3 sits inside the plateau
    // rather than on its peak, because the peak moves with the room.
    float mu = 0.3f;

    bool residual_suppression = true;

    // Floor on the suppressor's AMPLITUDE gain. Not zero on purpose: driving a
    // bin to silence produces the warbling "musical noise" that makes a VAD
    // twitch, and 0.03 (-30 dB) is already far below anything Silero scores.
    float suppression_floor = 0.03f;

    // Over-subtraction on the residual power estimate. 1.0 is the Wiener
    // solution; slightly above buys margin against an under-estimated leak at
    // the cost of clipping the quietest near-end syllables.
    float over_subtraction = 1.5f;
};

// The seam. A capture path holds this type, never the implementation, so a
// SpeexDSP or AEC3 backend can be dropped in without touching a caller.
class IEchoCanceller {
public:
    virtual ~IEchoCanceller() = default;
    IEchoCanceller(const IEchoCanceller&) = delete;
    IEchoCanceller& operator=(const IEchoCanceller&) = delete;

    // `near_end` and `far_end` are BLOCK-SYNCHRONOUS: `count` samples each, at
    // the same sample rate, covering the same span of wall clock. Keeping that
    // true is the caller's job and is the single most important thing about
    // integrating an AEC -- see AecCaptureFilter, which is what does it here.
    //
    // Writes exactly `count` samples to `out`, delayed by latency_samples().
    // `out` may alias `near_end`. noexcept and allocation-free.
    virtual void Process(const float* near_end, const float* far_end,
                         float* out, std::size_t count) noexcept = 0;

    // Drops the learned response and all history. For a device restart, not for
    // a barge-in: barge-in is precisely when the learned response is most
    // valuable, because the room is still ringing with the tail of our own
    // voice while the user talks over it.
    virtual void Reset() noexcept = 0;

    // Smoothed echo return loss enhancement, dB. How much of the microphone
    // energy the linear stage is removing while the far end is active; 0 means
    // it has learned nothing. THE number to look at when an AEC "does not work".
    virtual float erle_db() const noexcept = 0;

    // Fixed delay Process() imposes on the near-end path.
    virtual std::size_t latency_samples() const noexcept = 0;

protected:
    IEchoCanceller() = default;
};

// Partitioned-block frequency-domain adaptive filter (overlap-save), NLMS in the
// frequency domain, plus the WOLA residual suppressor described above.
class BlockFdafEchoCanceller final : public IEchoCanceller {
public:
    // INIT tier: throws std::invalid_argument on a config that cannot work
    // (non-power-of-two block, empty tail).
    explicit BlockFdafEchoCanceller(const EchoCancellerConfig& cfg);
    ~BlockFdafEchoCanceller() override;

    void Process(const float* near_end, const float* far_end,
                 float* out, std::size_t count) noexcept override;
    void Reset() noexcept override;
    float erle_db() const noexcept override;
    std::size_t latency_samples() const noexcept override;

    // Times the filter was found to be AMPLIFYING the echo and was zeroed. A
    // nonzero value that keeps climbing means the reference is not aligned with
    // the microphone -- look at the far-end plumbing, not at the step size.
    std::uint64_t divergence_resets() const noexcept;

    // Smoothed residual-to-echo ratio the suppressor is acting on, in [0,1].
    // 1 means "the linear stage is removing nothing yet" (the startup state);
    // small values mean it has converged and the suppressor is barely needed.
    float leak_estimate() const noexcept;

private:
    struct Impl;                       // hides the FFT tables and the weight sets
    std::unique_ptr<Impl> impl_;
};

}  // namespace blackwell::audio_rt
