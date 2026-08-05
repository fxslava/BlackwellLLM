#pragma once
// -----------------------------------------------------------------------------
// RationalResampler — streaming L/M polyphase FIR rate conversion.
//
// WHY THIS EXISTS. An echo canceller subtracts a reference from a microphone
// signal, and it can only do that if both are on the SAME clock. They are not:
// the capture path runs at 16 kHz (what Whisper's mel front-end and Silero both
// want) and F5-TTS emits at 24 kHz. Feeding the canceller a 24 kHz reference
// against a 16 kHz mic does not merely degrade it -- the "echo" it is asked to
// model is then a TIME-SCALED copy of its reference, and no linear
// time-invariant filter can produce one. The adaptive filter would chase a
// moving target forever and never cancel anything, which looks exactly like
// "the AEC is not working" and is impossible to debug from the outside.
//
// So the reference is converted to the near-end rate before it ever reaches the
// canceller. 24 kHz -> 16 kHz reduces to L=2 / M=3.
//
// GROUP DELAY IS PART OF THE CONTRACT. The FIR is linear phase, so it adds a
// constant delay of (L*T - 1) / (2*L) input samples -- 0.65 ms at the default
// 16 taps/phase. That delay lands on the REFERENCE, which means the canceller
// sees the echo arriving 0.65 ms sooner relative to its reference than it
// physically does. This is safe only because an adaptive FIR can model positive
// delays and not negative ones, and the real speaker->mic path (device buffer +
// flight time) is one to two orders of magnitude larger. Raising taps_per_phase
// far past the default without re-checking that margin is how a working AEC
// stops cancelling.
//
// THREADING. Not thread-safe and holds per-call state; one instance belongs to
// exactly one stream on exactly one thread. Allocation happens ONCE, in the
// constructor -- Process() never allocates, so it is legal on the DSP worker
// that must not stall the capture drain.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <vector>

namespace blackwell::audio_rt {

class RationalResampler {
public:
    // `in_rate` / `out_rate` are reduced by their GCD; only the ratio matters.
    // `taps_per_phase` is the FIR length of ONE polyphase branch: 16 gives a
    // stopband floor around -70 dB at 24->16, which is far below anything the
    // canceller's own residual reaches. Throws std::invalid_argument on a
    // non-positive rate (INIT tier).
    RationalResampler(int in_rate, int out_rate, int taps_per_phase = 16);

    // How many samples Process() will emit for `in_count` inputs, given the
    // phase it is currently in. EXACT, not an upper bound -- call it to size the
    // destination rather than guessing, because a rational converter emits a
    // different count on different calls (2, 1, 2, 1, ... for 3:2).
    std::size_t OutputCountFor(std::size_t in_count) const noexcept;

    // Converts `in_count` samples. Writes at most `out_capacity` samples and
    // returns how many it wrote; a short return means the destination was too
    // small and the excess was DROPPED (which desynchronises the stream, so
    // size the destination with OutputCountFor). Never allocates.
    std::size_t Process(const float* in, std::size_t in_count,
                        float* out, std::size_t out_capacity) noexcept;

    // Forgets all history and returns to phase 0. Use on a stream discontinuity
    // (a device restart), never mid-stream: it is a step change in the output.
    void Reset() noexcept;

    int interpolation() const noexcept { return l_; }
    int decimation() const noexcept { return m_; }

    // Constant group delay the filter adds, in INPUT samples. See the header.
    double group_delay_in_samples() const noexcept;

private:
    int l_ = 1;                  // interpolation factor (rate ratio numerator)
    int m_ = 1;                  // decimation factor
    std::size_t taps_ = 0;       // taps per polyphase branch (T)
    std::vector<float> h_;       // L*T coefficients, branch p at stride L from p

    // Input history addressed by ABSOLUTE sample index masked into the buffer,
    // so a branch never has to care where a call boundary fell. Capacity is a
    // power of two and at least T + the largest chunk Process() consumes in one
    // pass, which is what guarantees every tap a phase needs is still resident.
    std::vector<float> hist_;
    std::size_t mask_ = 0;

    std::uint64_t total_in_ = 0;   // absolute count of samples ever pushed
    std::uint64_t out_n_ = 0;      // absolute index of the next output sample
};

}  // namespace blackwell::audio_rt
