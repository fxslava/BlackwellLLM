#pragma once
// -----------------------------------------------------------------------------
// SileroVAD — neural voice-activity detection (Silero v5/v6 ONNX) over the
// ONNXRuntime CPU execution provider.
//
// WHY CPU: the model is 2.3 MB and one inference over a 32 ms chunk costs well
// under a millisecond on a single core. A GPU execution provider would consume
// VRAM and bandwidth the 8B AWQ backbone needs and would drag a second CUDA
// runtime into the process for no latency win. This class never touches CUDA.
//
// STATE (the part that makes this an RNN, not a classifier): Silero carries a
// [2, 1, 128] float recurrent state between chunks — the probability for chunk
// N depends on every chunk before it. The state is owned here, threaded through
// every Session::Run, and ping-ponged between two buffers so a step needs no
// copy-back. reset_state() zeroes it and MUST be called whenever the audio is
// discontinuous (a new utterance, a mode flip, a seek) — feeding a fresh
// utterance into a state trained on the previous one produces garbage for the
// first few hundred milliseconds.
//
// CHUNK GEOMETRY: the caller supplies EXACTLY kChunkSamples (512 @ 16 kHz =
// 32 ms) of NEW audio per step. The tensor actually fed to the model is larger:
// Silero v5+ expects kContextSamples (64) samples of the PREVIOUS chunk
// prepended, i.e. [context | new] = 576 floats. That prepend is not optional and
// not cosmetic — feeding a bare 512 produces ~0.0009 for fluent speech (measured)
// instead of ~1.0, because the model's first convolution then sees a window
// short of the receptive field it was exported with. The context is maintained
// internally; callers never see it.
//
// The live pipeline blocks audio at 160 samples (10 ms), so feed() adapts: it
// accumulates arbitrary sample counts and runs an inference every 512, returning
// the most recent probability in between ("sticky"). Detection latency is
// therefore <= 32 ms and the pipeline's 10 ms hangover resolution is unchanged.
//
// ERROR DOCTRINE (CLAUDE.md extension pattern #4), split by phase:
//   INIT    — the ctor throws std::runtime_error (missing model, unexpected
//             graph signature, ORT failure). Callers construct inside try/catch
//             and fall back to the threshold VAD.
//   RUNTIME — process_chunk()/feed() are noexcept. An ORT exception mid-stream
//             is caught, counted (inference_errors()), and reported as the
//             last-good probability: a VAD hiccup must degrade boundary quality,
//             never unwind the audio thread.
//
// THREADING: ONE owning thread calls feed()/process_chunk()/reset_state() (the
// DSP worker in audio_translator). threshold()/set_threshold()/
// last_probability()/inference_errors() are atomic and callable from any thread
// — that is the UI-slider seam, and the only cross-thread surface.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace blackwell::vad {

// The model's fixed operating point. Silero also ships an 8 kHz mode (256-sample
// chunks); this project is 16 kHz end to end (Whisper front-end), so 16 kHz is
// the only rate wired up.
inline constexpr int    kSampleRate   = 16000;
inline constexpr size_t kChunkSamples = 512;                    // 32 ms @ 16 kHz
inline constexpr size_t kStateElems   = 2u * 1u * 128u;         // [2, batch=1, 128]
// Samples of the previous chunk the model requires prepended to each step (see
// the header block). 64 @ 16 kHz. Exposed because it explains why the first
// chunk after reset_state() is scored with a zero-filled lead-in.
inline constexpr size_t kContextSamples = 64;

// Returned by process_chunk() when the input is malformed (wrong sample count or
// null pointer). Negative so it can never be confused with a probability, and so
// `p > threshold` reads false without a special case at every call site.
inline constexpr float kInvalidProbability = -1.0f;

class SileroVAD {
public:
    // INIT tier: loads silero_vad.onnx and validates that the graph exposes the
    // v5/v6 signature this wrapper binds to (input/state/sr -> output/stateN).
    // Throws std::runtime_error on a missing file, a pre-v5 model (those expose
    // separate h/c state tensors), or any ORT failure.
    explicit SileroVAD(const std::string& model_path);
    ~SileroVAD();

    // Movable (the session travels with the PIMPL). Standard moved-from
    // semantics: only destruction and assignment are valid on the source.
    SileroVAD(SileroVAD&&) noexcept;
    SileroVAD& operator=(SileroVAD&&) noexcept;
    SileroVAD(const SileroVAD&) = delete;
    SileroVAD& operator=(const SileroVAD&) = delete;

    // RUNTIME tier. Runs ONE inference over exactly kChunkSamples samples and
    // advances the recurrent state. Returns the speech probability in [0, 1], or
    // kInvalidProbability if count != kChunkSamples / samples == nullptr (the
    // state is then left untouched). Allocation-free at steady state.
    float process_chunk(const float* samples, size_t count) noexcept;
    float process_chunk(const std::vector<float>& chunk) noexcept {
        return process_chunk(chunk.data(), chunk.size());
    }

    // RUNTIME tier. The streaming adapter: accepts ANY sample count, runs an
    // inference per accumulated kChunkSamples, and returns the latest
    // probability (the previous one while a chunk is still filling). Never
    // returns kInvalidProbability. This is what the audio pipeline calls.
    float feed(const float* samples, size_t count) noexcept;

    // Drops the recurrent state AND the partial chunk accumulator: the next
    // chunk is scored as the start of a fresh utterance. Call on every
    // discontinuity in the audio.
    void reset_state() noexcept;

    // ---- cross-thread seam (UI slider + readout) ----------------------------
    // Speech decision threshold in [0, 1]. Higher = less sensitive (fewer false
    // triggers on noise, more risk of clipping quiet speech onsets).
    void  set_threshold(float t) noexcept;
    float threshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }

    // Most recent probability produced by feed()/process_chunk(); 0 before the
    // first inference. For UI display only — the decision uses is_speech().
    float last_probability() const noexcept {
        return last_probability_.load(std::memory_order_relaxed);
    }

    // Count of inferences that failed and fell back to the last-good value.
    // Monotone; a nonzero value means the neural path is degraded.
    uint64_t inference_errors() const noexcept {
        return inference_errors_.load(std::memory_order_relaxed);
    }

    // Convenience decision over the last probability and the live threshold.
    bool is_speech() const noexcept { return last_probability() > threshold(); }

    static constexpr float kDefaultThreshold = 0.5f;

private:
    // Runs one inference over the chunk buffer the caller has already filled and
    // publishes the result. The single point where the RUNTIME-tier fallback
    // (count the failure, report the last-good probability) is applied.
    float run_current_chunk() noexcept;

    struct Impl;                       // holds the ORT env/session/tensors
    std::unique_ptr<Impl> impl_;       // PIMPL: keeps ONNXRuntime out of this header

    std::atomic<float>    threshold_{kDefaultThreshold};
    std::atomic<float>    last_probability_{0.0f};
    std::atomic<uint64_t> inference_errors_{0};
};

}  // namespace blackwell::vad
