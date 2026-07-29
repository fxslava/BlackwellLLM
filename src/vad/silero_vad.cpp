// -----------------------------------------------------------------------------
// SileroVAD implementation — the ONLY translation unit in the project that sees
// <onnxruntime_cxx_api.h>. Everything above it talks to the PIMPL'd header, so
// no consumer inherits ONNXRuntime's include path or its exception types.
//
// Graph bound here (Silero v5/v6 silero_vad.onnx):
//     inputs   input [1, 576] f32 | state [2, 1, 128] f32 | sr [1] i64
//     outputs  output [1, 1] f32  | stateN [2, 1, 128] f32
// Pre-v5 models exposed separate h/c tensors and will fail validate_signature().
//
// The 576 is kContextSamples(64) + kChunkSamples(512): the model is exported to
// receive the tail of the PREVIOUS chunk ahead of the new one. Measured on the
// LibriSpeech reference clip, omitting the prepend collapses fluent speech from
// p~1.00 to p~0.0009 — it fails silently, with plausible-looking low
// probabilities, which is why silero_vad_test.cpp asserts on real audio.
// -----------------------------------------------------------------------------
#include "silero_vad.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include <onnxruntime_cxx_api.h>

namespace blackwell::vad {
namespace {

// The graph's tensor names, in the binding order used for every Run().
constexpr std::array<const char*, 3> kInputNames{"input", "state", "sr"};
constexpr std::array<const char*, 2> kOutputNames{"output", "stateN"};

// UTF-8 std::string -> the wide path ORT wants on Windows. Goes through char8_t
// so a non-ASCII checkout path survives (the plain std::filesystem::path(
// std::string) ctor would reinterpret it in the active code page, and
// std::filesystem::u8path is deprecated in C++20).
std::filesystem::path to_path(const std::string& utf8) {
    return std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(utf8.c_str()), utf8.size()));
}

}  // namespace

// =============================================================================
// Impl — the ORT session plus every tensor it binds, all pre-allocated.
//
// STEADY-STATE ALLOCATION-FREE. The chunk buffer, both state buffers, the sample
// rate and the probability slot are members; the Ort::Value views onto them are
// built ONCE in the ctor. Run() writes into caller-owned output slots, so a step
// performs no heap traffic at all.
//
// STATE PING-PONG. Silero is an RNN: each step reads `state` and writes `stateN`,
// and the two may not alias. So the state lives in TWO buffers and the bindings
// are pre-built once per phase — phase p reads state[p] and writes state[1 - p].
// A step is then "Run the phase, flip the phase", with no copy-back and no
// per-call tensor construction. (The chunk / sr / probability views are repeated
// in both phases; they are non-owning wrappers over the same memory, so this
// costs two extra 40-byte handles and keeps the step down to two lines.)
// =============================================================================
struct SileroVAD::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "blackwell_vad"};
    Ort::SessionOptions options;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> session;

    // Bound buffers (owned here; the Ort::Values below are views onto them).
    //
    // window = [ context (64) | new chunk (512) ]. The model reads all 576; the
    // caller only ever writes the chunk region, which is what chunk() returns.
    // After each step the tail of the chunk becomes the next step's context.
    static constexpr size_t kWindow = kContextSamples + kChunkSamples;
    std::array<float, kWindow> window{};
    std::array<std::array<float, kStateElems>, 2> state{};
    int64_t sample_rate = kSampleRate;
    std::array<float, 1> probability{};

    float* chunk() noexcept { return window.data() + kContextSamples; }

    // in_[p] / out_[p] are the complete binding for phase p.
    std::array<std::array<Ort::Value, 3>, 2> in_{
        std::array<Ort::Value, 3>{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}},
        std::array<Ort::Value, 3>{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}}};
    std::array<std::array<Ort::Value, 2>, 2> out_{
        std::array<Ort::Value, 2>{Ort::Value{nullptr}, Ort::Value{nullptr}},
        std::array<Ort::Value, 2>{Ort::Value{nullptr}, Ort::Value{nullptr}}};

    size_t phase = 0;  // which buffer holds the LIVE recurrent state
    size_t fill  = 0;  // partial-chunk accumulator for feed(); [0, kChunkSamples)

    explicit Impl(const std::string& model_path) {
        // One core, sequential. The caller is a real-time audio worker: an ORT
        // thread pool would spin-wait on other cores for a 2.3 MB model whose
        // inference is shorter than the pool's own wake-up latency.
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const std::filesystem::path path = to_path(model_path);
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("SileroVAD: model not found: " + model_path);
        }
        session = std::make_unique<Ort::Session>(env, path.c_str(), options);
        validate_signature(model_path);

        const std::array<int64_t, 2> window_shape{1, static_cast<int64_t>(kWindow)};
        const std::array<int64_t, 3> state_shape{2, 1, 128};
        const std::array<int64_t, 1> sr_shape{1};
        const std::array<int64_t, 2> prob_shape{1, 1};

        for (size_t p = 0; p < 2; ++p) {
            in_[p][0] = Ort::Value::CreateTensor<float>(
                memory_info, window.data(), window.size(), window_shape.data(),
                window_shape.size());
            in_[p][1] = Ort::Value::CreateTensor<float>(
                memory_info, state[p].data(), kStateElems, state_shape.data(), state_shape.size());
            in_[p][2] = Ort::Value::CreateTensor<int64_t>(
                memory_info, &sample_rate, 1, sr_shape.data(), sr_shape.size());

            out_[p][0] = Ort::Value::CreateTensor<float>(
                memory_info, probability.data(), probability.size(), prob_shape.data(),
                prob_shape.size());
            out_[p][1] = Ort::Value::CreateTensor<float>(
                memory_info, state[1 - p].data(), kStateElems, state_shape.data(),
                state_shape.size());
        }
    }

    // Fails loudly on any model whose graph is not the one bound above — a
    // silently mis-bound tensor would produce plausible-looking garbage
    // probabilities, which is far worse than not loading at all.
    void validate_signature(const std::string& model_path) const {
        Ort::AllocatorWithDefaultOptions alloc;
        const size_t n_in  = session->GetInputCount();
        const size_t n_out = session->GetOutputCount();
        if (n_in != kInputNames.size() || n_out != kOutputNames.size()) {
            throw std::runtime_error(
                "SileroVAD: unexpected graph arity in " + model_path + " (got " +
                std::to_string(n_in) + " inputs / " + std::to_string(n_out) +
                " outputs, expected 3/2). Silero v5+ is required; pre-v5 models "
                "expose separate h/c state tensors and are not supported.");
        }
        for (size_t i = 0; i < n_in; ++i) {
            const Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, alloc);
            if (std::string(name.get()) != kInputNames[i]) {
                throw std::runtime_error(
                    "SileroVAD: input " + std::to_string(i) + " of " + model_path + " is '" +
                    name.get() + "', expected '" + kInputNames[i] + "'.");
            }
        }
        for (size_t i = 0; i < n_out; ++i) {
            const Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, alloc);
            if (std::string(name.get()) != kOutputNames[i]) {
                throw std::runtime_error(
                    "SileroVAD: output " + std::to_string(i) + " of " + model_path + " is '" +
                    name.get() + "', expected '" + kOutputNames[i] + "'.");
            }
        }
    }

    // One inference over `window`. Throws Ort::Exception on failure (the caller
    // converts that to the last-good value).
    float run() {
        session->Run(Ort::RunOptions{nullptr}, kInputNames.data(), in_[phase].data(),
                     in_[phase].size(), kOutputNames.data(), out_[phase].data(),
                     out_[phase].size());
        // Carry the tail of the chunk we just consumed into the context slot for
        // the next step. Source [kChunkSamples, kWindow) and destination
        // [0, kContextSamples) cannot overlap because kChunkSamples > kContextSamples.
        std::copy_n(window.data() + kChunkSamples, kContextSamples, window.data());
        phase = 1 - phase;  // the freshly written buffer is now the live state
        return probability[0];
    }

    void reset() noexcept {
        state[0].fill(0.0f);
        state[1].fill(0.0f);
        window.fill(0.0f);  // drops the carried context along with the chunk
        fill  = 0;
        phase = 0;
    }
};

// =============================================================================
// SileroVAD
// =============================================================================
SileroVAD::SileroVAD(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

SileroVAD::~SileroVAD() = default;

// The atomic members are not themselves movable, so the moves are hand-written:
// the session moves with the PIMPL, the published scalars are transferred by
// value. Relaxed ordering is right here — a move is not a concurrent operation
// (the source must already be quiescent), so these loads/stores only need to be
// atomic, not ordered against anything.
SileroVAD::SileroVAD(SileroVAD&& other) noexcept
    : impl_(std::move(other.impl_)),
      threshold_(other.threshold_.load(std::memory_order_relaxed)),
      last_probability_(other.last_probability_.load(std::memory_order_relaxed)),
      inference_errors_(other.inference_errors_.load(std::memory_order_relaxed)) {}

SileroVAD& SileroVAD::operator=(SileroVAD&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        threshold_.store(other.threshold_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        last_probability_.store(other.last_probability_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        inference_errors_.store(other.inference_errors_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
    }
    return *this;
}

float SileroVAD::process_chunk(const float* samples, size_t count) noexcept {
    if (samples == nullptr || count != kChunkSamples) return kInvalidProbability;
    std::copy_n(samples, kChunkSamples, impl_->chunk());
    return run_current_chunk();
}

// Runs the inference over impl_->chunk (already filled) and publishes the
// result. RUNTIME tier: an ORT failure is counted and reported as the last-good
// probability rather than unwinding the audio thread.
float SileroVAD::run_current_chunk() noexcept {
    try {
        const float p = impl_->run();
        last_probability_.store(p, std::memory_order_relaxed);
        return p;
    } catch (const std::exception&) {  // Ort::Exception derives from std::exception
        inference_errors_.fetch_add(1, std::memory_order_relaxed);
        return last_probability_.load(std::memory_order_relaxed);
    }
}

float SileroVAD::feed(const float* samples, size_t count) noexcept {
    if (samples == nullptr) return last_probability_.load(std::memory_order_relaxed);

    size_t offset = 0;
    while (offset < count) {
        const size_t take = std::min(kChunkSamples - impl_->fill, count - offset);
        std::copy_n(samples + offset, take, impl_->chunk() + impl_->fill);
        impl_->fill += take;
        offset += take;
        if (impl_->fill == kChunkSamples) {
            impl_->fill = 0;
            (void)run_current_chunk();
        }
    }
    // Sticky: while a chunk is still filling, the previous probability stands.
    return last_probability_.load(std::memory_order_relaxed);
}

void SileroVAD::reset_state() noexcept {
    impl_->reset();
    last_probability_.store(0.0f, std::memory_order_relaxed);
}

void SileroVAD::set_threshold(float t) noexcept {
    threshold_.store(std::clamp(t, 0.0f, 1.0f), std::memory_order_relaxed);
}

}  // namespace blackwell::vad
