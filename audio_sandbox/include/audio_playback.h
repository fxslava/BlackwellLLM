#pragma once
// -----------------------------------------------------------------------------
// audio_playback.h — real-time f32 playback via miniaudio. The output half of
// the duplex path; audio_capture.h is the input half and this deliberately
// mirrors its shape (PIMPL over ma_device, throws on INIT, one callback thread).
//
// SEPARATE DEVICE, NOT DUPLEX. Rejected ma_device_type_duplex for the reasons in
// docs/TTS_INTEGRATION_AUDIT.md §3.2: duplex would force a rewrite of
// AudioCapture (shared with audio_realtime and untouched by design) and is
// incompatible with ma_device_type_loopback, which is half of this app's capture
// story. The one real argument FOR duplex is that acoustic echo cancellation
// wants a playback reference sample-aligned to the capture clock -- when AEC
// lands, that decision has to be revisited, and this comment is the pointer.
//
// THE CALLBACK IS A HARD REAL-TIME CONTEXT. `PlaybackPullFn` runs on the WASAPI
// thread. It must not allocate, lock, block, or throw: a missed deadline is an
// audible click, and clicks are the entire perceived quality of a speech
// feature. TTSDuplexBridge::PullForPlayback satisfies this (lock-free ring read
// plus one ring write); anything else routed here must too.
//
// A raw function pointer + void* rather than std::function, matching the
// SpeechVadScoreFn seam in the bridge: no indirection through a type-erased
// heap object on the one thread in this process with a sub-millisecond budget.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <memory>
#include <string>

namespace rt {

// Fills `dst` with EXACTLY `frames` mono f32 samples. Underrun is the callee's
// problem to handle (pad with silence) -- the device must always be handed a
// full buffer.
using PlaybackPullFn = void (*)(void* user, float* dst, std::size_t frames);

class AudioPlayback {
public:
    AudioPlayback();
    ~AudioPlayback();
    AudioPlayback(const AudioPlayback&) = delete;
    AudioPlayback& operator=(const AudioPlayback&) = delete;

    // Opens + starts a mono f32 playback device at `sample_rate` (24000 for
    // F5-TTS). miniaudio converts to whatever the endpoint natively wants, so
    // the rate here is the rate the PULL sees -- no resampler on our side.
    // Throws std::runtime_error on failure (INIT tier): the caller disables TTS
    // and the app runs on, which is the posture the rest of this app takes for
    // every optional subsystem.
    void start(int sample_rate, PlaybackPullFn pull, void* user);

    // Stops the device and JOINS the callback. After this returns, `pull` is
    // guaranteed not to be running -- which is what makes it safe to destroy
    // whatever `user` points at.
    void stop();

    bool running() const noexcept { return running_; }
    const std::string& backend_name() const noexcept { return backend_name_; }

private:
    struct Impl;                       // hides ma_device / ma_context
    std::unique_ptr<Impl> impl_;
    std::string backend_name_;
    bool running_ = false;
};

}  // namespace rt
