#pragma once
// -----------------------------------------------------------------------------
// audio_capture.h — real-time f32 capture (mic or system loopback) via miniaudio.
//
// STAGE 1 of the pipeline: the miniaudio callback thread is the SOLE writer into
// SampleRing; the DSP worker is the sole reader. miniaudio is configured for
// 16 kHz / mono / f32 and does the resampling + downmix internally, so the ring
// already carries exactly what the Whisper DSP expects.
//
// miniaudio types are hidden behind a PIMPL so <miniaudio.h> (a very large TU)
// is compiled in exactly one place (audio_capture.cpp).
// -----------------------------------------------------------------------------
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

namespace rt {

enum class CaptureMode { Microphone, Loopback };

// Thread-safe FIFO of f32 samples shared between the audio callback (writer) and
// the DSP worker (reader). Mutex-guarded std::deque (per the plan). If the reader
// falls behind past `max_samples`, the OLDEST samples are dropped — a live
// visualiser must never block the audio callback.
class SampleRing {
public:
    explicit SampleRing(size_t max_samples) : max_(max_samples) {}

    void push(const float* data, size_t count);       // called from audio callback
    size_t pop(float* dst, size_t count);             // returns #samples written
    size_t size() const;

private:
    mutable std::mutex m_;
    std::deque<float> q_;
    size_t max_;
};

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();
    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    // Opens + starts a 16 kHz / mono / f32 capture. Throws std::runtime_error on
    // failure (INIT tier). `mode` selects mic vs. system loopback (WASAPI).
    void start(CaptureMode mode);
    void stop();

    SampleRing& ring() { return ring_; }
    const std::string& backend_name() const { return backend_name_; }

private:
    struct Impl;                     // hides ma_device / ma_context
    std::unique_ptr<Impl> impl_;
    SampleRing ring_;
    std::string backend_name_;
    bool running_ = false;
};

}  // namespace rt
