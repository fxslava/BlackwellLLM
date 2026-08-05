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
//
// DEVICE SELECTION. start() takes an optional endpoint NAME (see
// audio_devices.h for how to list them). It is a name and not an id because an
// id is a backend-specific blob that cannot be written into a settings file, and
// because the name is what a user can actually recognise. A name that does not
// resolve is NOT fatal: the system default is opened instead and the mismatch is
// reported through device_name(), because a laptop whose USB headset is unplugged
// must still hear the user rather than fail to start.
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
    //
    // TWO WAYS TO NAME THE ENDPOINT, and `device_index` WINS when both are set.
    // `device_index` is a zero-based position in the list print_audio_devices()
    // shows; -1 means "not selected by index". `device_name` is an endpoint name
    // from enumerate_audio_devices(); empty means the system default. The
    // precedence lives in rt::detail::resolve_device_selection, shared with
    // AudioPlayback so the two cannot disagree about it.
    //
    // Neither selector is fatal: an unresolvable name, an out-of-range index, or
    // an endpoint that will not open falls back to the default rather than
    // throwing -- see the header block.
    //
    // IN LOOPBACK MODE BOTH SELECTORS ADDRESS THE PLAYBACK LIST, since loopback
    // taps a speaker rather than a microphone. So `input_device_index` is an
    // index into the PLAYBACK listing when loopback_capture is on -- the same
    // trap the name form has always had, now with a number that looks even more
    // like it must mean the capture list.
    void start(CaptureMode mode, const std::string& device_name = std::string(),
               int device_index = -1);
    void stop();

    // ---- hot swap (NOT the audio thread) ------------------------------------
    // The capture twin of AudioPlayback::hot_reload, and safe for the same
    // reason: stop() joins the callback, and the SampleRing lives on this object
    // rather than on the device -- so every consumer holding ring() keeps a
    // valid reference across the swap and simply sees a brief gap in samples.
    //
    // `mode` is a parameter because loopback-vs-microphone is itself a settings
    // change a user can make, and it is the same device reopen either way.
    bool hot_reload(CaptureMode mode, const std::string& device_name, int device_index);

    // ---- input gain + level (any thread) ------------------------------------
    // Applied in the capture callback, BEFORE the ring -- so the VAD, the AEC
    // and the ASR all see the adjusted signal, and the meter below reports what
    // they will actually get. Clamped to [0, 1]; relaxed atomic, read once per
    // callback, exactly like the playback gain.
    void set_input_gain(float g) noexcept;
    float input_gain() const noexcept;

    // Peak absolute amplitude of the most recent capture callback, POST-gain,
    // in [0, 1]. For a UI meter: it is a decaying peak rather than an RMS
    // because a meter has to move on a syllable, and RMS over a 10 ms block
    // barely does.
    float input_level() const noexcept;

    SampleRing& ring() { return ring_; }
    const std::string& backend_name() const { return backend_name_; }

    // The endpoint actually opened. Compare against what was requested to detect
    // a silent fallback -- "" means the system default was used.
    const std::string& device_name() const { return device_name_; }
    // True when a name was requested and something else was opened. The caller
    // decides how loudly to say so; this class only records it.
    bool device_fallback() const { return device_fallback_; }

private:
    struct Impl;                     // hides ma_device / ma_context
    std::unique_ptr<Impl> impl_;
    SampleRing ring_;
    std::string backend_name_;
    std::string device_name_;
    CaptureMode mode_ = CaptureMode::Microphone;   // remembered for hot_reload
    bool device_fallback_ = false;
    bool running_ = false;
};

}  // namespace rt
