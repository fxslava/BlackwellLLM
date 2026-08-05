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
// story. The one real argument FOR duplex was that acoustic echo cancellation
// wants a playback reference sample-aligned to the capture clock. AEC has since
// landed and the decision HELD: the reference is a separate loopback capture of
// the render endpoint, which gives that alignment without duplex -- and gives it
// post-mix and post-volume, which duplex would not have.
//
// THE CALLBACK IS A HARD REAL-TIME CONTEXT. `PlaybackPullFn` runs on the WASAPI
// thread. It must not allocate, lock, block, or throw: a missed deadline is an
// audible click, and clicks are the entire perceived quality of a speech
// feature. TTSDuplexBridge::PullForPlayback satisfies this (one lock-free ring
// read); anything else routed here must too.
//
// A raw function pointer + void* rather than std::function, matching the
// SpeechVadScoreFn seam in the bridge: no indirection through a type-erased
// heap object on the one thread in this process with a sub-millisecond budget.
//
// =============================================================================
// VOLUME LIVES HERE, AND THE ECHO CANCELLER NO LONGER HAS TO CARE
// =============================================================================
// The gain is applied in the device callback, AFTER the pull -- which is the
// last point before the samples become sound, and the only one where "what the
// speaker emits" is definitively known.
//
// That used to matter to the AEC. When the reference was tapped at
// TTSDuplexBridge::PullForPlayback it carried the PRE-gain signal while the room
// heard the post-gain one, so the canceller had to be handed the same value
// through AecCaptureFilter::SetReferenceGain() or it would re-converge every
// time the slider moved.
//
// That coupling is GONE. The far end is now a WASAPI loopback of the render
// endpoint, which is downstream of this gain and therefore already carries it.
// Volume is a local concern of this class again.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rt {

// Fills `dst` with EXACTLY `frames` mono f32 samples. Underrun is the callee's
// problem to handle (pad with silence) -- the device must always be handed a
// full buffer.
//
// RETURNS how many of those frames were REAL audio rather than padding. The
// device does not need the number to play the buffer; the diagnostics do. A
// starved sink and an idle one are both silence at the speaker, and telling
// them apart from outside is impossible -- only the callee knows whether it had
// nothing to say or had something and could not produce it in time. Returning
// the count is what makes underruns countable WITHOUT the callback doing any
// I/O of its own (see playback_stats() below).
using PlaybackPullFn = std::size_t (*)(void* user, float* dst, std::size_t frames);

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
    //
    // TWO WAYS TO NAME THE ENDPOINT, and `device_index` WINS when both are set.
    // `device_index` is a zero-based position in the list print_audio_devices()
    // shows; -1 means "not selected by index". `device_name` is an endpoint name
    // from enumerate_audio_devices(); empty means the system default. The
    // precedence and the reasoning behind it live in one place --
    // rt::detail::resolve_device_selection -- so that capture and playback
    // cannot drift apart on it.
    //
    // Neither selector is fatal. A name that does not resolve, an index past the
    // end of the list, or an endpoint that resolves but will not open all fall
    // back to the default and are reported through device_fallback() -- speech
    // that comes out of the wrong speaker beats speech that does not come out at
    // all.
    void start(int sample_rate, PlaybackPullFn pull, void* user,
               const std::string& device_name = std::string(), int device_index = -1);

    // Stops the device and JOINS the callback. After this returns, `pull` is
    // guaranteed not to be running -- which is what makes it safe to destroy
    // whatever `user` points at.
    void stop();

    // ---- hot swap (NOT the audio thread; the settings/UI thread) ------------
    // Moves playback to a different endpoint WITHOUT disturbing anything else in
    // the process. Closes the ma_device (joining its callback), reopens on the
    // new selection, and rewires the SAME pull target and sample rate -- so the
    // producer feeding it never learns that the device changed.
    //
    // WHY IT IS SAFE TO DO UNDER A LIVE PRODUCER. stop() joins the callback, so
    // while this runs nothing is pulling; the producer keeps writing into its
    // ring and the ring simply fills. Reopening drains it again. The audible
    // result is a gap the length of a device open (~10-50 ms on WASAPI), which
    // is a click, not a restart.
    //
    // WHAT IT COSTS ELSEWHERE. The echo canceller's learned impulse response
    // describes the OLD speaker and is wrong the instant this returns -- the
    // caller is responsible for resetting it (main.cpp does). Volume is
    // preserved across the swap; it lives on this object, not on the device.
    //
    // Returns false if the new endpoint could not be opened, in which case it
    // falls back the same way start() does rather than leaving playback dead.
    bool hot_reload(const std::string& device_name, int device_index);

    // The sample rate the device was opened at, so a hot swap can reuse it.
    int sample_rate() const noexcept { return sample_rate_; }

    // ---- volume (any thread, effective on the next callback) ----------------
    // Clamped to [0, 1]. Applied to every sample the callback hands the device.
    // A plain relaxed atomic: the callback reads it once per buffer, and a
    // change landing one buffer later than another thread thinks it did is
    // inaudible at 10 ms buffers -- ordering it against anything else would buy
    // nothing and cost a fence on the real-time thread.
    void set_volume(float v) noexcept;
    float volume() const noexcept;

    // ---- playback telemetry (any thread) ------------------------------------
    // WHY THIS IS COUNTERS AND NOT A LOG LINE. The obvious way to see whether
    // the sink is starving is to print the requested and available frame counts
    // from the device callback. Doing that CREATES the fault it is looking for:
    // the callback is a hard real-time context (see the header block), printf
    // takes a lock and can touch the heap, and a callback that misses its
    // deadline underruns -- so the instrument would manufacture its own
    // readings, most spectacularly under exactly the GPU load being blamed.
    //
    // So the callback only ever bumps relaxed atomics, and whoever wants a
    // report reads them from an ordinary thread.
    struct PlaybackStats {
        std::uint64_t callbacks = 0;        // device callbacks served
        std::uint64_t frames_requested = 0; // frames the device asked for
        std::uint64_t frames_served = 0;    // of those, real audio
        std::uint64_t starved_callbacks = 0;// callbacks that got less than asked
        std::uint32_t last_requested = 0;   // most recent buffer, for a live read
        std::uint32_t last_available = 0;   // real frames the pull had for it
    };
    PlaybackStats stats() const noexcept;
    void reset_stats() noexcept;

    bool running() const noexcept { return running_; }
    const std::string& backend_name() const noexcept { return backend_name_; }
    // The endpoint actually opened; "" means the system default.
    const std::string& device_name() const noexcept { return device_name_; }
    // True when a name was requested and something else was opened.
    bool device_fallback() const noexcept { return device_fallback_; }

private:
    struct Impl;                       // hides ma_device / ma_context
    std::unique_ptr<Impl> impl_;
    std::string backend_name_;
    std::string device_name_;
    bool device_fallback_ = false;
    bool running_ = false;
    // Remembered from start() so hot_reload() can reopen with the same contract
    // without making the caller restate it.
    int sample_rate_ = 0;
    PlaybackPullFn pull_ = nullptr;
    void* pull_user_ = nullptr;
};

}  // namespace rt
