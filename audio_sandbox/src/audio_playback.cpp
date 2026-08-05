// -----------------------------------------------------------------------------
// audio_playback.cpp — see audio_playback.h for the real-time contract.
//
// NOTE THE ABSENT MACRO. This TU includes miniaudio for its DECLARATIONS only;
// MINIAUDIO_IMPLEMENTATION is defined in exactly one place in this project
// (audio_sandbox/src/audio_capture.cpp) and defining it a second time is a wall
// of duplicate-symbol link errors.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
// miniaudio pulls in <windows.h> for WASAPI, and the min/max macros it brings
// break std::min/std::max in this TU. Same guard audio_capture.cpp uses.
#define NOMINMAX
#endif

#include "audio_playback.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "miniaudio.h"

#include "ma_device_select.h"

namespace rt {

struct AudioPlayback::Impl {
    // Explicit, for the same reason AudioCapture owns one: a device id is only
    // meaningful on the context that produced it (ma_device_select.h).
    ma_context context{};
    bool context_ready = false;
    ma_device device{};
    bool device_ready = false;

    PlaybackPullFn pull = nullptr;
    void*          user = nullptr;

    std::atomic<float> volume{1.0f};

    // ---- the playback time-lock (see audio_playback.h) ----------------------
    // `playing` is what the outside world reads. `silence_frames` is the
    // callback's own accumulator and is never read from anywhere else, but it is
    // atomic anyway: hot_reload() swaps the device underneath it, and the joining
    // callback's last write must not race the reset.
    std::atomic<bool> playing{false};
    std::atomic<std::uint32_t> silence_frames{0};
    // kAcousticTailMs converted to frames at the rate the device was opened at.
    // Written once in start(), before ma_device_start, so the callback only ever
    // reads a settled value.
    std::atomic<std::uint32_t> tail_frames{0};

    // Telemetry. All relaxed: these are counters nobody makes a decision on
    // mid-stream, and an acquire/release pair per callback would be ordering
    // work on the one thread that must do none. See audio_playback.h on why
    // this is atomics and not a printf.
    std::atomic<std::uint64_t> cb_count{0};
    std::atomic<std::uint64_t> frames_requested{0};
    std::atomic<std::uint64_t> frames_served{0};
    std::atomic<std::uint64_t> starved{0};
    std::atomic<std::uint32_t> last_requested{0};
    std::atomic<std::uint32_t> last_available{0};

    // THE real-time callback. Everything it may legally do is in
    // audio_playback.h. A static member rather than a free function because Impl
    // is private to AudioPlayback, and widening that just to let a file-scope
    // function reach it would leak the PIMPL for no benefit.
    static void on_data(ma_device* device, void* output, const void* /*input*/,
                        ma_uint32 frame_count) {
        auto* impl = static_cast<Impl*>(device->pUserData);
        auto* dst = static_cast<float*>(output);
        if (dst == nullptr) return;
        if (impl == nullptr || impl->pull == nullptr) {
            // Silence rather than stale buffer contents: an un-wired device must
            // be quiet, not play whatever WASAPI last left in the mapped region.
            for (ma_uint32 i = 0; i < frame_count; ++i) dst[i] = 0.0f;
            return;
        }
        const std::size_t served =
            impl->pull(impl->user, dst, static_cast<std::size_t>(frame_count));

        // Six relaxed increments. Deliberately no branch on a "diagnostics on"
        // flag: the branch would cost as much as the counters and would make the
        // measurement conditional on remembering to arm it.
        impl->cb_count.fetch_add(1, std::memory_order_relaxed);
        impl->frames_requested.fetch_add(frame_count, std::memory_order_relaxed);
        impl->frames_served.fetch_add(served, std::memory_order_relaxed);
        impl->last_requested.store(frame_count, std::memory_order_relaxed);
        impl->last_available.store(static_cast<std::uint32_t>(served),
                                   std::memory_order_relaxed);
        // "Starved" means the sink had SOMETHING and ran short, or had nothing
        // while the stream was live. An idle device serving 0 of 480 every
        // callback is not starvation, so the caller reads this together with
        // chunks_spoken -- the counter cannot tell idle from starved on its own
        // and does not pretend to.
        if (served < frame_count) impl->starved.fetch_add(1, std::memory_order_relaxed);

        // ---- the playback time-lock -----------------------------------------
        // `served` is the number of REAL samples the pull had, so served > 0 is
        // the exact statement "the ring was not empty" -- the condition the lock
        // is specified against, taken from the one place that cannot be wrong
        // about it.
        //
        // RELEASE on the store that RAISES the lock, so a consumer that observes
        // `playing` also observes everything the callback wrote before it. The
        // lowering store is relaxed on purpose: it publishes nothing, and it is
        // the one that runs every callback of an idle device.
        if (served > 0) {
            impl->silence_frames.store(0, std::memory_order_relaxed);
            impl->playing.store(true, std::memory_order_release);
        } else if (impl->playing.load(std::memory_order_relaxed)) {
            // Saturating, not wrapping: an idle device sits here forever, and a
            // counter that wraps past the tail would drop the lock back on for
            // one callback every few hours. Nothing else re-arms it.
            const std::uint32_t tail = impl->tail_frames.load(std::memory_order_relaxed);
            std::uint32_t n = impl->silence_frames.load(std::memory_order_relaxed);
            if (n < tail) {
                n = (tail - n > frame_count) ? n + frame_count : tail;
                impl->silence_frames.store(n, std::memory_order_relaxed);
            }
            if (n >= tail) impl->playing.store(false, std::memory_order_relaxed);
        }

        // ---- software volume -------------------------------------------------
        // Read ONCE per buffer, not per sample: re-reading would let the gain
        // change mid-buffer, which is a step discontinuity and therefore a click.
        // A whole buffer at the old value is inaudible; a click is not.
        //
        // No ramp, deliberately. At 10 ms buffers a slider drag arrives as a
        // staircase of steps small enough to be inaudible on speech, and a ramp
        // would need per-sample state on the one thread that must stay trivial.
        const float g = impl->volume.load(std::memory_order_relaxed);
        if (g != 1.0f) {
            for (ma_uint32 i = 0; i < frame_count; ++i) dst[i] *= g;
        }
    }
};

AudioPlayback::AudioPlayback() : impl_(std::make_unique<Impl>()) {}

AudioPlayback::~AudioPlayback() {
    stop();
}

void AudioPlayback::start(int sample_rate, PlaybackPullFn pull, void* user,
                          const std::string& device_name, int device_index) {
    if (running_) return;
    if (pull == nullptr) {
        throw std::runtime_error("AudioPlayback::start: null pull callback");
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("AudioPlayback::start: sample_rate must be positive");
    }

    device_name_.clear();
    device_fallback_ = false;

    // Remembered for hot_reload(), which reopens on the same terms.
    sample_rate_ = sample_rate;
    pull_ = pull;
    pull_user_ = user;

    // Wired BEFORE ma_device_start so the callback can never observe a
    // half-initialised Impl.
    impl_->pull = pull;
    impl_->user = user;

    // The time-lock's tail, in frames at THIS device's rate. Computed here and
    // not in the callback because the callback may not divide. A zero tail is
    // impossible for any sane rate, but the max() makes the lock's contract
    // ("false only AFTER a tail has elapsed") hold even if one appeared: with
    // tail_frames == 0 the very first silent callback would drop it.
    impl_->tail_frames.store(
        std::max(1u, static_cast<unsigned>(
                         (static_cast<std::int64_t>(sample_rate) * kAcousticTailMs) / 1000)),
        std::memory_order_relaxed);
    impl_->silence_frames.store(0, std::memory_order_relaxed);
    impl_->playing.store(false, std::memory_order_relaxed);

    const auto unwire = [this] {
        impl_->pull = nullptr;
        impl_->user = nullptr;
    };

    if (ma_context_init(nullptr, 0, nullptr, &impl_->context) != MA_SUCCESS) {
        unwire();
        throw std::runtime_error("AudioPlayback: ma_context_init failed (no audio backend?)");
    }
    impl_->context_ready = true;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 1;      // TTS is mono; miniaudio upmixes to the device
    cfg.sampleRate        = static_cast<ma_uint32>(sample_rate);
    cfg.dataCallback      = &Impl::on_data;
    cfg.pUserData         = impl_.get();

    ma_device_id selected{};
    bool have_selected = false;
    if (device_index >= 0 || !device_name.empty()) {
        std::string resolved;
        std::string reason;
        const detail::DeviceLookup r = detail::resolve_device_selection(
            &impl_->context, /*playback_list=*/true, device_index, device_name, &selected,
            &resolved, &reason);
        if (r == detail::DeviceLookup::Matched) {
            cfg.playback.pDeviceID = &selected;
            have_selected = true;
            device_name_ = resolved;
        } else {
            device_fallback_ = true;
            // See the capture-side twin: the message names the selector that was
            // actually consulted, so a stale name in settings.json cannot be
            // mistaken for the cause of an index failure.
            if (device_index >= 0) {
                std::fprintf(stderr,
                             "[audio] output device index %d: %s -- falling back to the "
                             "system default. The device list printed at startup is the "
                             "one these indices count into.\n",
                             device_index, reason.c_str());
            } else {
                std::fprintf(stderr,
                             "[audio] output device \"%s\": %s -- falling back to the "
                             "system default. Run with --list-audio-devices to see the "
                             "exact names.\n",
                             device_name.c_str(), reason.c_str());
            }
        }
    }

    ma_result res = ma_device_init(&impl_->context, &cfg, &impl_->device);
    if (res != MA_SUCCESS && have_selected) {
        // The name resolved but the endpoint will not open -- another app holds
        // it exclusively, or it was unplugged since enumeration. Retry on the
        // default rather than leaving the assistant mute.
        std::fprintf(stderr,
                     "[audio] output device \"%s\" failed to open -- falling back to the "
                     "system default.\n",
                     device_name_.c_str());
        cfg.playback.pDeviceID = nullptr;
        device_name_.clear();
        device_fallback_ = true;
        res = ma_device_init(&impl_->context, &cfg, &impl_->device);
    }
    if (res != MA_SUCCESS) {
        ma_context_uninit(&impl_->context);
        impl_->context_ready = false;
        unwire();
        throw std::runtime_error("AudioPlayback: ma_device_init failed (no output device?)");
    }
    impl_->device_ready = true;

    backend_name_ = ma_get_backend_name(impl_->device.pContext->backend);

    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        ma_device_uninit(&impl_->device);
        impl_->device_ready = false;
        ma_context_uninit(&impl_->context);
        impl_->context_ready = false;
        unwire();
        throw std::runtime_error("AudioPlayback: ma_device_start failed");
    }
    running_ = true;
}

void AudioPlayback::stop() {
    if (impl_ == nullptr) return;
    if (impl_->device_ready) {
        // ma_device_uninit stops the device and JOINS its thread, which is what
        // lets the caller destroy the pull target immediately afterwards.
        ma_device_uninit(&impl_->device);
        impl_->device_ready = false;
    }
    if (impl_->context_ready) {
        ma_context_uninit(&impl_->context);
        impl_->context_ready = false;
    }
    impl_->pull = nullptr;
    impl_->user = nullptr;
    // Dropped AFTER the callback is joined, so nothing can raise it again. A
    // stopped device emits no sound, so leaving the lock set would deafen the
    // VAD permanently -- the one failure mode of a suppressor is that it never
    // lifts, and a device swap is exactly when it would happen.
    impl_->playing.store(false, std::memory_order_release);
    impl_->silence_frames.store(0, std::memory_order_relaxed);
    running_ = false;
}

bool AudioPlayback::hot_reload(const std::string& device_name, int device_index) {
    // Nothing was ever started: there is no device to move, and start() is the
    // caller's job. Reporting false here rather than silently opening one keeps
    // "playback is off" from being resurrected by a settings change.
    if (pull_ == nullptr) return false;

    const float keep_volume = volume();
    const int rate = sample_rate_;
    PlaybackPullFn pull = pull_;
    void* user = pull_user_;

    stop();   // joins the callback; the producer's ring just fills meanwhile

    try {
        start(rate, pull, user, device_name, device_index);
    } catch (const std::exception& e) {
        // start() already falls back to the default endpoint internally; getting
        // here means even that failed, so playback is genuinely gone. Say so and
        // let the app continue mute rather than propagating into a settings save.
        std::fprintf(stderr, "[audio] hot reload failed to reopen playback: %s\n", e.what());
        return false;
    }
    // Restored AFTER the device exists: set_volume writes through to Impl, which
    // stop() left intact, but ordering it this way keeps the invariant obvious.
    set_volume(keep_volume);
    return true;
}

AudioPlayback::PlaybackStats AudioPlayback::stats() const noexcept {
    PlaybackStats s;
    if (impl_ == nullptr) return s;
    // Read one by one, so the snapshot is not atomic as a whole. That is fine
    // and worth stating: these are monotone counters sampled for a report, and
    // the cost of making the set consistent (a lock on the audio thread) is
    // exactly what this design exists to avoid.
    s.callbacks         = impl_->cb_count.load(std::memory_order_relaxed);
    s.frames_requested  = impl_->frames_requested.load(std::memory_order_relaxed);
    s.frames_served     = impl_->frames_served.load(std::memory_order_relaxed);
    s.starved_callbacks = impl_->starved.load(std::memory_order_relaxed);
    s.last_requested    = impl_->last_requested.load(std::memory_order_relaxed);
    s.last_available    = impl_->last_available.load(std::memory_order_relaxed);
    return s;
}

void AudioPlayback::reset_stats() noexcept {
    if (impl_ == nullptr) return;
    impl_->cb_count.store(0, std::memory_order_relaxed);
    impl_->frames_requested.store(0, std::memory_order_relaxed);
    impl_->frames_served.store(0, std::memory_order_relaxed);
    impl_->starved.store(0, std::memory_order_relaxed);
    impl_->last_requested.store(0, std::memory_order_relaxed);
    impl_->last_available.store(0, std::memory_order_relaxed);
}

bool AudioPlayback::is_playing_tts() const noexcept {
    if (impl_ == nullptr) return false;
    return impl_->playing.load(std::memory_order_acquire);
}

void AudioPlayback::set_volume(float v) noexcept {
    // Clamped HERE rather than trusted, because this value arrives from a
    // settings file and a CLI flag. A gain above 1 would clip a synthesiser
    // whose output is already normalised, and a negative one inverts the phase
    // of the AEC reference -- which turns the canceller into an echo ADDER.
    if (!(v >= 0.0f)) v = 0.0f;   // also catches NaN, which the comparison rejects
    impl_->volume.store(std::min(v, 1.0f), std::memory_order_relaxed);
}

float AudioPlayback::volume() const noexcept {
    return impl_->volume.load(std::memory_order_relaxed);
}

}  // namespace rt
