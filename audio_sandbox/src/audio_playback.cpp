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

#include <stdexcept>
#include <string>

#include "miniaudio.h"

namespace rt {

struct AudioPlayback::Impl {
    ma_device device{};
    bool device_ready = false;

    PlaybackPullFn pull = nullptr;
    void*          user = nullptr;

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
        impl->pull(impl->user, dst, static_cast<std::size_t>(frame_count));
    }
};

AudioPlayback::AudioPlayback() : impl_(std::make_unique<Impl>()) {}

AudioPlayback::~AudioPlayback() {
    stop();
}

void AudioPlayback::start(int sample_rate, PlaybackPullFn pull, void* user) {
    if (running_) return;
    if (pull == nullptr) {
        throw std::runtime_error("AudioPlayback::start: null pull callback");
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("AudioPlayback::start: sample_rate must be positive");
    }

    // Wired BEFORE ma_device_start so the callback can never observe a
    // half-initialised Impl.
    impl_->pull = pull;
    impl_->user = user;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 1;      // TTS is mono; miniaudio upmixes to the device
    cfg.sampleRate        = static_cast<ma_uint32>(sample_rate);
    cfg.dataCallback      = &Impl::on_data;
    cfg.pUserData         = impl_.get();

    if (ma_device_init(nullptr, &cfg, &impl_->device) != MA_SUCCESS) {
        impl_->pull = nullptr;
        impl_->user = nullptr;
        throw std::runtime_error("AudioPlayback: ma_device_init failed (no output device?)");
    }
    impl_->device_ready = true;

    backend_name_ = ma_get_backend_name(impl_->device.pContext->backend);

    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        ma_device_uninit(&impl_->device);
        impl_->device_ready = false;
        impl_->pull = nullptr;
        impl_->user = nullptr;
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
    impl_->pull = nullptr;
    impl_->user = nullptr;
    running_ = false;
}

}  // namespace rt
