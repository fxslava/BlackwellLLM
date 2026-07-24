// -----------------------------------------------------------------------------
// audio_capture.cpp — the single translation unit that compiles miniaudio.
// -----------------------------------------------------------------------------
// miniaudio.h pulls in <windows.h> (WASAPI); block the min/max macros so
// std::min/std::max below still parse.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio_capture.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace rt {

// --- SampleRing --------------------------------------------------------------
void SampleRing::push(const float* data, size_t count) {
    std::lock_guard<std::mutex> lk(m_);
    q_.insert(q_.end(), data, data + count);
    if (q_.size() > max_) {
        // Drop oldest — never stall the audio thread waiting on the visualiser.
        q_.erase(q_.begin(), q_.begin() + (q_.size() - max_));
    }
}

size_t SampleRing::pop(float* dst, size_t count) {
    std::lock_guard<std::mutex> lk(m_);
    const size_t n = std::min(count, q_.size());
    for (size_t i = 0; i < n; ++i) dst[i] = q_[i];
    q_.erase(q_.begin(), q_.begin() + n);
    return n;
}

size_t SampleRing::size() const {
    std::lock_guard<std::mutex> lk(m_);
    return q_.size();
}

// --- AudioCapture ------------------------------------------------------------
struct AudioCapture::Impl {
    ma_device device{};
    bool device_inited = false;
};

// miniaudio hands us data already converted to the CONFIGURED format/rate/
// channels (f32 / 16 kHz / mono), so we can forward it verbatim to the ring.
static void data_callback(ma_device* device, void* /*output*/, const void* input,
                          ma_uint32 frame_count) {
    auto* ring = static_cast<SampleRing*>(device->pUserData);
    if (input != nullptr && ring != nullptr) {
        ring->push(static_cast<const float*>(input),
                   static_cast<size_t>(frame_count) * device->capture.channels);
    }
}

AudioCapture::AudioCapture()
    : impl_(std::make_unique<Impl>()),
      ring_(/*max_samples=*/16000 * 8) {}  // ~8 s of 16 kHz mono headroom

AudioCapture::~AudioCapture() { stop(); }

void AudioCapture::start(CaptureMode mode) {
    if (running_) return;

    const ma_device_type type =
        (mode == CaptureMode::Loopback) ? ma_device_type_loopback : ma_device_type_capture;
    ma_device_config cfg = ma_device_config_init(type);
    cfg.capture.format = ma_format_f32;   // CRITICAL: f32 samples
    cfg.capture.channels = 1;             // mono (miniaudio downmixes)
    cfg.sampleRate = 16000;               // CRITICAL: miniaudio resamples to 16 kHz
    cfg.dataCallback = data_callback;
    cfg.pUserData = &ring_;

    if (ma_device_init(nullptr, &cfg, &impl_->device) != MA_SUCCESS) {
        throw std::runtime_error(
            mode == CaptureMode::Loopback
                ? "ma_device_init failed for loopback (WASAPI loopback unavailable?)"
                : "ma_device_init failed for microphone capture");
    }
    impl_->device_inited = true;
    backend_name_ = ma_get_backend_name(impl_->device.pContext->backend);

    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        ma_device_uninit(&impl_->device);
        impl_->device_inited = false;
        throw std::runtime_error("ma_device_start failed");
    }
    running_ = true;
}

void AudioCapture::stop() {
    if (impl_ && impl_->device_inited) {
        ma_device_uninit(&impl_->device);  // stops + frees
        impl_->device_inited = false;
    }
    running_ = false;
}

}  // namespace rt
