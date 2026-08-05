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
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_devices.h"
#include "ma_device_select.h"

namespace rt {

// --- device enumeration -------------------------------------------------------
// Lives in THIS TU because this is the one place miniaudio's implementation is
// compiled, and it is linked into every target that opens an audio device -- so
// audio_playback.cpp can reach the shared resolver without a fourth source file
// appearing in three CMake source lists.
std::vector<AudioDeviceInfo> enumerate_audio_devices(AudioDeviceKind kind) {
    std::vector<AudioDeviceInfo> out;

    // A context of its own, torn down before returning: this is a query, and
    // holding a backend open afterwards would make a diagnostic call have a
    // lifetime. The device classes own their own contexts for their own devices.
    ma_context ctx{};
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) return out;

    ma_device_info* playback = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info* capture = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&ctx, &playback, &playback_count, &capture, &capture_count) ==
        MA_SUCCESS) {
        const bool want_playback = (kind == AudioDeviceKind::Playback);
        const ma_device_info* list = want_playback ? playback : capture;
        const ma_uint32 count = want_playback ? playback_count : capture_count;
        for (ma_uint32 i = 0; i < count && list != nullptr; ++i) {
            AudioDeviceInfo info;
            info.name = list[i].name;
            info.is_default = list[i].isDefault != 0;
            out.push_back(std::move(info));
        }
    }
    ma_context_uninit(&ctx);
    return out;
}

void print_audio_devices() {
    const auto show = [](const char* title, const char* index_key, AudioDeviceKind kind) {
        const std::vector<AudioDeviceInfo> devices = enumerate_audio_devices(kind);
        std::printf("[audio] %s (%s):\n", title, index_key);
        if (devices.empty()) {
            std::printf("  (none -- backend unavailable)\n");
            return;
        }
        for (std::size_t i = 0; i < devices.size(); ++i) {
            // The [N] is the VECTOR POSITION, which is what the index setting
            // selects with -- see audio_devices.h on why it is derived here and
            // never stored.
            //
            // Names print VERBATIM and unquoted-but-delimited: this is the
            // string a user copies into settings.json, and a trailing space
            // swallowed by the terminal is a name that then fails to resolve.
            std::printf("  [%zu] %s \"%s\"\n", i, devices[i].is_default ? "*" : " ",
                        devices[i].name.c_str());
        }
    };
    show("Playback devices", "output_device_index / output_device_name",
         AudioDeviceKind::Playback);
    show("Capture devices", "input_device_index / input_device_name", AudioDeviceKind::Capture);
    // Spelling out the default twice over: -1 and "" are the same instruction
    // given to the two different selectors, and a user who sets one but not the
    // other has to be able to tell that they have not accidentally set both.
    std::printf("[audio] * = system default. index -1 and an empty name both mean "
                "\"follow the default\";\n"
                "        an index >= 0 wins over a name.\n");
}

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
    // An EXPLICIT context, not the implicit one ma_device_init(NULL, ...)
    // creates. Device ids are only valid on the context that enumerated them
    // (see ma_device_select.h), so selection is impossible without owning one.
    ma_context context{};
    bool context_inited = false;
    ma_device device{};
    bool device_inited = false;

    // Gain and meter. Relaxed atomics for the same reason the playback gain is:
    // the capture callback is a real-time context and must not order anything.
    std::atomic<float> gain{1.0f};
    std::atomic<float> level{0.0f};

    // The destination ring, reached THROUGH Impl so pUserData stays a single
    // pointer. It belongs to the AudioCapture object and outlives every device
    // opened on it, which is what makes hot_reload() safe for consumers holding
    // ring().
    SampleRing* ring = nullptr;

    // Scratch for the gain pass. Sized once at start() to the largest block the
    // device will hand us; the callback must not allocate.
    std::vector<float> scaled;

    // A static member rather than a free function for the same reason
    // AudioPlayback::Impl::on_data is one: Impl is private, and widening it just
    // so a file-scope function could reach these fields would leak the PIMPL.
    static void on_data(ma_device* device, void* output, const void* input,
                        ma_uint32 frame_count);
};

// miniaudio hands us data already converted to the CONFIGURED format/rate/
// channels (f32 / 16 kHz / mono), so we can forward it verbatim to the ring.
// A REAL-TIME CONTEXT, like the playback callback: no allocation, no locks, no
// throwing. The gain pass writes into a buffer sized at start(), and the meter
// is one relaxed store.
void AudioCapture::Impl::on_data(ma_device* device, void* /*output*/, const void* input,
                                 ma_uint32 frame_count) {
    auto* impl = static_cast<AudioCapture::Impl*>(device->pUserData);
    if (input == nullptr || impl == nullptr || impl->ring == nullptr) return;

    const auto* src = static_cast<const float*>(input);
    const std::size_t count = static_cast<std::size_t>(frame_count) * device->capture.channels;

    // Read ONCE per block, not per sample: a gain that changes mid-block is a
    // step discontinuity, i.e. a click, for the same reason playback reads its
    // volume once per buffer.
    const float g = impl->gain.load(std::memory_order_relaxed);

    // Peak, not RMS -- see input_level(). Measured POST-gain so the meter shows
    // what the VAD will actually be handed.
    float peak = 0.0f;
    if (g == 1.0f) {
        for (std::size_t i = 0; i < count; ++i) {
            const float a = src[i] < 0.0f ? -src[i] : src[i];
            if (a > peak) peak = a;
        }
        impl->ring->push(src, count);
    } else if (count <= impl->scaled.size()) {
        for (std::size_t i = 0; i < count; ++i) {
            const float v = src[i] * g;
            impl->scaled[i] = v;
            const float a = v < 0.0f ? -v : v;
            if (a > peak) peak = a;
        }
        impl->ring->push(impl->scaled.data(), count);
    } else {
        // The device handed us a bigger block than start() provisioned for.
        // Forward it UNSCALED rather than dropping it or allocating here: a
        // wrong-gain block is a blemish, a dropped one is a hole in the ASR's
        // input and an allocation is a deadline miss.
        for (std::size_t i = 0; i < count; ++i) {
            const float a = src[i] < 0.0f ? -src[i] : src[i];
            if (a > peak) peak = a;
        }
        impl->ring->push(src, count);
    }
    impl->level.store(peak, std::memory_order_relaxed);
}

AudioCapture::AudioCapture()
    : impl_(std::make_unique<Impl>()),
      ring_(/*max_samples=*/16000 * 8) {}  // ~8 s of 16 kHz mono headroom

AudioCapture::~AudioCapture() { stop(); }

void AudioCapture::start(CaptureMode mode, const std::string& device_name, int device_index) {
    if (running_) return;
    mode_ = mode;   // remembered so hot_reload() can reopen in the same mode

    device_name_.clear();
    device_fallback_ = false;

    if (ma_context_init(nullptr, 0, nullptr, &impl_->context) != MA_SUCCESS) {
        throw std::runtime_error("AudioCapture: ma_context_init failed (no audio backend?)");
    }
    impl_->context_inited = true;

    const ma_device_type type =
        (mode == CaptureMode::Loopback) ? ma_device_type_loopback : ma_device_type_capture;
    ma_device_config cfg = ma_device_config_init(type);
    cfg.capture.format = ma_format_f32;   // CRITICAL: f32 samples
    cfg.capture.channels = 1;             // mono (miniaudio downmixes)
    cfg.sampleRate = 16000;               // CRITICAL: miniaudio resamples to 16 kHz
    cfg.dataCallback = &Impl::on_data;
    // Impl, not the ring: the callback needs the gain and the meter too, and one
    // pUserData pointer has to carry all three.
    impl_->ring = &ring_;
    // 4096 frames is far above any WASAPI shared-mode period at 16 kHz; the
    // callback falls back to unscaled rather than allocating if it is ever
    // exceeded.
    impl_->scaled.assign(4096, 0.0f);
    cfg.pUserData = impl_.get();

    // ---- explicit endpoint selection ----------------------------------------
    // In Loopback mode the id names a PLAYBACK endpoint: WASAPI loopback taps
    // the output of a speaker, so searching the capture list would find nothing
    // and the fallback would look like a missing microphone.
    ma_device_id selected{};
    bool have_selected = false;
    if (device_index >= 0 || !device_name.empty()) {
        std::string resolved;
        std::string reason;
        const detail::DeviceLookup r = detail::resolve_device_selection(
            &impl_->context, /*playback_list=*/mode == CaptureMode::Loopback, device_index,
            device_name, &selected, &resolved, &reason);
        if (r == detail::DeviceLookup::Matched) {
            cfg.capture.pDeviceID = &selected;
            have_selected = true;
            device_name_ = resolved;
        } else {
            device_fallback_ = true;
            // Echoes back WHICH selector was actually used, not both: a user who
            // set an index sees the index in the message, so they know the name
            // sitting in their settings file was not what failed.
            if (device_index >= 0) {
                std::fprintf(stderr,
                             "[audio] input device index %d: %s -- falling back to the "
                             "system default. The device list printed at startup is the "
                             "one these indices count into.\n",
                             device_index, reason.c_str());
            } else {
                std::fprintf(stderr,
                             "[audio] input device \"%s\": %s -- falling back to the system "
                             "default. Run with --list-audio-devices to see the exact "
                             "names.\n",
                             device_name.c_str(), reason.c_str());
            }
        }
    }

    ma_result res = ma_device_init(&impl_->context, &cfg, &impl_->device);
    if (res != MA_SUCCESS && have_selected) {
        // Resolved but would not OPEN: exclusive-mode contention, or a device
        // unplugged between enumeration and init. Distinct from "not found" and
        // worth its own line -- the name is right, the hardware is not available.
        std::fprintf(stderr,
                     "[audio] input device \"%s\" failed to open -- falling back to the "
                     "system default.\n",
                     device_name_.c_str());
        cfg.capture.pDeviceID = nullptr;
        device_name_.clear();
        device_fallback_ = true;
        res = ma_device_init(&impl_->context, &cfg, &impl_->device);
    }
    if (res != MA_SUCCESS) {
        ma_context_uninit(&impl_->context);
        impl_->context_inited = false;
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
        ma_context_uninit(&impl_->context);
        impl_->context_inited = false;
        throw std::runtime_error("ma_device_start failed");
    }
    running_ = true;
}

void AudioCapture::stop() {
    if (impl_ && impl_->device_inited) {
        ma_device_uninit(&impl_->device);  // stops + frees
        impl_->device_inited = false;
    }
    // The context outlives the device it opened, and only by one line: it holds
    // the backend the device's id came from.
    if (impl_ && impl_->context_inited) {
        ma_context_uninit(&impl_->context);
        impl_->context_inited = false;
    }
    running_ = false;
}

bool AudioCapture::hot_reload(CaptureMode mode, const std::string& device_name,
                              int device_index) {
    const float keep_gain = input_gain();
    stop();          // joins the callback; ring_ and every reference to it survive
    try {
        start(mode, device_name, device_index);
    } catch (const std::exception& e) {
        // start() already retries on the default endpoint, so reaching here means
        // capture is genuinely gone. The app keeps running deaf rather than
        // taking a settings save down with it.
        std::fprintf(stderr, "[audio] hot reload failed to reopen capture: %s\n", e.what());
        return false;
    }
    set_input_gain(keep_gain);
    return true;
}

void AudioCapture::set_input_gain(float g) noexcept {
    // Clamped here rather than trusted: this arrives from a settings file and a
    // web slider. NaN-safe -- `!(g >= 0)` catches it where `g < 0` does not, and
    // a NaN gain would poison every sample the VAD and the ASR see.
    if (!(g >= 0.0f)) g = 0.0f;
    impl_->gain.store(g > 1.0f ? 1.0f : g, std::memory_order_relaxed);
}

float AudioCapture::input_gain() const noexcept {
    return impl_->gain.load(std::memory_order_relaxed);
}

float AudioCapture::input_level() const noexcept {
    return impl_->level.load(std::memory_order_relaxed);
}

}  // namespace rt
