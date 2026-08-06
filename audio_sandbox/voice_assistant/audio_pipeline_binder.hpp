#pragma once
// -----------------------------------------------------------------------------
// audio_pipeline_binder.hpp — everything between the microphone and the speaker,
// owned in one place and torn down in one order.
//
// WHAT IT BINDS TOGETHER
//   CAPTURE SIDE   WASAPI microphone -> input gain -> RealTimeDSP worker ->
//                  [ the PCM tap ] -> the active ISpeechMode.
//   FAR END        a SECOND WASAPI capture in LOOPBACK mode on the RENDER
//                  endpoint, drained by the same tap into the canceller's ring.
//   SUBTRACTION    AecCaptureFilter (AEC3 when the package was found, the
//                  built-in partitioned-block filter otherwise) sits in the tap
//                  between the two.
//   OUTPUT SIDE    TtsRuntime: chunker -> F5 synthesis -> the speaker ring.
//
// THE ORDERING IS THE POINT, and it is why this is a class rather than a set of
// free functions. Member DECLARATION order is construction order and reverse
// destruction order (CLAUDE.md extension pattern #3, applied to an object that
// owns two audio devices and a filter holding a reference into a ring). Two of
// those relationships will crash on shutdown if reversed and no test would catch
// it, so each declaration site below states why it is where it is.
//
// TWO PHASES, NOT ONE, and that is a behavioural requirement rather than tidiness:
//   ctor                  the CAPTURE half. Runs BEFORE the engine, because
//                         WhisperDSP has to exist for bring_up_real_engine to
//                         borrow, and because failing on a bad mel geometry must
//                         happen before a 5.3 GB load.
//   start_speech_output() the OUTPUT half. Runs AFTER the engine, so a TTS
//                         failure cannot delay the thing the app is actually for,
//                         and BEFORE the UI, so the first reply can be spoken.
//   bind_speech_mode()    installs the tap and starts the DSP worker. Runs once
//                         the mode exists, because the tap calls into it.
//
// ERROR TIER: mixed, deliberately. The ctor is INIT (a missing mel_filters.bin or
// a dead capture device is fatal -- there is no assistant without a microphone).
// start_speech_output() is NON-FATAL throughout: no speech output, or speech with
// no canceller, both leave a working assistant and say so loudly. "The assistant
// cannot speak" must never become "the assistant cannot answer".
//
// THREADING. The tap runs on the DSP worker, which is this pipeline's single
// producer, so its scratch buffers need no synchronisation. Everything else here
// is called from the UI thread (settings pushes, hot reload) or from main's
// startup path, and touches only device handles and atomics -- never the engine.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio_capture.h"
#include "audio_recorder.h"   // realtime_dsp.h only forward-declares AudioRecorder
#include "realtime_dsp.h"
#include "whisper_dsp.h"

#include "app_context.hpp"
#include "settings_store.hpp"
#include "speech_mode.hpp"    // rt::ISpeechMode

namespace rt {

class AudioPipelineBinder {
public:
    // `n_mels` is resolved by the caller because it depends on which PIPELINE is
    // running: on the legacy real path the mel geometry MUST match the projector
    // the audio head was trained with, while cascade mode takes the default and
    // does not consult the projector at all (whisper.cpp computes its own log-mel
    // internally from the PCM it is handed). The DSP is still constructed on the
    // cascade path, because RealTimeDSP is what HOSTS the PCM tap both modes read
    // from.
    //
    // `ctx` is BORROWED and must outlive this object: start_speech_output()
    // publishes the TTS / AEC / loopback handles into it, which is how the answer
    // stream and the settings handler reach them.
    //
    // INIT tier: throws on a missing mel_filters.bin or a capture device that will
    // not open.
    AudioPipelineBinder(const AssistantSettings& settings, int n_mels,
                        const std::string& data_dir, AppContext& ctx);
    ~AudioPipelineBinder();

    AudioPipelineBinder(const AudioPipelineBinder&) = delete;
    AudioPipelineBinder& operator=(const AudioPipelineBinder&) = delete;

    // The DSP the engine bootstrap borrows. Must outlive the engine stack, which
    // is exactly what declaring this object before it in main() gives.
    whisper::WhisperDSP& dsp() noexcept { return dsp_; }
    std::uint32_t sample_rate() const noexcept { return static_cast<std::uint32_t>(cfg_.sample_rate); }

    // Speech output + the capture-side half of full duplex. NON-FATAL: reports and
    // returns on any failure, leaving the app text-only (or speaking without a
    // canceller, which it warns about). No-op in a build without the TTS stack.
    void start_speech_output(const AssistantSettings& settings, int device_id);

    // Install the PCM tap and start the DSP worker. `mode` is BORROWED and must
    // outlive the call to stop(): the tap dereferences it on every block.
    void bind_speech_mode(ISpeechMode* mode);

    // Stop the DSP worker and both capture devices. Idempotent; called from the
    // shutdown path BEFORE the engine thread is joined, because the tap calls into
    // the mode the engine thread is pumping.
    void stop() noexcept;

    // ---- live settings -------------------------------------------------------
    // The capture-side twin of tts_volume, live for the same reason: the callback
    // reads it once per block.
    void set_input_gain(float gain) noexcept { capture_.set_input_gain(gain); }
    float input_level() const noexcept { return capture_.input_level(); }

    // The slider folded with the dock's mute. MUTED MEANS NOT SYNTHESISED (the
    // answer never reaches the solver), so this is the BACKSTOP -- it still
    // matters for everything that bypasses the answer path (the test tone) and for
    // the instant between the mute edge and the barge-in drain.
    float effective_tts_volume(float slider) const noexcept;

    // Apply the AEC enable flag and the folded speaker volume. Called from every
    // settings push; safe before start_speech_output() (both handles are null then
    // and the launch values are applied at construction instead).
    void apply_live_settings(const AssistantSettings& s);

    // ---- audio hot reload ----------------------------------------------------
    // Move the two ma_devices to whatever endpoints the settings now name, and
    // NOTHING else. Reopening two ma_devices takes tens of milliseconds and
    // touches no VRAM: the engine keeps its weights, the KV pool stays allocated
    // and the F5 ONNX session is not reloaded. That is the entire reason these
    // settings are their own tier.
    //
    // ORDER: playback first, then capture, then the loopback reference, then the
    // AEC reset. Playback is what the canceller's reference describes, so
    // resetting the filter before the new speaker exists would just make it
    // converge on the old one for another few hundred milliseconds.
    void apply_audio_reload(const AssistantSettings& s);

    // ---- shutdown telemetry --------------------------------------------------
    // Printed by main's summary block. Split out so main does not have to reach
    // through two optionals and an #if to read three counters.
    void print_shutdown_summary() const;

private:
    // ---- DECLARATION ORDER IS DESTRUCTION ORDER (reversed). Do not reorder. ----

    whisper::DspConfig cfg_{};
    AppContext&        ctx_;

    // The capture half, in the order main() built it.
    whisper::WhisperDSP dsp_;
    // The spectrogram no longer has a viewer -- it is kept because RealTimeDSP
    // writes into it unconditionally, and shrinking it is a DSP change, not a UI
    // one.
    SpectrogramBuffer spectrogram_;
    AudioRecorder     recorder_;
    AudioCapture      capture_;
    RealTimeDSP       realtime_;

#if defined(VOICE_ASSISTANT_HAS_TTS)
    std::optional<TtsRuntime> tts_;

    // ---- THE FAR END: a WASAPI loopback of the render endpoint ---------------
    // The canceller's reference is what the SPEAKER emits, taken from the speaker
    // rather than from us. miniaudio hands it back at the same 16 kHz mono f32 the
    // microphone runs at, so there is NO resampler on this path at all.
    //
    // WHAT THIS BUYS over tapping PullForPlayback: the reference is post-mix and
    // post-volume -- it already contains the software gain (so the canceller never
    // has to be told about it, and never re-converges when the slider moves) and
    // it contains audio THIS PROCESS DID NOT PRODUCE, all of which used to reach
    // the microphone as uncancellable echo.
    //
    // WHAT IT DOES NOT BUY, stated so nobody looks for it: the microphone and the
    // render endpoint are still two devices on two independent clocks, so
    // reference/mic drift is unchanged and AecCaptureFilter's backlog policy is
    // still doing that job.
    AudioCapture loopback_capture_;
    // The far-end ring the filter reads. ~3 s at 16 kHz: it only ever holds the
    // drift between the loopback callback and the DSP worker, and the filter's own
    // ceiling discards anything older than max_lag_ms.
    blackwell::audio_rt::SpscRing<float> loopback_ref_{48000};

    // DECLARED AFTER both, DELIBERATELY. It holds a reference into loopback_ref_,
    // so it must be destroyed BEFORE the ring it points at -- which reverse
    // declaration order is exactly what gives.
    std::optional<blackwell::audio_rt::AecCaptureFilter> aec_;

    // Sized ONCE in the ctor: RealTimeDSP pops at most 4096 samples per call, so
    // the resize in the tap never fires again after the first block and the tap
    // stays allocation-free on the path it shares with the capture drain.
    std::vector<float> aec_out_;
    std::vector<float> far_scratch_;
#endif

    // BORROWED. Set by bind_speech_mode(); read by the tap on the DSP worker.
    ISpeechMode* mode_ = nullptr;
    bool         stopped_ = false;
};

}  // namespace rt
