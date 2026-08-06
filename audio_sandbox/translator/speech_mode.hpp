#pragma once
// -----------------------------------------------------------------------------
// ISpeechMode — the ONE seam between audio_translator's plumbing and the two
// mutually exclusive ways this app can turn speech into text.
//
//   Mode A  CONVERSATIONAL  listen -> silence -> generate -> barge-in.
//                           The classic path: speech_pipeline_* drives
//                           RealEngineControl through IEngineControl.
//   Mode B  SIMULTANEOUS    continuous rolling re-translation, no bot voice.
//                           SpeechSegmenter -> AbsoluteAudioRing ->
//                           RetranslationSession (docs/CONTINUOUS_STREAMING.md).
//   Mode C  WHISPER CASCADE listen -> silence -> whisper.cpp transcribes ->
//                           the TEXT is published as an intent. No audio ever
//                           reaches the backbone (voice_assistant only; see
//                           voice_assistant/whisper_cascade_mode.hpp).
//
// WHICH BINARY BUILDS WHICH, because the interface describes more modes than any
// one executable links and a reader should not have to infer that from the
// absence of a construction site:
//
//   audio_translator   Mode A, Mode B
//   voice_assistant    Mode A, Mode C
//
// Neither app silently substitutes. voice_assistant REFUSES a Mode B request with
// a std::runtime_error naming the reason (AppLifecycleManager::bring_up_speech_mode),
// and refuses a Mode C request it cannot serve -- a missing GGML model -- rather
// than degrading to Mode A behind the user's back.
//
// WHY THE SEAM IS HERE AND NOT AT IEngineControl. The obvious-looking move is to
// make Mode B another IEngineControl implementation, mapping Partial->warm_prefill
// and Final->commit_and_decode. Every one of those verbs is wrong for it:
//   * warm_prefill is APPEND-ONLY speculative growth; a Partial is a
//     rewind-and-refeed. Opposite direction.
//   * commit_and_decode assumes one reply per utterance plus a barge-in epoch;
//     a Final is a redraft PLUS a commit-pointer advance.
//   * cancel_generation has no meaning in Mode B — its draft is disposable BY
//     DESIGN, so there is nothing to interrupt.
//   * SpeechPipelineState has no state that describes "always drafting".
// What the two modes genuinely share is narrower and lower: a stream of 10 ms PCM
// blocks in, text out, one engine thread. That is exactly this interface.
//
// THREADING — each method names its thread, and the whole point of the split is
// that the single-engine-thread doctrine (CLAUDE.md) stays trivially satisfied:
// there is still exactly one thread calling into the engine, in either mode.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>

namespace rt {

// What a mode publishes for the UI badge, without the UI knowing any mode's
// internals. Fields a given mode does not have simply stay zero.
struct ModeTelemetry {
    const char*   name       = "";
    std::uint64_t utterances = 0;   // committed utterances (both modes)
    std::uint64_t drafts     = 0;   // Mode B only: redrafts that did not commit
    std::uint64_t evictions  = 0;   // Mode B only
    std::uint32_t kv_tail    = 0;   // Mode B only: live KV length
};

class ISpeechMode {
public:
    virtual ~ISpeechMode() = default;

    virtual const char* name() const noexcept = 0;

    // ENGINE THREAD, exactly once, before the first pump_engine(). Prefills this
    // mode's frozen system prefix — which DIFFERS per mode (a conversational
    // assistant and a verbatim transcriber want different prompts), and is why a
    // mode switch is necessarily a session boundary rather than a free toggle.
    virtual void start_on_engine_thread() = 0;

    // AUDIO/DSP THREAD, once per block. MUST NOT touch the engine, CUDA, COM or a
    // window — it may only append to a ring / score VAD / enqueue (the producer
    // edge). Kept noexcept because the DSP worker has nowhere to put an exception.
    virtual void on_pcm_block(const float* samples, std::size_t count) noexcept = 0;

    // ENGINE THREAD. ONE iteration of the mode's work loop. May block until work
    // arrives (Mode A parks in wait_and_pump at 0% CPU); must return promptly once
    // stop() has been called, or the runner cannot join.
    virtual void pump_engine() = 0;

    // ANY THREAD. Unblock pump_engine() and abort anything in flight so the engine
    // thread can leave its loop. Idempotent.
    virtual void stop() noexcept = 0;

    virtual ModeTelemetry telemetry() const noexcept = 0;

    // ---- LIVE settings (any thread; effective on the next VAD block) ---------
    // The four knobs an app's settings panel retunes on a RUNNING mode. They are
    // virtual with DEFAULT NO-OP bodies rather than pure, for two reasons:
    //
    //   * Not every mode has every knob. A warm-prefill throttle is meaningless
    //     to a mode that never speculatively prefills, and the honest
    //     implementation of "retune a knob I do not have" is to ignore it -- not
    //     to force each mode to write an empty override.
    //   * They are ADDITIVE to an interface that already has implementors. A
    //     pure virtual here would break every one of them at once for the
    //     benefit of no caller.
    //
    // Their existence is what lets a settings push say `active->set_...` instead
    // of reaching through a mode-specific handle (audio_translator still calls
    // ConversationalMode::pipeline() directly, which stays public and unchanged).
    // Each mode documents which of these it honours.
    virtual void set_vad_threshold(float /*probability*/) noexcept {}
    virtual void set_silence_hangover_ms(std::uint32_t /*ms*/) noexcept {}
    virtual void set_warm_prefill_interval_ms(std::uint32_t /*ms*/) noexcept {}
    // Push-to-talk: while enabled, no automatic VAD transition may fire. PCM
    // keeps flowing either way -- this mutes the DECISIONS, never the stream.
    virtual void set_manual_mode(bool /*enabled*/) noexcept {}
};

}  // namespace rt
