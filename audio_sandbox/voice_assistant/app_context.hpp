#pragma once
// -----------------------------------------------------------------------------
// app_context.hpp — the handle the speech-pipeline and answer-stream callbacks
// reach state through, plus the three C-ABI callbacks themselves.
//
// WHY THESE TRAVEL TOGETHER. AppContext is not a god object by accident; it is
// the `void* user` the speech pipeline's C ABI hands back to every callback, so
// it is exactly the set of things a callback fired from the ENGINE, DSP or
// DISPATCHER thread is allowed to touch. Keeping the struct and its callbacks in
// one header is what makes that set auditable: if a member appears here, some
// non-UI thread reaches it, and the comment at its declaration says how.
//
// THE RULE FOR EVERY CALLBACK BELOW: they may only hand results to the
// thread-safe view, or to an object whose own header documents cross-thread
// use. Never a window, never COM, never CUDA, never a re-entrant pipeline call.
//
// LAYOUT IS BUILD-DEPENDENT (the members under VOICE_ASSISTANT_HAS_TTS), which
// is why build_features.hpp is included FIRST and unconditionally -- see the
// argument in that header about what a disagreeing feature macro does to a
// struct shared between six translation units.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "bridge/speculative_bridge_api.h"  // SpeechTokenEvent, SpeechPipelineState, SpeechVadScoreFn
#include "engine_control_bridge.hpp"        // blackwell::bridge::EngineControlBridge

#include "audio_capture.h"                  // rt::AudioCapture (the loopback reference)
#include "assistant_view.hpp"               // rt::AssistantView

namespace rt {

struct AppContext {
    AssistantView* view = nullptr;
    // The BASE type: this is what makes the callbacks backend-agnostic. Both
    // RealEngineControl and SimulatedEngineControl derive from it, so nothing
    // below ever branches on which decode loop is live.
    blackwell::bridge::EngineControlBridge* control = nullptr;

    // THE SPEAKER MUTE (the dock's speaker icon), and the one piece of state here
    // that is NOT inside the TTS guard: the volume fold that reads it is compiled
    // in both builds, and a flag whose home moves with a build option is a flag
    // someone will read from the wrong side of an #if.
    //
    // Session state, deliberately absent from `settings`: an app that starts up
    // silent because of a tap three days ago reads as broken, and save_settings
    // physically cannot reach a variable that is not in the struct.
    //
    // ATOMIC because the writer and the readers are different threads: the UI
    // thread sets it from the audio hot-update, while the answer stream reads it
    // on the ENGINE thread (local leg) or the DISPATCHER thread (remote leg).
    //
    // MUTED MEANS NOT SYNTHESISED, not synthesised-at-zero-gain. The gate is on
    // the three answer-stream edges (Resume / PushToken / EndOfTurn), so a muted
    // turn costs no solver steps at all; the zero volume it also produces is the
    // backstop, not the mechanism.
    std::atomic<bool> tts_muted{false};

#if defined(VOICE_ASSISTANT_HAS_TTS)
    // Null when speech output is unavailable (no models, no GPU, ctor threw).
    // Every use is guarded, because "the assistant cannot speak" must never
    // become "the assistant cannot answer".
    TtsRuntime* tts = nullptr;
    // The echo canceller sitting in the capture path. Null exactly when `tts` is
    // null: with no loudspeaker there is no echo and no reference to cancel it
    // with. It is a POINTER here rather than being reached through `tts` because
    // it belongs to the capture side -- the settings handler and the PCM tap
    // touch it, and neither has any business in the TTS stack.
    blackwell::audio_rt::AecCaptureFilter* aec = nullptr;
    // The loopback device supplying that filter's far end. A POINTER for the same
    // reason `aec` is one: the audio hot-reload has to move it when the OUTPUT
    // endpoint changes.
    AudioCapture* loopback = nullptr;

    // The answer text as it is handed to the speaker, accumulated purely to be
    // LOGGED once the turn ends. It is not the source of what gets spoken --
    // PushToken already streamed every fragment as it arrived, which is what
    // gives time-to-first-audio; re-pushing this at the end would say everything
    // twice.
    //
    // THREADING. The text sink runs on the ENGINE thread for the local leg and
    // the DISPATCHER thread for the remote one, and the completion edge on the
    // dispatcher thread either way. For the local leg the dispatcher is parked
    // inside send() for the whole generation, so the two never actually overlap
    // -- but "never overlaps" here is a property of another header's blocking
    // behaviour, and a mutex costs nothing on a per-token path that already takes
    // one inside PushToken.
    std::mutex answer_mu;
    std::string answer_text;
#endif
};

// ---- speech-pipeline callbacks (engine / VAD threads) -----------------------
// Passed to ConversationalMode / WhisperCascadeMode as raw function pointers, so
// they have external linkage and live in app_context.cpp.

// THIS STREAM IS THE USER'S OWN WORDS. Do not connect it to the TTS.
//
// Everything arriving here is decoded by session B, the ephemeral audio sequence
// (RealEngineControl's `audio_`), whose job is to write down what the user just
// said. publish_turn() runs extract_transcript() over it and offers the result to
// the commit gate as an INTENT; the reply to that intent is a separate generation
// entirely, and it surfaces through the dispatcher's callbacks -- which is where
// speech output is wired.
//
// An earlier version gated this on event->is_translation, which reads as if it
// separated the two. It does not: speech_pipeline_controller.cpp sets that flag
// to a hardcoded `true` for ALL decode-loop output, because at that layer there
// is only one decode loop and it cannot see which session ran it. So the gate was
// always open and the assistant recited the user's sentence back at them.
void on_token(void* user, const SpeechTokenEvent* event, std::uint64_t gen_id);

// Pipeline state transitions. Owns the BARGE-IN edge for speech output.
void on_state(void* user, SpeechPipelineState prev, SpeechPipelineState next,
              std::uint64_t gen_id);

#if defined(VOICE_ASSISTANT_HAS_SILERO)
// Runs on the DSP worker, once per 10 ms block. That thread is the SINGLE owner
// of the SileroVAD instance (the class's threading contract); the UI thread only
// reads its published atomics.
//
// SELF-BARGE-IN IS ANSWERED IN THE DSP, NOT HERE. The loudspeaker feeds the
// microphone and Silero scores the assistant's own voice as speech -- correctly,
// because it IS speech. Two guards were tried in this function and both were
// removed: a raised probability bar (a statistical bound that fails whenever the
// reference drifts out of alignment) and a hard mute keyed on playback state
// (deterministic, but it starved a recurrent model of blocks and withheld
// barge-in without asking).
//
// Both made the DETECTOR lie about what it heard. The answer lives one layer
// down, where the problem actually is: the assistant's own voice is SUBTRACTED
// from the capture stream by the echo canceller, fed a WASAPI loopback reference
// of what the speaker is really emitting. This function does one thing and does
// it unconditionally.
struct VadContext {
    blackwell::vad::SileroVAD* vad = nullptr;
};

float vad_score(void* user, const float* block, std::size_t count);
#endif

}  // namespace rt
