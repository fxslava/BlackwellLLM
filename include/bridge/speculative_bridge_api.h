#ifndef BLACKWELL_BRIDGE_SPECULATIVE_BRIDGE_API_H
#define BLACKWELL_BRIDGE_SPECULATIVE_BRIDGE_API_H
// =============================================================================
// bridge/speculative_bridge_api.h — C-ABI for the unified speculative speech
// streaming pipeline (the audio analogue of poc_overlay's LiveTranslationTracker).
//
// Maps the overlay's edit-stream doctrine onto speech:
//     Keystrokes        -> VAD frames / streaming ASR tokens (push_pcm)
//     CaretDebounce     -> VAD silence hangover (a STABLE boundary)  -> on_silence_timeout
//     TrackUpdate       -> speculative KV-cache warming prefill      (while speaking)
//     TriggerGeneration -> commit soft-tokens + begin LLM decode     (on_silence_timeout)
//     Barge-in          -> monotone gen-id bump + KV micro-rewind    (on_speech_start)
//
// THREADING (single-thread engine doctrine, unchanged)
//   The audio/VAD thread only pushes PCM and flags boundary events through this
//   API. Those calls are lock-free: they bump atomics and MARSHAL engine work
//   (rewind / prefill / decode) onto the one engine thread — they never touch
//   CUDA, the KV cache, or the decode loop directly. Token/state callbacks fire
//   FROM the engine thread and must only hand results off (never re-enter the API).
//
// Pure C: fixed-width ints, stdbool, opaque handle, function-pointer callbacks —
// no C++ / CUDA / <windows.h> — so a gRPC/WebSocket/FFI layer links it directly.
// Reuses BridgeStatus + EngineHandle/AudioStreamHandle from engine_api.h.
// =============================================================================

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bridge/engine_api.h"  /* BridgeStatus, EngineHandle, AudioStreamHandle */

#ifndef BRIDGE_API
#  if defined(_WIN32) && defined(BLACKWELL_BRIDGE_SHARED)
#    if defined(BLACKWELL_BRIDGE_EXPORTS)
#      define BRIDGE_API __declspec(dllexport)
#    else
#      define BRIDGE_API __declspec(dllimport)
#    endif
#  else
#    define BRIDGE_API   /* static lib / non-Windows: no decoration */
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Pipeline state (ABI-frozen). Transitions are driven by VAD boundaries and     */
/* barge-in; INTERRUPTION_REWIND is the transient a barge-in passes through while */
/* the cancel signal fires and the KV micro-rewind is marshaled.                  */
/* -------------------------------------------------------------------------- */
typedef enum SpeechPipelineState {
    SPEECH_STATE_IDLE                = 0,  /* no speech; ring quiescent                    */
    SPEECH_STATE_PREFILL_SPEAKING    = 1,  /* user speaking; warming KV cache speculatively */
    SPEECH_STATE_DECODE_TRANSLATING  = 2,  /* stable boundary reached; LLM decoding output  */
    SPEECH_STATE_INTERRUPTION_REWIND = 3   /* barge-in: cancelling + rewinding to verified  */
} SpeechPipelineState;

/* One emitted token — either a streaming ASR partial or a committed translation */
/* piece (is_translation distinguishes them). `text` is UTF-8, valid ONLY for the */
/* duration of the callback (copy it if retained).                                */
typedef struct SpeechTokenEvent {
    const char* text;         /* NUL-terminated UTF-8 piece                         */
    uint32_t    token_id;     /* vocabulary id (0 if not surfaced at this layer)    */
    uint32_t    position;     /* 0-based index in the emitted stream               */
    bool        is_translation; /* false = streaming ASR partial; true = translation */
} SpeechTokenEvent;

/* Pipeline construction descriptor (POD "DESC" idiom). Holds the bridge handles  */
/* the pipeline drives plus the VAD policy that shapes its boundaries.             */
typedef struct SpeechPipelineConfig {
    EngineHandle      engine;                 /* drives the single CUDA engine thread     */
    AudioStreamHandle audio_stream;           /* pinned SPSC ring this pipeline feeds      */

    uint32_t sample_rate;                     /* PCM sample rate (16000)                   */
    float    vad_threshold_db;                /* onset threshold, dBFS (e.g. -40.0)        */
    float    vad_release_db;                  /* release threshold for hysteresis; <= onset;
                                                 0 => same as vad_threshold_db             */
    uint32_t silence_hangover_ms;             /* stable-boundary hangover (e.g. 800)       */
    uint32_t warm_prefill_interval_ms;        /* throttle for speculative prefill (e.g. 320);
                                                 0 => disable speculative warming           */
} SpeechPipelineConfig;

/* Opaque pipeline handle (points at the internal SpeechPipelineController). */
typedef struct SpeechPipelineOpaque* SpeechPipelineHandle;

/* Token callback — invoked on the ENGINE thread, once per emitted token, carrying
 * the generation id the token belongs to (so a stale-epoch token can be dropped
 * by the app after a barge-in). Must return promptly; must NOT re-enter the API. */
typedef void (*SpeechTokenCallback)(void* user, const SpeechTokenEvent* event, uint64_t gen_id);

/* State-change callback — invoked on whichever thread performs the transition
 * (audio thread for VAD/barge-in, engine thread for decode completion). */
typedef void (*SpeechStateCallback)(void* user, SpeechPipelineState prev,
                                    SpeechPipelineState next, uint64_t gen_id);

/* -------------------------------------------------------------------------- */
/* Lifecycle.                                                                   */
/* -------------------------------------------------------------------------- */

/* Create a pipeline over an existing engine + audio stream. Validates the config
 * and handles; no CUDA allocation here (the stream's ring already exists). On
 * BRIDGE_OK, *out_handle must later be released with speech_pipeline_destroy. */
BRIDGE_API BridgeStatus speech_pipeline_create(const SpeechPipelineConfig* config,
                                               SpeechPipelineHandle* out_handle);

/* Destroy a pipeline. Does NOT destroy the underlying engine or audio stream
 * (the app owns those). Must not race with in-flight push/event calls. */
BRIDGE_API BridgeStatus speech_pipeline_destroy(SpeechPipelineHandle handle);

/* Register token + state callbacks (either may be NULL). Call once during setup,
 * before streaming begins. `user` is passed back verbatim to both callbacks. */
BRIDGE_API BridgeStatus speech_pipeline_register_callbacks(SpeechPipelineHandle handle,
                                                           SpeechTokenCallback token_cb,
                                                           SpeechStateCallback state_cb,
                                                           void* user);

/* -------------------------------------------------------------------------- */
/* Streaming edge (audio/VAD thread). All lock-free; none touch CUDA directly.  */
/* -------------------------------------------------------------------------- */

/* Push 16 kHz mono float32 PCM. Forwards samples to the stream's SPSC ring and
 * runs the internal VAD: crossing the onset threshold auto-fires speech-start,
 * accumulated silence past the hangover auto-fires silence-timeout, and while
 * speaking it throttles speculative warm prefills. Lock-free, non-blocking;
 * returns BRIDGE_ERR_BUFFER_FULL only if the ring is full (samples dropped). */
BRIDGE_API BridgeStatus speech_pipeline_push_pcm(SpeechPipelineHandle handle,
                                                 const float* samples, size_t count);

/* Explicit barge-in / speech-onset (an external VAD/ASR may call this instead of
 * relying on the internal VAD). Bumps the generation id (invalidating in-flight
 * decode), signals cancel, marshals a KV micro-rewind to the verified prefix,
 * resets speculative state, and enters PREFILL_SPEAKING. No-op if already
 * PREFILL_SPEAKING. Callable from any thread. */
BRIDGE_API BridgeStatus speech_pipeline_on_speech_start(SpeechPipelineHandle handle);

/* Explicit stable-boundary / silence-timeout. Commits the buffered audio as
 * soft-tokens (projector + inject_audio_embeddings splice) and begins the LLM
 * decode loop, entering DECODE_TRANSLATING. No-op unless PREFILL_SPEAKING. */
BRIDGE_API BridgeStatus speech_pipeline_on_silence_timeout(SpeechPipelineHandle handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* BLACKWELL_BRIDGE_SPECULATIVE_BRIDGE_API_H */
