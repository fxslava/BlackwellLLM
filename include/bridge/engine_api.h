#ifndef BLACKWELL_BRIDGE_ENGINE_API_H
#define BLACKWELL_BRIDGE_ENGINE_API_H
// =============================================================================
// bridge/engine_api.h  —  C-ABI high-level bridge for multimodal streaming.
//
// The stable, language-neutral edge an application layer (gRPC service,
// WebSocket gateway, C#/Python/Rust FFI) links against to stream audio into the
// engine and receive generated tokens. Deliberately pure C: no C++ types, no
// CUDA, no <windows.h> in any signature — only fixed-width ints, opaque handles,
// and a function-pointer callback — so any ABI/CRT/runtime can sit across it.
//
// THREADING MODEL (this is the contract that keeps the engine correct)
//   The engine control plane is single-threaded: exactly one thread ever runs
//   the CUDA inference loop. This API is split accordingly:
//
//     * audio_stream_push_pcm  == PRODUCER edge. Callable from ANY thread (the
//       socket's receive callback), but exactly ONE producer thread per stream.
//       It is lock-free and returns immediately — it only appends to that
//       stream's SPSC AudioRingBuffer. It NEVER blocks, and NEVER touches the
//       engine, so a slow/fast network peer cannot stall decoding (Rule #1).
//
//     * engine_generate_multimodal_async == marshals a job onto the engine
//       thread and returns immediately. Generation, audio DSP, projector, splice
//       and decode all run on that one engine thread, draining the ring.
//
//     * CallbackFn fires FROM the engine thread as tokens are produced. Like the
//       poc_overlay worker rule, the callback MUST only hand results off (enqueue
//       to the app's own queue / socket write); it must NOT call back into any
//       engine_* function synchronously, and must not block (it stalls decode).
//
//   Handle ownership is single-owner (mirrors the COM edge: no refcounting — a
//   shareable handle would only invite violations of the single-thread doctrine).
//
// ERROR MODEL
//   Every call returns a BridgeStatus by value (no errno, no exceptions across
//   the edge). It mirrors blackwell::EngineStatus and maps 1:1 to the engine's
//   HRESULTs one layer down; bridge_status_to_string() gives a static label.
//
// ZERO-ALLOCATION / LIFECYCLE
//   The per-stream pinned ring and all decode workspaces are allocated when the
//   stream/engine is created (INIT tier). The steady-state push/generate/callback
//   path performs no allocation (Rule #2).
// =============================================================================

#include <stddef.h>   /* size_t */
#include <stdint.h>   /* int32_t */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Status codes (ABI-frozen; mirror blackwell::EngineStatus + bridge-specific  */
/* backpressure). Non-zero == failure.                                         */
/* -------------------------------------------------------------------------- */
typedef enum BridgeStatus {
    BRIDGE_OK                 = 0,
    BRIDGE_ERR_INVALID_HANDLE = 1,  /* null / stale / wrong-type opaque handle       */
    BRIDGE_ERR_INVALID_ARG    = 2,  /* null pointer, bad length, no <|audio|> in prompt */
    BRIDGE_ERR_BUFFER_FULL    = 3,  /* producer backpressure: ring had no room (retry)  */
    BRIDGE_ERR_OUT_OF_MEMORY  = 4,  /* pinned/VRAM allocation failed at create time     */
    BRIDGE_ERR_STATE          = 5,  /* op illegal in current state (e.g. gen in flight) */
    BRIDGE_ERR_CUDA           = 6,  /* CUDA runtime/kernel fault in the decode path      */
    BRIDGE_ERR_INTERNAL       = 7   /* unexpected engine fault (mapped from unknown)     */
} BridgeStatus;

/* -------------------------------------------------------------------------- */
/* Opaque handles. The concrete types live entirely behind the DLL; callers    */
/* only ever hold these pointers. Single-owner (no AddRef/Release semantics).   */
/* -------------------------------------------------------------------------- */
typedef struct BridgeEngineOpaque*      EngineHandle;       /* wraps the single-thread engine + audio pipeline */
typedef struct BridgeAudioStreamOpaque* AudioStreamHandle;  /* wraps one SPSC AudioRingBuffer + stream state   */

/* -------------------------------------------------------------------------- */
/* Token callback. Invoked on the ENGINE thread, once per produced token.       */
/*   user_data   : opaque context the caller passed to the async generate call. */
/*   token_utf8  : NUL-terminated UTF-8 piece for this step (valid only for the  */
/*                 duration of the call — copy it if you keep it).               */
/*   token_index : 0-based decode step index.                                    */
/*   is_final    : non-zero on the LAST callback for this request (EOS / stop /   */
/*                 error). No further callbacks fire for the request after it.    */
/*   status      : BRIDGE_OK while streaming; a failure code on the final call if */
/*                 generation aborted (token_utf8 may be empty then).             */
/* The callee MUST return promptly and MUST NOT re-enter any engine_* function.  */
/* -------------------------------------------------------------------------- */
typedef void (*CallbackFn)(void*        user_data,
                           const char*  token_utf8,
                           int32_t      token_index,
                           int32_t      is_final,
                           BridgeStatus status);

/* -------------------------------------------------------------------------- */
/* Audio stream lifecycle.                                                      */
/* -------------------------------------------------------------------------- */

/* Create a new audio input stream bound to `handle`. Allocates the stream's
 * pinned+mapped SPSC ring up front (INIT tier). On BRIDGE_OK, *stream_out holds
 * a handle the caller must later release with engine_destroy_audio_stream.
 * One engine may own several streams (e.g. multiplexed sessions), but each
 * stream still has exactly one producer thread and is consumed by the one
 * engine thread. Returns BRIDGE_ERR_INVALID_HANDLE / _INVALID_ARG / _OUT_OF_MEMORY. */
BridgeStatus engine_create_audio_stream(EngineHandle handle, AudioStreamHandle* stream_out);

/* Destroy a stream created above and free its pinned ring. Must NOT be called
 * while a generation using this stream is in flight (BRIDGE_ERR_STATE). Idempotent
 * against a null handle (returns BRIDGE_ERR_INVALID_HANDLE). */
BridgeStatus engine_destroy_audio_stream(AudioStreamHandle stream);

/* -------------------------------------------------------------------------- */
/* Producer edge — push 16 kHz mono float32 PCM. Lock-free, non-blocking.       */
/*   Appends `num_samples` from `pcm_data` to the stream's ring. If the ring is  */
/*   full it writes NOTHING and returns BRIDGE_ERR_BUFFER_FULL (the caller applies*/
/*   its own policy: retry after the consumer drains, or drop). Safe to call from */
/*   a realtime socket callback; never touches the engine or CUDA state.         */
/*   Exactly one thread may call this per stream.                                */
/* -------------------------------------------------------------------------- */
BridgeStatus audio_stream_push_pcm(AudioStreamHandle stream,
                                   const float*      pcm_data,
                                   size_t            num_samples);

/* -------------------------------------------------------------------------- */
/* Consumer edge — start multimodal generation asynchronously.                  */
/*   Marshals a job onto the engine thread and returns immediately (does not     */
/*   block on decode). `prompt_text` is UTF-8 and MUST contain exactly one        */
/*   `<|audio|>` placeholder marking where the streamed audio is spliced; the     */
/*   engine snapshots the audio buffered on `stream` at job start, runs the        */
/*   Whisper DSP + Ultravox projector ONCE, and (for speculative decoding) shares */
/*   the resulting embeddings between Draft and Target via a borrow view (see      */
/*   prompt_injector.hpp) — no double projector execution.                        */
/*   Tokens are delivered through `callback` on the engine thread; `user_data` is  */
/*   passed back verbatim. Returns BRIDGE_ERR_STATE if a generation is already in  */
/*   flight on this engine, BRIDGE_ERR_INVALID_ARG if the prompt lacks/duplicates  */
/*   the audio placeholder or callback is null.                                    */
/* -------------------------------------------------------------------------- */
BridgeStatus engine_generate_multimodal_async(EngineHandle      handle,
                                              AudioStreamHandle stream,
                                              const char*       prompt_text,
                                              CallbackFn        callback,
                                              void*             user_data);

/* Static human-readable label for a status code (never NULL; not localized). */
const char* bridge_status_to_string(BridgeStatus status);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* BLACKWELL_BRIDGE_ENGINE_API_H */
