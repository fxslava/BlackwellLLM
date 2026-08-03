#pragma once
// -----------------------------------------------------------------------------
// TtsStatus — the RUNTIME-tier result type for the speech-synthesis stack
// (CLAUDE.md extension pattern #4, the hybrid error doctrine).
//
// THE SPLIT, restated for this subsystem so nobody has to guess:
//   INIT tier    — loading a voice, creating an ORT session, opening an audio
//                  device, starting the worker thread. These THROW
//                  std::runtime_error. The caller catches, disables TTS, and the
//                  app runs on: this is the same graceful-degradation posture
//                  translator/main.cpp already takes for the neural VAD and the
//                  audio head, and it is why a missing 60 MB voice file is a
//                  greyed-out button rather than a failed launch.
//   RUNTIME tier — phonemisation, synthesis, resampling, ring traffic. These are
//                  noexcept and return one of these codes. NOTHING on a live
//                  audio path may unwind: the miniaudio callback and the DSP
//                  worker have nowhere to put an exception, and by Phase 4 an
//                  aborted callback also desynchronises the echo canceller from
//                  its reference stream.
//
// Statuses are for the CALLER to branch on, and for counters/UI to surface. They
// are deliberately coarse: a code exists here only where a caller would
// genuinely act differently. Diagnostics that merely explain a code belong in a
// counter next to it (the SileroVAD::inference_errors() idiom), not in a new
// enumerator.
// -----------------------------------------------------------------------------
#include <cstdint>

namespace blackwell::tts {

enum class TtsStatus : std::int32_t {
    Success             = 0,
    InvalidArgument     = 1,  // caller error: null buffer, or a voice with no symbol table
    UnsupportedLanguage = 2,  // no voice installed for the requested language
    EmptyResult         = 3,  // input held nothing this voice can pronounce (see below)
    NotInitialized      = 4,  // used before its INIT-tier setup ran, or after teardown
    RuntimeFailure      = 5,  // backend (ORT / resampler) failed mid-stream; degrade, do not die
    Interrupted         = 6,  // barge-in: abandoned on request, NOT a fault (see below)
};

inline const char* to_string(TtsStatus status) noexcept {
    switch (status) {
        case TtsStatus::Success:             return "Success";
        case TtsStatus::InvalidArgument:     return "InvalidArgument";
        case TtsStatus::UnsupportedLanguage: return "UnsupportedLanguage";
        case TtsStatus::EmptyResult:         return "EmptyResult";
        case TtsStatus::NotInitialized:      return "NotInitialized";
        case TtsStatus::RuntimeFailure:      return "RuntimeFailure";
        case TtsStatus::Interrupted:         return "Interrupted";
    }
    return "UnknownTtsStatus";
}

// EmptyResult vs InvalidArgument is the one distinction worth stating outright,
// because they look alike at a call site and mean opposite things:
//   InvalidArgument — the CALLER is wrong (null buffer, unusable voice). A bug.
//   EmptyResult     — the caller is fine and the input was simply unpronounceable
//                     by this voice: empty text, whitespace only, or a string
//                     whose every symbol is missing from the voice's table (the
//                     classic case being Cyrillic text handed to an English-only
//                     voice). NOT a bug, and NOT silence to be played — the
//                     correct response is to skip the utterance and leave the
//                     unknown-symbol counter to explain why.
// Malformed UTF-8 is neither: the offending bytes are skipped and counted, and
// the request succeeds on whatever remained pronounceable. Only if that leaves
// nothing does it surface, as EmptyResult.
//
// Interrupted is the third member of that family and the one most likely to be
// mishandled, because it is the only status here that means "everything worked".
// A barge-in abandons a solve that was proceeding correctly, so it must NOT
// advance an error counter, must NOT be surfaced as a failure in the panel, and
// must NOT be retried — the user asked for the audio to stop, and re-speaking a
// line they just talked over is the exact behaviour barge-in exists to prevent.
// Callers branch on it to skip the utterance silently.
inline bool is_success(TtsStatus status) noexcept { return status == TtsStatus::Success; }

// True when a request ended without producing audio AND without anything being
// wrong. The predicate exists so call sites stop spelling out the disjunction and
// getting it wrong in one of the two places.
inline bool is_benign_empty(TtsStatus status) noexcept {
    return status == TtsStatus::EmptyResult || status == TtsStatus::Interrupted;
}

}  // namespace blackwell::tts
