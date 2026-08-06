#pragma once
// -----------------------------------------------------------------------------
// build_features.hpp — the ONE place that turns this build's CMake options into
// the VOICE_ASSISTANT_HAS_* macros, and pulls the headers each one gates.
//
// WHY IT EXISTS. Every optional half of this app (speech output, the neural VAD,
// the whisper.cpp cascade, the cloud leg, AEC3) is a compile-time presence test
// written as a chain of #ifs over a define some OTHER target exports. While the
// app was one translation unit those chains lived at the top of main.cpp and were
// correct by construction. They are now spread over six TUs, and a feature macro
// that is defined in one and not another is not a build error -- it is a struct
// whose layout differs between two object files, which links cleanly and then
// misbehaves at runtime in a way no warning will find.
//
// So the derivation happens exactly once, here, and every voice_assistant TU
// includes this header FIRST. The rule for adding a feature: derive the macro
// here, include its header here under the same guard, and never write
// `#if defined(BLACKWELL_HAVE_...)` anywhere else in this directory.
//
// THE ONE TRAP, restated because it already cost a silent feature loss:
// BLACKWELL_HAVE_TTS_F5 is blackwell_tts_f5's OWN public define, and it exists
// only when the ONNXRuntime CUDA execution provider was fetched
// (-DBLACKWELL_ORT_GPU=ON). Do NOT gate speech output on BLACKWELL_HAVE_ORT_CUDA:
// that define belongs to blackwell::onnxruntime, which blackwell_tts_f5 links
// PRIVATE, so it never reaches this header and the whole integration vanishes
// from the binary without a word.
// -----------------------------------------------------------------------------

// ---- Mode C: the whisper.cpp cascade ----------------------------------------
// VOICE_ASSISTANT_HAS_WHISPER comes straight from CMake (the target exists only
// with -DUSE_WHISPER_CPP=ON, the default). Nothing to derive; the include is
// here so the pairing cannot drift.
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
#include "whisper_cascade_mode.hpp"     // rt::WhisperCascadeMode
#endif

// ---- Silero neural VAD -------------------------------------------------------
// The model path is baked in by cmake/OnnxRuntime.cmake only when blackwell_vad
// was built; its absence means the pipeline's built-in RMS detector runs instead.
#if defined(BLACKWELL_VAD_MODEL_PATH)
#include "silero_vad.hpp"
#define VOICE_ASSISTANT_HAS_SILERO 1
#endif

// ---- F5-TTS speech output, and the capture-side half of full duplex ----------
// The DiT measures ~0.2x realtime on CPU, so a CPU-provider build would produce
// audio slower than it plays -- worse than no speech at all. Hence the gate is on
// the CUDA-provider define and not on "was ORT found".
#if defined(BLACKWELL_HAVE_TTS_F5)
#include "tts_runtime.hpp"
// Gated WITH the TTS because it is the loudspeaker that creates the problem it
// solves: with no speech output there is no echo, and no far-end reference to
// cancel one with.
#include "aec_capture_filter.hpp"
#if defined(BLACKWELL_HAVE_AEC3)
// PIMPL'd: this pulls in no WebRTC header, only the seam and a unique_ptr.
#include "aec3_echo_canceller.hpp"
#endif
#define VOICE_ASSISTANT_HAS_TTS 1
#endif
