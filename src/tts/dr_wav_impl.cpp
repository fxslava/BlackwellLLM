// -----------------------------------------------------------------------------
// dr_wav_impl.cpp — the dr_wav amalgamation, and NOTHING else.
//
// WHY THIS FILE EXISTS AS A SEPARATE TU. DR_WAV_IMPLEMENTATION expands a whole
// third-party C library into whichever TU defines it, and that code is not
// /W4-clean (C4701 "potentially uninitialized local" at dr_wav.h:3726, among
// others). The compiler-hygiene skill's usual move — wrapping the #include in
// #pragma warning(push, 0) / pop — does NOT work for C4701: it is emitted during
// CODE GENERATION, and MSVC resolves codegen-time warnings against the warning
// state at the END of the translation unit, by which point the pop has already
// restored /W4. The pragma is simply not consulted.
//
// The remaining options were (a) disable the warning for the whole TU, which in
// f5_tts_cli.cpp would also blind our own code, or (b) give the vendored library
// its own TU so the suppression cannot reach anything first-party. (b) is this
// file. The /W0 is applied per-SOURCE in src/tts/CMakeLists.txt — not per-target
// — so test_f5_tts_cuda_cpp itself stays at /W4 /WX.
//
// Keep this file at exactly two lines of content. Anything added here would
// silently inherit the suppression.
// -----------------------------------------------------------------------------
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
