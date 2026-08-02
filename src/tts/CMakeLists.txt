# =============================================================================
# blackwell_tts -- on-demand speech synthesis (docs/TTS_INTEGRATION_AUDIT.md).
#
# CURRENT CONTENT: the TEXT FRONTEND only -- the IPhonemizer seam, the voice
# descriptor, and the RUNTIME-tier status type. The ONNX session, the worker
# thread and the resampler land in later phases of that document.
#
# DELIBERATELY NOT GATED ON blackwell_onnxruntime, unlike src/vad/. Nothing here
# touches ONNXRuntime yet, so gating it would produce a target that vanishes for
# reasons unrelated to anything it contains -- and would make the CPU-only
# phonemizer tests disappear with -DUSE_SILERO_VAD=OFF for no reason. The gate
# belongs on the TU that actually opens a session, and moves here (or splits this
# target) when that TU arrives.
#
# ENGINE-FREE AND CUDA-FREE, and must stay that way: this is a leaf that the
# audio sandbox links, not part of the inference stack. It is NOT a party to the
# single-engine-thread doctrine because it never touches the engine -- see the
# threading contract in phonemizer.hpp.
#
# Explicit source list (src/kernels/ convention; never file(GLOB)).
# =============================================================================
add_library(blackwell_tts STATIC phonemizer.cpp)

# The headers ARE the public surface; there is no PIMPL here because there is no
# third-party type to hide yet. When the ORT session lands it follows the
# src/vad/ pattern exactly: one .cpp sees <onnxruntime_cxx_api.h>, and the
# dependency stays PRIVATE so consumers inherit neither its headers nor its
# include path.
target_include_directories(blackwell_tts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    # REQUIRED, not cosmetic. The symbol tables and the tests carry Cyrillic
    # UTF-8 literals -- the primary target language of the translator. Without
    # /utf-8 MSVC reads them in the active code page, the phonemizer's map keys
    # stop matching the input bytes, and every Russian utterance degrades to
    # unknown symbols. /WX would also trip C4819 on the non-ASCII source.
    target_compile_options(blackwell_tts PRIVATE /utf-8)
endif()
