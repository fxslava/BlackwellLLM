# =============================================================================
# blackwell_audio_rt -- header-only real-time audio primitives.
#
# The lock-free structures that sit ON the audio callback: SpscRing today, the
# playback sink and the echo-canceller seam as the TTS/AEC work lands. Kept in
# its OWN target rather than inside blackwell_tts because its consumers are not
# all TTS: the capture path and the echo canceller use the same ring, and a
# capture-side include of a "tts" target would be a layering lie.
#
# DEPENDENCY-FREE BY CONSTRUCTION -- STL only. No CUDA, no ONNXRuntime, no
# miniaudio, no engine headers. That is what lets it be linked from the audio
# sandbox (which must stay CUDA-free in its standalone configuration) and from
# the engine-linked targets alike, and it is why this target is NOT gated on
# anything: there is nothing here that can fail to be available.
#
# INTERFACE (no sources yet). It becomes STATIC the moment a primitive here needs
# a TU of its own; consumers link the same name either way.
# =============================================================================
add_library(blackwell_audio_rt INTERFACE)
add_library(blackwell::audio_rt ALIAS blackwell_audio_rt)

# NOT SYSTEM: this is first-party code and is held to the root's /W4 /WX budget
# in every target that includes it (see the compiler-hygiene skill).
target_include_directories(blackwell_audio_rt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
