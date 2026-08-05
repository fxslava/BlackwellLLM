# =============================================================================
# blackwell_audio_rt -- real-time audio primitives.
#
# The lock-free structures that sit ON the audio callback plus the echo
# canceller that sits BESIDE it: SpscRing, the AEC seam and its
# partitioned-block implementation, and the rate converter the AEC needs to put
# its two streams on one clock. Kept in its OWN target rather than inside
# blackwell_tts because its consumers are not all TTS: the capture path holds
# the canceller and the reference ring, and a capture-side include of a "tts"
# target would be a layering lie.
#
# DEPENDENCY-FREE BY CONSTRUCTION -- STL only. No CUDA, no ONNXRuntime, no
# miniaudio, no engine headers, and no fetched third-party include paths. That
# is what lets it be linked from the audio sandbox (which must stay CUDA-free in
# its standalone configuration) and from the engine-linked targets alike, and it
# is why this target is NOT gated on anything: there is nothing here that can
# fail to be available.
#
# THE FFT IS THE ONE PLACE THAT COSTS SOMETHING. echo_canceller.cpp carries its
# own 256-point radix-2 transform rather than reaching for pocketfft through
# blackwell::sandbox_headers, which would put a fetched dependency on this
# target's otherwise empty link line. Same trade, same reason, as
# src/tts/f5_mel_extractor.cpp.
#
# STATIC as of the AEC landing: it was INTERFACE while SpscRing was the only
# thing here, and the comment then said it would become STATIC the moment a
# primitive needed a TU of its own. Consumers link the same name either way.
# =============================================================================
add_library(blackwell_audio_rt STATIC
    echo_canceller.cpp
    rational_resampler.cpp
    aec_capture_filter.cpp)
add_library(blackwell::audio_rt ALIAS blackwell_audio_rt)

# NOT SYSTEM: this is first-party code and is held to the root's /W4 /WX budget
# in every target that includes it (see the compiler-hygiene skill).
target_include_directories(blackwell_audio_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# =============================================================================
# blackwell_audio_aec3 -- the WebRTC AEC3 backend, in its OWN target.
#
# It is separate from blackwell_audio_rt above for one reason, and it is the
# reason stated in that target's header block: blackwell_audio_rt is
# dependency-free by construction, and that is what lets the CUDA-free audio
# sandbox and the engine-linked targets both link it without inheriting anything.
# WebRTC brings an include tree, abseil, and its own compile definitions. Putting
# it on blackwell_audio_rt would make every consumer of SpscRing pay for a
# canceller backend it may not use, and would make the target gateable -- which
# its comment specifically says it is not.
#
# So the split mirrors src/vad/: one TU sees the heavy dependency, everyone else
# sees a PIMPL'd header. AecCaptureFilter takes the backend by injection
# (aec_capture_filter.hpp), so blackwell_audio_rt never names AEC3 at all and
# this target is a pure addition on top of it.
# =============================================================================
if(BLACKWELL_HAVE_WEBRTC_AEC3)
    add_library(blackwell_audio_aec3 STATIC aec3_echo_canceller.cpp)
    add_library(blackwell::audio_aec3 ALIAS blackwell_audio_aec3)

    target_include_directories(blackwell_audio_aec3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    # PUBLIC audio_rt: the header exposes IEchoCanceller in its base clause, so a
    # consumer of aec3_echo_canceller.hpp needs echo_canceller.hpp too.
    # PRIVATE webrtc: nobody downstream should inherit that include path -- the
    # PIMPL exists precisely so they do not.
    target_link_libraries(blackwell_audio_aec3
        PUBLIC  blackwell::audio_rt
        PRIVATE blackwell::webrtc_aec3)

    # Consumers branch on this rather than on USE_WEBRTC_AEC3, which says what was
    # ASKED for; this says what was actually found and built.
    target_compile_definitions(blackwell_audio_aec3 PUBLIC BLACKWELL_HAVE_AEC3=1)
endif()
