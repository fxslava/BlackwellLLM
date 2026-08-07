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
# f5_mel_extractor.cpp lives HERE and not in blackwell_tts_f5 on purpose: it is
# pure DSP (STFT + mel filterbank + log), with no ONNXRuntime, no CUDA and no
# engine. It carries its own radix-2 FFT rather than reaching for pocketfft via
# blackwell::sandbox_headers, which keeps this target's link line EMPTY -- the
# property the block above is protecting.
#
# THE TEXT FRONTEND IS FOUR STAGES AND THEY ALL LIVE HERE (the order is fixed
# and documented above DuplexConfig in tts_duplex_bridge.hpp):
#   text_chunker      punctuation-only splitting of the LLM stream
#   speech_text       Markdown / symbol / stress-notation normalisation
#   text_normalizer   digits and symbols -> spoken Russian words
#   stress_marker     '+' accent placement (dictionary; ONNX behind IAccentor)
# All four are STL-only: no ONNXRuntime, no CUDA, no engine, which is what lets
# tts_tests exercise the whole frontend on a machine with no GPU.
add_library(blackwell_tts STATIC
    phonemizer.cpp
    f5_mel_extractor.cpp
    f5_tokenizer.cpp
    text_chunker.cpp
    speech_text.cpp
    text_normalizer.cpp
    stress_marker.cpp
    tts_duplex_bridge.cpp)

# blackwell::audio_rt is a HEADER-ONLY INTERFACE target (SpscRing) -- no objects,
# no ONNXRuntime, no CUDA -- so linking it does not violate the engine-free /
# CUDA-free contract above. It is PUBLIC because tts_duplex_bridge.hpp names
# SpscRing<float> in its own interface.
#
# NOTE ON THE BRIDGE'S PLACEMENT. tts_duplex_bridge.cpp orchestrates synthesis
# but does NOT name F5TtsEngine: it talks to the ISynthesizer seam, and the
# adapter that binds that seam to the CUDA engine (f5_engine_synthesizer.hpp)
# lives with blackwell_tts_f5. That is what lets the bridge sit here without
# dragging a 350 MB runtime into the text frontend -- and what lets the barge-in
# tests run on a machine with no GPU and no checkpoint.
target_link_libraries(blackwell_tts PUBLIC blackwell::audio_rt)

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

# =============================================================================
# blackwell_tts_f5 -- F5-TTS synthesis on the ONNXRuntime CUDA execution
# provider. A SEPARATE TARGET, and the separation is the point.
#
# WHY NOT JUST ADD f5_tts_engine.cpp TO blackwell_tts. Because the block at the
# top of this file states that blackwell_tts is "ENGINE-FREE AND CUDA-FREE, and
# must stay that way", and that claim is load-bearing rather than decorative:
# blackwell_tts is what the phonemizer tests link, what the text frontend lives
# in, and what a future espeak-ng backend would sit behind. Adding ORT + cudart
# to it would drag a ~350 MB runtime and a CUDA dependency into every consumer
# of a pure text-processing library, and would make the CPU-only tts_tests
# unbuildable on a machine with no CUDA.
#
# So the dependency runs ONE WAY: blackwell_tts_f5 -> blackwell_tts (for
# TtsStatus), never the reverse. blackwell_tts's own link line is untouched
# above, and CI should keep it that way.
#
# GATED ON THE CUDA EP, not merely on ONNXRuntime existing. With
# -DBLACKWELL_ORT_GPU=OFF the CPU archive has no CUDA provider, so this engine
# would compile, link, and then silently run its DiT on the CPU at ~0.2x
# realtime -- audio that arrives slower than it plays. A target that cannot
# work is worse than a target that is absent, so it is absent: consumers guard
# with if(TARGET blackwell_tts_f5), exactly as they already do for
# blackwell_vad and blackwell_tts.
# =============================================================================
if(NOT TARGET blackwell_onnxruntime)
    return()
endif()

get_target_property(_ort_gpu blackwell_onnxruntime BLACKWELL_ORT_GPU_ENABLED)
if(NOT _ort_gpu)
    message(STATUS
        "blackwell_tts_f5: skipped (ONNXRuntime has no CUDA execution provider; "
        "configure with -DBLACKWELL_ORT_GPU=ON to build the F5-TTS engine)")
    return()
endif()

# CUDA::cudart -- for the three cudaMemcpy/cudaMemset calls that seed the device
# conditioning, noise and text buffers. ORT can ALLOCATE device memory and BIND
# it, but exposes no way to write a device tensor out of band; the alternative
# is re-uploading unchanged conditioning on every solver step. All of it sits
# outside the ODE loop by construction (see the cudart-seam block in
# f5_tts_engine.cpp).
find_package(CUDAToolkit REQUIRED)

add_library(blackwell_tts_f5 STATIC f5_tts_engine.cpp f5_engine_synthesizer.cpp)

target_include_directories(blackwell_tts_f5 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(blackwell_tts_f5
    # PUBLIC: f5_tts_engine.hpp includes spsc_ring.hpp in its own interface (the
    # streaming GenerateAudio overload takes a SpscRing<float>&), so consumers
    # need audio_rt's include path too. Header-only INTERFACE target -- no code.
    PUBLIC  blackwell::audio_rt
            blackwell_tts                     # TtsStatus; the one-way dependency
    # PRIVATE: the PIMPL earns its keep here. f5_tts_engine.cpp is the only TU
    # that sees <onnxruntime_cxx_api.h> and <cuda_runtime.h>, so no consumer
    # inherits ORT's headers, ORT's exception types, or the CUDA include path.
    PRIVATE blackwell::onnxruntime
            CUDA::cudart)

# The target's OWN capability define, PUBLIC so consumers can gate on it.
#
# It cannot be BLACKWELL_HAVE_ORT_CUDA: that one rides on blackwell::onnxruntime,
# which is linked PRIVATE here precisely so ORT's headers do not leak into
# consumers -- and PRIVATE means its INTERFACE definitions do not reach them
# either. A consumer gating on it compiles its entire integration OUT while the
# build stays green, which is exactly what happened to voice_assistant the first
# time. This define exists so "the engine is available" is expressible without
# exposing the engine's dependencies.
target_compile_definitions(blackwell_tts_f5 PUBLIC BLACKWELL_HAVE_TTS_F5)

if(MSVC)
    # Same /utf-8 rationale as blackwell_tts above: this TU's error strings name
    # model paths, which on this box are Cyrillic-capable.
    target_compile_options(blackwell_tts_f5 PRIVATE /utf-8)
endif()

# =============================================================================
# test_f5_tts_cuda_cpp -- the end-to-end smoke tool (reference WAV + text ->
# synthesised speech, on the GPU).
#
# An EXECUTABLE and not a ctest case, deliberately: it needs a ~1.3 GB model, a
# CUDA device and a reference clip. Putting that in the `validation` suite would
# make a fast CPU-only suite depend on all three. It graduates to `integration`
# when those assets have a pinned home.
#
# dr_wav comes from blackwell::sandbox_headers (header-only, SYSTEM includes, so
# outside the /W4 /WX budget). This TU is the ONLY one in the root build that
# defines DR_WAV_IMPLEMENTATION -- audio_sandbox's capture path defines
# MINIAUDIO_IMPLEMENTATION, not this one, so there is no duplicate-symbol clash.
# =============================================================================
add_executable(test_f5_tts_cuda_cpp f5_tts_cli.cpp dr_wav_impl.cpp)

# Per-SOURCE /W0 on the vendored amalgamation ONLY. Not per-target: f5_tts_cli.cpp
# stays under the full /W4 /WX budget. dr_wav's implementation emits C4701 during
# code generation, which MSVC resolves against the end-of-TU warning state, so no
# pragma around the #include can reach it -- hence isolation plus this flag.
# See dr_wav_impl.cpp for the full reasoning.
if(MSVC)
    set_source_files_properties(dr_wav_impl.cpp PROPERTIES COMPILE_OPTIONS "/W0")
endif()
target_link_libraries(test_f5_tts_cuda_cpp PRIVATE
    blackwell_tts_f5              # engine (brings blackwell_tts + audio_rt)
    blackwell::sandbox_headers    # dr_wav
    nlohmann_json::nlohmann_json) # f5_tts_contract.json

if(MSVC)
    # REQUIRED here above all: this tool's whole purpose is Cyrillic text, and
    # the --text/--ref-text literals and diagnostics are UTF-8.
    target_compile_options(test_f5_tts_cuda_cpp PRIVATE /utf-8)
endif()

# The CUDA EP provider DLLs must sit next to the exe or ORT silently falls back
# to the CPU provider -- which for this model means ~0.2x realtime instead of a
# failure. See cmake/OnnxRuntime.cmake.
blackwell_copy_onnxruntime_dlls(test_f5_tts_cuda_cpp)
# ...and cuDNN 9, which the GPU archive does NOT contain. Without it the session
# builds fine and then dies at the first Conv node. No-op when cuDNN was not
# found at configure time (the message says so there).
blackwell_copy_cudnn_dlls(test_f5_tts_cuda_cpp)
