# =============================================================================
# whisper.cpp acquisition + the blackwell::whisper_cpp INTERFACE target.
#
# Included from the root CMakeLists AFTER USE_WHISPER_CPP / BLACKWELL_WHISPER_CUDA
# are defined, and BEFORE the /W4 /WX directory property is imposed -- see the
# WARNINGS section below, which is the one ordering constraint in this file that
# will silently cost an afternoon if it is broken.
#
# WHAT IT IS FOR. The Cascade ASR pipeline: a GGML Whisper model (large-v3-turbo
# class, e.g. Whisper-Turbo-Platinum-F16.bin) transcribes a VAD-bounded utterance
# to UTF-8, and that text is routed to the local text backbone. It is the
# ALTERNATIVE to the Ultravox audio head, never a companion to it -- cascade mode
# does not load the audio tower at all, which is where its VRAM comes from.
#
# Everything below is a no-op when the option is OFF, and NOTHING here is
# FATAL_ERROR on absence: a tree without whisper.cpp must still build, and the
# app must still run on the legacy Ultravox path. Same posture as
# cmake/WebRtcAec3.cmake and the audio head itself.
#
# =============================================================================
# WHY FetchContent AND NOT find_package
# =============================================================================
# whisper.cpp is a CMake project that vendors ggml, has no system packaging worth
# depending on, and is pinned by commit rather than by release cadence. That is
# exactly the shape cmake/Dependencies.cmake and cmake/OnnxRuntime.cmake already
# fetch. WebRTC is the exception in this repo, not the rule, and it is one only
# because it does not build with CMake at all.
#
# =============================================================================
# THE STATIC-CRT TRAP (read this before bumping the pin)
# =============================================================================
# The root pins a STATIC CRT for every first-party target
# (CMAKE_MSVC_RUNTIME_LIBRARY = MultiThreaded$<$<CONFIG:Debug>:Debug>), and
# whisper.cpp is linked STATICALLY into our executables -- so unlike
# onnxruntime.dll (an import library, no CRT objects linked in), its object files
# land in our binaries and its CRT choice MUST match ours. A mismatch is not a
# warning: it is LNK2038 "RuntimeLibrary=MT_StaticRelease vs MD_DynamicRelease",
# and it surfaces at the very end of a very long build.
#
# CMAKE_POLICY_DEFAULT_CMP0091=NEW below is what makes the subproject honour the
# abstract MSVC_RUNTIME_LIBRARY property instead of whatever its own CMAKE_*_FLAGS
# say. It is set as a DEFAULT (not just our own policy) precisely because the
# subproject's cmake_minimum_required may otherwise pull it back to OLD.
#
# =============================================================================
# WARNINGS: this must be included BEFORE the root's /W4 /WX block
# =============================================================================
# That block is a DIRECTORY property (add_compile_options at root scope), so it
# applies to every target created after it -- including the targets a later
# FetchContent_MakeAvailable creates. whisper.cpp and ggml do not compile clean
# under /W4 /WX and it is not ours to fix. Including this module alongside the
# other dependency modules (before the block) is what keeps the zero-warning
# policy pointed at first-party code. The SYSTEM include dirs below handle the
# other half: consumers of blackwell::whisper_cpp see its headers at /external:W0.
# =============================================================================

# --- Pin (bump deliberately) --------------------------------------------------
# A TAG, exposed as a cache variable so a build can be moved onto a specific
# commit without editing this file. whisper.cpp moves fast and its ggml submodule
# moves faster; treat a bump as a change that needs the cascade smoke test rerun,
# not as routine maintenance.
set(BLACKWELL_WHISPER_CPP_TAG "v1.8.2" CACHE STRING
    "whisper.cpp git tag or commit SHA to build the Cascade ASR against.")
set(BLACKWELL_WHISPER_CPP_REPO "https://github.com/ggml-org/whisper.cpp.git" CACHE STRING
    "whisper.cpp repository URL.")

# --- The GGML model path (NOT fetched) ----------------------------------------
# Deliberately not a FetchContent/file(DOWNLOAD) asset, unlike silero_vad.onnx:
# a turbo-class F16 model is ~1.6 GB and has no canonical, hash-stable URL this
# project can pin. It is configured, and it is ALSO resolvable at runtime from
# the settings file / --whisper-model / next to the exe (resolve_asset_file in
# voice_assistant/main.cpp), so a build with no model still produces a binary
# that runs -- on the legacy path, saying why.
set(BLACKWELL_WHISPER_MODEL "" CACHE FILEPATH
    "Path to the GGML Whisper model (e.g. Whisper-Turbo-Platinum-F16.bin). Empty = configure it at runtime.")

# Copies the configured GGML model next to <target>'s executable, so the app can
# find it beside itself with no path configuration at all (the first entry in the
# resolution order voice_assistant uses). ALWAYS DEFINED -- consumers call it
# unconditionally rather than carrying their own if(USE_WHISPER_CPP) guards, the
# same contract blackwell_copy_vad_model() and blackwell_copy_dstorage_dlls()
# offer. No-op when whisper.cpp is off or no model was configured.
#
# NOTE it is a copy of a ~1.6 GB file. copy_if_different makes that a one-time
# cost per build directory, but it is a real cost: leave BLACKWELL_WHISPER_MODEL
# empty and point the app at the model by settings/flag if you would rather not
# pay it.
function(blackwell_copy_whisper_model target)
    if(NOT TARGET blackwell_whisper_cpp)
        return()
    endif()
    get_target_property(_wmodel blackwell_whisper_cpp BLACKWELL_WHISPER_MODEL_PATH)
    if(NOT _wmodel)
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_wmodel}"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying the GGML Whisper model next to ${target}...")
endfunction()

if(NOT USE_WHISPER_CPP)
    message(STATUS "whisper.cpp: disabled (USE_WHISPER_CPP=OFF) -- Cascade ASR unavailable, "
                   "voice_assistant runs the legacy Ultravox path only")
    return()
endif()

# --- Subproject configuration -------------------------------------------------
# Every one of these is set as a normal (non-cache) variable ahead of
# FetchContent_MakeAvailable so the subproject's own option() calls see them as
# already-defined and do not override them (CMP0077 NEW, which cmake>=3.28 gives
# us). Nothing here leaks back out: they are scoped to this directory.
set(BUILD_SHARED_LIBS       OFF)  # static into our exes; see the CRT trap above
set(WHISPER_BUILD_TESTS     OFF)
set(WHISPER_BUILD_EXAMPLES  OFF)
set(WHISPER_BUILD_SERVER    OFF)
set(GGML_BUILD_TESTS        OFF)
set(GGML_BUILD_EXAMPLES     OFF)
# The CLI tools pull SDL2 and a lot else; we consume the library only.
set(WHISPER_SDL2            OFF)

# THE CRT POLICY. See the header block -- without this the subproject may build
# /MD against our /MT and fail at final link.
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# --- CUDA backend -------------------------------------------------------------
# ON by default for this project: Cascade mode exists to lower latency, and the
# CPU backend puts a turbo-class encode at several hundred milliseconds per
# utterance. The cost is stated rather than hidden:
#
#   * A THIRD CUDA CONTEXT in-process (engine + ONNXRuntime CUDA EP + ggml), each
#     with its own allocator and stream pool. ggml owns its streams; there is no
#     API to hand it ours, so the isolation is by context, not by stream.
#   * ~1.6 GB of VRAM for a turbo F16 model, which is affordable ONLY because
#     cascade mode does not load the Ultravox audio head. Watch the [vram] lines.
#   * A long first build: ggml-cuda is a large translation-unit set, compiled for
#     every arch in CMAKE_CUDA_ARCHITECTURES.
#
# SM CONTENTION IS REAL AND IS MANAGED ABOVE THIS LAYER, not here: the ASR runs
# on its own worker thread, off the decode loop's critical path, and is bounded
# to one in-flight utterance (see whisper_cascade_mode.hpp). Nothing in CMake can
# make two CUDA contexts not share SMs.
if(BLACKWELL_WHISPER_CUDA)
    set(GGML_CUDA ON)
    # ggml derives its own arch list when this is unset, and its default is a
    # broad matrix that would multiply an already-long build. Inherit the
    # project's pinned matrix instead -- 120-real (Blackwell) today.
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES_SAVED "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    message(STATUS "whisper.cpp: fetching ${BLACKWELL_WHISPER_CPP_TAG} with the ggml CUDA backend "
                   "(arch: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(GGML_CUDA OFF)
    message(STATUS "whisper.cpp: fetching ${BLACKWELL_WHISPER_CPP_TAG} with the ggml CPU backend "
                   "(BLACKWELL_WHISPER_CUDA=OFF)")
endif()

include(FetchContent)
FetchContent_Declare(whisper_cpp
    GIT_REPOSITORY "${BLACKWELL_WHISPER_CPP_REPO}"
    GIT_TAG        "${BLACKWELL_WHISPER_CPP_TAG}"
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(whisper_cpp)

if(NOT TARGET whisper)
    message(STATUS
        "whisper.cpp: fetched, but no 'whisper' target was created -- the upstream "
        "target layout changed at ${BLACKWELL_WHISPER_CPP_TAG}.\n"
        "             Cascade ASR is unavailable; the legacy Ultravox path is unaffected.")
    return()
endif()

# --- The usage-requirements target --------------------------------------------
# Pure usage requirements, matching blackwell::onnxruntime's and
# blackwell::webrtc_aec3's shape: a consumer links ONE name and inherits the
# headers (quarantined), the archives, and the BLACKWELL_HAVE_WHISPER_CPP define
# that gates every use site.
add_library(blackwell_whisper_cpp INTERFACE)
add_library(blackwell::whisper_cpp ALIAS blackwell_whisper_cpp)
target_link_libraries(blackwell_whisper_cpp INTERFACE whisper)
target_compile_definitions(blackwell_whisper_cpp INTERFACE BLACKWELL_HAVE_WHISPER_CPP)

# SYSTEM, deliberately. whisper.h and the ggml headers are held to NONE of our
# /W4 /WX budget (/external:I -> W0), exactly as ORT's and WebRTC's are. Taken
# from the target rather than guessed from the source dir, because whisper.cpp
# has moved its public headers between layouts more than once.
get_target_property(_whisper_inc whisper INTERFACE_INCLUDE_DIRECTORIES)
if(_whisper_inc)
    target_include_directories(blackwell_whisper_cpp SYSTEM INTERFACE ${_whisper_inc})
endif()
unset(_whisper_inc)

# Only the CUDA backend is behind this define. Code that merely needs a context
# (the ASR wrapper) must keep compiling either way, so nothing gates on it except
# the places that must select a device or account for VRAM.
if(BLACKWELL_WHISPER_CUDA)
    target_compile_definitions(blackwell_whisper_cpp INTERFACE BLACKWELL_HAVE_WHISPER_CUDA)
endif()

# The model path rides along as a property for the copy helper above, so no
# directory-scope variable leaks between scripts (the DirectStorage.cmake idiom).
set_target_properties(blackwell_whisper_cpp PROPERTIES
    BLACKWELL_WHISPER_MODEL_PATH "${BLACKWELL_WHISPER_MODEL}"
    BLACKWELL_WHISPER_CUDA_ENABLED "${BLACKWELL_WHISPER_CUDA}")

# Tuck the fetched targets under a solution folder so they do not clutter the
# top level of Solution Explorer next to first-party projects (USE_FOLDERS is on).
foreach(_t whisper ggml ggml-base ggml-cpu ggml-cuda)
    if(TARGET ${_t})
        set_target_properties(${_t} PROPERTIES FOLDER "ThirdParty/whisper.cpp")
    endif()
endforeach()

message(STATUS "whisper.cpp integrated from: ${whisper_cpp_SOURCE_DIR}")
if(BLACKWELL_WHISPER_MODEL)
    if(EXISTS "${BLACKWELL_WHISPER_MODEL}")
        message(STATUS "Whisper GGML model: ${BLACKWELL_WHISPER_MODEL}")
    else()
        # Not fatal: the app resolves the model at runtime too, and a configure
        # that fails over a missing 1.6 GB asset would be a worse trade than a
        # binary that starts and says the model is absent.
        message(STATUS "Whisper GGML model: ${BLACKWELL_WHISPER_MODEL} -- NOT FOUND at configure "
                       "time; set it at runtime (Settings -> Whisper model, or --whisper-model)")
    endif()
else()
    message(STATUS "Whisper GGML model: not configured -- set -DBLACKWELL_WHISPER_MODEL=<path> "
                   "to deploy it next to the exe, or configure it at runtime")
endif()
