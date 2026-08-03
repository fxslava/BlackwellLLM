# ONNXRuntime acquisition + the blackwell::onnxruntime INTERFACE target, plus
# the pinned silero_vad.onnx model fetch. Included from the root CMakeLists
# AFTER the USE_SILERO_VAD and BLACKWELL_ORT_GPU options are defined. Everything
# below is a no-op when USE_SILERO_VAD is OFF -- but
# blackwell_copy_onnxruntime_dlls() and blackwell_copy_vad_model() are ALWAYS
# defined, so consumers call them unconditionally instead of carrying their own
# if(USE_SILERO_VAD) guards (same contract as blackwell_copy_dstorage_dlls in
# DirectStorage.cmake).
#
# WHICH BUILD, AND WHY THE DEFAULT MOVED (2026-08).
# This module used to fetch the CPU archive unconditionally, and the argument
# for that was sound FOR ITS ONLY CONSUMER: the VAD model is 2.3 MB and runs in
# well under a millisecond per 32 ms chunk on one core, so a GPU provider would
# burn VRAM and bandwidth the 8B AWQ backbone needs for no latency win. That
# reasoning still holds for blackwell_vad and is not being retracted.
#
# What changed is that a SECOND consumer arrived with the opposite profile.
# blackwell_tts_f5 runs an F5-TTS flow-matching DiT -- ~330 M parameters
# evaluated 16-32 times per utterance -- which measures ~1.8 s PER STEP on this
# box's CPU (~0.2x realtime, measured via scripts/export_f5_tts_onnx.py's
# verification path). CPU synthesis is therefore not slow, it is unusable: the
# playback sink starves permanently. That consumer needs the CUDA EP, and the
# CUDA EP is only in the GPU archive.
#
# So the GPU build is now the default and the CPU build is the fallback. The
# cost is honest and worth stating: 349-434 MB fetched instead of 75 MB, a
# second CUDA runtime in-process, and real SM contention with the decode loop
# (see the header block of src/tts/f5_tts_engine.hpp). Configure with
# -DBLACKWELL_ORT_GPU=OFF to get the old behaviour; blackwell_vad is identical
# either way, and blackwell_tts_f5 simply does not exist.
#
# CUDA MAJOR VERSION IS PART OF THE PIN. From 1.28.0 Microsoft ships SEPARATE
# GPU archives per CUDA major (gpu_cuda12 / gpu_cuda13) -- there is no plain
# "onnxruntime-win-x64-gpu-<ver>.zip" any more, and a URL guessed on the old
# naming 404s. Worse, the two are NOT interchangeable at runtime: loading a
# provider DLL built against the wrong CUDA major fails inside LoadLibrary with
# a message that names neither CUDA nor the version. The check below turns that
# into a configure-time error instead.
#
# CRT NOTE (/MT vs /MD). The root pins a STATIC CRT for every first-party
# target, while the prebuilt onnxruntime.dll links the DYNAMIC CRT. That is not
# the LNK2038 hazard it looks like: onnxruntime.lib is an IMPORT library, so no
# CRT objects are linked into our binaries, and the ORT C API never passes CRT
# state (FILE*, malloc'd blocks, C++ exceptions) across the DLL seam -- tensors
# are allocated through ORT's own allocator and errors come back as OrtStatus*.
# This is exactly the argument the root CMakeLists already makes for the SHARED
# blackwell_core.dll. The price is that onnxruntime.dll must sit next to every
# consumer executable, which is what the copy helper below is for.

# --- Pins (bump deliberately; re-record the hashes when you do) ---------------
# All three digests below are the ones GitHub reports for the v1.28.0 release
# assets (the `digest` field of the releases API), which is the same SHA256
# URL_HASH verifies. Re-record them from
#   https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v<version>
# rather than by hand -- and note the CPU digest is unchanged from the pin this
# file has always carried, which is what confirms the source is the right one.
set(BLACKWELL_ORT_VERSION "1.28.0")
set(BLACKWELL_ORT_SHA256_CPU
    "abef733dacbe2f571547a7150b479b5cb9cc0df22f96c24983a42cadb1b4f8bc")   #  75.1 MB
set(BLACKWELL_ORT_SHA256_GPU_CUDA12
    "6b7bf16d6d30180db7f386fb179aa4e4f1313f0924531a2879b7b090b56518c1")   # 434.3 MB
set(BLACKWELL_ORT_SHA256_GPU_CUDA13
    "137f0822a4923b1d84d3e09496e0792ebbb221eb3a61a0657f71a12ab68ab1e2")   # 348.9 MB

# Which GPU archive. Defaults to the major of the CUDA toolkit this build is
# already using, because that is the only value that can possibly be right --
# see the CUDA-major note in the header block.
set(BLACKWELL_ORT_CUDA_MAJOR "" CACHE STRING
    "CUDA major for the ONNXRuntime GPU archive: 12 or 13. Empty = match the detected CUDA toolkit.")
set_property(CACHE BLACKWELL_ORT_CUDA_MAJOR PROPERTY STRINGS "" "12" "13")
# snakers4/silero-vad release tag + the SHA256 of that tag's silero_vad.onnx.
# v6.x keeps the v5 I/O signature this project's wrapper binds to:
#   input  input [B,N] f32 | state [2,B,128] f32 | sr <scalar> i64
#   output output [B,1] f32 | stateN [2,B,128] f32
# (The pre-v5 models exposed SEPARATE h/c state tensors -- a wrapper written
# against those will NOT bind to this file. Do not repin below v5.)
set(BLACKWELL_SILERO_TAG    "v6.2.1")
set(BLACKWELL_SILERO_SHA256 "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3")

# Copies the ONNXRuntime runtime DLLs next to <target>'s executable (POST_BUILD,
# copy_if_different). No-op when the VAD is disabled.
#
# WHICH DLLS, AND WHY IT IS NOT JUST ONE. On a CPU build, onnxruntime.dll is the
# whole runtime -- the CPU execution provider is compiled INTO it. On a GPU
# build the CUDA EP is deliberately NOT: it lives in
# onnxruntime_providers_cuda.dll and reaches the core through
# onnxruntime_providers_shared.dll, which is the ABI shim ORT uses to keep
# provider plugins loadable across builds. All three must sit beside the exe.
#
# The failure mode when they do not is the reason this is spelled out: ORT does
# not error at session creation. It silently falls back to the CPU provider and
# the app runs -- 50-100x slower, which on a synthesis path reads as "the model
# is broken" rather than "a DLL is missing". Nothing in the log says CUDA.
#
# TensorRT's provider DLL is deliberately NOT copied: it additionally requires a
# TensorRT installation this project does not depend on, and copying it would
# only move the failure later.
#
# The exact list is resolved once at configure time (existence-checked against
# the extracted archive) and carried on the target as a property, so this
# function never has to re-derive the CPU/GPU distinction.
function(blackwell_copy_onnxruntime_dlls target)
    if(NOT TARGET blackwell_onnxruntime)
        return()
    endif()
    get_target_property(ort_dlls blackwell_onnxruntime BLACKWELL_ORT_RUNTIME_DLLS)
    if(NOT ort_dlls)
        return()
    endif()
    list(LENGTH ort_dlls _n)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ort_dlls}
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying ${_n} ONNXRuntime DLL(s) next to ${target}...")
endfunction()

# Copies the cuDNN 9 runtime next to <target>'s executable.
#
# WHY THIS IS SEPARATE FROM THE ORT COPY, AND WHY IT IS NEEDED AT ALL. The
# ONNXRuntime GPU archive does NOT contain cuDNN -- Microsoft expects it on the
# system. Without it the CUDA EP loads, the session is created, warm-up passes,
# and then the FIRST Conv node fails at run time with
#     NOT_IMPLEMENTED : cuDNN is unavailable or disabled for CUDA Execution
#     Provider: LoadLibrary failed for cudnn64_9.dll with error 2
# i.e. after the model is already resident in VRAM. Measured on this box before
# the copy existed.
#
# WHERE IT COMES FROM. There is no canonical Windows location, so this probes,
# in order: an explicit -DBLACKWELL_CUDNN_DIR, the CUDA toolkit's own bin, and
# the NVIDIA cuDNN installer's default tree. It deliberately does NOT scavenge a
# Python package's copy: torch ships cuDNN under site-packages, which works on a
# dev box and is not a defensible build dependency for something this repo
# installs (deploy/installer.iss). Point BLACKWELL_CUDNN_DIR at a real cuDNN 9
# redistributable to ship.
#
# NOT FATAL when absent: the build still produces a working CPU-EP binary, and
# saying so at configure time is far cheaper than the runtime error above.
function(blackwell_copy_cudnn_dlls target)
    if(NOT TARGET blackwell_onnxruntime)
        return()
    endif()
    get_target_property(_gpu blackwell_onnxruntime BLACKWELL_ORT_GPU_ENABLED)
    if(NOT _gpu)
        return()
    endif()
    get_target_property(_cudnn_dlls blackwell_onnxruntime BLACKWELL_CUDNN_DLLS)
    if(NOT _cudnn_dlls)
        return()
    endif()
    list(LENGTH _cudnn_dlls _n)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_cudnn_dlls}
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying ${_n} cuDNN DLL(s) next to ${target}...")
endfunction()

# Copies silero_vad.onnx next to <target>'s executable, so a consumer can find
# the model beside itself with no path configuration at all (the first entry in
# the resolution order the apps use). No-op when the VAD is disabled.
function(blackwell_copy_vad_model target)
    if(NOT TARGET blackwell_onnxruntime)
        return()
    endif()
    get_target_property(vad_model blackwell_onnxruntime BLACKWELL_VAD_MODEL_PATH)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${vad_model}"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying silero_vad.onnx next to ${target}...")
endfunction()

if(NOT (USE_SILERO_VAD AND WIN32))
    return()
endif()

# --- pick the archive ---------------------------------------------------------
# Resolved into: _ort_flavour (FetchContent name, distinct per variant so
# flipping the option cannot collide with a previously extracted tree),
# _ort_archive (asset stem) and _ort_sha256.
if(BLACKWELL_ORT_GPU)
    # Decide the CUDA major. An explicit -DBLACKWELL_ORT_CUDA_MAJOR wins;
    # otherwise match the toolkit this build already compiles CUDA with, since
    # that is the only value with a chance of being right.
    set(_ort_cuda_major "${BLACKWELL_ORT_CUDA_MAJOR}")
    if(NOT _ort_cuda_major)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(_ort_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
        elseif(CMAKE_CUDA_COMPILER_VERSION)
            string(REGEX MATCH "^[0-9]+" _ort_cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
        else()
            message(FATAL_ERROR
                "BLACKWELL_ORT_GPU=ON but no CUDA toolkit was detected, so the "
                "correct ONNXRuntime GPU archive cannot be chosen.\n"
                "  Set -DBLACKWELL_ORT_CUDA_MAJOR=12 or =13 explicitly, or "
                "configure with -DBLACKWELL_ORT_GPU=OFF.")
        endif()
    endif()

    if(_ort_cuda_major STREQUAL "13")
        set(_ort_sha256 "${BLACKWELL_ORT_SHA256_GPU_CUDA13}")
    elseif(_ort_cuda_major STREQUAL "12")
        set(_ort_sha256 "${BLACKWELL_ORT_SHA256_GPU_CUDA12}")
    else()
        # Fail rather than fall back to CPU. A silent downgrade here surfaces as
        # a 50-100x slowdown with nothing in the log naming CUDA, which is the
        # single most expensive way this could go wrong.
        message(FATAL_ERROR
            "ONNXRuntime ${BLACKWELL_ORT_VERSION} ships GPU archives for CUDA 12 "
            "and 13 only, but this build's CUDA toolkit is major "
            "'${_ort_cuda_major}'.\n"
            "  Install a CUDA 12.x or 13.x toolkit, pin the archive explicitly "
            "with -DBLACKWELL_ORT_CUDA_MAJOR=<12|13>, or build without the CUDA "
            "execution provider via -DBLACKWELL_ORT_GPU=OFF.")
    endif()

    set(_ort_flavour "onnxruntime_gpu_cuda${_ort_cuda_major}")
    set(_ort_archive "onnxruntime-win-x64-gpu_cuda${_ort_cuda_major}-${BLACKWELL_ORT_VERSION}")
    message(STATUS
        "Fetching ONNXRuntime ${BLACKWELL_ORT_VERSION} "
        "(GPU / CUDA ${_ort_cuda_major} execution provider, win-x64)...")
else()
    set(_ort_flavour "onnxruntime_cpu")
    set(_ort_archive "onnxruntime-win-x64-${BLACKWELL_ORT_VERSION}")
    set(_ort_sha256  "${BLACKWELL_ORT_SHA256_CPU}")
    message(STATUS "Fetching ONNXRuntime ${BLACKWELL_ORT_VERSION} (CPU-only, win-x64)...")
endif()

include(FetchContent)
FetchContent_Declare(${_ort_flavour}
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${BLACKWELL_ORT_VERSION}/${_ort_archive}.zip"
    # Pinned hash: the TLS_VERIFY-ON house rule extended to content integrity --
    # a swapped release asset fails the configure instead of the build.
    URL_HASH SHA256=${_ort_sha256}
    # The archive ships no CMakeLists, so MakeAvailable only POPULATES it.
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(${_ort_flavour})
FetchContent_GetProperties(${_ort_flavour} SOURCE_DIR ORT_DIR)

# --- silero_vad.onnx (2.3 MB, fetched once at CONFIGURE time) -----------------
# Lands in the SOURCE tree under models/ (gitignored) rather than the build tree:
# it is a stable, hash-pinned asset shared by every build directory, and keeping
# it out of out/build means a clean reconfigure does not re-download it.
# EXPECTED_HASH makes this self-healing AND idempotent -- CMake verifies an
# existing file and skips the download when it already matches.
set(BLACKWELL_VAD_MODEL "${CMAKE_SOURCE_DIR}/models/silero_vad.onnx")
file(DOWNLOAD
    "https://raw.githubusercontent.com/snakers4/silero-vad/${BLACKWELL_SILERO_TAG}/src/silero_vad/data/silero_vad.onnx"
    "${BLACKWELL_VAD_MODEL}"
    EXPECTED_HASH SHA256=${BLACKWELL_SILERO_SHA256}
    TLS_VERIFY ON
    STATUS _silero_dl)
list(GET _silero_dl 0 _silero_rc)
if(NOT _silero_rc EQUAL 0)
    list(GET _silero_dl 1 _silero_msg)
    message(FATAL_ERROR
        "Failed to fetch silero_vad.onnx (${_silero_msg}).\n"
        "Fix the network/proxy and reconfigure, download it manually to\n"
        "  ${BLACKWELL_VAD_MODEL}\n"
        "from https://github.com/snakers4/silero-vad/blob/${BLACKWELL_SILERO_TAG}/src/silero_vad/data/silero_vad.onnx\n"
        "or configure with -DUSE_SILERO_VAD=OFF to build without the neural VAD.")
endif()

# Pure usage-requirements target: linking it (even PRIVATE) gives a consumer the
# ORT headers, the import lib, and the BLACKWELL_HAVE_SILERO_VAD define that
# gates the wrapper's TU. The runtime DLL dir and the model path ride along as
# properties for the copy helpers above, so no directory-scope variable leaks
# between scripts (the DirectStorage.cmake idiom).
# --- runtime DLL set, resolved and CHECKED once -------------------------------
# Verified here rather than trusted, because the archive layout is the one thing
# in this file that upstream can change without changing the version number --
# and a missing provider DLL degrades silently to the CPU provider at runtime
# (see blackwell_copy_onnxruntime_dlls). Failing the configure is the only place
# this is cheap to notice.
set(_ort_required_dlls "onnxruntime.dll")
if(BLACKWELL_ORT_GPU)
    list(APPEND _ort_required_dlls
        "onnxruntime_providers_shared.dll"     # the provider-plugin ABI shim
        "onnxruntime_providers_cuda.dll")      # the CUDA EP itself
endif()

set(BLACKWELL_ORT_DLLS "")
foreach(_dll IN LISTS _ort_required_dlls)
    if(NOT EXISTS "${ORT_DIR}/lib/${_dll}")
        message(FATAL_ERROR
            "ONNXRuntime archive '${_ort_archive}' does not contain lib/${_dll}.\n"
            "  Extracted to: ${ORT_DIR}\n"
            "  The archive layout or asset naming changed upstream. Re-check the "
            "release assets and update the pins at the top of cmake/OnnxRuntime.cmake.")
    endif()
    list(APPEND BLACKWELL_ORT_DLLS "${ORT_DIR}/lib/${_dll}")
endforeach()

# --- cuDNN 9 discovery (GPU builds only) --------------------------------------
# See blackwell_copy_cudnn_dlls above for why this is needed and why the search
# order is what it is.
set(BLACKWELL_CUDNN_DIR "" CACHE PATH
    "Directory containing cudnn64_9.dll and friends. Empty = probe the CUDA toolkit and the cuDNN installer default.")
set(BLACKWELL_CUDNN_DLLS "")
if(BLACKWELL_ORT_GPU)
    set(_cudnn_candidates "${BLACKWELL_CUDNN_DIR}")
    if(CUDAToolkit_BIN_DIR)
        list(APPEND _cudnn_candidates "${CUDAToolkit_BIN_DIR}")
    endif()
    file(GLOB _cudnn_installed "C:/Program Files/NVIDIA/CUDNN/v9*/bin/*")
    list(APPEND _cudnn_candidates ${_cudnn_installed})

    foreach(_dir IN LISTS _cudnn_candidates)
        if(_dir AND EXISTS "${_dir}/cudnn64_9.dll")
            # The whole family, not just the loader: cudnn64_9.dll is a ~300 KB
            # shim that dlopens cudnn_graph/cudnn_engines_*/cudnn_ops at runtime,
            # so copying it alone reproduces the same failure one level deeper.
            file(GLOB BLACKWELL_CUDNN_DLLS "${_dir}/cudnn*.dll")
            message(STATUS "cuDNN 9 found: ${_dir}")
            break()
        endif()
    endforeach()

    if(NOT BLACKWELL_CUDNN_DLLS)
        message(STATUS
            "cuDNN 9 NOT found -- the ONNXRuntime CUDA execution provider needs it and "
            "will fail at the first Conv node with 'LoadLibrary failed for cudnn64_9.dll'. "
            "Install the cuDNN 9 redistributable or configure with "
            "-DBLACKWELL_CUDNN_DIR=<dir containing cudnn64_9.dll>.")
    endif()
endif()

add_library(blackwell_onnxruntime INTERFACE)
add_library(blackwell::onnxruntime ALIAS blackwell_onnxruntime)
# SYSTEM: ORT's headers are held to NONE of our /W4 /WX budget (/external:I -> W0).
target_include_directories(blackwell_onnxruntime SYSTEM INTERFACE "${ORT_DIR}/include")
target_link_libraries(blackwell_onnxruntime INTERFACE "${ORT_DIR}/lib/onnxruntime.lib")
target_compile_definitions(blackwell_onnxruntime INTERFACE BLACKWELL_HAVE_SILERO_VAD)
# Only the CUDA EP is behind this define -- code that merely needs a session
# (the VAD) must keep compiling either way, so nothing gates on it except the
# CUDA-bound TTS engine.
if(BLACKWELL_ORT_GPU)
    target_compile_definitions(blackwell_onnxruntime INTERFACE BLACKWELL_HAVE_ORT_CUDA)
endif()
set_target_properties(blackwell_onnxruntime PROPERTIES
    BLACKWELL_ORT_BIN_DIR      "${ORT_DIR}/lib"
    BLACKWELL_ORT_RUNTIME_DLLS "${BLACKWELL_ORT_DLLS}"
    BLACKWELL_ORT_GPU_ENABLED  "${BLACKWELL_ORT_GPU}"
    BLACKWELL_CUDNN_DLLS       "${BLACKWELL_CUDNN_DLLS}"
    BLACKWELL_VAD_MODEL_PATH   "${BLACKWELL_VAD_MODEL}")
message(STATUS "ONNXRuntime integrated from: ${ORT_DIR}")
list(LENGTH BLACKWELL_ORT_DLLS _ort_dll_count)
message(STATUS "ONNXRuntime runtime DLLs (${_ort_dll_count}): ${_ort_required_dlls}")
message(STATUS "Silero VAD model: ${BLACKWELL_VAD_MODEL} (${BLACKWELL_SILERO_TAG})")
