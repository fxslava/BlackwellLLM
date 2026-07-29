# ONNXRuntime (CPU-only) acquisition + the blackwell::onnxruntime INTERFACE
# target, plus the pinned silero_vad.onnx model fetch. Included from the root
# CMakeLists AFTER the USE_SILERO_VAD option is defined. Everything below is a
# no-op when the option is OFF -- but blackwell_copy_onnxruntime_dlls() and
# blackwell_copy_vad_model() are ALWAYS defined, so consumers call them
# unconditionally instead of carrying their own if(USE_SILERO_VAD) guards
# (same contract as blackwell_copy_dstorage_dlls in DirectStorage.cmake).
#
# CPU-ONLY BY DESIGN. The VAD model is 2.3 MB and runs in well under a
# millisecond per 32 ms chunk on one core; a GPU execution provider would burn
# VRAM and bandwidth that the 8B AWQ backbone needs, and would drag a second
# CUDA runtime into the process. The CPU archive is also 75 MB against 434 MB
# for the CUDA 12 build.
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

# --- Pins (bump deliberately; re-record the hash when you do) -----------------
set(BLACKWELL_ORT_VERSION "1.28.0")
set(BLACKWELL_ORT_SHA256  "abef733dacbe2f571547a7150b479b5cb9cc0df22f96c24983a42cadb1b4f8bc")
# snakers4/silero-vad release tag + the SHA256 of that tag's silero_vad.onnx.
# v6.x keeps the v5 I/O signature this project's wrapper binds to:
#   input  input [B,N] f32 | state [2,B,128] f32 | sr <scalar> i64
#   output output [B,1] f32 | stateN [2,B,128] f32
# (The pre-v5 models exposed SEPARATE h/c state tensors -- a wrapper written
# against those will NOT bind to this file. Do not repin below v5.)
set(BLACKWELL_SILERO_TAG    "v6.2.1")
set(BLACKWELL_SILERO_SHA256 "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3")

# Copies the ONNXRuntime runtime DLL next to <target>'s executable (POST_BUILD,
# copy_if_different). No-op when the VAD is disabled. Only onnxruntime.dll is
# needed: the CPU execution provider is built INTO it, and the separate
# onnxruntime_providers_shared.dll exists solely for the out-of-process
# CUDA/TensorRT providers we deliberately do not use.
function(blackwell_copy_onnxruntime_dlls target)
    if(NOT TARGET blackwell_onnxruntime)
        return()
    endif()
    get_target_property(ort_bin blackwell_onnxruntime BLACKWELL_ORT_BIN_DIR)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ort_bin}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying onnxruntime.dll next to ${target}...")
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

include(FetchContent)
message(STATUS "Fetching ONNXRuntime ${BLACKWELL_ORT_VERSION} (CPU-only, win-x64)...")
FetchContent_Declare(onnxruntime_cpu
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${BLACKWELL_ORT_VERSION}/onnxruntime-win-x64-${BLACKWELL_ORT_VERSION}.zip"
    # Pinned hash: the TLS_VERIFY-ON house rule extended to content integrity --
    # a swapped release asset fails the configure instead of the build.
    URL_HASH SHA256=${BLACKWELL_ORT_SHA256}
    # The archive ships no CMakeLists, so MakeAvailable only POPULATES it.
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(onnxruntime_cpu)
FetchContent_GetProperties(onnxruntime_cpu SOURCE_DIR ORT_DIR)

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
add_library(blackwell_onnxruntime INTERFACE)
add_library(blackwell::onnxruntime ALIAS blackwell_onnxruntime)
# SYSTEM: ORT's headers are held to NONE of our /W4 /WX budget (/external:I -> W0).
target_include_directories(blackwell_onnxruntime SYSTEM INTERFACE "${ORT_DIR}/include")
target_link_libraries(blackwell_onnxruntime INTERFACE "${ORT_DIR}/lib/onnxruntime.lib")
target_compile_definitions(blackwell_onnxruntime INTERFACE BLACKWELL_HAVE_SILERO_VAD)
set_target_properties(blackwell_onnxruntime PROPERTIES
    BLACKWELL_ORT_BIN_DIR    "${ORT_DIR}/lib"
    BLACKWELL_VAD_MODEL_PATH "${BLACKWELL_VAD_MODEL}")
message(STATUS "ONNXRuntime integrated from: ${ORT_DIR}")
message(STATUS "Silero VAD model: ${BLACKWELL_VAD_MODEL} (${BLACKWELL_SILERO_TAG})")
