# =============================================================================
# audio_sandbox — Whisper large-v3-turbo log-mel front-end (the audio encoder
# used by fixie-ai/ultravox-v0_5-llama-3_2-1b). Two products, one verified DSP:
#
#   parity_check    CPU offline parity tester vs the PyTorch golden dump.
#   audio_realtime  Windows real-time visualiser: miniaudio capture + Direct2D
#                   spectrogram + Dear ImGui controls + VAD .wav recorder.
#
# INTEGRATED into the root BlackwellLLM build via add_subdirectory(audio_sandbox).
# The third-party deps (dr_wav, pocketfft, miniaudio, imgui) are the SHARED root
# targets fetched ONCE in cmake/Dependencies.cmake -- this dir no longer declares
# its own FetchContent. It stays CUDA-free with no engine deps.
#
# STILL STANDALONE-CAPABLE: configuring this directory as the top-level source
# re-uses the very same dependency module, so the old flow keeps working:
#     cmake -S audio_sandbox -B audio_sandbox/out
#     cmake --build audio_sandbox/out --config Release
# =============================================================================

# --- Standalone bootstrap: only when this dir is the top-level CMake source ----
# When built from the root, project()/deps/warning-policy all come from the root
# CMakeLists; this block runs ONLY for a direct `cmake -S audio_sandbox` build.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.28)
    project(audio_sandbox LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()

    # Match the root's static-CRT deployment so a standalone build behaves the
    # same as the integrated one.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # Pull the SHARED third-party targets (imgui, blackwell::sandbox_headers, ...)
    # from the single source of truth so nothing is duplicated across build modes.
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Dependencies.cmake)

    # No root /W4 /WX directory property exists in a standalone build, so opt the
    # first-party targets in explicitly (see the per-target flags at the bottom).
    set(_sandbox_standalone TRUE)
endif()

# The shared deps must exist by now (root or standalone bootstrap provided them).
if(NOT TARGET blackwell::sandbox_headers)
    message(FATAL_ERROR
        "audio_sandbox: shared third-party targets missing. Build from the repo "
        "root (add_subdirectory) or configure audio_sandbox as the top-level "
        "source so cmake/Dependencies.cmake runs.")
endif()

# --- DSP library (the VERIFIED core; shared by both products) -----------------
add_library(whisper_dsp STATIC src/whisper_dsp.cpp)
target_include_directories(whisper_dsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(whisper_dsp PRIVATE blackwell::sandbox_headers)

# --- parity_check: offline golden parity tester ------------------------------
add_executable(parity_check src/parity_check.cpp)
target_link_libraries(parity_check PRIVATE whisper_dsp blackwell::sandbox_headers)

# --- audio_realtime: capture + Direct2D + ImGui + VAD recorder (Windows only) -
if(WIN32)
    add_executable(audio_realtime
        src/main.cpp
        src/audio_capture.cpp
        src/audio_recorder.cpp
        src/realtime_dsp.cpp
        src/window_d2d.cpp)
    # imgui is the SHARED root static lib (core + Win32/DX11 backends); its SYSTEM
    # PUBLIC include dirs propagate here, quarantined from our /W4 budget.
    target_link_libraries(audio_realtime PRIVATE
        whisper_dsp blackwell::sandbox_headers imgui
        d3d11 dxgi d2d1 d3dcompiler user32 ole32)  # DX11 + DXGI + D2D + ImGui shaders + Win32 + COM
    # UNICODE aligns the generic Win32 macros (IDC_ARROW etc.) with the explicit
    # -W APIs we call. (NOMINMAX / WIN32_LEAN_AND_MEAN are set in the sources that
    # include <windows.h>, so they are not repeated here.)
    target_compile_definitions(audio_realtime PRIVATE UNICODE _UNICODE)
    if(MSVC)
        target_compile_definitions(audio_realtime PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()

# Default the Visual Studio "Startup Project" to a runnable EXE (not the
# whisper_dsp .lib, which VS otherwise picks alphabetically and then fails to
# launch with "not a valid Win32 application"). Also anchor the debugger working
# directory to the sandbox root so the default "data" arg (mel_filters.bin etc.)
# resolves under F5 instead of the build/output folder.
if(MSVC)
    if(TARGET audio_realtime)
        set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT audio_realtime)
        set_target_properties(audio_realtime PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    else()
        set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT parity_check)
    endif()
    set_target_properties(parity_check PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# --- audio_translator: real-time speech translation on the LIVE speech pipeline --
# (include/bridge/speculative_bridge_api.h). UNLIKE audio_realtime, this links the
# engine control plane (blackwell_bridge) + the engine white-box tier
# (blackwell_core_obj, which pulls CUDA), so it is built ONLY in the integrated
# (root) build where those targets already exist. A standalone `cmake -S
# audio_sandbox` skips it and stays CUDA-free. A GPU-free MockEngineControl drives
# the REAL EngineControlBridge (SPSC command ring + barge-in epoch + wait/pump).
# It now drives a REAL BlackwellEngine (8B AWQ) via RealEngineControl: --model-dir
# selects the checkpoint, config.json is validated against the projector width
# before any CUDA alloc, and each commit streams a genuine GPU decode. The live
# audio->embeddings splice (Whisper encoder -> projector) is the one marked seam.
if(WIN32 AND TARGET blackwell_bridge AND TARGET blackwell_core_obj)
    add_executable(audio_translator
        translator/main.cpp
        # Reuse the sandbox capture/DSP/GUI sources by recompiling them into this
        # target, so audio_realtime's own target stays byte-for-byte untouched. The
        # optional PCM-tap / extra-panel hooks in these files are inert unless a
        # consumer sets them (audio_realtime never does).
        src/audio_capture.cpp
        src/audio_recorder.cpp
        src/realtime_dsp.cpp
        src/window_d2d.cpp)

    # Public bridge headers arrive via blackwell_bridge (PUBLIC include/); the
    # white-box bridge/engine headers it consumes are PRIVATE, so declare them here
    # (mirrors tests/bridge). translator/ holds the app-local mock + transcript view.
    target_include_directories(audio_translator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/translator  # transcript_view.hpp, mock_engine_control.hpp
        ${CMAKE_SOURCE_DIR}/src                  # common.h
        ${CMAKE_SOURCE_DIR}/src/core             # device_buffer.h, engine_impl.h (white-box step_*)
        ${CMAKE_SOURCE_DIR}/src/audio            # ping_pong_audio_buffer.h (audio prefill consumer)
        ${CMAKE_SOURCE_DIR}/src/bridge)          # engine_control_bridge.hpp, bridge_internal.hpp

    # whisper_dsp/sandbox_headers/imgui = the sandbox front-end (SYSTEM includes
    # quarantined from /W4); blackwell_bridge + blackwell_kernels + blackwell_core_obj
    # = the speech C-ABI and the engine symbols its control plane references (the
    # same trio tests/bridge links; the engine is never invoked by the mock).
    target_link_libraries(audio_translator PRIVATE
        whisper_dsp blackwell::sandbox_headers imgui
        d3d11 dxgi d2d1 d3dcompiler user32 ole32
        blackwell_bridge blackwell_kernels blackwell_core_obj
        # audio frontend: WhisperEncoder + UltravoxProjector (the audio-head
        # pipeline real_engine_control loads); pulls CUDA::cublas transitively.
        blackwell_audio
        # config.json parsing (cli_config.hpp) -- header-only, angle-bracket
        # include keeps it outside the /W4 /WX budget (/external:W0).
        nlohmann_json::nlohmann_json)

    # Silero neural VAD. Optional by construction: with -DUSE_SILERO_VAD=OFF the
    # target does not exist, cli_config.hpp's BLACKWELL_VAD_MODEL_PATH stays
    # undefined, and the app runs the pipeline's built-in RMS threshold detector.
    # blackwell_vad's PUBLIC include dir carries silero_vad.hpp; ONNXRuntime
    # itself stays PRIVATE to that lib (PIMPL), so nothing here sees ORT headers.
    if(TARGET blackwell_vad)
        target_link_libraries(audio_translator PRIVATE blackwell_vad)
        get_target_property(_vad_model blackwell_onnxruntime BLACKWELL_VAD_MODEL_PATH)
        target_compile_definitions(audio_translator PRIVATE
            BLACKWELL_VAD_MODEL_PATH="${_vad_model}")
        # The DLL must sit next to the exe; the model is copied too so a deployed
        # build resolves it beside the binary even without the configure-time path.
        blackwell_copy_onnxruntime_dlls(audio_translator)
        blackwell_copy_vad_model(audio_translator)
    endif()

    target_compile_definitions(audio_translator PRIVATE UNICODE _UNICODE)
    if(MSVC)
        target_compile_definitions(audio_translator PRIVATE _CRT_SECURE_NO_WARNINGS)
        # The mock's canned Russian translations are UTF-8 string literals: tell
        # MSVC the source AND execution charset is UTF-8 so ImGui receives correct
        # bytes and /WX never trips C4819 on the non-ASCII source.
        target_compile_options(audio_translator PRIVATE /utf-8)
    endif()

    # DirectStorage runtime DLLs next to the exe (blackwell_core_obj may load them).
    blackwell_copy_dstorage_dlls(audio_translator)

    if(MSVC)
        # Anchor the debugger working dir so the default "data" arg (mel_filters.bin)
        # resolves under F5, matching audio_realtime.
        set_target_properties(audio_translator PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()

# --- warnings: strict /W4 /WX on ALL first-party code, quarantining third-party -
# When integrated, the root CMakeLists already imposes /W4 /WX /external:W0 as a
# directory property (audio_sandbox is added after that block), so we add nothing.
# In a STANDALONE build there is no such property, so opt each first-party target
# in explicitly. Third-party headers are quarantined either way: the SYSTEM
# include dirs on blackwell::sandbox_headers + imgui map to /external:I -> W0.
if(_sandbox_standalone)
    if(MSVC)
        set(FIRST_PARTY_WARN /W4 /WX /external:W0 /external:anglebrackets)
        target_compile_options(whisper_dsp  PRIVATE ${FIRST_PARTY_WARN})
        target_compile_options(parity_check PRIVATE ${FIRST_PARTY_WARN})
        target_compile_definitions(parity_check PRIVATE _CRT_SECURE_NO_WARNINGS)
        if(TARGET audio_realtime)
            target_compile_options(audio_realtime PRIVATE ${FIRST_PARTY_WARN})
        endif()
    else()
        target_compile_options(whisper_dsp  PRIVATE -Wall -Wextra)
        target_compile_options(parity_check PRIVATE -Wall -Wextra)
    endif()
endif()
