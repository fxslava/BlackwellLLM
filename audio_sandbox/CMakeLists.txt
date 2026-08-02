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
# audio_sandbox` skips it and stays CUDA-free. A GPU-free SimulatedEngineControl drives
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
    # (mirrors tests/bridge). translator/ holds the simulated control + transcript view.
    target_include_directories(audio_translator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/translator  # transcript_view.hpp, simulated_engine_control.hpp
        ${CMAKE_SOURCE_DIR}/src                  # common.h
        ${CMAKE_SOURCE_DIR}/src/core             # device_buffer.h, engine_impl.h (white-box step_*)
        ${CMAKE_SOURCE_DIR}/src/audio            # ping_pong_audio_buffer.h (audio prefill consumer)
        ${CMAKE_SOURCE_DIR}/src/bridge           # engine_control_bridge.hpp, continuous_streaming_config.hpp
        ${CMAKE_SOURCE_DIR}/src/vad)             # silero_vad.hpp + speech_segmenter.hpp (header-only;
                                                 # the ORT link stays PRIVATE to blackwell_vad)

    # whisper_dsp/sandbox_headers/imgui = the sandbox front-end (SYSTEM includes
    # quarantined from /W4); blackwell_bridge + blackwell_kernels + blackwell_core_obj
    # = the speech C-ABI and the engine symbols its control plane references (the
    # same trio tests/bridge links; the simulated control never invokes the engine).
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
        # The simulated control's canned Russian replies are UTF-8 literals: tell
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

# --- voice_assistant: the Local Router voice loop, standalone -------------------
# listen -> pause -> local generate -> [barge-in aborts | EOS commits] -> remote.
#
# THE UI IS CHROMIUM, NOT IMGUI. The main window is a WebView2 host
# (voice_assistant/assistant_window.cpp) whose client area is the messenger in
# voice_assistant/web/ -- the same virtual-host-mapped arrangement poc_overlay's
# settings window uses, deployed next to the exe by the POST_BUILD copy below.
# This target therefore links NEITHER imgui NOR the D3D11/D2D swap-chain stack,
# and does not compile src/window_d2d.cpp: nothing here draws a spectrogram.
#
# TWO BACKENDS, ONE PIPELINE. A configured model directory brings up
# RealEngineControl (real CUDA decode on a real checkpoint, audio head armed);
# without one it falls back to SimulatedEngineControl (no GPU, no checkpoint).
# Both derive from EngineControlBridge, so the app holds one base pointer and the
# commit gate, the SPSC command ring, the barge-in epoch and the dispatcher are
# byte-identical on both paths -- which is what makes the simulated path a
# rehearsal of the real one rather than a parallel implementation.
#
# It therefore links the same engine stack audio_translator does. The GPU-free
# configuration is preserved at RUNTIME (no checkpoint configured), not by
# omitting the dependency -- a link-time split would have meant two binaries and
# two code paths to keep honest.
if(WIN32 AND TARGET blackwell_bridge AND TARGET blackwell_core_obj)
    if(NOT TARGET blackwell::webview2)
        message(FATAL_ERROR
            "voice_assistant needs blackwell::webview2 (cmake/WebView2.cmake), "
            "included from the root CMakeLists.")
    endif()

    add_executable(voice_assistant
        voice_assistant/main.cpp
        voice_assistant/assistant_window.cpp   # WebView2 host for the messenger UI
        # Recompiled here so audio_realtime's and audio_translator's own targets
        # stay untouched (same convention as audio_translator above).
        src/audio_capture.cpp
        src/audio_recorder.cpp
        src/realtime_dsp.cpp)

    target_include_directories(voice_assistant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/voice_assistant  # assistant_view/window, settings_store
        ${CMAKE_CURRENT_SOURCE_DIR}/translator       # conversational_mode.hpp, cli_config.hpp,
                                                     # engine_bootstrap.hpp, real/simulated control
        ${CMAKE_SOURCE_DIR}/src                      # common.h
        ${CMAKE_SOURCE_DIR}/src/core                 # device_buffer.h, engine_impl.h (white-box step_*)
        ${CMAKE_SOURCE_DIR}/src/audio                # ping_pong_audio_buffer.h (audio prefill consumer)
        ${CMAKE_SOURCE_DIR}/src/bridge               # engine_control_bridge.hpp, intent_commit.hpp
        ${CMAKE_SOURCE_DIR}/src/cloud                # intent_dispatcher.hpp, intent_transport.hpp
        ${CMAKE_SOURCE_DIR}/src/vad)                 # silero_vad.hpp (declaration only; ORT stays
                                                     # PRIVATE to blackwell_vad)

    target_link_libraries(voice_assistant PRIVATE
        whisper_dsp blackwell::sandbox_headers
        # Win32 + COM + the shell folder picker the Settings modal calls into.
        user32 gdi32 ole32 shell32
        blackwell::webview2   # Chromium UI: headers + static loader + version.lib
        blackwell_bridge blackwell_kernels blackwell_core_obj
        # audio frontend: WhisperEncoder + UltravoxProjector (the audio head
        # engine_bootstrap loads); pulls CUDA::cublas transitively.
        blackwell_audio
        # config.json parsing (cli_config.hpp) -- header-only, angle-bracket
        # include keeps it outside the /W4 /WX budget (/external:W0).
        nlohmann_json::nlohmann_json)

    # The LIVE Claude transport is opt-in and additive: without it the binary
    # still builds and runs offline, which is the whole point of the seam.
    if(TARGET blackwell_cloud)
        target_link_libraries(voice_assistant PRIVATE blackwell_cloud)
        target_compile_definitions(voice_assistant PRIVATE BLACKWELL_HAVE_CLOUD_CLIENT)
    endif()

    # Silero neural VAD, same optional wiring as audio_translator: with
    # -DUSE_SILERO_VAD=OFF the target does not exist, BLACKWELL_VAD_MODEL_PATH
    # stays undefined, and the app runs the pipeline's built-in RMS detector.
    if(TARGET blackwell_vad)
        target_link_libraries(voice_assistant PRIVATE blackwell_vad)
        get_target_property(_va_vad_model blackwell_onnxruntime BLACKWELL_VAD_MODEL_PATH)
        target_compile_definitions(voice_assistant PRIVATE
            BLACKWELL_VAD_MODEL_PATH="${_va_vad_model}")
        blackwell_copy_onnxruntime_dlls(voice_assistant)
        blackwell_copy_vad_model(voice_assistant)
    endif()

    target_compile_definitions(voice_assistant PRIVATE UNICODE _UNICODE)
    if(MSVC)
        target_compile_definitions(voice_assistant PRIVATE _CRT_SECURE_NO_WARNINGS)
        # SimulatedEngineControl's canned replies carry UTF-8 literals, and the UI
        # strings that reach the WebView2 channel must be real UTF-8 bytes.
        target_compile_options(voice_assistant PRIVATE /utf-8)
    endif()

    # The messenger UI ships as external web assets served by WebView2 through a
    # virtual-host->folder mapping, so iterating on the CSS/JS needs a reload and
    # not a recompile. Deploy them next to the exe, where AssistantWindow resolves
    # <exe dir>\web at runtime.
    add_custom_command(TARGET voice_assistant POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/voice_assistant/web"
            "$<TARGET_FILE_DIR:voice_assistant>/web"
        COMMENT "Deploying voice_assistant UI assets (web/ -> next to the exe)")

    blackwell_copy_dstorage_dlls(voice_assistant)

    if(MSVC)
        # Anchor to the sandbox root, NOT the repo root: the default positional
        # data dir is "data", and mel_filters.bin lives at audio_sandbox/data.
        # Matches audio_translator so F5 works identically for both.
        set_target_properties(voice_assistant PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()

# --- offline_retranslate: the T5 offline driver for continuous streaming --------
# (docs/CONTINUOUS_STREAMING.md). The SAME engine trio + audio frontend as
# audio_translator, but HEADLESS: no window, no mic, no ImGui/Direct2D — it reads
# a WAV and drives Silero -> SpeechSegmenter -> AbsoluteAudioRing ->
# RetranslationSession -> the 8B backbone on one thread. Gated on the same targets
# (a standalone `cmake -S audio_sandbox` skips it and stays CUDA-free) but NOT on
# WIN32: nothing here touches the Win32 GUI stack.
if(TARGET blackwell_bridge AND TARGET blackwell_core_obj)
    add_executable(offline_retranslate
        translator/offline_retranslate.cpp
        # dr_wav lives here (load_wav_mono16k); recompiled into this target so
        # audio_realtime's and audio_translator's own targets stay untouched.
        src/audio_recorder.cpp)

    target_include_directories(offline_retranslate PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/translator  # retranslation_engine.hpp, cli_config.hpp
        ${CMAKE_SOURCE_DIR}/src                  # common.h
        ${CMAKE_SOURCE_DIR}/src/core             # device_buffer.h, engine_impl.h, rope_config.h
        ${CMAKE_SOURCE_DIR}/src/audio            # audio_embedding_pipeline.h
        ${CMAKE_SOURCE_DIR}/src/bridge           # kv_ledger / retranslation_session / ring
        ${CMAKE_SOURCE_DIR}/src/vad)             # silero_vad.hpp + speech_segmenter.hpp

    target_link_libraries(offline_retranslate PRIVATE
        whisper_dsp blackwell::sandbox_headers
        blackwell_bridge blackwell_kernels blackwell_core_obj
        blackwell_audio                          # WhisperEncoder + UltravoxProjector
        nlohmann_json::nlohmann_json)            # cli_config.hpp config.json parsing

    # Silero is NOT optional here, unlike the live app: the driver's whole point is
    # to exercise the real segmentation policy, and the RMS fallback would change
    # WHERE the boundaries fall. Without it the target still builds, and main()
    # exits with a clear message rather than silently measuring something else.
    if(TARGET blackwell_vad)
        target_link_libraries(offline_retranslate PRIVATE blackwell_vad)
        get_target_property(_vad_model_ofl blackwell_onnxruntime BLACKWELL_VAD_MODEL_PATH)
        target_compile_definitions(offline_retranslate PRIVATE
            BLACKWELL_VAD_MODEL_PATH="${_vad_model_ofl}")
        blackwell_copy_onnxruntime_dlls(offline_retranslate)
        blackwell_copy_vad_model(offline_retranslate)
    endif()

    if(MSVC)
        target_compile_definitions(offline_retranslate PRIVATE _CRT_SECURE_NO_WARNINGS)
        # The decoded drafts are UTF-8 (Cyrillic targets); tell MSVC the source AND
        # execution charset so /WX never trips C4819 and the console gets real bytes.
        target_compile_options(offline_retranslate PRIVATE /utf-8)
        set_target_properties(offline_retranslate PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    blackwell_copy_dstorage_dlls(offline_retranslate)
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
