# =============================================================================
# audio_sandbox — Whisper large-v3-turbo log-mel front-end (the audio encoder
# used by fixie-ai/ultravox-v0_5-llama-3_2-1b). Two products, one verified DSP:
#
#   parity_check    CPU offline parity tester vs the PyTorch golden dump.
#   audio_realtime  Windows real-time visualiser: miniaudio capture + Direct2D
#                   spectrogram + Dear ImGui controls + VAD .wav recorder.
#
# STANDALONE by design: its own project(), no CUDA, no BlackwellLLM engine deps.
#     cmake -S audio_sandbox -B audio_sandbox/out -DCMAKE_BUILD_TYPE=Release
#     cmake --build audio_sandbox/out --config Release
#
# Third-party deps (dr_wav, pocketfft, miniaudio, imgui) are pulled via
# FetchContent. NOTE: GIT_TAG uses upstream branches/tags for a first build. For
# reproducibility, replace each with the exact commit hash you validated against
# (the repo's convention is pinned FetchContent — see cmake/Dependencies.cmake).
# =============================================================================
cmake_minimum_required(VERSION 3.20)
project(audio_sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(FetchContent)

# --- dr_wav: single-header WAV loader (mackron/dr_libs) -----------------------
FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        master  # replace with a validated commit hash to pin
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(dr_libs)  # header-only: no CMake project, just populate

# --- pocketfft: header-only C++ FFT (mreineck/pocketfft, 'cpp' branch) --------
FetchContent_Declare(pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        cpp
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(pocketfft)

# --- miniaudio: single-header capture/playback (mackron/miniaudio) ------------
# SOURCE_SUBDIR points at a non-existent dir so MakeAvailable only POPULATES the
# header and never add_subdirectory()s miniaudio's own build (tests/examples).
FetchContent_Declare(miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        master  # replace with a validated commit hash to pin
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake-noop-header-only)
FetchContent_MakeAvailable(miniaudio)

# --- Dear ImGui: immediate-mode GUI (ocornut/imgui) --------------------------
# Ships no CMakeLists, so MakeAvailable just populates the sources; we build our
# own static lib with the Win32 + DX11 backends.
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.0  # replace with a validated commit hash to pin
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(imgui)

# Interface target bundling the single-header include dirs (marked SYSTEM so
# their warnings are quarantined under /W4 — see the warnings block below).
add_library(sandbox_thirdparty INTERFACE)
target_include_directories(sandbox_thirdparty SYSTEM INTERFACE
    ${dr_libs_SOURCE_DIR}
    ${pocketfft_SOURCE_DIR}
    ${miniaudio_SOURCE_DIR})

# --- DSP library (the VERIFIED core; shared by both products) -----------------
add_library(whisper_dsp STATIC src/whisper_dsp.cpp)
target_include_directories(whisper_dsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(whisper_dsp PRIVATE sandbox_thirdparty)

# --- parity_check: offline golden parity tester ------------------------------
add_executable(parity_check src/parity_check.cpp)
target_link_libraries(parity_check PRIVATE whisper_dsp sandbox_thirdparty)

# --- audio_realtime: capture + Direct2D + ImGui + VAD recorder (Windows only) -
if(WIN32)
    # Dear ImGui static lib (core + Win32/DX11 backends). Its own headers are
    # SYSTEM so ImGui/DirectX warnings never count against our /W4 budget.
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp)
    target_include_directories(imgui SYSTEM PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends)

    add_executable(audio_realtime
        src/main.cpp
        src/audio_capture.cpp
        src/audio_recorder.cpp
        src/realtime_dsp.cpp
        src/window_d2d.cpp)
    target_link_libraries(audio_realtime PRIVATE
        whisper_dsp sandbox_thirdparty imgui
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

# --- warnings: strict /W4 on ALL first-party code, quarantining third-party ---
# /external:W0 + /external:anglebrackets silences <>-included SDK/STL headers;
# the SYSTEM include dirs above (/external:I) cover the quote-included single-
# header + ImGui deps. Result: only OUR .cpp/"".h files are held to /W4.
if(MSVC)
    set(FIRST_PARTY_WARN /W4 /external:W0 /external:anglebrackets)
    target_compile_options(whisper_dsp  PRIVATE ${FIRST_PARTY_WARN})
    target_compile_options(parity_check PRIVATE ${FIRST_PARTY_WARN})
    if(TARGET audio_realtime)
        target_compile_options(audio_realtime PRIVATE ${FIRST_PARTY_WARN})
    endif()
    target_compile_definitions(parity_check PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(whisper_dsp  PRIVATE -Wall -Wextra)
    target_compile_options(parity_check PRIVATE -Wall -Wextra)
endif()
