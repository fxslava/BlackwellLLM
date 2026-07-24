# =============================================================================
# audio_sandbox — Whisper large-v3-turbo log-mel front-end (the audio encoder
# used by fixie-ai/ultravox-v0_5-llama-3_2-1b). Two products, one verified DSP:
#
#   parity_check    CPU offline parity tester vs the PyTorch golden dump.
#   audio_realtime  Windows real-time visualiser (miniaudio capture + Direct2D).
#
# STANDALONE by design: its own project(), no CUDA, no BlackwellLLM engine deps.
#     cmake -S audio_sandbox -B audio_sandbox/out -DCMAKE_BUILD_TYPE=Release
#     cmake --build audio_sandbox/out --config Release
#
# Third-party single-header deps (dr_wav, pocketfft, miniaudio) are pulled via
# FetchContent. NOTE: GIT_TAG uses upstream branches for a first build. For
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

# Interface target bundling the single-header include dirs.
add_library(sandbox_thirdparty INTERFACE)
target_include_directories(sandbox_thirdparty INTERFACE
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

# --- audio_realtime: real-time capture + Direct2D visualiser (Windows only) ---
if(WIN32)
    add_executable(audio_realtime
        src/main.cpp
        src/audio_capture.cpp
        src/realtime_dsp.cpp
        src/window_d2d.cpp)
    target_link_libraries(audio_realtime PRIVATE
        whisper_dsp sandbox_thirdparty
        d2d1 user32 ole32)  # Direct2D + Win32 windowing + COM (miniaudio WASAPI)
    # UNICODE aligns the generic Win32 macros (IDC_ARROW etc.) with the explicit
    # -W APIs we call. (NOMINMAX / WIN32_LEAN_AND_MEAN are set in the sources that
    # include <windows.h>, so they are not repeated here.)
    target_compile_definitions(audio_realtime PRIVATE UNICODE _UNICODE)
    if(MSVC)
        # miniaudio.h + <windows.h>/<d2d1.h> are noisy under /W4; keep this target
        # at the default warning level (the DSP core stays strict below).
        target_compile_definitions(audio_realtime PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()

# --- warnings: strict on our first-party code, lax on third-party TUs ---------
if(MSVC)
    target_compile_definitions(parity_check PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(whisper_dsp  PRIVATE /W4)
    target_compile_options(parity_check PRIVATE /W4)
else()
    target_compile_options(whisper_dsp  PRIVATE -Wall -Wextra)
    target_compile_options(parity_check PRIVATE -Wall -Wextra)
endif()
