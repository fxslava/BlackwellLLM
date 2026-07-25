# Third-party dependencies fetched at configure time. Pinned release tags only
# (never branches); TLS verification stays ON for every fetch.
include(FetchContent)

# GoogleTest: MSVC adaptation -- no pthreads probing, and the same DLL/MD CRT
# as the rest of the VS2022 build.
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# nlohmann/json: Safetensors header parsing + config loading.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ============================================================================
# Audio-frontend third-party deps (single source of truth).
# Fetched ONCE here so the engine apps, the tests, and audio_sandbox/ all share
# the SAME populated sources / built libraries instead of each re-declaring
# FetchContent (the old audio_sandbox/CMakeLists.txt duplicated all four).
#
# Declared BEFORE the root /W4 /WX directory property (root CMakeLists includes
# this module first), so the imgui object library below is NEVER held to our
# first-party warning budget -- and every include dir here is marked SYSTEM so
# the single-header deps are quarantined under /external:W0 as well.
#
# NOTE: GIT_TAG follows the repo's pinning convention (release tag where one
# exists; upstream branch for the header-only repos that don't tag). Replace a
# branch with the exact validated commit hash when reproducibility demands it.
# ============================================================================

# --- dr_libs: single-header WAV loader (mackron/dr_libs) ----------------------
FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(dr_libs)  # header-only: populate, no add_subdirectory

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
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake-noop-header-only)
FetchContent_MakeAvailable(miniaudio)

# --- Dear ImGui: immediate-mode GUI (ocornut/imgui) --------------------------
# Ships no CMakeLists, so MakeAvailable just populates the sources; we build our
# own static lib (below) with the Win32 + DX11 backends.
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.0
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(imgui)

# INTERFACE target bundling the header-only single-header include dirs. SYSTEM so
# their warnings are quarantined under /W4 (/external:I -> /external:W0). Every
# consumer (whisper_dsp, audio_realtime, and any future engine app) links this
# ONE target instead of hard-coding the *_SOURCE_DIR paths.
add_library(blackwell_sandbox_headers INTERFACE)
target_include_directories(blackwell_sandbox_headers SYSTEM INTERFACE
    ${dr_libs_SOURCE_DIR}
    ${pocketfft_SOURCE_DIR}
    ${miniaudio_SOURCE_DIR})
add_library(blackwell::sandbox_headers ALIAS blackwell_sandbox_headers)

# Dear ImGui static lib (core + Win32/DX11 backends) -- Windows-only because the
# backends we compile are Win32 + DX11. Built ONCE here and shared by every GUI
# consumer. Its headers are SYSTEM PUBLIC so ImGui/DirectX warnings never count
# against a linking target's /W4 budget. This target is created before the root
# /W4 /WX block, so it is not itself held to that budget.
if(WIN32)
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
endif()
