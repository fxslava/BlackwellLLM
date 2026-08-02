# ==============================================================================
# poc_overlay: the live overlay translator -- the full pipeline:
#   WH_KEYBOARD_LL hook (producer) -> UI Automation caret lookup (COM thread)
#   -> TranslationService (BlackwellEngine on its own worker thread, driven by
#      the 1-step AgentOrchestrator ReAct loops)
#   -> transparent Direct2D overlay window (consumer) -> SendInput/UIA text
#   replacement, with a WebView2 (Chromium) settings window.
#
# This target now links the REAL inference stack (blackwell_core + CUDA +
# agent_orchestrator), mirroring agent_playground. The BlackwellLLMAdapter is
# compiled in from the playground directory until it moves to a shared library
# target (it has no playground-specific dependencies).
# ==============================================================================

add_executable(poc_overlay WIN32
    main.cpp
    hook_manager.cpp
    caret_tracker.cpp
    overlay_window.cpp
    tray_icon.cpp
    settings_dialog.cpp
    text_injector.cpp
    spotlight_composer.cpp    # Companion Mode: centered Tier-4 clipboard-inject composer
    config.cpp
    translation_service.cpp
    live_translation_tracker.cpp
    smoke_test.cpp
    app.rc                 # embeds Icon.ico (taskbar/Explorer) + version info
    ../playground/blackwell_llm_adapter.cpp
)

# Product identity: the executable ships as TypeTranslate.exe. The CMake target
# and source directory keep the historical `poc_overlay` name; only the emitted
# binary is rebranded, so build scripts that reference the target are unaffected.
set_target_properties(poc_overlay PROPERTIES OUTPUT_NAME "TypeTranslate")

target_compile_features(poc_overlay PRIVATE cxx_std_20)

# The prefix-cache substrate is exported by the engine (BlackwellEngine::
# prefix_cache() / prefill_driver()), so translation_service.cpp warm-starts
# the AOT prompt library (.bkv) for real. The paging headers it pulls include
# <cuda_runtime.h>, hence the CUDAToolkit include dirs below.
target_compile_definitions(poc_overlay PRIVATE BLACKWELL_ENGINE_HAS_PREFIX_CACHE)

find_package(CUDAToolkit REQUIRED)

target_include_directories(poc_overlay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../agent_orchestrator  # orchestrator headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../playground             # blackwell_llm_adapter.h
    ${CMAKE_SOURCE_DIR}/src/core                          # engine_prefill_coordinator.h, paging/
    ${CMAKE_SOURCE_DIR}/src                               # paging/ headers reach kernels/*.cuh + common.h
    ${CUDAToolkit_INCLUDE_DIRS}                           # paging headers -> cuda_runtime.h
)

# The whole PoC calls the explicit *W (wide) Win32 APIs directly, but
# built-in resource macros like IDC_ARROW/IDI_INFORMATION are UNICODE-gated
# by the SDK headers themselves -- without this define they decay to ANSI
# (LPSTR) resource IDs, which then fail to convert to the LPCWSTR the *W
# functions expect.
target_compile_definitions(poc_overlay PRIVATE UNICODE _UNICODE)

# The WebView2 (Chromium) settings UI links `blackwell::webview2` -- headers +
# the static loader, fetched ONCE in cmake/WebView2.cmake so this target and
# voice_assistant cannot pin two different SDK versions. (The WebView2 *runtime*
# still has to be installed on the machine; it ships with modern Windows/Edge,
# and settings_dialog.cpp reports its absence at runtime.)
if(NOT TARGET blackwell::webview2)
    message(FATAL_ERROR
        "poc_overlay needs blackwell::webview2 (cmake/WebView2.cmake). It is "
        "Windows-only and included from the root CMakeLists.")
endif()

target_link_libraries(poc_overlay PRIVATE
    user32
    gdi32
    ole32
    oleaut32          # SAFEARRAY helpers (SafeArrayAccessData/GetLBound/...)
    uuid               # CLSID_CUIAutomation / IID_IUIAutomation GUID definitions
    UIAutomationCore   # IUIAutomation client COM classes
    d2d1
    dwrite
    shell32            # Shell_NotifyIconW
    nlohmann_json::nlohmann_json                       # config.json + WebView2 IPC
    blackwell::webview2  # Chromium settings window (headers + static loader + version.lib)
    CUDA::cudart_static  # smoke_test.cpp calls the CUDA Runtime API directly
                         # (cudaGetDeviceCount/cudaSetDevice) from a CXX TU.
                         # STATIC to match the rest of the build (-cudart static,
                         # the CMake default): linking the dynamic CUDA::cudart
                         # here duplicates cudaGetExportTable against the
                         # transitively-linked cudart_static.lib (LNK2005) under
                         # the MSBuild/VS CUDA integration.
    blackwell_core_obj   # white-box engine layer: the tracker/service drive the
                         # EnginePrefillCoordinator + paging C++ API, which the
                         # DLL's COM boundary does not (yet) cover
    agent_orchestrator   # AgentOrchestrator / ToolParser / ToolRegistry
)

# The Settings UI ships as external web assets (settings.html/.css/.js) served
# by WebView2 via a virtual-host->folder mapping. Deploy the web/ folder next to
# the exe so SettingsWindow can find it at <exe dir>\web at runtime.
add_custom_command(TARGET poc_overlay POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/web"
    "$<TARGET_FILE_DIR:poc_overlay>/web"
    COMMENT "Deploying poc_overlay Settings web assets (web/ -> next to the exe)")

# When the core lib was built with DirectStorage, the engine needs the DS
# runtime DLLs next to the executable (no-op otherwise; the helper comes from
# cmake/DirectStorage.cmake).
blackwell_copy_dstorage_dlls(poc_overlay)
