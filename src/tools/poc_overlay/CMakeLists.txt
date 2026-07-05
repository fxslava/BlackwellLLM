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
    config.cpp
    translation_service.cpp
    live_translation_tracker.cpp
    ../playground/blackwell_llm_adapter.cpp
)

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
    ${CUDAToolkit_INCLUDE_DIRS}                           # paging headers -> cuda_runtime.h
)

# The whole PoC calls the explicit *W (wide) Win32 APIs directly, but
# built-in resource macros like IDC_ARROW/IDI_INFORMATION are UNICODE-gated
# by the SDK headers themselves -- without this define they decay to ANSI
# (LPSTR) resource IDs, which then fail to convert to the LPCWSTR the *W
# functions expect.
target_compile_definitions(poc_overlay PRIVATE UNICODE _UNICODE)

# ------------------------------------------------------------------------------
# WebView2 SDK (Chromium-based settings UI), fetched straight from NuGet -- same
# approach the core uses for the DirectStorage SDK. The package is just a ZIP;
# we point the compiler at its headers and link the STATIC loader so no extra
# WebView2Loader.dll has to ship next to the exe. (The WebView2 *runtime* still
# has to be installed on the machine; it ships with modern Windows/Edge.)
# ------------------------------------------------------------------------------
include(FetchContent)
message(STATUS "Fetching Microsoft.Web.WebView2 SDK from NuGet...")
FetchContent_Declare(
    webview2_sdk
    URL "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.2792.45"
    DOWNLOAD_NAME "webview2.zip"
    TLS_VERIFY OFF
)
FetchContent_MakeAvailable(webview2_sdk)
FetchContent_GetProperties(webview2_sdk SOURCE_DIR WV2_DIR)

target_include_directories(poc_overlay PRIVATE ${WV2_DIR}/build/native/include)

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
    version            # required by the WebView2 static loader
    nlohmann_json::nlohmann_json                       # config.json + WebView2 IPC
    ${WV2_DIR}/build/native/x64/WebView2LoaderStatic.lib
    blackwell_core       # CUDA engine + tokenizer (PUBLIC-propagates include/ + kernels)
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

# When the core lib was built with DirectStorage, the engine needs the DS runtime
# DLLs next to the executable (same block as agent_playground; DS_BIN_DIR is set
# in src/CMakeLists.txt and inherited here).
if(USE_DIRECT_STORAGE AND WIN32)
    add_custom_command(TARGET poc_overlay POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DS_BIN_DIR}/dstorage.dll"
        "${DS_BIN_DIR}/dstoragecore.dll"
        $<TARGET_FILE_DIR:poc_overlay>
        COMMENT "Copying DirectStorage runtime DLLs next to poc_overlay.exe")
endif()
