# ==============================================================================
# poc_overlay: standalone PoC proving out the OS-level pipeline a live overlay
# translator needs:
#   WH_KEYBOARD_LL hook (producer) -> UI Automation caret lookup (COM thread)
#   -> transparent Direct2D overlay window (consumer) -> SendInput/UIA text
#   replacement, with a WebView2 (Chromium) settings window.
#
# Deliberately has ZERO dependency on blackwell_core / CUDA / the tokenizer --
# this only exercises Win32/COM/UIA/Direct2D/WebView2 plumbing, so it builds and
# runs standalone without touching the inference engine at all.
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
)
