# ==============================================================================
# poc_overlay: standalone PoC proving out the OS-level pipeline a live overlay
# translator needs:
#   WH_KEYBOARD_LL hook (producer) -> UI Automation caret lookup (COM thread)
#   -> transparent Direct2D overlay window (consumer).
#
# Deliberately has ZERO dependency on blackwell_core / CUDA / the tokenizer --
# this only exercises Win32/COM/UIA/Direct2D plumbing, so it builds and runs
# standalone without touching the inference engine at all.
# ==============================================================================

add_executable(poc_overlay WIN32
    main.cpp
    hook_manager.cpp
    caret_tracker.cpp
    overlay_window.cpp
    tray_icon.cpp
    settings_dialog.cpp
    text_injector.cpp
)

# The whole PoC calls the explicit *W (wide) Win32 APIs directly, but
# built-in resource macros like IDC_ARROW/IDI_INFORMATION are UNICODE-gated
# by the SDK headers themselves -- without this define they decay to ANSI
# (LPSTR) resource IDs, which then fail to convert to the LPCWSTR the *W
# functions expect.
target_compile_definitions(poc_overlay PRIVATE UNICODE _UNICODE)

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
    comctl32           # InitCommonControlsEx + msctls_hotkey32 (settings dialog)
)
