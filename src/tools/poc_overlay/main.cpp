// poc_overlay: standalone PoC for the OS-level pipeline behind a live overlay
// translator -- NO inference/translation logic here, on purpose. It proves out
// a "UIA-driven single source of truth" pipeline:
//
//   [UI thread]      WH_KEYBOARD_LL hook (lightweight TRIGGER + fallback buffer)
//                        |  RequestUpdate(fallbackText)     [condvar signal]
//                        v
//   [COM/UIA thread]  CaretTracker: reads the REAL text from the focused control
//                      (IUIAutomationTextPattern line-up-to-caret, or
//                      IUIAutomationValuePattern), plus the caret rectangle. Only
//                      UIA-opaque apps fall back to the hook's typed buffer.
//                        |  PostUpdate(text, caretPos)      [PostMessage]
//                        v
//   [UI thread]       OverlayWindow: layered, click-through, D2D-rendered popup
//                      whose string END anchors just above the caret.
//
// A system tray icon provides the only way to exit (no visible window/taskbar
// entry otherwise).
#include <windows.h>
#include <commctrl.h>  // HOTKEYF_* for the default shortcut
#include <objbase.h>

#include <string>

#include "caret_tracker.h"
#include "hook_manager.h"
#include "overlay_window.h"
#include "settings_dialog.h"
#include "tray_icon.h"

namespace {

constexpr UINT kTrayIconId = 1;

// Mock stand-ins for the eventual inference calls -- surfaced via
// OutputDebugString so the pipeline can be observed under a debugger / DebugView
// without any UI of their own.
void LogMock(const wchar_t* tag, const std::wstring& text) {
    OutputDebugStringW((std::wstring(L"[poc_overlay] ") + tag + L": \"" + text + L"\"\n").c_str());
}

LRESULT CALLBACK ControllerWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* trayIcon = reinterpret_cast<TrayIcon*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
        case TrayIcon::WM_TRAYICON: {
            if (!trayIcon) {
                return 0;
            }
            switch (trayIcon->HandleMessage(wParam, lParam)) {
                case TrayCommand::ShowSettings: {
                    Shortcut current = HookManager::Instance().GetCommitShortcut();
                    const HINSTANCE hInst =
                        reinterpret_cast<HINSTANCE>(GetWindowLongPtrW(hwnd, GWLP_HINSTANCE));
                    if (ShowSettingsDialog(hwnd, hInst, current)) {
                        HookManager::Instance().SetCommitShortcut(current);
                    }
                    break;
                }
                case TrayCommand::Exit:
                    DestroyWindow(hwnd);
                    break;
                case TrayCommand::None:
                    break;
            }
            return 0;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int) {
    // Without this, the process defaults to DPI-unaware and Windows silently
    // virtualizes/rescales screen coordinates for it -- inconsistently across
    // monitors with different scale factors. That breaks the whole pipeline:
    // UIA reports the caret's real physical-pixel position, but our own
    // SetWindowPos/UpdateLayeredWindow calls would land in the *virtualized*
    // coordinate space, so the overlay ends up nowhere near the caret except
    // by coincidence on a single-monitor, 100%-scale setup. Must be set
    // before any window is created.
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    // D2D/DWrite/tray-icon setup on this thread don't themselves require COM,
    // but initialize an apartment anyway since it's harmless and conventional
    // for a GUI thread. UI Automation's own STA lives on CaretTracker's
    // dedicated worker thread -- it is NOT this one.
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    OverlayWindow overlay;
    if (!overlay.Create(hInstance)) {
        MessageBoxW(nullptr, L"Failed to create the overlay window (Direct2D/DirectWrite init failed).",
                    L"poc_overlay", MB_ICONERROR);
        return 1;
    }

    // Mock "translation" applied at commit: locale-aware UPPERCASE. std::towupper
    // mishandles Cyrillic and other scripts, so use the Win32 CharUpperBuffW,
    // which upper-cases the whole buffer per the current locale's rules. Runs on
    // the CaretTracker STA thread against the freshly re-resolved source text.
    auto commitTransform = [](const std::wstring& source) -> std::wstring {
        std::wstring upper = source;
        if (!upper.empty()) {
            CharUpperBuffW(&upper[0], static_cast<DWORD>(upper.size()));
        }
        LogMock(L"CommitTranslation", upper);  // later -> real translation
        return upper;
    };
    // Brackets the injection window on the hook so it flags synthetic input.
    auto injectionGuard = [](bool active) { HookManager::Instance().SetInjecting(active); };

    // CaretTracker resolves the authoritative text + caret position on ITS OWN
    // (COM/UIA STA) thread, and also performs the commit replacement there (so the
    // UIA interfaces never cross apartments). The update callback runs on that
    // thread and must not touch the overlay directly -- PostUpdate() marshals the
    // result back to the overlay window's owning (UI) thread via PostMessage.
    CaretTracker caretTracker(
        [&](const CaretUpdate& update) {
            if (update.caretFound) {
                overlay.PostUpdate(update.text, update.caretScreenPos);
            }
            // Word boundary -> mock "OnWordComplete" on the authoritative UIA text;
            // the gate where the full buffer will later be sent to inference.
            if (update.wordBoundary && !update.text.empty()) {
                LogMock(L"OnWordComplete", update.text);
            }
        },
        commitTransform, injectionGuard);

    // The hook callbacks run inline in the global hook chain on THIS thread and
    // must stay fast. They only hand work to the caret tracker's queue (an O(1)
    // lock + condvar notify) -- the UIA round-trip, D2D paint, and the whole
    // 3-tier replacement happen off this call stack entirely.
    HookManager::Callbacks callbacks;
    callbacks.onTrigger = [&caretTracker](const std::wstring& fallbackText, bool wordBoundary) {
        caretTracker.RequestUpdate(fallbackText, wordBoundary);  // "field changed -> poll UIA"
    };
    callbacks.onCommit = [&](const std::wstring& fallbackText) {
        overlay.Hide();
        caretTracker.RequestCommit(fallbackText);  // re-resolve + replace on the STA thread
    };
    callbacks.onReset = [&overlay]() { overlay.Hide(); };

    const bool hookInstalled = HookManager::Instance().Install(std::move(callbacks));
    if (!hookInstalled) {
        MessageBoxW(nullptr, L"Failed to install the low-level keyboard/mouse hooks.", L"poc_overlay",
                    MB_ICONERROR);
        return 1;
    }

    // Default commit/trigger shortcut: Ctrl+Enter. Rebindable via the tray
    // "Settings..." dialog.
    HookManager::Instance().SetCommitShortcut(Shortcut{HOTKEYF_CONTROL, VK_RETURN});

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &ControllerWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"BlackwellPocOverlayController";
    RegisterClassExW(&wc);

    // Invisible controller window: exists solely to own the tray icon and
    // receive its callback messages (WM_TRAYICON).
    HWND controller = CreateWindowExW(0, wc.lpszClassName, L"poc_overlay controller", WS_OVERLAPPED,
                                       0, 0, 0, 0, nullptr, nullptr, hInstance, nullptr);

    TrayIcon trayIcon(controller, kTrayIconId);
    SetWindowLongPtrW(controller, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&trayIcon));
    trayIcon.Create(L"Blackwell PoC: Input Hook -> UIA Caret -> Overlay");

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    HookManager::Instance().Uninstall();
    CoUninitialize();
    return static_cast<int>(msg.wParam);
}
