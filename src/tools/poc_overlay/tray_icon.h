#pragma once
#include <windows.h>

// What the user picked from the tray context menu (or nothing).
enum class TrayCommand {
    None,
    ShowSettings,
    Exit,
};

// Minimal Shell_NotifyIcon-based tray icon with a right-click context menu
// offering "Settings" and "Exit", so the background process (no taskbar
// button, since the overlay is a tool window) can be configured and shut down.
// It piggybacks on an owner window's message queue -- see main.cpp's controller
// window, which forwards WM_TRAYICON to HandleMessage().
class TrayIcon {
public:
    static constexpr UINT WM_TRAYICON = WM_APP + 100;

    TrayIcon(HWND ownerWindow, UINT iconId);
    ~TrayIcon();

    bool Create(const wchar_t* tooltip);

    // Call from the owner window's WndProc on WM_TRAYICON. Returns which menu
    // item (if any) the user selected.
    TrayCommand HandleMessage(WPARAM wParam, LPARAM lParam);

private:
    HWND owner_;
    UINT iconId_;
    bool added_ = false;
};
