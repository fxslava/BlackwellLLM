// -----------------------------------------------------------------------------
// tray_icon.cpp — the notification-area icon and its menu.
// -----------------------------------------------------------------------------
#include "tray_icon.hpp"

#include <shellapi.h>

#include <cstdio>

namespace rt {
namespace {

// Icon id within this window. One icon, so the value only has to be stable
// between the NIM_ADD and the NIM_DELETE.
constexpr UINT kIconId = 1;

// Menu command ids. Local to the tracked popup, so they cannot collide with
// anything else the app posts.
constexpr UINT kCmdShow    = 100;
constexpr UINT kCmdHide    = 101;
constexpr UINT kCmdConsole = 102;
constexpr UINT kCmdExit    = 103;

}  // namespace

TrayIcon::~TrayIcon() { destroy(); }

bool TrayIcon::create(HWND owner, const std::wstring& tooltip) {
    if (added_) return true;
    owner_ = owner;
    tooltip_ = tooltip;

    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd = owner;
    nid.uID = kIconId;
    nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    nid.uCallbackMessage = kTrayCallbackMessage;
    // THE APPLICATION'S OWN ICON, loaded from the executable, with the shell's
    // generic application icon as the fallback. LoadIconW(instance, IDI_*) with
    // no icon resource returns null, and a null hIcon makes Shell_NotifyIcon add
    // an invisible entry -- an icon the user cannot find but which still holds a
    // slot. So the fallback is not defensive, it is the common case for a build
    // with no .rc file.
    nid.hIcon = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
    if (nid.hIcon == nullptr) {
        nid.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    }
    wcsncpy_s(nid.szTip, tooltip.c_str(), _TRUNCATE);

    added_ = Shell_NotifyIconW(NIM_ADD, &nid) != FALSE;
    if (!added_) {
        std::fprintf(stderr, "[tray] the notification area refused the icon.\n");
        return false;
    }
    // Version 4 gives the modern message encoding (the cursor position arrives in
    // wParam rather than requiring GetCursorPos) and reliable balloon behaviour.
    // Failing this is not fatal: the classic encoding still works, and
    // handle_message reads the values that are correct under both.
    nid.uVersion = NOTIFYICON_VERSION_4;
    Shell_NotifyIconW(NIM_SETVERSION, &nid);
    return true;
}

void TrayIcon::destroy() noexcept {
    if (!added_) return;
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd = owner_;
    nid.uID = kIconId;
    Shell_NotifyIconW(NIM_DELETE, &nid);
    added_ = false;
}

void TrayIcon::set_tooltip(const std::wstring& tooltip) {
    tooltip_ = tooltip;
    if (!added_) return;
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd = owner_;
    nid.uID = kIconId;
    nid.uFlags = NIF_TIP;
    wcsncpy_s(nid.szTip, tooltip.c_str(), _TRUNCATE);
    Shell_NotifyIconW(NIM_MODIFY, &nid);
}

void TrayIcon::notify(const std::wstring& title, const std::wstring& text) {
    if (!added_) return;
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd = owner_;
    nid.uID = kIconId;
    nid.uFlags = NIF_INFO;
    nid.dwInfoFlags = NIIF_INFO;
    wcsncpy_s(nid.szInfoTitle, title.c_str(), _TRUNCATE);
    wcsncpy_s(nid.szInfo, text.c_str(), _TRUNCATE);
    Shell_NotifyIconW(NIM_MODIFY, &nid);
}

bool TrayIcon::handle_message(WPARAM wp, LPARAM lp) {
    // Under NOTIFYICON_VERSION_4 the event is in the LOW word of lParam and the
    // icon id in the high word; under the classic encoding lParam IS the event.
    // Taking the low word reads correctly under both, which is why create()
    // tolerating a failed NIM_SETVERSION is safe.
    const UINT event = LOWORD(lp);
    (void)wp;

    switch (event) {
        case WM_LBUTTONUP:
        case WM_LBUTTONDBLCLK:
            // Single click summons. A double click would too, but binding both
            // means the second click of a double does not toggle it back off --
            // which is what happens if only the click is bound and the user
            // double-clicks out of habit.
            if (cb_.on_show) cb_.on_show();
            return true;

        case WM_RBUTTONUP:
        case WM_CONTEXTMENU:
            show_menu();
            return true;

        default:
            return false;
    }
}

void TrayIcon::show_menu() {
    HMENU menu = CreatePopupMenu();
    if (menu == nullptr) return;

    AppendMenuW(menu, MF_STRING, kCmdShow, L"&Show assistant");
    AppendMenuW(menu, MF_STRING, kCmdHide, L"&Hide to tray");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, kCmdConsole, L"Toggle &console");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, kCmdExit, L"E&xit");
    // The default item is what a click activates and what the shell renders in
    // bold; making it Show matches the left-click gesture.
    SetMenuDefaultItem(menu, kCmdShow, FALSE);

    POINT pt{};
    GetCursorPos(&pt);

    // REQUIRED, and its absence is a classic tray bug: without foregrounding the
    // owner first, the menu does not dismiss when the user clicks elsewhere and
    // sits on screen until something else takes focus.
    SetForegroundWindow(owner_);

    // TPM_RETURNCMD makes this synchronous -- the command comes back as the
    // return value rather than as a WM_COMMAND the owner would have to route.
    // That keeps the whole menu self-contained in this class, which is the
    // reason the owner only has to forward one message.
    const int cmd = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
                                   pt.x, pt.y, 0, owner_, nullptr);
    DestroyMenu(menu);

    switch (cmd) {
        case kCmdShow:    if (cb_.on_show) cb_.on_show(); break;
        case kCmdHide:    if (cb_.on_hide) cb_.on_hide(); break;
        case kCmdConsole: if (cb_.on_toggle_console) cb_.on_toggle_console(); break;
        case kCmdExit:    if (cb_.on_exit) cb_.on_exit(); break;
        default: break;
    }
}

}  // namespace rt
