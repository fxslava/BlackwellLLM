#include "tray_icon.h"

#include <shellapi.h>

namespace {
constexpr UINT kSettingsMenuId = 1;
constexpr UINT kExitMenuId = 2;
}  // namespace

TrayIcon::TrayIcon(HWND ownerWindow, UINT iconId) : owner_(ownerWindow), iconId_(iconId) {}

TrayIcon::~TrayIcon() {
    if (added_) {
        NOTIFYICONDATAW nid{sizeof(nid)};
        nid.hWnd = owner_;
        nid.uID = iconId_;
        Shell_NotifyIconW(NIM_DELETE, &nid);
    }
}

bool TrayIcon::Create(const wchar_t* tooltip) {
    NOTIFYICONDATAW nid{sizeof(nid)};
    nid.hWnd = owner_;
    nid.uID = iconId_;
    nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    nid.uCallbackMessage = WM_TRAYICON;
    nid.hIcon = LoadIconW(nullptr, IDI_INFORMATION);
    wcsncpy_s(nid.szTip, tooltip, _TRUNCATE);

    added_ = Shell_NotifyIconW(NIM_ADD, &nid) != FALSE;
    return added_;
}

TrayCommand TrayIcon::HandleMessage(WPARAM /*wParam*/, LPARAM lParam) {
    if (LOWORD(lParam) != WM_RBUTTONUP && LOWORD(lParam) != WM_CONTEXTMENU) {
        return TrayCommand::None;
    }

    POINT pt;
    GetCursorPos(&pt);

    HMENU menu = CreatePopupMenu();
    AppendMenuW(menu, MF_STRING, kSettingsMenuId, L"Settings...");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(menu, MF_STRING, kExitMenuId, L"Exit PoC");

    // Required so the popup menu closes correctly when the user clicks away
    // from it (standard tray-icon menu dance).
    SetForegroundWindow(owner_);
    const UINT clicked =
        TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD, pt.x, pt.y, 0, owner_, nullptr);
    DestroyMenu(menu);

    switch (clicked) {
        case kSettingsMenuId:
            return TrayCommand::ShowSettings;
        case kExitMenuId:
            return TrayCommand::Exit;
        default:
            return TrayCommand::None;
    }
}
