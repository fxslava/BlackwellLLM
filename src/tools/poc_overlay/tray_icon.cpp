#include "tray_icon.h"

#include <shellapi.h>

#include "resource.h"

namespace {
constexpr UINT kSettingsMenuId = 1;
constexpr UINT kExitMenuId = 2;

// Load the embedded TypeTranslate application icon (app.rc / IDI_APP_ICON) at
// the small-icon size the notification area expects. Falls back to a stock icon
// if the resource is missing so the tray entry always appears.
HICON LoadTrayIcon() {
    HICON icon = static_cast<HICON>(LoadImageW(
        GetModuleHandleW(nullptr), MAKEINTRESOURCEW(IDI_APP_ICON), IMAGE_ICON,
        GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR));
    return icon ? icon : LoadIconW(nullptr, IDI_INFORMATION);
}
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
    nid.hIcon = LoadTrayIcon();
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
    AppendMenuW(menu, MF_STRING, kExitMenuId, L"Exit");

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
