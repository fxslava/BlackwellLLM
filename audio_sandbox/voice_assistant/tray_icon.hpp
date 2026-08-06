#pragma once
// -----------------------------------------------------------------------------
// tray_icon.hpp — the notification-area presence: the icon, its menu, and the
// gestures that summon the window back.
//
// WHY IT IS A SEPARATE CLASS AND NOT PART OF AssistantWindow. It needs the
// window's HWND (Shell_NotifyIcon delivers its callbacks as window messages, so
// it must have a window to deliver to) but nothing else about the window, and
// keeping it separate is what lets the tray outlive a hidden frame conceptually
// -- the whole point of this feature is that the app is alive when the window is
// not. The owner forwards one message id here and the class answers everything
// else itself.
//
// THE MESSAGE IT NEEDS. Shell_NotifyIcon posts to `uCallbackMessage` on the
// owner window, so AssistantWindow::handle() must route kTrayCallbackMessage to
// TrayIcon::handle_message(). That one line is the entire integration.
//
// THREADING. UI thread only, like every other Win32 surface here. The icon is
// created on the thread that owns the window and must be removed on it too --
// an icon whose owner window is destroyed without a NIM_DELETE leaves a ghost in
// the notification area until the user hovers over it, which is the single most
// recognisable symptom of an app that did not clean up.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <functional>
#include <string>

namespace rt {

// The private message Shell_NotifyIcon posts to the owner. WM_APP+2, chosen to
// sit clear of AssistantWindow's own kMsgDrain (WM_APP+1).
constexpr UINT kTrayCallbackMessage = WM_APP + 2;

struct TrayCallbacks {
    // Left click, or "Show" from the menu. Un-hide, restore and foreground.
    std::function<void()> on_show;
    // "Hide" from the menu. The complement of on_show, so a user who summoned
    // the window by accident can put it back without reaching for the [X].
    std::function<void()> on_hide;
    // Toggle the log console. Present here specifically so the console is
    // reachable when its global hotkey failed to register -- which is the case
    // where the user most needs a log.
    std::function<void()> on_toggle_console;
    // "Exit". THE ONLY WAY OUT of the app once WM_CLOSE has been intercepted,
    // which is why it is the last item and separated by a rule.
    std::function<void()> on_exit;
};

class TrayIcon {
public:
    TrayIcon() = default;
    ~TrayIcon();

    TrayIcon(const TrayIcon&) = delete;
    TrayIcon& operator=(const TrayIcon&) = delete;

    void set_callbacks(TrayCallbacks cb) { cb_ = std::move(cb); }

    // Add the icon. `owner` receives kTrayCallbackMessage and must forward it.
    // Returns false when the shell refused (a notification area that is not
    // running yet at login is the usual cause) -- non-fatal, and the caller
    // should keep the window visible if it happens, because hiding to a tray
    // that does not exist would make the app unreachable.
    bool create(HWND owner, const std::wstring& tooltip);

    // Remove it. Idempotent, and called from the owner's WM_DESTROY as well as
    // from the destructor: whichever runs first wins, and the second is a no-op.
    void destroy() noexcept;

    [[nodiscard]] bool alive() const noexcept { return added_; }

    // Route kTrayCallbackMessage here. Returns true when handled.
    bool handle_message(WPARAM wp, LPARAM lp);

    // A balloon, used once: the first time the window hides to the tray. An app
    // that vanishes on [X] with no explanation reads as a crash, and the one
    // notification that prevents that is worth the interruption. The caller is
    // responsible for only asking once.
    void notify(const std::wstring& title, const std::wstring& text);

    // Reflect the mic state in the tooltip, so hovering answers "is it
    // listening" without opening anything.
    void set_tooltip(const std::wstring& tooltip);

private:
    void show_menu();

    HWND          owner_ = nullptr;
    bool          added_ = false;
    std::wstring  tooltip_;
    TrayCallbacks cb_;
};

}  // namespace rt
