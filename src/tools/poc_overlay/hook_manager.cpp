#include "hook_manager.h"

#include <commctrl.h>  // HOTKEYF_* modifier flags

#include <cwctype>

#include "text_injector.h"  // TextInjector::kInjectedSignature

namespace {
// SetWindowsHookEx requires raw function pointers, so the singleton instance
// is stashed here for the static callbacks to route through.
HookManager* g_instance = nullptr;

// True while any Ctrl/Alt/Win key is physically down. Shift is intentionally
// excluded -- it is a normal part of typing capitals and shifted punctuation.
// NOTE: AltGr surfaces as Ctrl+Alt on many layouts, so AltGr-produced glyphs
// are treated as a chord (context break) -- an accepted PoC limitation; UIA
// still reports such characters correctly when it is the source of truth.
bool IsChordModifierHeld() {
    return (GetAsyncKeyState(VK_CONTROL) & 0x8000) || (GetAsyncKeyState(VK_MENU) & 0x8000) ||
           (GetAsyncKeyState(VK_LWIN) & 0x8000) || (GetAsyncKeyState(VK_RWIN) & 0x8000);
}

// Standalone modifier/lock keydowns: neither typing nor a context break.
bool IsModifierKey(DWORD vk) {
    switch (vk) {
        case VK_SHIFT: case VK_LSHIFT: case VK_RSHIFT:
        case VK_CONTROL: case VK_LCONTROL: case VK_RCONTROL:
        case VK_MENU: case VK_LMENU: case VK_RMENU:
        case VK_LWIN: case VK_RWIN:
        case VK_CAPITAL: case VK_NUMLOCK: case VK_SCROLL:
            return true;
        default:
            return false;
    }
}

// Caret navigation / focus-changing keys. In the UIA model these no longer need
// per-key bookkeeping; they simply break the continuous-input flow so we hide
// the overlay and drop the fallback buffer.
bool IsContextBreakKey(DWORD vk) {
    switch (vk) {
        case VK_LEFT: case VK_RIGHT: case VK_UP: case VK_DOWN:
        case VK_HOME: case VK_END: case VK_PRIOR: case VK_NEXT:
        case VK_DELETE: case VK_INSERT:
        case VK_ESCAPE: case VK_TAB:
        case VK_APPS: case VK_SNAPSHOT: case VK_PAUSE:
            return true;
        default:
            return vk >= VK_F1 && vk <= VK_F24;  // function keys
    }
}

// True when the foreground window belongs to our own process (e.g. the WebView2
// settings window). We must not treat typing there as user input into a target
// app, or the overlay/commit would fire against our own UI.
bool ForegroundIsOwnProcess() {
    DWORD pid = 0;
    GetWindowThreadProcessId(GetForegroundWindow(), &pid);
    return pid == GetCurrentProcessId();
}

// Exact match of the currently-held modifiers against a shortcut's HOTKEYF_*
// mask (so e.g. Ctrl+Shift+Enter does not fire a Ctrl+Enter binding).
bool ModifiersMatch(UINT hotkeyFlags) {
    const bool wantCtrl = (hotkeyFlags & HOTKEYF_CONTROL) != 0;
    const bool wantAlt = (hotkeyFlags & HOTKEYF_ALT) != 0;
    const bool wantShift = (hotkeyFlags & HOTKEYF_SHIFT) != 0;
    const bool haveCtrl = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
    const bool haveAlt = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
    const bool haveShift = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    return wantCtrl == haveCtrl && wantAlt == haveAlt && wantShift == haveShift;
}
}  // namespace

HookManager& HookManager::Instance() {
    static HookManager instance;
    return instance;
}

HookManager::~HookManager() {
    Uninstall();
}

bool HookManager::Install(Callbacks callbacks) {
    callbacks_ = std::move(callbacks);
    fallbackBuffer_.clear();
    g_instance = this;

    const HINSTANCE module = GetModuleHandleW(nullptr);
    keyboardHook_ = SetWindowsHookExW(WH_KEYBOARD_LL, &HookManager::LowLevelKeyboardProc, module, 0);
    mouseHook_ = SetWindowsHookExW(WH_MOUSE_LL, &HookManager::LowLevelMouseProc, module, 0);

    if (!keyboardHook_ || !mouseHook_) {
        Uninstall();
        return false;
    }
    return true;
}

void HookManager::Uninstall() {
    if (keyboardHook_) {
        UnhookWindowsHookEx(keyboardHook_);
        keyboardHook_ = nullptr;
    }
    if (mouseHook_) {
        UnhookWindowsHookEx(mouseHook_);
        mouseHook_ = nullptr;
    }
    g_instance = nullptr;
}

void HookManager::ResetFallback() {
    fallbackBuffer_.clear();
    if (callbacks_.onReset) {
        callbacks_.onReset();
    }
}

LRESULT CALLBACK HookManager::LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode == HC_ACTION && g_instance) {
        const auto* info = reinterpret_cast<const KBDLLHOOKSTRUCT*>(lParam);
        // Pass our own injected keystrokes straight through: recognized by the
        // dwExtraInfo tag (robust) and, belt-and-suspenders, the injecting_ flag.
        // TextInjector::Replace runs on this same (UI) thread, so the synthetic
        // events re-enter here while injecting_ is still set.
        const bool ownInjection =
            info->dwExtraInfo == TextInjector::kInjectedSignature ||
            g_instance->injecting_.load(std::memory_order_relaxed);
        if (!ownInjection && g_instance->HandleKeyEvent(wParam, *info)) {
            return 1;  // consume the commit shortcut so the app never sees it
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

LRESULT CALLBACK HookManager::LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam) {
    // Only track the mouse while Translation Mode is active, and never react to
    // clicks in our own UI (settings window).
    if (nCode == HC_ACTION && g_instance && g_instance->enabled_ && !ForegroundIsOwnProcess()) {
        const auto* ms = reinterpret_cast<const MSLLHOOKSTRUCT*>(lParam);
        // A click on the overlay's own chrome (header bar / open dropdown) is UI
        // interaction, not a context break: let it reach the overlay window
        // untouched -- no reset (which would hide the overlay out from under the
        // click) and no selection check.
        if (g_instance->hasInteractiveRect_ && ms &&
            PtInRect(&g_instance->interactiveRect_, ms->pt)) {
            return CallNextHookEx(nullptr, nCode, wParam, lParam);
        }
        // A button press can reposition the caret / dismiss a live popup: break
        // the input flow (cancels any in-flight generation, hides the overlay).
        if (wParam == WM_LBUTTONDOWN || wParam == WM_RBUTTONDOWN || wParam == WM_MBUTTONDOWN ||
            wParam == WM_XBUTTONDOWN) {
            g_instance->ResetFallback();
        } else if (wParam == WM_LBUTTONUP && g_instance->callbacks_.onSelectionCandidate) {
            // The user may have just finished a drag-selection -- go check UIA.
            g_instance->callbacks_.onSelectionCandidate();
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

bool HookManager::HandleKeyEvent(WPARAM wParam, const KBDLLHOOKSTRUCT& info) {
    if (wParam != WM_KEYDOWN && wParam != WM_SYSKEYDOWN) {
        return false;
    }

    const DWORD vk = info.vkCode;

    // Never react to typing in our own windows (e.g. the settings UI).
    if (ForegroundIsOwnProcess()) {
        return false;
    }

    // Master activation shortcut: toggle Translation Mode on/off. Works even
    // while disabled (it is checked before the enabled_ gate below). We do NOT
    // fire onReset here -- onActivationToggle owns the visual (the HUD banner),
    // and an onReset would hide it.
    const Shortcut activation = activationShortcut_;
    if (activation.vk != 0 && vk == activation.vk && ModifiersMatch(activation.modifiers)) {
        enabled_ = !enabled_;
        fallbackBuffer_.clear();
        if (callbacks_.onActivationToggle) {
            callbacks_.onActivationToggle(enabled_);
        }
        return true;  // consume
    }
    // While disabled, swallow nothing but do nothing -- only the toggle above works.
    if (!enabled_) {
        return false;
    }

    // Configurable commit/trigger shortcut. Checked first so its modifier combo
    // isn't mistaken for a context-breaking chord.
    const Shortcut commit = commitShortcut_;
    if (commit.vk != 0 && vk == commit.vk && ModifiersMatch(commit.modifiers)) {
        std::wstring snapshot = fallbackBuffer_;
        fallbackBuffer_.clear();
        if (callbacks_.onCommit) {
            callbacks_.onCommit(snapshot);
        }
        return true;  // consume
    }

    // Language switcher shortcut: cycle the active translation direction.
    const Shortcut cycle = cycleLanguageShortcut_;
    if (cycle.vk != 0 && vk == cycle.vk && ModifiersMatch(cycle.modifiers)) {
        if (callbacks_.onCycleLanguage) {
            callbacks_.onCycleLanguage();
        }
        return true;  // consume
    }

    // Force-override hotkeys: Alt+1 .. Alt+9 pin language-pair index 0..8,
    // suppressing OS-layout auto-routing. Checked BEFORE the chord-reset logic
    // below (Alt is a chord modifier) so it isn't mistaken for a context break.
    // Bare Alt only -- no Ctrl/Shift/Win -- so it never collides with the
    // Alt+Shift+* shortcuts above. Bounded by languagePairCount_ so an Alt+<N>
    // with no matching pair falls through to the app untouched.
    if (vk >= '1' && vk <= '9') {
        const bool altOnly = (GetAsyncKeyState(VK_MENU) & 0x8000) &&
                             !(GetAsyncKeyState(VK_CONTROL) & 0x8000) &&
                             !(GetAsyncKeyState(VK_SHIFT) & 0x8000) &&
                             !(GetAsyncKeyState(VK_LWIN) & 0x8000) &&
                             !(GetAsyncKeyState(VK_RWIN) & 0x8000);
        const int index = vk - '1';
        if (altOnly && index < languagePairCount_) {
            if (callbacks_.onForceLanguage) {
                callbacks_.onForceLanguage(index);
            }
            return true;  // consume so the app never sees the Alt+<N> chord
        }
    }

    if (IsModifierKey(vk)) {
        return false;  // lone Shift/Ctrl/... press
    }
    if (IsChordModifierHeld()) {
        ResetFallback();  // Ctrl+A, Ctrl+V, Alt+Tab, ...
        return false;
    }
    if (IsContextBreakKey(vk)) {
        ResetFallback();  // arrows / Home / End / Esc / Tab / ...
        return false;
    }

    // Backspace: nudge the fallback and re-poll. UIA is authoritative, so even
    // if our fallback is now empty the actual field may still hold text. Never a
    // word boundary -- deleting must not trigger inference.
    if (vk == VK_BACK) {
        if (!fallbackBuffer_.empty()) {
            fallbackBuffer_.pop_back();
        }
        if (callbacks_.onTrigger) {
            callbacks_.onTrigger(fallbackBuffer_, /*wordBoundary=*/false);
        }
        return false;
    }

    // Translate the raw virtual key to Unicode using the FOREGROUND window's
    // keyboard layout (not our own thread's), so the fallback buffer stays
    // correct across layouts. ToUnicodeEx is allocation-free and safe to call
    // from inside a low-level hook.
    const DWORD fgThread = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
    const HKL layout = GetKeyboardLayout(fgThread);
    BYTE keyState[256] = {};
    if (!GetKeyboardState(keyState)) {
        return false;
    }

    wchar_t chars[8] = {};
    const int result = ToUnicodeEx(vk, info.scanCode, keyState, chars, 8, 0, layout);
    if (result <= 0) {
        return false;  // dead key / no printable character
    }

    bool typed = false;
    bool wordBoundary = false;
    for (int i = 0; i < result; ++i) {
        const wchar_t c = chars[i];
        if (std::iswcntrl(c)) {
            // Enter inserts a newline in multiline fields -- mirror it into the
            // fallback (overlay renders '\n' as a line break) and treat it as a
            // word boundary; drop other control characters.
            if (vk == VK_RETURN) {
                fallbackBuffer_.push_back(L'\n');
                typed = true;
                wordBoundary = true;
            }
            continue;
        }
        fallbackBuffer_.push_back(c);
        typed = true;
        if (c == L' ' || std::iswpunct(c)) {
            wordBoundary = true;
        }
    }

    // Any printable keystroke (alphanumeric / space / punctuation / newline) is
    // a trigger: tell CaretTracker to re-read the real text from UIA. Separators
    // additionally flag a word boundary for the inference-gating consumer.
    if (typed && callbacks_.onTrigger) {
        callbacks_.onTrigger(fallbackBuffer_, wordBoundary);
    }
    return false;
}
