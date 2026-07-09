#pragma once
#include <windows.h>

#include <atomic>
#include <functional>
#include <string>

// A commit/trigger shortcut expressed the same way the Win32 msctls_hotkey32
// control reports it: `modifiers` is a HOTKEYF_* bitmask (HOTKEYF_CONTROL /
// HOTKEYF_ALT / HOTKEYF_SHIFT), `vk` is a virtual-key code. A vk of 0 means
// "unbound" and is never matched.
struct Shortcut {
    UINT modifiers = 0;  // HOTKEYF_* bitmask
    UINT vk = 0;         // virtual-key code
};

// A lightweight input TRIGGER (not a source of truth).
//
// In the UIA-driven architecture the actual typed text is read back from the
// focused OS control by CaretTracker; this hook's only jobs are:
//
//   * Notice that the text field probably changed (any printable key, Enter,
//     or Backspace) and fire onTrigger so CaretTracker re-polls UIA.
//   * Maintain a small FALLBACK buffer -- the exact characters the user typed,
//     translated with ToUnicodeEx against the *foreground* window's keyboard
//     layout -- for apps (e.g. Telegram/Qt) that expose no usable UIA text or
//     value pattern. onTrigger carries this buffer so CaretTracker can use it
//     when UIA comes back empty.
//   * Detect the configurable commit shortcut and a few clear context-breakers
//     (caret navigation, Ctrl/Alt/Win chords, mouse clicks) to hide the overlay.
//
// The hook no longer tracks caret motion, selection, or editing state in
// detail -- UIA is authoritative for the displayed string.
//
// All state is touched only on the thread that installs the hooks: low-level
// hook procs run in the context of the installing (UI) thread as it pumps
// messages, and SetCommitShortcut is likewise called from that thread, so no
// locking is required.
class HookManager {
public:
    using TextCallback = std::function<void(const std::wstring& fallbackText)>;
    // `wordBoundary` is true when the keystroke that fired the trigger closed a
    // word (Space / punctuation / Enter). It rides along so the resolved-text
    // consumer can gate "run inference now" separately from mere repositioning.
    using TriggerCallback = std::function<void(const std::wstring& fallbackText, bool wordBoundary)>;
    using VoidCallback = std::function<void()>;
    using ToggleCallback = std::function<void(bool active)>;

    struct Callbacks {
        // "The field changed -- go poll UIA." Carries the current fallback buffer
        // and whether this keystroke was a word boundary.
        TriggerCallback onTrigger;
        // Commit shortcut fired. Carries a snapshot of the fallback buffer.
        TextCallback onCommit;
        // Input flow broken (navigation / chord / mouse) -- hide the overlay.
        VoidCallback onReset;
        // Master activation shortcut toggled Translation Mode on (true) / off.
        ToggleCallback onActivationToggle;
        // A left-mouse-button release: a text selection may now exist -- go check
        // it via UIA. Only fired while Translation Mode is active.
        VoidCallback onSelectionCandidate;
        // Language switcher shortcut: cycle to the next translation direction.
        // Only fired while Translation Mode is active.
        VoidCallback onCycleLanguage;
        // A force-override hotkey (Alt+1 -> 0, Alt+2 -> 1, ...) was pressed:
        // pin that language-pair index (CaretTracker::SetLanguageOverride),
        // suppressing OS-layout auto-routing. Only fired while Translation Mode
        // is active, and only for an index that actually maps to a configured
        // pair (see SetLanguagePairCount). Carries the 0-based index.
        std::function<void(int index)> onForceLanguage;
    };

    static HookManager& Instance();

    bool Install(Callbacks callbacks);
    void Uninstall();

    void SetCommitShortcut(const Shortcut& shortcut) { commitShortcut_ = shortcut; }
    Shortcut GetCommitShortcut() const { return commitShortcut_; }

    // Toggles the whole assistant on/off. While disabled the hook ignores all
    // input except the activation shortcut itself.
    void SetActivationShortcut(const Shortcut& shortcut) { activationShortcut_ = shortcut; }
    Shortcut GetActivationShortcut() const { return activationShortcut_; }

    // Cycles the active translation direction (fires onCycleLanguage). Honored
    // only while Translation Mode is active.
    void SetCycleLanguageShortcut(const Shortcut& shortcut) { cycleLanguageShortcut_ = shortcut; }
    Shortcut GetCycleLanguageShortcut() const { return cycleLanguageShortcut_; }

    // Number of configured language pairs. Bounds the Alt+<N> force-override
    // hotkeys: only Alt+1 .. Alt+<count> (max 9) fire onForceLanguage / are
    // consumed, so pressing Alt+5 with two pairs configured passes through to the
    // app untouched. Set at startup on the hook-owning (UI) thread.
    void SetLanguagePairCount(int count) { languagePairCount_ = count; }

    // Programmatic master toggle (the activateOnStartup launch path): puts the
    // hook in the given Translation Mode WITHOUT firing onActivationToggle --
    // the caller owns the accompanying HUD/engine work, exactly as the hotkey
    // handler owns them when the user toggles. Hook-owning (UI) thread only.
    void SetTranslationMode(bool enabled) {
        enabled_ = enabled;
        fallbackBuffer_.clear();
    }

    // Current Translation-Mode state (the master toggle). Read on the UI thread to
    // decide whether the caret pipeline should wake when Companion Mode releases
    // it (see ReapplyInputArchitecture).
    bool IsEnabled() const { return enabled_; }

    // Gate for the inline commit/substitution shortcut (Ctrl+Enter surgical
    // replace). Companion Mode turns this OFF so the caret-driven inline-replace
    // path is fully bypassed while the Spotlight composer owns text entry, and
    // restores it when Companion Mode is switched off. Hook-owning (UI) thread
    // only -- same thread the hook procs run on -- so no locking is required.
    void SetInlineSubstitutionEnabled(bool enabled) { inlineSubstitution_ = enabled; }

    // Guards a text-injection window: while set (and, more robustly, for any event
    // carrying TextInjector::kInjectedSignature), the keyboard hook passes input
    // straight through without treating it as user typing. Set it around a
    // TextInjector::Replace() call so we never re-trigger on our own keystrokes.
    void SetInjecting(bool injecting) { injecting_.store(injecting, std::memory_order_relaxed); }

    // The overlay's clickable chrome (header bar / open dropdown) in SCREEN
    // coordinates. While a mouse button-down/up lands inside it, the mouse hook
    // treats the click as UI interaction -- it does NOT fire the context-break
    // reset (which would hide the overlay) or the selection candidate, letting
    // the click reach the overlay window instead. Pass nullptr to clear. Called
    // on the hook-owning (UI) thread only -- same thread the hook procs run on --
    // so no locking is required.
    void SetInteractiveRect(const RECT* rect) {
        if (rect) {
            interactiveRect_ = *rect;
            hasInteractiveRect_ = true;
        } else {
            hasInteractiveRect_ = false;
        }
    }

private:
    HookManager() = default;
    ~HookManager();
    HookManager(const HookManager&) = delete;
    HookManager& operator=(const HookManager&) = delete;

    static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam);
    static LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam);

    // Returns true if the key event was consumed and must NOT reach the focused
    // application -- currently only the commit shortcut.
    bool HandleKeyEvent(WPARAM wParam, const KBDLLHOOKSTRUCT& info);
    void ResetFallback();

    HHOOK keyboardHook_ = nullptr;
    HHOOK mouseHook_ = nullptr;
    Callbacks callbacks_;
    std::wstring fallbackBuffer_;
    Shortcut commitShortcut_;
    Shortcut activationShortcut_;
    Shortcut cycleLanguageShortcut_;
    int languagePairCount_ = 0;  // bounds the Alt+<N> force-override hotkeys
    bool enabled_ = false;  // Translation Mode: OFF until the master toggle fires
    bool inlineSubstitution_ = true;  // Ctrl+Enter inline replace; OFF in Companion Mode
    std::atomic<bool> injecting_{false};

    RECT interactiveRect_{};            // overlay chrome (screen coords); see SetInteractiveRect
    bool hasInteractiveRect_ = false;
};
