#pragma once
#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

struct IUIAutomation;         // <UIAutomation.h> is only pulled into the .cpp
struct IUIAutomationElement;

struct CaretUpdate {
    std::wstring text;       // resolved string to display (UIA text, or fallback)
    POINT caretScreenPos{};  // the caret point (left/top of the caret rect), screen coords
    bool caretFound = false;
    bool wordBoundary = false;  // the triggering keystroke closed a word (gates inference)
};

// Owns a dedicated COM/UI-Automation STA worker thread and is the SINGLE SOURCE
// OF TRUTH for the displayed text.
//
// On each RequestUpdate the worker queries the focused element and, from its
// IUIAutomationTextPattern (primary) or IUIAutomationValuePattern (secondary),
// reads back the text of the active line up to the caret plus the caret's
// screen rectangle. Because it reads the control's real state, the string is
// perfectly synchronized regardless of keyboard layout, cut/paste, or
// selection changes. Only when a control exposes no usable UIA text/value
// pattern (e.g. Telegram/Qt) does it fall back to the buffer supplied by the
// keyboard hook.
//
// The keyboard hook (UI thread) must never block on a UIA round-trip against a
// foreign process, so RequestUpdate is fire-and-forget and results are reported
// asynchronously via the callback -- which runs ON THIS WORKER THREAD. Callers
// that touch a window must marshal back themselves (see OverlayWindow::PostUpdate).
class CaretTracker {
public:
    using UpdateCallback = std::function<void(const CaretUpdate&)>;
    // Mock "translation" applied to the source text at commit time, on the STA
    // thread (so it runs against the freshly re-resolved authoritative text).
    using TransformCallback = std::function<std::wstring(const std::wstring&)>;
    // Brackets the text-injection window (true=begin, false=end) so the keyboard
    // hook can flag that synthetic input is in flight. Wired to HookManager.
    using InjectionGuard = std::function<void(bool)>;

    CaretTracker(UpdateCallback callback, TransformCallback commitTransform,
                 InjectionGuard injectionGuard);
    ~CaretTracker();

    // `fallbackText` is used only if UIA yields no text. `wordBoundary` marks a
    // word-closing keystroke. Coalesces to the latest request if calls arrive
    // faster than UIA can be queried; a coalesced word boundary is never lost
    // (the pending flag is sticky until consumed).
    void RequestUpdate(std::wstring fallbackText, bool wordBoundary);

    // Commit shortcut fired: re-resolve the current text on the STA thread and
    // replace it (transform -> TextInjector 3-tier). `fallbackText` is the hook's
    // typed buffer, used for the Tier-3 length when no UIA is available.
    void RequestCommit(std::wstring fallbackText);

private:
    void ThreadMain();
    // Reads the authoritative text + caret position for the focused control.
    // Returns true if a caret position was resolved (so the overlay can be
    // placed); update.text is set to the UIA text when available, otherwise to
    // `fallbackText`.
    bool Resolve(IUIAutomation* automation, const std::wstring& fallbackText, CaretUpdate& update);
    // Runs the commit replacement on the STA thread (gathers UIA context here so
    // the interfaces stay in their owning apartment).
    void PerformCommit(IUIAutomation* automation, const std::wstring& fallbackText);
    // Given the full UIA text before the caret, returns just the segment typed
    // since the last commit (the part the overlay should show). Resets the commit
    // boundary if the text no longer starts with the committed prefix.
    std::wstring StripCommittedPrefix(const std::wstring& fullText);

    UpdateCallback callback_;
    TransformCallback transform_;
    InjectionGuard injectionGuard_;
    std::thread thread_;

    // State split: `committedPrefix_` is the text already committed/translated and
    // still sitting in the field, so the overlay hides it and tracks only the new
    // segment. `inferenceContext_` keeps the full ORIGINAL history for the AI.
    // Both are touched only on the worker thread.
    std::wstring committedPrefix_;
    std::wstring inferenceContext_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::wstring pendingFallback_;
    bool pendingWordBoundary_ = false;
    bool pendingCommit_ = false;
    std::wstring pendingCommitFallback_;
    bool hasPending_ = false;
    std::atomic<bool> stop_{false};
};
