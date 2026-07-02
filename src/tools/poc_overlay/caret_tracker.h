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

    explicit CaretTracker(UpdateCallback callback);
    ~CaretTracker();

    // `fallbackText` is used only if UIA yields no text. `wordBoundary` marks a
    // word-closing keystroke. Coalesces to the latest request if calls arrive
    // faster than UIA can be queried; a coalesced word boundary is never lost
    // (the pending flag is sticky until consumed).
    void RequestUpdate(std::wstring fallbackText, bool wordBoundary);

private:
    void ThreadMain();
    // Reads the authoritative text + caret position for the focused control.
    // Returns true if a caret position was resolved (so the overlay can be
    // placed); update.text is set to the UIA text when available, otherwise to
    // `fallbackText`.
    bool Resolve(IUIAutomation* automation, const std::wstring& fallbackText, CaretUpdate& update);

    UpdateCallback callback_;
    std::thread thread_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::wstring pendingFallback_;
    bool pendingWordBoundary_ = false;
    bool hasPending_ = false;
    std::atomic<bool> stop_{false};
};
