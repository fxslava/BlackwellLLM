#pragma once
#include <windows.h>

#include <string>

struct IUIAutomationValuePattern;  // <UIAutomation.h> is only pulled into the .cpp
struct IUIAutomationTextRange;

// Replaces the text sitting before the caret using a robust, multi-tier
// strategy. Backspace-spam is deliberately abandoned: Chromium collapses rapid
// synthetic VK_BACK events, so we select-then-replace instead.
//
//   Tier 1  ValuePattern SetValue      -- instant, whole-value replace (only when
//                                          the entire control value IS the text we
//                                          replace, so outer context is preserved).
//   Tier 2  UIA range Select + paste   -- highlight the exact IUIAutomationTextRange
//                                          in the OS, then Ctrl+V a clipboard swap.
//   Tier 3  Shift+Left select + paste  -- UIA-opaque apps (Qt/Telegram): select N
//                                          chars by keystroke, then the same paste.
//   Tier 4  Paste over live selection  -- selectionActive only: the OS/user ALREADY
//                                          highlights the target range, so a bare
//                                          Ctrl+V overwrites it (no re-select).
//
// Every synthetic event is tagged with kInjectedSignature in dwExtraInfo so the
// app's own WH_KEYBOARD_LL hook recognizes and ignores it.
//
// THREADING: Tiers 1 and 2 call the passed UIA COM interfaces, which are
// apartment-bound, so Replace() MUST run on the same STA thread that produced
// them (CaretTracker's worker). SendInput/clipboard work from any thread. The
// call is synchronous and inserts small sleeps around clipboard/paste steps.
namespace TextInjector {

// Arbitrary magic stamped into INPUT::ki.dwExtraInfo to mark our own input.
constexpr ULONG_PTR kInjectedSignature = 0xB1AC4E11;  // "BLACKELL"

enum class Tier {
    None = 0,
    ValuePattern = 1,
    RangeSelectPaste = 2,
    KeystrokeSelectPaste = 3,
    SelectionPaste = 4,
};

struct Request {
    IUIAutomationValuePattern* valuePattern = nullptr;    // Tier 1 candidate (borrowed, nullable)
    IUIAutomationTextRange* selectionRange = nullptr;     // Tier 2 candidate (borrowed, nullable)
    std::wstring source;       // text currently before the caret (what we replace)
    std::wstring replacement;  // new text to put in its place
    // Selection-commit mode: the OS/user ALREADY has the target text highlighted
    // (a mouse drag-selection). When set, the keystroke Shift+Left fallback is
    // NOT used -- re-selecting would move or clear the live range -- so if no UIA
    // range does the job, the fallback is a bare Ctrl+V over the current
    // selection (Tier 4). `source` may be empty in this mode.
    bool selectionActive = false;
};

// Runs the tiers in order, stopping at the first that succeeds. Returns which
// tier did the work (Tier::None if nothing could be done).
Tier Replace(const Request& request);

}  // namespace TextInjector
