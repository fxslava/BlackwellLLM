// The visual contract between the CaretTracker state machine (producer, STA
// thread) and the OverlayWindow renderer (consumer, UI thread). One immutable
// snapshot per transition, marshaled with PostMessage -- the overlay renders
// whatever the latest snapshot says and holds no state-machine logic itself.
#pragma once
#include <windows.h>

#include <string>

#include "token_info.h"

// Mirrors CaretTracker's phases. The first four are the caret-anchored typing
// pipeline; CenterHud is the master-toggle banner; the Selection* pair is the
// passive-selection popup anchored near the mouse.
enum class OverlayPhase {
    Hidden,                // hide the pill
    Typing,                // caret pill: `source` dimmed/italic ("what is captured")
    Translating,           // caret pill: dimmed `source` + loading/streamed partial
    Ready,                 // caret pill: `translation` prominent (Ctrl+Enter commits)
    CenterHud,             // large screen-centered banner (`message`), optional fade
    SelectionTranslating,  // cursor-anchored popup: loading/streamed partial, NO source
    SelectionReady,        // cursor-anchored popup: `translation`, NO source
};

struct OverlaySnapshot {
    OverlayPhase phase = OverlayPhase::Hidden;
    std::wstring source;       // captured source_raw (Typing / Translating)
    std::wstring translation;  // streamed partial or final (Translating / Ready / Selection*)
    std::wstring message;      // CenterHud banner text
    POINT anchor{};            // caret point (Typing*) or mouse point (Selection*)
    bool anchorValid = false;  // false = nothing to anchor to -> treat as Hidden
    bool fade = false;         // CenterHud only: hold briefly, then fade out and hide
    // Currently-effective translation direction (index into the app's language
    // pairs) for the caret-anchored typing pipeline. -1 = unknown/not applicable
    // (selection popup, HUD). Drives the overlay's header bar + override dropdown;
    // the OverlayWindow maps the index to a label from its own configured list.
    int language = -1;
    // How that direction was chosen: false = OS keyboard-layout auto-routing
    // ("[Auto]"), true = a manual override pinned via the dropdown or an Alt+<N>
    // force hotkey ("[Pinned]"). Shown in the header so the user always knows WHY
    // this direction is active.
    bool languagePinned = false;

    // ---- Developer Mode (all inert unless `developerMode` is true) -----------
    bool developerMode = false;
    // The focused UIA text element's bounding box, in SCREEN pixels -- drawn as a
    // red border ("what UIA reports as the edit control"). Valid only when read
    // from the provider this transition.
    RECT uiaBounds{};
    bool uiaBoundsValid = false;
    // The exact caret rectangle from IUIAutomationTextRange (screen pixels): a
    // zero-width sliver at the caret, drawn as a green line ("what the caret
    // hook actually anchored to"). Valid only when the provider gave a caret.
    RECT caretRect{};
    bool caretRectValid = false;
    // Per-token confidence for the heatmap behind the text. `sourceTokens`
    // concatenate to `source`; `translationTokens` concatenate to `translation`.
    // Empty = no heatmap for that run (render it flat).
    TokenHeatmap sourceTokens;
    TokenHeatmap translationTokens;
};
