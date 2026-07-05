// The visual contract between the CaretTracker state machine (producer, STA
// thread) and the OverlayWindow renderer (consumer, UI thread). One immutable
// snapshot per transition, marshaled with PostMessage -- the overlay renders
// whatever the latest snapshot says and holds no state-machine logic itself.
#pragma once
#include <windows.h>

#include <string>

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
};
