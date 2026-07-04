// The visual contract between the CaretTracker state machine (producer, STA
// thread) and the OverlayWindow renderer (consumer, UI thread). One immutable
// snapshot per transition, marshaled with PostMessage -- the overlay renders
// whatever the latest snapshot says and holds no state-machine logic itself.
#pragma once
#include <windows.h>

#include <string>

// Mirrors CaretTracker's Typing / Translating / Ready states, plus Hidden for
// "nothing captured" (empty field, focus lost, commit completed).
enum class OverlayPhase {
    Hidden,       // hide the pill
    Typing,       // show `source` dimmed/italic: "this is what is captured"
    Translating,  // dimmed `source` + loading indicator (or streamed partial)
    Ready,        // show `translation` prominently: Ctrl+Enter will commit it
};

struct OverlaySnapshot {
    OverlayPhase phase = OverlayPhase::Hidden;
    std::wstring source;       // captured source_raw (Typing / Translating)
    std::wstring translation;  // streamed partial (Translating) or final (Ready)
    POINT anchor{};            // caret point the pill anchors above
    bool anchorValid = false;  // false = nothing to anchor to -> treat as Hidden
};
