#pragma once
#include <windows.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "config.h"         // CaptureGranularity
#include "overlay_state.h"  // OverlaySnapshot / OverlayPhase

struct IUIAutomation;         // <UIAutomation.h> is only pulled into the .cpp
struct IUIAutomationElement;

// Owns a dedicated COM/UI-Automation STA worker thread, is the SINGLE SOURCE
// OF TRUTH for the displayed text, and runs the debounce-driven interaction
// state machine:
//
//        keystroke                     idle timer fires
//   Idle ---------> Typing ----------------------------> Translating
//    ^                ^  ^       (TriggerGeneration)          |
//    |                |  |                                    | stream done
//    |     keystroke  |  +--- keystroke (cancel inference) ---+------+
//    |                |                                       v      |
//    +---- commit ----+----------------------------------- Ready <--+
//         (surgical TextInjector replace, Ctrl+Enter)
//
//   Typing       every keystroke re-resolves the UIA text, re-extracts the
//                capture area (backwards from the caret to the configured
//                granularity boundary), re-arms the idle deadline, AND fires
//                a speculative background prefill (TrackUpdate) that warms
//                the engine's radix tree for the growing capture -- so the
//                Translating step below has (ideally) nothing left to
//                prefill. The overlay shows the capture dimmed.
//   Translating  the idle deadline expired: the capture (source_raw) went to
//                LiveTranslationTracker via TriggerGeneration, bypassing
//                AgentOrchestrator entirely. The overlay shows a loading
//                state and the streamed partial as it arrives.
//   Ready        the stream completed (translation_raw). Ctrl+Enter now
//                performs a host-side surgical replace -- NO inference on the
//                commit path.
//   Interrupt    ANY keystroke in Translating/Ready cancels the in-flight
//                generation, clears the translation and returns to Typing.
//
// The debounce runs on this same worker thread as a cv_.wait_until deadline --
// no extra timer thread, no detached std::async, no re-entrancy: timer expiry
// is just another event in the one serialized event loop, so it can never race
// a keystroke or a commit.
//
// The keyboard hook (UI thread) must never block on a UIA round-trip against a
// foreign process, so all Request* entry points are O(1) fire-and-forget. All
// callbacks fire ON THIS WORKER THREAD and must only enqueue/marshal.
class CaretTracker {
public:
    // OS-aware translation-direction routing. The typing pipeline picks a
    // language-pair index from the FOREGROUND keyboard layout (locked for the
    // whole typing session so an accidental layout tap mid-sentence never flips
    // it); a highlighted-text selection ignores the layout entirely and uses the
    // asymmetric `selectionReading` direction (read foreign text while keeping
    // your layout ready to reply). All fields are indices into the app's
    // languagePairs (see config.h) -- the exact same indices the
    // TranslationService/LiveTranslationTracker address their .bkv branches by.
    struct LanguageRouting {
        int count = 0;             // number of configured language pairs
        int typingCyrillic = 0;    // index used when typing on a Cyrillic (RU) layout -> RU->EN
        int typingLatin = 0;       // index used when typing on a Latin (EN) layout   -> EN->RU
        int selectionReading = 0;  // index used for a selection (asymmetric reading) -> EN->RU
    };

    struct Callbacks {
        // State changed -> repaint. Wire to OverlayWindow::PostState (PostMessage-
        // marshaled, safe from this thread).
        std::function<void(const OverlaySnapshot&)> render;
        // The effective translation direction changed (typing-session lock, a
        // selection's reading direction, or a manual override). Fired on the
        // worker thread BEFORE the matching trackUpdate/triggerGeneration so the
        // engine addresses the correct .bkv branch. Wire to
        // TranslationService::SetActiveLanguage (an atomic store; O(1)).
        std::function<void(int languageIndex)> setActiveLanguage;
        // EVERY keystroke while a capture area exists: fire-and-forget
        // speculative background prefill of `source` (+ durable committed
        // `context`) that warms the engine's radix tree token-by-token as the
        // user types, so the debounce-triggered generation below has
        // (ideally) nothing left to prefill. Wire to
        // TranslationService::TrackUpdate -- bypasses AgentOrchestrator
        // entirely, talks straight to the adapter/engine.
        std::function<void(const std::wstring& source, const std::wstring& context)>
            trackUpdate;
        // Idle timer fired: generate now on `source` with the durable
        // committed `context`. Wire to TranslationService::TriggerGeneration.
        std::function<void(const std::wstring& source, const std::wstring& context)>
            triggerGeneration;
        // Keystroke interrupted Translating/Ready, or a focus/caret reset:
        // stop paying for any pending/in-flight decode. Wire to
        // TranslationService::Cancel.
        std::function<void()> cancelGeneration;
        // Brackets the text-injection window (true=begin, false=end) so the
        // keyboard hook can flag that synthetic input is in flight.
        std::function<void(bool)> injectionGuard;
    };

    CaretTracker(Callbacks callbacks, CaptureGranularity granularity, int idleTimerMs,
                 LanguageRouting routing);
    ~CaretTracker();

    // Keystroke trigger. `fallbackText` is used only if UIA yields no text.
    // Coalesces to the latest request if calls arrive faster than UIA can be
    // queried. Every call is an interrupt: it resets the idle timer and cancels
    // any in-flight/displayed translation.
    void RequestUpdate(std::wstring fallbackText, bool wordBoundary);

    // Commit shortcut fired: if (and only if) the state machine is in Ready,
    // surgically replace source_raw behind the caret with translation_raw via
    // the 3-tier TextInjector. Ignored in any other state.
    void RequestCommit(std::wstring fallbackText);

    // Input flow broken (navigation / chord / mouse): cancel everything and
    // return to Idle (overlay hidden). A no-op against a CenterHud banner.
    void RequestReset();

    // A left-mouse-button release fired while Translation Mode is active: check
    // the UIA selection and, if a non-empty range is selected, translate it in
    // the cursor-anchored SelectionPopup (immediate -- no debounce). O(1),
    // fire-and-forget; ignored unless the tracker is Active.
    void RequestSelectionCheck();

    // Master Translation-Mode gate. When set false, cancels any in-flight
    // generation and stops the tracker producing translations/renders (so a
    // CenterHud banner owns the overlay). Typing/selection are only honored
    // while Active. Marshals into the worker loop; O(1).
    void SetActive(bool active);

    // Manually override the active translation direction (index into the app's
    // languagePairs), overriding the OS-aware typing heuristic and the selection
    // reading default. Pass a valid index to pin a direction (e.g. the overlay
    // header dropdown, or the Alt+Shift+L cycle); pass -1 to clear the override
    // and return to fully automatic OS-aware routing. Marshals into the worker
    // loop; O(1). Takes effect immediately for an in-progress typing session.
    void SetLanguageOverride(int index);

    // Hot-reload the routing table after a Settings save changed the language
    // pairs (NO app restart). An out-of-range manual override is cleared and a
    // live typing session relocks its direction against the new table, so no
    // stale index can ever reach the engine. Marshals into the worker loop.
    void SetLanguageRouting(LanguageRouting routing);

    // Show a large, screen-centered HUD banner (master-toggle feedback). `fade`
    // = hold briefly, then dissolve and hide; !fade = persist until replaced
    // (e.g. "Initializing..."). Marshals into the worker loop.
    void ShowHud(std::wstring message, bool fade);

    // Translation stream delivery. THREAD-SAFE: called from the
    // TranslationService worker; marshals into this worker's event loop.
    // done=false carries a growing partial; done=true the final text (empty
    // final = the run failed/was dropped -> falls back to Typing).
    void OnPreviewResult(std::wstring text, bool done);

    // Live-apply capture settings from the settings window (UI thread).
    void SetCaptureSettings(CaptureGranularity granularity, int idleTimerMs) {
        granularity_.store(static_cast<int>(granularity), std::memory_order_relaxed);
        idleTimerMs_.store(idleTimerMs > 0 ? idleTimerMs : 1, std::memory_order_relaxed);
    }

    // Pure helper, exposed for tests: extract the capture area -- the suffix of
    // `segment` after the last granularity boundary, left-trimmed. A trailing
    // run of whitespace/boundary characters is skipped BEFORE searching, so a
    // just-typed terminator keeps its own clause/sentence captured.
    static std::wstring ExtractCapture(const std::wstring& segment,
                                       CaptureGranularity granularity);

private:
    // Idle/Typing/Translating/Ready are the caret-anchored typing pipeline;
    // Hud is the master-toggle banner; SelTranslating/SelReady are the
    // passive-selection popup (translating a mouse selection near the cursor).
    enum class Phase { Idle, Typing, Translating, Ready, Hud, SelTranslating, SelReady };

    void ThreadMain();
    // Reads the authoritative text + caret position for the focused control.
    // Returns true if usable; fills `segment` (text since last commit, up to
    // the caret) and the caret anchor.
    bool ResolveSegment(IUIAutomation* automation, const std::wstring& fallbackText,
                        std::wstring& segment, bool& fromUia);
    std::wstring StripCommittedPrefix(const std::wstring& fullText);

    // --- state-machine event handlers; all run on the worker thread ----------
    void HandleKeystroke(IUIAutomation* automation, const std::wstring& fallback);
    void HandleIdleExpired();
    void HandleTranslation(const std::wstring& text, bool done);
    void HandleCommit(IUIAutomation* automation);
    void HandleSelectionCheck(IUIAutomation* automation);   // mouse-up -> translate selection
    void HandleSelectionCommit(IUIAutomation* automation);  // Ctrl+Enter over a selection
    void HandleSetActive(bool active);
    void HandleShowHud(const std::wstring& message, bool fade);
    void HandleSetOverride(int index);
    void HandleSetRouting(const LanguageRouting& routing);
    void ResetToIdle();
    void Render() const;

    // --- OS-aware language routing (worker thread only) -----------------------
    // Index of the language pair implied by the FOREGROUND window's current
    // keyboard layout: Cyrillic (Russian) layout -> typingCyrillic, otherwise
    // typingLatin. Reads GetKeyboardLayout for the focused thread.
    int DetectTypingLanguage() const;
    // Resolve the direction to use for a NEW typing session and lock it into
    // sessionLanguage_: the manual override if one is set, else the OS layout.
    // Idempotent within a session (only recomputed when a fresh session begins).
    int ResolveTypingLanguage(bool newSession);
    // Resolve the direction for a selection (asymmetric): the manual override if
    // set, else the reading default -- never the OS layout.
    int ResolveSelectionLanguage() const;
    // Push `index` to the engine (setActiveLanguage callback) if it changed since
    // the last push, so a burst of same-direction keystrokes costs one store.
    void PublishLanguage(int index);

    // Read once from `read_selection_via_clipboard`: UIA-opaque controls (Qt /
    // Telegram, WinUI3 / Notepad) expose no usable text SELECTION, so fall back
    // to a backup-clipboard + synthetic Ctrl+C + restore round-trip. Returns true
    // and fills `out` only on a non-empty capture. Runs on the worker thread;
    // brackets the synthetic copy with the injection guard so our own keyboard
    // hook ignores it.
    bool ReadSelectionViaClipboard(std::wstring& out);

    Callbacks callbacks_;
    std::thread thread_;

    std::atomic<int> granularity_;
    std::atomic<int> idleTimerMs_;

    // --- worker-thread-only state machine ------------------------------------
    Phase phase_ = Phase::Idle;
    bool active_ = false;              // Translation Mode gate (set via SetActive)
    std::wstring sourceRaw_;           // current capture / selected text (what inference sees)
    std::wstring translationPartial_;  // streamed partial while (Sel)Translating
    std::wstring translationRaw_;      // final translation while (Sel)Ready
    std::wstring hudMessage_;          // CenterHud banner text (Phase::Hud)
    bool hudFade_ = false;             // CenterHud: auto-fade
    POINT anchor_{};                   // caret point (Typing*) or cursor point (Sel*)
    bool anchorValid_ = false;
    bool idleArmed_ = false;
    std::chrono::steady_clock::time_point idleDeadline_{};

    // State split: `committedPrefix_` is the text already committed/translated
    // and still sitting in the field, so capture tracks only the new segment.
    // `inferenceContext_` keeps the full ORIGINAL history for the AI. Both are
    // touched only on the worker thread.
    std::wstring committedPrefix_;
    std::wstring inferenceContext_;

    // --- language routing (worker-thread-only) --------------------------------
    LanguageRouting routing_;                  // index map from config (hot-reloadable)
    std::optional<int> languageOverride_;      // manual pin (dropdown / cycle); nullopt = auto
    int sessionLanguage_ = -1;                 // direction locked for the CURRENT typing session
    int publishedLanguage_ = -1;               // last index handed to setActiveLanguage

    // --- cross-thread event slots (guarded by mutex_) -------------------------
    std::mutex mutex_;
    std::condition_variable cv_;
    std::wstring pendingFallback_;
    bool hasPendingUpdate_ = false;
    bool pendingCommit_ = false;
    bool pendingReset_ = false;
    bool pendingSelectionCheck_ = false;
    std::optional<bool> pendingActive_;                               // SetActive slot
    std::optional<int> pendingOverride_;                              // SetLanguageOverride slot
    std::optional<LanguageRouting> pendingRouting_;                   // SetLanguageRouting slot
    std::optional<std::pair<std::wstring, bool>> pendingHud_;         // message, fade
    std::optional<std::pair<std::wstring, bool>> pendingTranslation_;  // text, done
    bool hasPending_ = false;
    std::atomic<bool> stop_{false};
};
