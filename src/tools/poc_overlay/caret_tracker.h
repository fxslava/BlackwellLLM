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
//    ^                ^  ^        (RequestPreview)            |
//    |                |  |                                    | stream done
//    |     keystroke  |  +--- keystroke (cancel inference) ---+------+
//    |                |                                       v      |
//    +---- commit ----+----------------------------------- Ready <--+
//         (surgical TextInjector replace, Ctrl+Enter)
//
//   Typing       every keystroke re-resolves the UIA text, re-extracts the
//                capture area (backwards from the caret to the configured
//                granularity boundary) and re-arms the idle deadline. The
//                overlay shows the capture dimmed.
//   Translating  the idle deadline expired: the capture (source_raw) went to
//                the TranslationService. The overlay shows a loading state and
//                the streamed partial as it arrives.
//   Ready        the stream completed (translation_raw). Ctrl+Enter now
//                performs a host-side surgical replace -- NO inference on the
//                commit path.
//   Interrupt    ANY keystroke in Translating/Ready cancels the in-flight
//                request, clears the translation and returns to Typing.
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
    struct Callbacks {
        // State changed -> repaint. Wire to OverlayWindow::PostState (PostMessage-
        // marshaled, safe from this thread).
        std::function<void(const OverlaySnapshot&)> render;
        // Idle timer fired: run preview inference on `source` with the durable
        // committed `context`. Wire to TranslationService::RequestPreview.
        std::function<void(const std::wstring& source, const std::wstring& context)>
            requestPreview;
        // Keystroke interrupted Translating/Ready: stop paying for the decode.
        // Wire to TranslationService::CancelPending.
        std::function<void()> cancelPreview;
        // Brackets the text-injection window (true=begin, false=end) so the
        // keyboard hook can flag that synthetic input is in flight.
        std::function<void(bool)> injectionGuard;
    };

    CaretTracker(Callbacks callbacks, CaptureGranularity granularity, int idleTimerMs);
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
    // return to Idle (overlay hidden).
    void RequestReset();

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
    enum class Phase { Idle, Typing, Translating, Ready };

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
    void ResetToIdle();
    void Render() const;

    Callbacks callbacks_;
    std::thread thread_;

    std::atomic<int> granularity_;
    std::atomic<int> idleTimerMs_;

    // --- worker-thread-only state machine ------------------------------------
    Phase phase_ = Phase::Idle;
    std::wstring sourceRaw_;           // current capture area (what inference sees)
    std::wstring translationPartial_;  // streamed partial while Translating
    std::wstring translationRaw_;      // final translation while Ready
    POINT anchor_{};                   // caret point from the last resolve
    bool anchorValid_ = false;
    bool idleArmed_ = false;
    std::chrono::steady_clock::time_point idleDeadline_{};

    // State split: `committedPrefix_` is the text already committed/translated
    // and still sitting in the field, so capture tracks only the new segment.
    // `inferenceContext_` keeps the full ORIGINAL history for the AI. Both are
    // touched only on the worker thread.
    std::wstring committedPrefix_;
    std::wstring inferenceContext_;

    // --- cross-thread event slots (guarded by mutex_) -------------------------
    std::mutex mutex_;
    std::condition_variable cv_;
    std::wstring pendingFallback_;
    bool hasPendingUpdate_ = false;
    bool pendingCommit_ = false;
    bool pendingReset_ = false;
    std::optional<std::pair<std::wstring, bool>> pendingTranslation_;  // text, done
    bool hasPending_ = false;
    std::atomic<bool> stop_{false};
};
