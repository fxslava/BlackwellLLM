#pragma once
#define NOMINMAX
#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

class LiveTranslationTracker;  // <live_translation_tracker.h> is pulled into the .cpp only

// SpotlightComposer -- "Companion Mode": a standalone, centered input overlay
// (macOS-Spotlight-shaped) that is the app's Tier-4 escape hatch for targets
// that strictly block in-place text mutation. Telegram/Discord/secure browsers
// deny UIAutomation ValuePattern/TextPattern writes AND swallow synthetic
// per-control WM_CHAR; the only channel left is "make the OS believe the user
// pressed Ctrl+V in the real foreground window". So instead of editing text
// where the caret is, the user composes here, sees the live translation, and
// commits -- we hide, hand focus back to the app that was in front, and paste.
//
// Relationship to the rest of the pipeline:
//   * The GLOBAL Alt+Space hotkey is owned by HookManager (the app's single
//     WH_KEYBOARD_LL owner). It calls Summon() -- see the wiring note at the
//     bottom of this header. The composer NEVER installs its own hook.
//   * Keystrokes typed INTO this window are ordinary WM_KEYDOWN/WM_CHAR (we hold
//     the foreground focus while summoned), NOT hook traffic. Every mutation
//     drives LiveTranslationTracker, exactly the same "spinal cord" the
//     caret-driven path uses (speculative TrackUpdate on every edit; a debounced
//     TriggerGeneration decodes). The tracker's Continuous-Speculative-Tracking
//     session does the BPE-seam-safe LCP diff + Copy-on-Write KV-cache truncation
//     for us -- we just feed it the current buffer.
//
// THREADING. The window and every field below live on ONE thread: the UI thread
// that created the composer (the same thread that pumps the message loop and
// owns HookManager's hooks). Summon() is the sole public entry safe to call from
// another thread -- it PostMessage-marshals to the window. The tracker's
// StreamCallback fires on the TRACKER's worker thread and does the one thing that
// contract permits: PostMessage a payload back to this window (see OnTranslation).
//
// The engine control plane is single-threaded (project doctrine): the tracker
// this composer drives is the SAME tracker the caret pipeline drives, so the two
// must not both be live at once. Summon()/Dismiss() report visibility through the
// visibility observer so the integrator can suspend the caret path while the
// composer holds the stage (see SetVisibilityObserver).
class SpotlightComposer {
public:
    SpotlightComposer() = default;
    ~SpotlightComposer();

    SpotlightComposer(const SpotlightComposer&) = delete;
    SpotlightComposer& operator=(const SpotlightComposer&) = delete;

    // Registers the window class, creates the (initially hidden, non-activating)
    // layered window, and initializes the Direct2D/DirectWrite render stack.
    // Returns false (and leaves the object inert) if any Win32/Direct2D resource
    // could not be created. The engine tracker is bound separately via
    // BindTracker once the async model load has produced it. UI thread.
    bool Create(HINSTANCE hInstance);

    // Late-bind (or clear) the engine "spinal cord". The tracker is constructed
    // asynchronously by TranslationService after the model finishes loading, so
    // it cannot be supplied at Create time; ReapplyInputArchitecture binds it
    // once TranslationService::LiveTracker() is non-null, and clears it (nullptr)
    // at teardown. BORROWED; must outlive the binding. UI thread. Until a tracker
    // is bound, Summon still shows the box but typing performs no translation.
    void BindTracker(LiveTranslationTracker* tracker) { tracker_ = tracker; }

    // Bring the composer up: cache the current foreground window (the paste
    // target), position at top-center of that window's monitor, grab focus, and
    // start a fresh composition. Thread-safe: if called off the UI thread it
    // marshals via PostMessage; on the UI thread it acts immediately. Idempotent
    // while already visible (re-centers, keeps the buffer).
    void Summon();

    // Tear the composer down. should_inject == true runs the Tier-4 clipboard +
    // synthetic-Ctrl+V injection of the current translation into the cached
    // foreground window; false just hides (Escape / focus-loss). UI thread.
    void Dismiss(bool should_inject);

    bool IsVisible() const { return visible_; }

    // Switch the active translation direction (index into the tracker's prompt
    // set). Forwarded straight to the tracker; thread-safe there.
    void SetActiveLanguage(int index);

    // Observer fired whenever the composer shows (true) or hides (false), on the
    // UI thread. The integrator wires this to suspend/resume the caret-driven
    // translation path, since both share the single-threaded engine. Optional.
    void SetVisibilityObserver(std::function<void(bool visible)> observer) {
        visibility_observer_ = std::move(observer);
    }

private:
    // Window messages we post to ourselves. WM_APP is the app-private base.
    static constexpr UINT kMsgSummon = WM_APP + 1;       // cross-thread Summon()
    static constexpr UINT kMsgTranslation = WM_APP + 2;  // tracker delta/final (lParam: StreamPayload*)

    static constexpr UINT_PTR kDebounceTimerId = 1;  // idle -> TriggerGeneration
    static constexpr UINT_PTR kCaretTimerId = 2;     // input-caret blink

    // Layout, in DPI-independent base pixels (scaled by dpi_scale_ at paint time).
    static constexpr int kBaseWidth = 650;
    static constexpr int kBaseHeight = 120;
    static constexpr int kBaseCorner = 22;   // squircle corner radius
    static constexpr int kBasePadX = 28;
    static constexpr int kBasePadY = 20;
    static constexpr int kBaseInputFont = 26;
    static constexpr int kBaseTransFont = 19;
    // Backing DIB is allocated once at the largest size we support (base * 2.0 for
    // 200%-DPI monitors); each frame blits only the scaled sub-rect from (0,0).
    static constexpr int kMaxDibWidth = kBaseWidth * 2;
    static constexpr int kMaxDibHeight = kBaseHeight * 2;

    // Debounce: how long the user must pause before a decode fires. Short enough
    // to feel live, long enough that a burst of keystrokes costs one generation.
    static constexpr UINT kDebounceMs = 130;

    // Streamed-translation payload marshaled from the tracker worker thread to the
    // window via PostMessage. `gen` lets a stale delivery (superseded by a newer
    // edit) be dropped on arrival.
    struct StreamPayload {
        std::uint64_t gen = 0;
        std::wstring text;
        bool done = false;
    };

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    LRESULT HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam);

    // --- Direct2D lifetime ----------------------------------------------------
    bool InitDirect2D();
    bool EnsureBackingBitmap();  // (re)creates the DIB when dpi_scale_ changes

    // --- composition / engine wiring -----------------------------------------
    void DoSummon();                 // UI-thread body of Summon()
    void OnInputChanged();           // any edit: speculative track + (re)arm debounce + repaint
    void IssueGeneration();          // debounce fired: commit + decode through the tracker
    void OnTranslation(StreamPayload* payload);  // kMsgTranslation handler (owns/deletes payload)
    void ResetComposition();         // clear buffers + cancel any in-flight tracker work

    // --- input handlers -------------------------------------------------------
    void OnChar(wchar_t ch);
    bool OnKeyDown(WPARAM vk);       // returns true if consumed
    void PasteFromClipboardIntoBuffer();  // Ctrl+V typed INTO the composer

    // --- rendering ------------------------------------------------------------
    // Composes the current buffers into the DIB and pushes it to the layered
    // window via UpdateLayeredWindow. Named OnPaint per the component spec; the
    // layered window has no WM_PAINT of its own (its pixels ARE the DIB).
    void OnPaint();
    void PositionForMonitor(HWND anchor);  // compute origin_/size for anchor's monitor

    // --- Tier-4 injection -----------------------------------------------------
    void InjectTranslation();        // restore foreground, clipboard + synthetic Ctrl+V

    HWND hwnd_ = nullptr;
    HWND prev_foreground_hwnd_ = nullptr;  // paste target, cached at Summon()
    LiveTranslationTracker* tracker_ = nullptr;

    std::wstring input_buffer_;        // raw user text
    std::wstring translation_buffer_;  // latest streamed translation for input_buffer_

    // Monotone request id: bumped on every IssueGeneration so a delayed callback
    // for a superseded buffer is dropped in OnTranslation. Atomic because the
    // tracker worker thread reads the captured copy; the window thread writes it.
    std::atomic<std::uint64_t> gen_{0};

    bool visible_ = false;
    bool summoning_ = false;   // guards the focus-steal handshake from self-dismissing
    bool caret_on_ = true;     // input-caret blink phase

    // Render geometry, recomputed per Summon from the target monitor's DPI.
    float dpi_scale_ = 1.0f;
    POINT origin_{};   // top-left of the window in screen (physical) pixels
    SIZE size_{};      // scaled window size in physical pixels
    int dib_scaled_w_ = 0;  // current DIB validity marker (scaled dims last built)
    int dib_scaled_h_ = 0;

    // Direct2D / DirectWrite. A DCRenderTarget bound to a top-down 32bpp DIB
    // feeds UpdateLayeredWindow's premultiplied-alpha requirement (same technique
    // as OverlayWindow -- an HwndRenderTarget cannot drive per-pixel window alpha).
    Microsoft::WRL::ComPtr<ID2D1Factory> d2d_factory_;
    Microsoft::WRL::ComPtr<IDWriteFactory> dwrite_factory_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> input_format_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> trans_format_;
    Microsoft::WRL::ComPtr<ID2D1DCRenderTarget> render_target_;
    HDC mem_dc_ = nullptr;
    HBITMAP dib_ = nullptr;

    std::function<void(bool visible)> visibility_observer_;
};

// ---------------------------------------------------------------------------
// Wiring into the existing app (main.cpp), for reference:
//
//   SpotlightComposer composer;
//   composer.Create(hInstance, tracker);          // `tracker` = the live tracker
//   composer.SetVisibilityObserver([&](bool up) { // share the single engine:
//       caretTracker.SetActive(!up);              //   pause the caret path while up
//   });
//
//   // HookManager gains an onSummonComposer callback fired on Alt+Space; it runs
//   // on the hook-owning (UI) thread, so it may call Summon() directly:
//   callbacks.onSummonComposer = [&composer] { composer.Summon(); };
//
// The composer consumes Alt+Space in the hook (return 1 / non-zero from the
// low-level proc) so the keystroke never reaches the foreground app.
// ---------------------------------------------------------------------------
