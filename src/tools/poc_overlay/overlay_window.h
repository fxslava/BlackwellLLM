#pragma once
#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <wrl/client.h>

#include <functional>
#include <string>
#include <vector>

#include "overlay_state.h"

// A transparent, click-through, never-activated overlay that renders one or
// more lines of text so that the END of the string (its bottom-right corner)
// sits just above a given screen point (the caret), mimicking inline
// autocomplete / Copilot-style suggestions. The box is sized to its content on
// every update and the text is right/bottom-aligned within it, so the last
// glyph lands directly above the caret and multi-line strings stack upward.
//
// The overlay is a dumb renderer for the CaretTracker state machine: it paints
// whatever the latest OverlaySnapshot says --
//   Typing       captured source text, dimmed + italic (what WILL be sent)
//   Translating  dimmed source, then the streamed partial translation (or an
//                ellipsis while the first tokens are still in flight) in the
//                accent color
//   Ready        the final translation, full-brightness (Ctrl+Enter commits)
//   Hidden       no pill
//
// Rendering path: an ID2D1DCRenderTarget is bound to a top-down 32bpp DIB (via
// BindDC) and painted with a transparent clear color plus premultiplied alpha.
// The DIB is pushed to the window with UpdateLayeredWindow(..., ULW_ALPHA),
// which requires premultiplied source pixels -- exactly what the DC render
// target produces. A plain HwndRenderTarget cannot drive a layered window's
// per-pixel alpha, hence this indirection for smooth anti-aliased text over a
// fully transparent background.
class OverlayWindow {
public:
    OverlayWindow() = default;
    ~OverlayWindow();

    OverlayWindow(const OverlayWindow&) = delete;
    OverlayWindow& operator=(const OverlayWindow&) = delete;

    bool Create(HINSTANCE hInstance);

    // Thread-safe: marshals the snapshot to the overlay's own window via
    // PostMessage so the D2D repaint always happens on the HWND's owning
    // thread. Called from the CaretTracker STA thread on every state change.
    void PostState(const OverlaySnapshot& state);

    void Hide();

    // Language-pair labels (index-aligned with the app's languagePairs), shown in
    // the header bar (active pair) and the override dropdown. Set once at startup
    // on the UI thread, before any state arrives. A leading "Auto" entry (OS-aware
    // routing) is added by the dropdown itself and maps to override index -1.
    void SetLanguageLabels(std::vector<std::wstring> labels) { labels_ = std::move(labels); }

    // A dropdown item was chosen: `index` is the language-pair index to pin, or
    // -1 for "Auto" (clear the manual override). Wire to
    // CaretTracker::SetLanguageOverride. Called on the UI thread.
    void SetOverrideSink(std::function<void(int index)> sink) { overrideSink_ = std::move(sink); }

    // Reports the current interactive screen rectangle (header, or header+menu
    // when the dropdown is open) so the global mouse hook can let those clicks
    // reach this window instead of treating them as a context-break reset. A
    // null rect means "nothing interactive right now". Wire to
    // HookManager::SetInteractiveRect. Called on the UI thread.
    void SetInteractiveRegionSink(std::function<void(const RECT*)> sink) {
        interactiveSink_ = std::move(sink);
    }

private:
    static constexpr UINT kMsgState = WM_APP + 1;
    static constexpr UINT_PTR kFadeTimerId = 1;  // drives the CenterHud fade-out
    // The backing DIB is allocated once at this maximum size; each frame only
    // blits the content-sized sub-rect from its top-left corner.
    static constexpr int kMaxWidth = 720;
    static constexpr int kMaxHeight = 360;   // header + drop-up menu + multi-line body
    static constexpr int kPadX = 14;         // horizontal text inset within the pill
    static constexpr int kPadY = 8;          // vertical text inset within the pill
    static constexpr int kGapAboveCaret = 6; // px between the text bottom and the caret
    static constexpr int kCursorOffsetX = 16; // selection popup offset from the cursor
    static constexpr int kCursorOffsetY = 22;
    static constexpr int kHeaderH = 24;      // clickable header-bar strip height
    static constexpr int kMenuItemH = 26;    // dropdown row height
    // CenterHud fade: hold at full opacity, then fade to zero over these spans.
    static constexpr UINT kFadeTimerMs = 30;
    static constexpr ULONGLONG kHudHoldMs = 900;
    static constexpr ULONGLONG kFadeDurationMs = 450;

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // How the pill's box is positioned relative to the snapshot's anchor.
    enum class AnchorMode { CaretPill, CursorPopup, ScreenCenter };

    bool InitDirect2D();
    bool InitBackingBitmap();
    void Repaint(const OverlaySnapshot& state);
    // Blits the current DIB sub-rect to the layered window at a constant alpha
    // (255 = opaque). Reused by both the initial paint and each fade step.
    void Blit(POINT dst, SIZE size, BYTE alpha);
    void RunFadeStep();  // WM_TIMER: advance / finish the CenterHud fade

    // Pre-draws the per-token probability heatmap ("thermograd") behind `layout`,
    // which must hold exactly the concatenation of `tokens[].text`. Colored
    // rectangles are filled at the token character ranges (measured via
    // HitTestTextRange) BEFORE the text is drawn on top. No-op if `tokens` is
    // empty. `origin` is where the layout is drawn in the DIB (its rects come
    // back in that space).
    void DrawHeatmap(IDWriteTextLayout* layout, const TokenHeatmap& tokens, D2D1_POINT_2F origin);

    // --- Developer-Mode debug overlay (second, whole-virtual-screen window) ----
    // The red UIA bounding box + green caret line live at arbitrary screen
    // coordinates and can be far larger than the caret pill, so they get their
    // own click-through layered window rather than fighting the pill's tiny,
    // caret-anchored DIB. Lazily created on the first dev-mode paint.
    bool EnsureDebugWindow();
    void RenderDebug(const OverlaySnapshot& state);
    void HideDebug();

    // --- interactive header bar + override dropdown --------------------------
    // True while the caret pill is showing a clickable header (typing pipeline
    // with a known language and configured labels).
    bool HeaderVisible(const OverlaySnapshot& state) const;
    // Hit-test a SCREEN point against the header / open menu rows. Returns the
    // region so WM_NCHITTEST can claim only those pixels (everything else stays
    // click-through) and WM_LBUTTONDOWN can act.
    enum class Hit { None, Header, MenuItem };
    Hit HitTest(POINT screenPt, int& outMenuIndex) const;
    void OnLeftButtonDown(POINT screenPt);
    // Push the current interactive rectangle (or null) to the hook via the sink.
    void PublishInteractiveRegion();

    HWND hwnd_ = nullptr;

    Microsoft::WRL::ComPtr<ID2D1Factory> d2dFactory_;
    Microsoft::WRL::ComPtr<IDWriteFactory> writeFactory_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> textFormat_;    // caret pill (trailing/far)
    Microsoft::WRL::ComPtr<IDWriteTextFormat> hudFormat_;     // center banner (center/center)
    Microsoft::WRL::ComPtr<IDWriteTextFormat> popupFormat_;   // selection popup (leading/near)
    Microsoft::WRL::ComPtr<IDWriteTextFormat> headerFormat_;  // header + menu (leading/center)
    Microsoft::WRL::ComPtr<ID2D1DCRenderTarget> renderTarget_;

    // Cached blit geometry so the fade timer can re-blit without re-rendering.
    POINT lastDst_{};
    SIZE lastSize_{};
    bool fadeActive_ = false;
    ULONGLONG fadeStart_ = 0;

    // Header / dropdown state. `lastState_` is the most recent NON-hidden
    // snapshot, so toggling the dropdown can re-render without a new snapshot.
    std::vector<std::wstring> labels_;   // language-pair labels (index-aligned)
    OverlaySnapshot lastState_;          // last rendered content (for dropdown re-render)
    bool dropdownOpen_ = false;
    // Interactive rectangles in SCREEN coordinates, recomputed each paint. The
    // menu maps row -> override index (row 0 = "Auto" = -1, row i = pair i-1).
    RECT headerRect_{};
    bool headerRectValid_ = false;
    struct MenuRow { RECT rect; int index; };  // index: -1 = Auto, >=0 = pair
    std::vector<MenuRow> menuRows_;

    std::function<void(int)> overrideSink_;
    std::function<void(const RECT*)> interactiveSink_;

    HDC memDC_ = nullptr;
    HBITMAP dib_ = nullptr;

    // Developer-Mode debug overlay: its own window + DIB + DC render target,
    // sized once to the virtual screen so any on-screen box fits. All null until
    // the first dev-mode paint (EnsureDebugWindow).
    HWND debugHwnd_ = nullptr;
    HDC debugDC_ = nullptr;
    HBITMAP debugDib_ = nullptr;
    Microsoft::WRL::ComPtr<ID2D1DCRenderTarget> debugRT_;
    int debugDibW_ = 0;
    int debugDibH_ = 0;
    bool debugVisible_ = false;
};
