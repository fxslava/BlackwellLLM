#pragma once
#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <wrl/client.h>

#include <string>

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

private:
    static constexpr UINT kMsgState = WM_APP + 1;
    // The backing DIB is allocated once at this maximum size; each frame only
    // blits the content-sized sub-rect from its top-left corner.
    static constexpr int kMaxWidth = 720;
    static constexpr int kMaxHeight = 260;
    static constexpr int kPadX = 14;         // horizontal text inset within the pill
    static constexpr int kPadY = 8;          // vertical text inset within the pill
    static constexpr int kGapAboveCaret = 6; // px between the text bottom and the caret

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    bool InitDirect2D();
    bool InitBackingBitmap();
    void Repaint(const OverlaySnapshot& state);

    HWND hwnd_ = nullptr;

    Microsoft::WRL::ComPtr<ID2D1Factory> d2dFactory_;
    Microsoft::WRL::ComPtr<IDWriteFactory> writeFactory_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> textFormat_;
    Microsoft::WRL::ComPtr<ID2D1DCRenderTarget> renderTarget_;

    HDC memDC_ = nullptr;
    HBITMAP dib_ = nullptr;
};
