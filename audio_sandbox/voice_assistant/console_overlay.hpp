#pragma once
// -----------------------------------------------------------------------------
// console_overlay.hpp — the Quake-style log console: a frameless, top-most,
// translucent Direct2D surface that slides down from the top of the screen on
// Ctrl+~ and shows what the app is printing.
//
// WHY DIRECT2D AND NOT A LISTBOX. The thing this renders is a 4000-line ring
// that changes several times a second while the app is talking, over a
// semi-transparent background, at full screen width. A GDI control redraws that
// by blitting text into a DC with no alpha and no subpixel control, and the
// per-frame cost during the slide animation is exactly when it matters. D2D +
// DirectWrite gives a hardware-composited surface, one text layout per visible
// line, and real per-pixel alpha -- which is the whole point of an overlay.
//
// LAYERED, AND WHY THAT DECIDES THE RENDER TARGET. WS_EX_LAYERED with
// SetLayeredWindowAttributes gives a WHOLE-WINDOW constant alpha, which is what
// the "85% opacity" setting means, and it composites a normal HWND render target
// for free. The alternative -- per-pixel alpha via UpdateLayeredWindow -- would
// require rendering to a DXGI surface and blitting it, buying per-glyph
// transparency this design does not use. So: HwndRenderTarget plus a constant
// layer alpha, which is the cheap correct answer for a uniform-translucency
// panel.
//
// WS_EX_NOACTIVATE IS LOAD-BEARING. The console must not steal focus: it is
// summoned over whatever the user is doing, and a panel that took the foreground
// would interrupt the very typing it is meant to sit behind. It is therefore
// never focused, has no caret, and reads input only through its global hotkey.
//
// THREADING. This is a UI-thread object like AssistantWindow: create(), the
// message loop and every D2D call belong to the thread that created it. The ONE
// crossing point is the LogBuffer it reads, which is thread-safe by construction
// -- and it is polled on a timer rather than pushed, so a producer thread never
// touches this class at all. That is deliberate: the alternative (a post from
// the writer) would put a wake-up on the engine thread's per-token path.
//
// THE ANIMATION is a WM_TIMER at ~60 Hz driving one scalar in [0, 1] through an
// ease-out cubic, with the window height set from it. It stops when it arrives,
// so an open-and-idle console costs one repaint every poll interval and a closed
// one costs nothing at all.
//
// ERROR TIER: INIT for create() (returns false, non-fatal -- the app runs
// without a console), RUNTIME for everything after: a lost device recreates its
// resources on the next frame, which is the standard D2D contract.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <string>

#include "log_buffer.hpp"
#include "settings_store.hpp"

namespace rt {

class ConsoleOverlay {
public:
    ConsoleOverlay() = default;
    ~ConsoleOverlay();

    ConsoleOverlay(const ConsoleOverlay&) = delete;
    ConsoleOverlay& operator=(const ConsoleOverlay&) = delete;

    // Create the window and the D2D/DirectWrite factories. `log` is BORROWED and
    // must outlive this object.
    //
    // Returns false when Direct2D or DirectWrite is unavailable, and that is a
    // DEGRADATION rather than a failure: the app is fully usable without a log
    // overlay, and taking startup down over one would be the wrong trade. The
    // reason is printed.
    //
    // The console registers its own global hotkey. It does NOT go through
    // AssistantWindow's registration for a reason worth stating: those ids are
    // scoped to that window, and routing this one through it would make the
    // console's availability depend on the main window existing -- which is
    // exactly backwards for the panel you open when the app is misbehaving.
    bool create(const AssistantSettings& s, LogBuffer* log);

    // Live re-configuration: opacity, font, size, height. Re-creates the text
    // format only when the font actually changed, so dragging an opacity slider
    // does not rebuild DirectWrite state.
    void apply_settings(const AssistantSettings& s);

    // Slide down / retract. Idempotent; calling toggle() mid-animation reverses
    // it from wherever it is rather than snapping.
    void show();
    void hide();
    void toggle();

    [[nodiscard]] bool visible() const noexcept { return target_ > 0.0f; }
    [[nodiscard]] HWND hwnd() const noexcept { return hwnd_; }

private:
    struct Impl;   // hides <d2d1.h> / <dwrite.h> from every consumer of this header

    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT handle(HWND, UINT, WPARAM, LPARAM);

    bool create_device_resources();
    void discard_device_resources();
    void render();
    void tick_animation();
    void reposition();
    void rebuild_text_format();

    HWND  hwnd_ = nullptr;
    Impl* impl_ = nullptr;

    // The animation scalar and its destination, both in [0, 1] where 1 is fully
    // down. Kept as two values rather than a bool + progress so that a toggle
    // mid-slide is just a change of `target_`.
    float position_ = 0.0f;
    float target_ = 0.0f;

    // The revision of the ring the last frame was laid out from. The poll timer
    // compares against it and skips the repaint when nothing has been logged,
    // which is what keeps an open console at ~0% CPU while the app is idle.
    std::uint64_t last_revision_ = 0;

    LogBuffer* log_ = nullptr;

    // Cached configuration. Held rather than read through a settings reference
    // because apply_settings may arrive from a save while a frame is rendering.
    std::wstring font_family_ = L"Cascadia Code";
    float        font_size_ = 14.0f;
    float        height_pct_ = 0.45f;
    float        opacity_ = 0.85f;
    // How far the view is scrolled back from the tail, in lines. 0 = pinned to
    // the newest line, which is where it returns whenever the console is opened:
    // a log you summon is a log you want the end of.
    int          scroll_back_ = 0;

    bool hotkey_registered_ = false;
};

}  // namespace rt
