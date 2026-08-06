// -----------------------------------------------------------------------------
// console_overlay.cpp — the Direct2D console. See the header for why it is
// layered-with-constant-alpha rather than per-pixel, and why it never activates.
// -----------------------------------------------------------------------------
#include "console_overlay.hpp"

#include <d2d1.h>
#include <dwrite.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#include "hotkey_spec.hpp"

namespace rt {
namespace {

constexpr wchar_t kClassName[] = L"BlackwellConsoleOverlay";

// Its own hotkey id. Window-scoped, so it cannot collide with AssistantWindow's
// 1/2/3 -- the OS namespaces hotkey ids per window, which is exactly why the
// console registering its own is free rather than a coordination problem.
constexpr int kHotkeyConsole = 1;

// The animation and the poll share one timer. 16 ms is ~60 Hz, which is the
// slide's frame rate; the log poll is decimated off it (see kPollEvery) because
// re-laying-out 40 text runs sixty times a second to discover nothing changed is
// the one thing that would make this overlay expensive.
constexpr UINT_PTR kTimerId = 1;
constexpr UINT     kTimerMs = 16;
constexpr int      kPollEvery = 4;   // ~15 Hz log poll while open

// Slide duration in frames. ~180 ms, which is long enough to read as a movement
// and short enough that a user toggling it repeatedly never waits.
constexpr float kSlideStep = 1.0f / 11.0f;

// Ease-out cubic. The console decelerates into place rather than stopping dead,
// which is what makes the motion read as a panel with weight rather than a
// rectangle being resized.
float ease_out_cubic(float t) {
    const float inv = 1.0f - t;
    return 1.0f - inv * inv * inv;
}

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                                      nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

}  // namespace

// The COM pointers, kept out of the header so no consumer of console_overlay.hpp
// pulls in <d2d1.h>.
struct ConsoleOverlay::Impl {
    ID2D1Factory*          d2d = nullptr;
    IDWriteFactory*        dwrite = nullptr;
    ID2D1HwndRenderTarget* rt = nullptr;
    IDWriteTextFormat*     format = nullptr;
    ID2D1SolidColorBrush*  brush_bg = nullptr;
    ID2D1SolidColorBrush*  brush_out = nullptr;
    ID2D1SolidColorBrush*  brush_err = nullptr;
    ID2D1SolidColorBrush*  brush_rule = nullptr;

    // Laid out once per revision change, not once per frame.
    std::vector<LogLine> lines;
    int                  frame = 0;
};

ConsoleOverlay::~ConsoleOverlay() {
    if (hotkey_registered_ && hwnd_ != nullptr) UnregisterHotKey(hwnd_, kHotkeyConsole);
    discard_device_resources();
    if (impl_ != nullptr) {
        if (impl_->format != nullptr) impl_->format->Release();
        if (impl_->dwrite != nullptr) impl_->dwrite->Release();
        if (impl_->d2d != nullptr) impl_->d2d->Release();
        delete impl_;
        impl_ = nullptr;
    }
    if (hwnd_ != nullptr) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
}

bool ConsoleOverlay::create(const AssistantSettings& s, LogBuffer* log) {
    log_ = log;
    impl_ = new Impl();

    font_family_ = widen(s.console_font);
    font_size_ = s.console_font_size;
    height_pct_ = s.console_height_pct;
    opacity_ = s.console_opacity;

    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &impl_->d2d))) {
        std::fprintf(stderr, "[console] Direct2D unavailable -- the log overlay is off.\n");
        return false;
    }
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                                   reinterpret_cast<IUnknown**>(&impl_->dwrite)))) {
        std::fprintf(stderr, "[console] DirectWrite unavailable -- the log overlay is off.\n");
        return false;
    }

    static bool registered = false;
    if (!registered) {
        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = &ConsoleOverlay::WndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = kClassName;
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        // NO BACKGROUND BRUSH. D2D paints every pixel of the client area, and a
        // class brush would produce a flash of solid colour between WM_ERASEBKGND
        // and the first present -- which on a translucent overlay is very visible.
        wc.hbrBackground = nullptr;
        RegisterClassExW(&wc);
        registered = true;
    }

    // WS_POPUP: no frame, no caption, no border -- the task's "frameless
    // top-level popup". TOOLWINDOW keeps it out of the Alt+Tab list, which a
    // panel that cannot be focused has no business appearing in. NOACTIVATE is
    // the one that matters most; see the header.
    hwnd_ = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        kClassName, L"Assistant Console", WS_POPUP,
        0, 0, 100, 100, nullptr, nullptr, GetModuleHandleW(nullptr), this);
    if (hwnd_ == nullptr) {
        std::fprintf(stderr, "[console] could not create the overlay window.\n");
        return false;
    }

    // The whole-window constant alpha. Applied to the layer rather than to the
    // brushes so that the text fades WITH the panel -- an overlay whose
    // background is translucent but whose glyphs are opaque reads as a rendering
    // bug rather than as a design.
    SetLayeredWindowAttributes(hwnd_, 0, static_cast<BYTE>(opacity_ * 255.0f), LWA_ALPHA);

    rebuild_text_format();
    reposition();

    const Hotkey hk = parse_hotkey(s.hotkey_console);
    if (hk.armed()) {
        // MOD_NOREPEAT, like the window's own three: without it, holding the
        // chord toggles the console once per key-repeat.
        if (RegisterHotKey(hwnd_, kHotkeyConsole, hk.mods | MOD_NOREPEAT, hk.vk)) {
            hotkey_registered_ = true;
        } else {
            // Not fatal, and the message names the likely cause: a global hotkey
            // fails almost exclusively because another process holds it.
            std::fprintf(stderr,
                         "[console] could not register '%s' (another application likely "
                         "holds it) -- the console is reachable only from the tray menu.\n",
                         s.hotkey_console.c_str());
        }
    }

    // Drives BOTH the slide animation and the log poll; see kPollEvery for why
    // the two share one clock.
    SetTimer(hwnd_, kTimerId, kTimerMs, nullptr);
    return true;
}

void ConsoleOverlay::rebuild_text_format() {
    if (impl_ == nullptr || impl_->dwrite == nullptr) return;
    if (impl_->format != nullptr) {
        impl_->format->Release();
        impl_->format = nullptr;
    }
    HRESULT hr = impl_->dwrite->CreateTextFormat(
        font_family_.c_str(), nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL, font_size_, L"", &impl_->format);
    if (FAILED(hr)) {
        // THE FALLBACK IS NOT COSMETIC. Cascadia Code is the default and ships
        // with Windows Terminal and Visual Studio -- neither of which is
        // guaranteed on a user's box. Consolas is present on every Windows
        // install since Vista, so this is the one that cannot fail.
        hr = impl_->dwrite->CreateTextFormat(
            L"Consolas", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, font_size_, L"", &impl_->format);
    }
    if (SUCCEEDED(hr) && impl_->format != nullptr) {
        // NO WRAPPING: a wrapped log line breaks column alignment and makes the
        // scrollback arithmetic (which counts lines, not rows) wrong. Long lines
        // are clipped at the right edge, which for a log is the correct trade.
        impl_->format->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    }
}

bool ConsoleOverlay::create_device_resources() {
    if (impl_ == nullptr || impl_->rt != nullptr) return true;

    RECT rc{};
    GetClientRect(hwnd_, &rc);
    const D2D1_SIZE_U size = D2D1::SizeU(static_cast<UINT32>(rc.right - rc.left),
                                         static_cast<UINT32>(rc.bottom - rc.top));
    if (size.width == 0 || size.height == 0) return false;

    if (FAILED(impl_->d2d->CreateHwndRenderTarget(D2D1::RenderTargetProperties(),
                                                  D2D1::HwndRenderTargetProperties(hwnd_, size),
                                                  &impl_->rt))) {
        return false;
    }
    // The palette. Deliberately the same near-black the main window's class brush
    // uses, so the two read as one application rather than two.
    impl_->rt->CreateSolidColorBrush(D2D1::ColorF(0.055f, 0.06f, 0.075f, 1.0f),
                                     &impl_->brush_bg);
    impl_->rt->CreateSolidColorBrush(D2D1::ColorF(0.82f, 0.85f, 0.88f, 1.0f),
                                     &impl_->brush_out);
    // stderr in warm red -- the one classification worth making, because a user
    // scanning a wall of log text is looking for exactly this.
    impl_->rt->CreateSolidColorBrush(D2D1::ColorF(1.0f, 0.45f, 0.40f, 1.0f),
                                     &impl_->brush_err);
    impl_->rt->CreateSolidColorBrush(D2D1::ColorF(0.35f, 0.75f, 0.95f, 1.0f),
                                     &impl_->brush_rule);
    return true;
}

void ConsoleOverlay::discard_device_resources() {
    if (impl_ == nullptr) return;
    auto release = [](auto*& p) {
        if (p != nullptr) { p->Release(); p = nullptr; }
    };
    release(impl_->brush_rule);
    release(impl_->brush_err);
    release(impl_->brush_out);
    release(impl_->brush_bg);
    release(impl_->rt);
}

void ConsoleOverlay::apply_settings(const AssistantSettings& s) {
    height_pct_ = s.console_height_pct;

    if (const BYTE a = static_cast<BYTE>(s.console_opacity * 255.0f);
        s.console_opacity != opacity_) {
        opacity_ = s.console_opacity;
        if (hwnd_ != nullptr) SetLayeredWindowAttributes(hwnd_, 0, a, LWA_ALPHA);
    }
    // ONLY on a real change: rebuilding the text format tears down and recreates
    // a DirectWrite object, and this function runs on every settings save.
    const std::wstring want = widen(s.console_font);
    if (want != font_family_ || s.console_font_size != font_size_) {
        font_family_ = want;
        font_size_ = s.console_font_size;
        rebuild_text_format();
    }
    if (visible()) reposition();

    // The chord may have moved. Re-registered unconditionally rather than
    // diffed, because RegisterHotKey is cheap and a stale binding is a hotkey
    // the user just changed and that still does the old thing.
    if (hwnd_ != nullptr) {
        if (hotkey_registered_) {
            UnregisterHotKey(hwnd_, kHotkeyConsole);
            hotkey_registered_ = false;
        }
        if (const Hotkey hk = parse_hotkey(s.hotkey_console); hk.armed()) {
            hotkey_registered_ =
                RegisterHotKey(hwnd_, kHotkeyConsole, hk.mods | MOD_NOREPEAT, hk.vk) != 0;
        }
    }
}

void ConsoleOverlay::reposition() {
    if (hwnd_ == nullptr) return;
    // THE PRIMARY MONITOR'S WORK AREA, not the virtual desktop: a console that
    // spanned every monitor would be unreadable on a multi-head setup, and the
    // Quake convention is one screen.
    RECT work{};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    const int screen_w = work.right - work.left;
    const int screen_h = work.bottom - work.top;

    const int full_h = static_cast<int>(static_cast<float>(screen_h) * height_pct_);
    const int shown = static_cast<int>(static_cast<float>(full_h) * ease_out_cubic(position_));

    if (shown <= 0) {
        ShowWindow(hwnd_, SW_HIDE);
        return;
    }
    // TOP-ANCHORED AND CLIPPED, not moved. The window's height grows from zero
    // rather than the window sliding in from off-screen: an off-screen window on
    // a multi-monitor desktop may be ON the monitor above, and the panel would
    // appear there during the slide.
    //
    // SWP_NOACTIVATE on every call -- the console must never take the foreground.
    SetWindowPos(hwnd_, HWND_TOPMOST, work.left, work.top, screen_w, shown,
                 SWP_NOACTIVATE | SWP_SHOWWINDOW);

    if (impl_ != nullptr && impl_->rt != nullptr) {
        impl_->rt->Resize(D2D1::SizeU(static_cast<UINT32>(screen_w),
                                      static_cast<UINT32>(shown)));
    }
}

void ConsoleOverlay::show() {
    // Pinned to the tail on every open: a log you summon is a log you want the
    // end of, and leaving it where it was scrolled last time means the first
    // thing a user sees is old news.
    scroll_back_ = 0;
    target_ = 1.0f;
}

void ConsoleOverlay::hide() { target_ = 0.0f; }

void ConsoleOverlay::toggle() {
    // Compared against the TARGET rather than the position, so a double-toggle
    // during the slide reverses cleanly instead of depending on which frame the
    // second press landed on.
    if (target_ > 0.0f) hide();
    else show();
}

void ConsoleOverlay::tick_animation() {
    if (position_ == target_) return;
    if (position_ < target_) {
        position_ = std::min(target_, position_ + kSlideStep);
    } else {
        position_ = std::max(target_, position_ - kSlideStep);
    }
    reposition();
    if (position_ > 0.0f) render();
}

void ConsoleOverlay::render() {
    // A RETRACTED CONSOLE DRAWS NOTHING. Not merely an optimisation: the window
    // still exists and still receives paint messages while hidden, and
    // presenting a D2D frame to a zero-height hidden surface is the work that
    // produced the paint storm described in WM_PAINT.
    if (!visible()) return;
    if (impl_ == nullptr || !create_device_resources()) return;

    ID2D1HwndRenderTarget* rt = impl_->rt;
    const D2D1_SIZE_F size = rt->GetSize();

    rt->BeginDraw();
    rt->Clear(D2D1::ColorF(0.055f, 0.06f, 0.075f, 1.0f));

    // The accent rule along the bottom edge: the one visual cue that this is an
    // overlay with an edge rather than the desktop having gone dark.
    rt->FillRectangle(D2D1::RectF(0.0f, size.height - 2.0f, size.width, size.height),
                      impl_->brush_rule);

    if (impl_->format != nullptr) {
        const float line_h = font_size_ * 1.35f;
        const float pad = 8.0f;
        // Bottom-anchored: the newest line sits just above the rule and history
        // scrolls up off the top, which is what every console does and what makes
        // a growing log stable to read.
        const int rows = std::max(1, static_cast<int>((size.height - pad * 2.0f) / line_h));

        const int total = static_cast<int>(impl_->lines.size());
        // scroll_back_ is clamped here rather than where it is changed, because
        // the number of rows depends on the window height and the height changes
        // during the slide.
        const int max_back = std::max(0, total - rows);
        scroll_back_ = std::min(scroll_back_, max_back);
        const int last = total - scroll_back_;
        const int first = std::max(0, last - rows);

        float y = pad;
        for (int i = first; i < last; ++i) {
            const LogLine& ln = impl_->lines[static_cast<std::size_t>(i)];
            const std::wstring w = widen(ln.text);
            if (!w.empty()) {
                rt->DrawTextW(w.c_str(), static_cast<UINT32>(w.size()), impl_->format,
                              D2D1::RectF(pad, y, size.width - pad, y + line_h),
                              ln.stream == LogStream::Err ? impl_->brush_err
                                                          : impl_->brush_out);
            }
            y += line_h;
        }
    }

    // D2DERR_RECREATE_TARGET is the device-lost signal (a driver reset, a display
    // change). Dropping the resources here makes the next frame rebuild them,
    // which is the standard contract and the reason nothing above checks for it.
    if (rt->EndDraw() == D2DERR_RECREATE_TARGET) {
        discard_device_resources();
    }
}

LRESULT ConsoleOverlay::handle(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_HOTKEY:
            if (static_cast<int>(wp) == kHotkeyConsole) {
                toggle();
                return 0;
            }
            break;

        case WM_TIMER:
            if (wp == kTimerId) {
                tick_animation();
                // The log poll, decimated off the animation clock. Only when the
                // console is actually on screen -- a hidden console reading a ring
                // it is not drawing is pure waste.
                if (visible() && impl_ != nullptr && ++impl_->frame >= kPollEvery) {
                    impl_->frame = 0;
                    if (log_ != nullptr && log_->revision() != last_revision_) {
                        last_revision_ = log_->revision();
                        impl_->lines = log_->snapshot();
                        render();
                    }
                }
                return 0;
            }
            break;

        case WM_MOUSEWHEEL: {
            // Scrollback. Three lines per notch, the Windows default.
            const int delta = GET_WHEEL_DELTA_WPARAM(wp);
            scroll_back_ = std::max(0, scroll_back_ + (delta > 0 ? 3 : -3));
            render();
            return 0;
        }

        case WM_PAINT: {
            // BeginPaint/EndPaint, NOT ValidateRect, and this is a fix rather
            // than a style preference. A retracted console still receives
            // WM_PAINT, and the render path declines to draw one (see below) --
            // so with ValidateRect the update region was never actually
            // consumed and the window re-invalidated immediately, producing a
            // continuous WM_PAINT storm.
            //
            // That storm was invisible but fatal to the feature: WM_TIMER is a
            // SYNTHESIZED low-priority message that Windows only produces when
            // nothing higher-priority is queued, and WM_PAINT outranks it. The
            // animation timer therefore never fired once, so the console could
            // not slide even though its hotkey was registering correctly.
            //
            // BeginPaint clears the update region unconditionally, whether or
            // not anything is drawn, which is exactly the property needed here.
            PAINTSTRUCT ps;
            BeginPaint(hwnd, &ps);
            if (visible()) render();
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_DISPLAYCHANGE:
            // The work area moved under us; the cached geometry is stale.
            reposition();
            return 0;

        // NEVER ACTIVATE. Returning this from WM_MOUSEACTIVATE means a click on
        // the console does not take focus from whatever the user was typing in,
        // which is the same promise WS_EX_NOACTIVATE makes for the keyboard.
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;

        case WM_ERASEBKGND:
            return 1;   // D2D paints every pixel; erasing would flash.

        case WM_DESTROY:
            KillTimer(hwnd, kTimerId);
            return 0;

        default:
            break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

LRESULT CALLBACK ConsoleOverlay::WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<ConsoleOverlay*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (self != nullptr) return self->handle(hwnd, msg, wp, lp);
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace rt
