// -----------------------------------------------------------------------------
// splash_screen.cpp — the startup splash. See the header for why it owns a
// thread rather than borrowing main's.
// -----------------------------------------------------------------------------
#include "splash_screen.hpp"

#include <algorithm>
#include <utility>

namespace rt {
namespace {

constexpr wchar_t kClassName[] = L"BlackwellSplash";

constexpr int kWidth  = 460;
constexpr int kHeight = 260;

constexpr UINT_PTR kTimerId = 1;
constexpr UINT     kTimerMs = 16;      // ~60 Hz: the fade and the pulse

// Fade rates, per frame, in 0..255 alpha. In gives ~130 ms, out ~200 ms -- the
// asymmetry is deliberate: appearing should feel immediate, leaving should feel
// like a transition rather than a window vanishing.
constexpr int kFadeInStep  = 32;
constexpr int kFadeOutStep = 20;

// The palette, matched to the messenger's own dark surface so the splash and the
// window that replaces it read as one application.
constexpr COLORREF kBg       = RGB(23, 24, 28);
constexpr COLORREF kAccent   = RGB(90, 190, 240);
constexpr COLORREF kTitleFg  = RGB(238, 240, 244);
constexpr COLORREF kSubtleFg = RGB(150, 156, 168);

}  // namespace

SplashScreen::~SplashScreen() { dismiss(); }

void SplashScreen::show(const std::wstring& title, const std::wstring& subtitle) {
    if (running_.load(std::memory_order_acquire)) return;
    title_ = title;
    subtitle_ = subtitle;
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this, title, subtitle] { thread_main(title, subtitle); });
}

void SplashScreen::set_status(const std::wstring& text) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_ = text;
    }
    // No repaint from here: the splash thread's timer picks it up on its next
    // frame. Posting one would mean a caller on the startup critical path
    // touching a window it does not own.
}

void SplashScreen::thread_main(std::wstring title, std::wstring subtitle) {
    title_ = std::move(title);
    subtitle_ = std::move(subtitle);

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &SplashScreen::WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = kClassName;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;   // fully painted in WM_PAINT; a brush would flash
    RegisterClassExW(&wc);

    // Centred on the PRIMARY monitor's work area. Not the virtual desktop: a
    // splash centred across two monitors lands on the seam between them.
    RECT work{};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    const int x = work.left + ((work.right - work.left) - kWidth) / 2;
    const int y = work.top + ((work.bottom - work.top) - kHeight) / 2;

    // TOPMOST so it is not buried by whatever had focus; TOOLWINDOW to stay out
    // of Alt+Tab; NOACTIVATE so launching the app does not steal the user's
    // focus mid-keystroke.
    HWND hwnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        kClassName, L"", WS_POPUP, x, y, kWidth, kHeight,
        nullptr, nullptr, GetModuleHandleW(nullptr), this);
    if (hwnd == nullptr) {
        running_.store(false, std::memory_order_release);
        return;
    }
    hwnd_.store(hwnd, std::memory_order_release);

    // Starts fully transparent and fades UP on the timer, so the splash does not
    // pop into existence at full opacity before its first paint has run.
    SetLayeredWindowAttributes(hwnd, 0, 0, LWA_ALPHA);
    ShowWindow(hwnd, SW_SHOWNOACTIVATE);
    SetTimer(hwnd, kTimerId, kTimerMs, nullptr);

    int alpha = 0;
    MSG msg;
    for (;;) {
        // The fade state machine lives in the loop rather than in the WndProc so
        // that the exit condition (fully faded out) can break out of the pump
        // directly -- a PostQuitMessage from a timer would work too, but this
        // keeps the whole lifecycle readable in one place.
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) goto done;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        if (fading_.load(std::memory_order_acquire)) {
            alpha -= kFadeOutStep;
            if (alpha <= 0) break;
        } else if (alpha < 255) {
            alpha = std::min(255, alpha + kFadeInStep);
        }
        SetLayeredWindowAttributes(hwnd, 0, static_cast<BYTE>(alpha), LWA_ALPHA);
        InvalidateRect(hwnd, nullptr, FALSE);
        UpdateWindow(hwnd);
        Sleep(kTimerMs);
    }

done:
    KillTimer(hwnd, kTimerId);
    hwnd_.store(nullptr, std::memory_order_release);
    DestroyWindow(hwnd);
    running_.store(false, std::memory_order_release);
}

void SplashScreen::paint(HWND hwnd) {
    PAINTSTRUCT ps;
    const HDC dc = BeginPaint(hwnd, &ps);

    RECT rc{};
    GetClientRect(hwnd, &rc);

    // DOUBLE-BUFFERED. This repaints at 60 Hz behind a status line that changes,
    // and painting the background and four text runs straight onto the window DC
    // flickers visibly.
    const HDC mem = CreateCompatibleDC(dc);
    const HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom);
    const HGDIOBJ old_bmp = SelectObject(mem, bmp);

    const HBRUSH bg = CreateSolidBrush(kBg);
    FillRect(mem, &rc, bg);
    DeleteObject(bg);

    // A hairline border: without it the splash has no edge against a dark
    // desktop and reads as a floating block of text.
    const HBRUSH edge = CreateSolidBrush(RGB(52, 56, 66));
    FrameRect(mem, &rc, edge);
    DeleteObject(edge);

    SetBkMode(mem, TRANSPARENT);

    const HFONT title_font =
        CreateFontW(34, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                    OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                    VARIABLE_PITCH, L"Segoe UI");
    const HFONT body_font =
        CreateFontW(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                    OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                    VARIABLE_PITCH, L"Segoe UI");

    RECT r = rc;
    r.top = 62;
    SelectObject(mem, title_font);
    SetTextColor(mem, kTitleFg);
    DrawTextW(mem, title_.c_str(), -1, &r, DT_CENTER | DT_SINGLELINE | DT_TOP);

    r.top = 108;
    SelectObject(mem, body_font);
    SetTextColor(mem, kSubtleFg);
    DrawTextW(mem, subtitle_.c_str(), -1, &r, DT_CENTER | DT_SINGLELINE | DT_TOP);

    // THE PULSE. An indeterminate bar rather than a percentage, because startup
    // has no measurable progress: the stages are a device open, a header parse,
    // a multi-gigabyte upload and a prefill, and any percentage assigned to them
    // would be invented. A moving bar says "working" without claiming to know
    // how much is left.
    const int bar_y = 168;
    const int bar_h = 3;
    const int bar_w = kWidth - 96;
    const int bar_x = 48;

    RECT track{bar_x, bar_y, bar_x + bar_w, bar_y + bar_h};
    const HBRUSH track_brush = CreateSolidBrush(RGB(42, 46, 56));
    FillRect(mem, &track, track_brush);
    DeleteObject(track_brush);

    const int span = bar_w / 3;
    const int phase = static_cast<int>((GetTickCount64() / 6) % (bar_w + span));
    const int lead = bar_x + phase;
    RECT fill{std::max(bar_x, lead - span), bar_y, std::min(bar_x + bar_w, lead),
              bar_y + bar_h};
    if (fill.right > fill.left) {
        const HBRUSH accent = CreateSolidBrush(kAccent);
        FillRect(mem, &fill, accent);
        DeleteObject(accent);
    }

    std::wstring status;
    {
        std::lock_guard<std::mutex> lk(mu_);
        status = status_;
    }
    r.top = 194;
    SetTextColor(mem, kSubtleFg);
    DrawTextW(mem, status.c_str(), -1, &r, DT_CENTER | DT_SINGLELINE | DT_TOP);

    BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);

    DeleteObject(title_font);
    DeleteObject(body_font);
    SelectObject(mem, old_bmp);
    DeleteObject(bmp);
    DeleteDC(mem);
    EndPaint(hwnd, &ps);
}

void SplashScreen::dismiss() {
    if (!running_.load(std::memory_order_acquire)) {
        // show() may never have been called, or the window failed to create. The
        // thread still has to be joined if it exists.
        if (thread_.joinable()) thread_.join();
        return;
    }
    // The fade runs ON the splash thread (it owns the window); this only asks.
    fading_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}

LRESULT CALLBACK SplashScreen::WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<SplashScreen*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
        case WM_PAINT:
            if (self != nullptr) {
                self->paint(hwnd);
                return 0;
            }
            break;
        case WM_ERASEBKGND:
            return 1;   // fully painted in WM_PAINT
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_NCHITTEST:
            // Click-through everywhere. The splash covers the middle of the
            // screen for several seconds and must not swallow a click meant for
            // whatever is behind it.
            return HTTRANSPARENT;
        default:
            break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace rt
