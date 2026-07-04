#include "overlay_window.h"

#include <algorithm>
#include <cmath>
#include <memory>

using Microsoft::WRL::ComPtr;

namespace {
const std::wstring kSpinner(1, L'\x2026');                       // "..." loading glyph
const D2D1_COLOR_F kAccent = D2D1::ColorF(0.55f, 0.75f, 1.0f);   // in-progress color

// Work area of the monitor currently under the mouse cursor (for centered HUDs).
RECT WorkAreaUnderCursor() {
    POINT c{};
    GetCursorPos(&c);
    HMONITOR mon = MonitorFromPoint(c, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi{sizeof(mi)};
    if (GetMonitorInfoW(mon, &mi)) {
        return mi.rcWork;
    }
    return RECT{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
}

// Keep a box fully on the monitor nearest `anchor`.
void ClampToMonitor(POINT& dst, int boxW, int boxH, POINT anchor) {
    HMONITOR mon = MonitorFromPoint(anchor, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi{sizeof(mi)};
    if (!GetMonitorInfoW(mon, &mi)) {
        return;
    }
    if (dst.x + boxW > mi.rcWork.right) dst.x = mi.rcWork.right - boxW;
    if (dst.y + boxH > mi.rcWork.bottom) dst.y = mi.rcWork.bottom - boxH;
    if (dst.x < mi.rcWork.left) dst.x = mi.rcWork.left;
    if (dst.y < mi.rcWork.top) dst.y = mi.rcWork.top;
}
}  // namespace

OverlayWindow::~OverlayWindow() {
    renderTarget_.Reset();
    if (dib_) {
        DeleteObject(dib_);
    }
    if (memDC_) {
        DeleteDC(memDC_);
    }
    if (hwnd_) {
        DestroyWindow(hwnd_);
    }
}

bool OverlayWindow::Create(HINSTANCE hInstance) {
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &OverlayWindow::WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"BlackwellPocOverlayWindow";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    // WS_EX_TRANSPARENT + WS_EX_NOACTIVATE: click-through and never steals focus
    // from whatever the user is typing into. WS_EX_LAYERED: required for
    // UpdateLayeredWindow per-pixel alpha. WS_EX_TOPMOST: stays above the target.
    // WS_EX_TOOLWINDOW: keeps it out of the taskbar and Alt+Tab.
    hwnd_ = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST |
                                 WS_EX_TOOLWINDOW,
                             wc.lpszClassName, L"", WS_POPUP, 0, 0, kMaxWidth, kMaxHeight, nullptr,
                             nullptr, hInstance, this);
    if (!hwnd_) {
        return false;
    }
    SetWindowLongPtrW(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

    return InitDirect2D() && InitBackingBitmap();
}

bool OverlayWindow::InitDirect2D() {
    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2dFactory_.GetAddressOf()))) {
        return false;
    }
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                                    reinterpret_cast<IUnknown**>(writeFactory_.GetAddressOf())))) {
        return false;
    }
    if (FAILED(writeFactory_->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
                                                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                                20.0f, L"en-us", textFormat_.GetAddressOf()))) {
        return false;
    }
    // Right-aligned + bottom-aligned so the string's end anchors at the caret and
    // multi-line strings grow upward. Wrapping keeps an over-long line inside the
    // max box (explicit '\n' from the buffer still forces hard line breaks).
    textFormat_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_TRAILING);
    textFormat_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_FAR);
    textFormat_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);

    // CenterHud banner: larger, centered on both axes.
    if (FAILED(writeFactory_->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
                                                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                                30.0f, L"en-us", hudFormat_.GetAddressOf()))) {
        return false;
    }
    hudFormat_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER);
    hudFormat_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    hudFormat_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);

    // Selection popup: left/top-aligned, sits below-right of the cursor.
    if (FAILED(writeFactory_->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
                                                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                                20.0f, L"en-us", popupFormat_.GetAddressOf()))) {
        return false;
    }
    popupFormat_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    popupFormat_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
    popupFormat_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);

    const D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED));
    return SUCCEEDED(d2dFactory_->CreateDCRenderTarget(&props, renderTarget_.GetAddressOf()));
}

bool OverlayWindow::InitBackingBitmap() {
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = kMaxWidth;
    bmi.bmiHeader.biHeight = -kMaxHeight;  // negative = top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    memDC_ = CreateCompatibleDC(nullptr);
    if (!memDC_) {
        return false;
    }

    void* bits = nullptr;
    dib_ = CreateDIBSection(memDC_, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!dib_) {
        return false;
    }
    SelectObject(memDC_, dib_);
    return true;
}

void OverlayWindow::PostState(const OverlaySnapshot& state) {
    auto* payload = new OverlaySnapshot(state);
    if (!PostMessageW(hwnd_, kMsgState, 0, reinterpret_cast<LPARAM>(payload))) {
        delete payload;  // window gone / queue full -- drop the update
    }
}

void OverlayWindow::Hide() {
    ShowWindow(hwnd_, SW_HIDE);
}

void OverlayWindow::Repaint(const OverlaySnapshot& state) {
    // A fresh state supersedes any in-progress fade.
    KillTimer(hwnd_, kFadeTimerId);
    fadeActive_ = false;

    // Compose the display string, the dim split point, the color, the text
    // format, and how the box anchors -- all phase-driven.
    std::wstring text;
    size_t dimLen = 0;  // leading chars rendered dimmed + italic (captured source)
    D2D1_COLOR_F fgColor = D2D1::ColorF(D2D1::ColorF::White);
    IDWriteTextFormat* format = textFormat_.Get();
    AnchorMode anchorMode = AnchorMode::CaretPill;
    bool fade = false;

    switch (state.phase) {
        case OverlayPhase::Hidden:
            ShowWindow(hwnd_, SW_HIDE);
            return;
        case OverlayPhase::Typing:
            text = state.source;
            dimLen = text.size();  // everything dimmed: "this is the capture"
            break;
        case OverlayPhase::Translating:
            // Dimmed source, then the streamed partial (or an ellipsis spinner
            // while the first tokens are in flight) in the accent color.
            text = state.source + L"\n" + (state.translation.empty() ? kSpinner : state.translation);
            dimLen = state.source.size();
            fgColor = kAccent;
            break;
        case OverlayPhase::Ready:
            text = state.translation;  // full-brightness: Ctrl+Enter commits this
            break;
        case OverlayPhase::CenterHud:
            text = state.message;
            format = hudFormat_.Get();
            anchorMode = AnchorMode::ScreenCenter;
            fade = state.fade;
            break;
        case OverlayPhase::SelectionTranslating:
            // No source (the OS already highlights it) -- just the spinner/partial.
            text = state.translation.empty() ? kSpinner : state.translation;
            fgColor = kAccent;
            format = popupFormat_.Get();
            anchorMode = AnchorMode::CursorPopup;
            break;
        case OverlayPhase::SelectionReady:
            text = state.translation;
            format = popupFormat_.Get();
            anchorMode = AnchorMode::CursorPopup;
            break;
    }

    // Nothing to show (empty text, or a caret/cursor-anchored pill with no
    // anchor) -- hide rather than flash an empty pill. Centered HUDs need no anchor.
    if (text.empty() || (anchorMode != AnchorMode::ScreenCenter && !state.anchorValid)) {
        ShowWindow(hwnd_, SW_HIDE);
        return;
    }

    // Measure the text to size the pill to its content (supports multiple lines).
    const float maxTextW = static_cast<float>(kMaxWidth - 2 * kPadX);
    const float maxTextH = static_cast<float>(kMaxHeight - 2 * kPadY);
    ComPtr<IDWriteTextLayout> layout;
    if (FAILED(writeFactory_->CreateTextLayout(text.c_str(), static_cast<UINT32>(text.length()),
                                                format, maxTextW, maxTextH, layout.GetAddressOf()))) {
        return;
    }
    // Captured-source range: italic (the brush split happens at draw time).
    if (dimLen > 0) {
        layout->SetFontStyle(DWRITE_FONT_STYLE_ITALIC,
                             DWRITE_TEXT_RANGE{0, static_cast<UINT32>(dimLen)});
    }
    DWRITE_TEXT_METRICS metrics{};
    layout->GetMetrics(&metrics);

    float textW = std::ceil(metrics.widthIncludingTrailingWhitespace);
    float textH = std::ceil(metrics.height);
    textW = std::clamp(textW, 1.0f, maxTextW);
    textH = std::clamp(textH, 1.0f, maxTextH);

    // Pin the layout box to the measured content so alignment hugs the pill's
    // inset (each line right-aligns under the caret; centered for the HUD).
    layout->SetMaxWidth(textW);
    layout->SetMaxHeight(textH);

    const int boxW = static_cast<int>(textW) + 2 * kPadX;
    const int boxH = static_cast<int>(textH) + 2 * kPadY;

    // Bind + paint only the top-left boxW x boxH sub-rect of the max-size DIB.
    const RECT bounds{0, 0, boxW, boxH};
    if (FAILED(renderTarget_->BindDC(memDC_, &bounds))) {
        return;
    }

    renderTarget_->BeginDraw();
    renderTarget_->Clear(D2D1::ColorF(0, 0, 0, 0));  // fully transparent background

    ComPtr<ID2D1SolidColorBrush> bgBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.05f, 0.05f, 0.05f, 0.82f),
                                          bgBrush.GetAddressOf());
    ComPtr<ID2D1SolidColorBrush> fgBrush;
    renderTarget_->CreateSolidColorBrush(fgColor, fgBrush.GetAddressOf());
    // Dimmed brush for the captured-source range (visually distinct from the
    // translation so the user always knows what is captured vs. produced).
    ComPtr<ID2D1SolidColorBrush> dimBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.72f, 0.74f, 0.78f, 0.85f),
                                          dimBrush.GetAddressOf());
    if (dimLen > 0 && dimBrush) {
        layout->SetDrawingEffect(dimBrush.Get(),
                                 DWRITE_TEXT_RANGE{0, static_cast<UINT32>(dimLen)});
    }

    const D2D1_ROUNDED_RECT bg = D2D1::RoundedRect(
        D2D1::RectF(0.0f, 0.0f, static_cast<float>(boxW), static_cast<float>(boxH)), 8.0f, 8.0f);
    if (bgBrush) {
        renderTarget_->FillRoundedRectangle(bg, bgBrush.Get());
    }
    if (fgBrush) {
        renderTarget_->DrawTextLayout(D2D1::Point2F(static_cast<float>(kPadX),
                                                     static_cast<float>(kPadY)),
                                       layout.Get(), fgBrush.Get());
    }
    renderTarget_->EndDraw();

    // Position the box per anchor mode.
    POINT dstPos{};
    switch (anchorMode) {
        case AnchorMode::CaretPill:
            // Anchor the string's END (bottom-right, inset by the padding) to the
            // caret: text right edge -> anchor.x, text bottom -> kGapAboveCaret px
            // above anchor.y. Floats up-and-to-the-left of the caret.
            dstPos = {state.anchor.x - boxW + kPadX, state.anchor.y - kGapAboveCaret - boxH + kPadY};
            break;
        case AnchorMode::CursorPopup:
            // Below-and-right of the cursor/selection end, clamped on-screen.
            dstPos = {state.anchor.x + kCursorOffsetX, state.anchor.y + kCursorOffsetY};
            ClampToMonitor(dstPos, boxW, boxH, state.anchor);
            break;
        case AnchorMode::ScreenCenter: {
            const RECT wa = WorkAreaUnderCursor();
            dstPos = {wa.left + ((wa.right - wa.left) - boxW) / 2,
                      wa.top + ((wa.bottom - wa.top) - boxH) / 3};  // upper third reads better
            break;
        }
    }

    Blit(dstPos, SIZE{boxW, boxH}, 255);
    lastDst_ = dstPos;
    lastSize_ = SIZE{boxW, boxH};
    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);

    // CenterHud with fade: hold at full opacity, then dissolve away.
    if (fade) {
        fadeActive_ = true;
        fadeStart_ = GetTickCount64();
        SetTimer(hwnd_, kFadeTimerId, kFadeTimerMs, nullptr);
    }
}

void OverlayWindow::Blit(POINT dst, SIZE size, BYTE alpha) {
    POINT srcPos{0, 0};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, alpha, AC_SRC_ALPHA};
    HDC screenDC = GetDC(nullptr);
    UpdateLayeredWindow(hwnd_, screenDC, &dst, &size, memDC_, &srcPos, 0, &blend, ULW_ALPHA);
    ReleaseDC(nullptr, screenDC);
}

void OverlayWindow::RunFadeStep() {
    if (!fadeActive_) {
        return;
    }
    const ULONGLONG elapsed = GetTickCount64() - fadeStart_;
    if (elapsed < kHudHoldMs) {
        return;  // still holding at full opacity
    }
    const ULONGLONG t = elapsed - kHudHoldMs;
    if (t >= kFadeDurationMs) {
        KillTimer(hwnd_, kFadeTimerId);
        fadeActive_ = false;
        ShowWindow(hwnd_, SW_HIDE);
        return;
    }
    const BYTE alpha = static_cast<BYTE>(255 - (255 * t) / kFadeDurationMs);
    Blit(lastDst_, lastSize_, alpha);
}

LRESULT CALLBACK OverlayWindow::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == kMsgState) {
        std::unique_ptr<OverlaySnapshot> payload(reinterpret_cast<OverlaySnapshot*>(lParam));
        auto* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self) {
            self->Repaint(*payload);
        }
        return 0;
    }
    if (msg == WM_TIMER && wParam == kFadeTimerId) {
        auto* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self) {
            self->RunFadeStep();
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}
