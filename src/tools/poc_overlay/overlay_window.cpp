#define NOMINMAX
#include "overlay_window.h"

#include <windowsx.h>  // GET_X_LPARAM / GET_Y_LPARAM

#include <algorithm>
#include <climits>  // LONG_MAX / LONG_MIN (union-rect seed)
#include <cmath>
#include <memory>
#include <vector>

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

// Thermograd ramp: map a token probability in [0, 1] to a semi-transparent
// background color. Confident tokens (cold) fade toward transparent faint blue;
// uncertain tokens (hot) run yellow -> red with rising opacity so the eye is
// drawn to exactly where the model was unsure.
//   p >= 0.90 : faint blue, barely there (cold)
//   0.50..0.90: blue -> yellow
//   p <  0.50 : yellow -> red, hotter (more opaque) as p -> 0
D2D1_COLOR_F HeatColor(float p) {
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    if (p >= 0.9f) {
        // Cold: a whisper of blue, mostly transparent.
        return D2D1::ColorF(0.30f, 0.55f, 1.0f, 0.10f);
    }
    if (p >= 0.5f) {
        // Blue -> yellow across [0.5, 0.9]; t=1 at the cold end.
        const float t = (p - 0.5f) / 0.4f;            // 0 (warm) .. 1 (cool)
        const float r = 0.95f * (1.0f - t) + 0.30f * t;
        const float g = 0.85f * (1.0f - t) + 0.55f * t;
        const float b = 0.20f * (1.0f - t) + 1.00f * t;
        return D2D1::ColorF(r, g, b, 0.28f);
    }
    // Hot: yellow -> red across [0, 0.5]; h=1 at the hottest (p=0).
    const float h = (0.5f - p) / 0.5f;                // 0 (yellow) .. 1 (red)
    const float r = 0.95f;
    const float g = 0.85f * (1.0f - h) + 0.20f * h;
    const float b = 0.20f * (1.0f - h);
    const float a = 0.30f + 0.35f * h;                // hotter = more opaque
    return D2D1::ColorF(r, g, b, a);
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
    debugRT_.Reset();
    if (dib_) {
        DeleteObject(dib_);
    }
    if (memDC_) {
        DeleteDC(memDC_);
    }
    if (debugDib_) {
        DeleteObject(debugDib_);
    }
    if (debugDC_) {
        DeleteDC(debugDC_);
    }
    if (debugHwnd_) {
        DestroyWindow(debugHwnd_);
    }
    if (hwnd_) {
        DestroyWindow(hwnd_);
    }
}

bool OverlayWindow::Create(HINSTANCE hInstance) {
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &OverlayWindow::WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"TypeTranslateOverlayWindow";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    // WS_EX_NOACTIVATE: never steals focus from whatever the user is typing into.
    // WS_EX_LAYERED: required for UpdateLayeredWindow per-pixel alpha. WS_EX_TOPMOST:
    // stays above the target. WS_EX_TOOLWINDOW: keeps it out of the taskbar and
    // Alt+Tab. NOTE: WS_EX_TRANSPARENT is deliberately NOT set -- the header bar /
    // override dropdown must receive clicks. Click-through is instead implemented
    // per-pixel in WM_NCHITTEST (HTTRANSPARENT everywhere but the interactive
    // header/menu rects), so the body still never intercepts the user's clicks.
    hwnd_ = CreateWindowExW(WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
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

    // Header bar + dropdown rows: small, left-aligned, vertically centered within
    // their strip; never wraps (a language label is short and single-line).
    if (FAILED(writeFactory_->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
                                                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                                13.0f, L"en-us", headerFormat_.GetAddressOf()))) {
        return false;
    }
    headerFormat_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    headerFormat_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    headerFormat_->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);

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
    dropdownOpen_ = false;
    headerRectValid_ = false;
    menuRows_.clear();
    PublishInteractiveRegion();
    ShowWindow(hwnd_, SW_HIDE);
}

bool OverlayWindow::HeaderVisible(const OverlaySnapshot& state) const {
    // The header/dropdown belongs only to the caret-anchored typing pipeline, and
    // only when there is a known language and a configured label set to show.
    const bool typingPhase = state.phase == OverlayPhase::Typing ||
                             state.phase == OverlayPhase::Translating ||
                             state.phase == OverlayPhase::Ready;
    return typingPhase && state.language >= 0 &&
           state.language < static_cast<int>(labels_.size());
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
            Hide();
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
        Hide();
        return;
    }

    // Remember the last drawn content so toggling the dropdown can re-render
    // without waiting for a fresh snapshot.
    lastState_ = state;

    const bool showHeader = HeaderVisible(state);
    if (!showHeader) {
        dropdownOpen_ = false;  // no header -> the menu can't be open
    }

    // Measure the body text to size the pill to its content (multiple lines OK).
    const float maxTextW = static_cast<float>(kMaxWidth - 2 * kPadX);
    const float maxTextH = static_cast<float>(kMaxHeight - kHeaderH - 2 * kPadY);
    ComPtr<IDWriteTextLayout> layout;
    if (FAILED(writeFactory_->CreateTextLayout(text.c_str(), static_cast<UINT32>(text.length()),
                                                format, maxTextW, maxTextH, layout.GetAddressOf()))) {
        return;
    }
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

    // --- header / dropdown geometry ------------------------------------------
    // The header spans the box width; the drop-up menu (Auto + one row per pair)
    // stacks ABOVE the header, so the body stays glued to the caret and the menu
    // grows upward into free screen space -- never over the caret or the text.
    const int headerH = showHeader ? kHeaderH : 0;
    const int menuCount = (showHeader && dropdownOpen_)
                              ? static_cast<int>(labels_.size()) + 1  // +1 for the Auto row
                              : 0;
    const int menuH = menuCount * kMenuItemH;

    // Width: at least the body, widened to fit the header label + any menu row.
    auto labelWidth = [&](const std::wstring& s) -> float {
        ComPtr<IDWriteTextLayout> l;
        if (FAILED(writeFactory_->CreateTextLayout(s.c_str(), static_cast<UINT32>(s.length()),
                                                    headerFormat_.Get(), maxTextW,
                                                    static_cast<float>(kHeaderH), l.GetAddressOf()))) {
            return 0.0f;
        }
        DWRITE_TEXT_METRICS m{};
        l->GetMetrics(&m);
        return m.widthIncludingTrailingWhitespace;
    };

    float chromeW = 0.0f;
    if (showHeader) {
        // Measure with the longer "[Pinned]" tag so the box never has to reflow
        // when the direction is pinned vs. auto (chevron + tag + pair label).
        chromeW = labelWidth(L"\x25BE  [Pinned]  " + labels_[state.language]);
        if (dropdownOpen_) {
            chromeW = std::max(chromeW, labelWidth(L"Auto (OS layout)"));
            for (const std::wstring& l : labels_) {
                chromeW = std::max(chromeW, labelWidth(l));
            }
        }
    }

    int boxW = std::max(static_cast<int>(textW) + 2 * kPadX,
                        static_cast<int>(std::ceil(chromeW)) + 2 * kPadX);
    boxW = std::clamp(boxW, 1, kMaxWidth);
    // Right-align the body within the (possibly header-widened) box so its last
    // glyph still lands under the caret.
    layout->SetMaxWidth(static_cast<float>(boxW - 2 * kPadX));
    layout->SetMaxHeight(textH);

    const int bodyBoxH = static_cast<int>(textH) + 2 * kPadY;
    int boxH = std::clamp(menuH + headerH + bodyBoxH, 1, kMaxHeight);
    const int bodyTop = menuH + headerH;  // body sits below the menu + header

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
    ComPtr<ID2D1SolidColorBrush> dimBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.72f, 0.74f, 0.78f, 0.85f),
                                          dimBrush.GetAddressOf());
    ComPtr<ID2D1SolidColorBrush> headerBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.62f, 0.72f, 0.90f, 0.95f),
                                          headerBrush.GetAddressOf());
    ComPtr<ID2D1SolidColorBrush> hiliteBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.30f, 0.45f, 0.70f, 0.55f),
                                          hiliteBrush.GetAddressOf());
    ComPtr<ID2D1SolidColorBrush> sepBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 1.0f, 0.10f),
                                          sepBrush.GetAddressOf());
    // Amber header tint for a PINNED (manually overridden) direction, so it reads
    // differently at a glance from the blue "[Auto]" (OS-layout) header.
    ComPtr<ID2D1SolidColorBrush> pinnedBrush;
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.96f, 0.78f, 0.38f, 0.98f),
                                          pinnedBrush.GetAddressOf());
    if (dimLen > 0 && dimBrush) {
        layout->SetDrawingEffect(dimBrush.Get(),
                                 DWRITE_TEXT_RANGE{0, static_cast<UINT32>(dimLen)});
    }

    const D2D1_ROUNDED_RECT bg = D2D1::RoundedRect(
        D2D1::RectF(0.0f, 0.0f, static_cast<float>(boxW), static_cast<float>(boxH)), 8.0f, 8.0f);
    if (bgBrush) {
        renderTarget_->FillRoundedRectangle(bg, bgBrush.Get());
    }

    // Draw a single strip of header/menu text via a layout (DrawTextLayout avoids
    // the windows.h DrawText macro ambiguity and gives us vertical centering for
    // free through headerFormat_'s PARAGRAPH_ALIGNMENT_CENTER).
    const auto drawStrip = [&](const std::wstring& s, float top, float height,
                               ID2D1SolidColorBrush* brush) {
        if (!brush) return;
        ComPtr<IDWriteTextLayout> l;
        if (FAILED(writeFactory_->CreateTextLayout(
                s.c_str(), static_cast<UINT32>(s.length()), headerFormat_.Get(),
                static_cast<float>(boxW - 2 * kPadX), height, l.GetAddressOf()))) {
            return;
        }
        renderTarget_->DrawTextLayout(D2D1::Point2F(static_cast<float>(kPadX), top), l.Get(), brush);
    };

    // Drop-up menu rows (top of the box), then the header strip, then the body.
    if (menuCount > 0 && headerBrush && hiliteBrush) {
        for (int i = 0; i < menuCount; ++i) {
            const int rowIndex = (i == 0) ? -1 : (i - 1);  // row 0 = Auto (-1)
            const float top = static_cast<float>(i * kMenuItemH);
            if (rowIndex == state.language) {  // mark the active pair
                renderTarget_->FillRectangle(
                    D2D1::RectF(1.0f, top, static_cast<float>(boxW) - 1.0f, top + kMenuItemH),
                    hiliteBrush.Get());
            }
            const std::wstring rowText =
                (i == 0) ? L"Auto (OS layout)" : labels_[static_cast<size_t>(rowIndex)];
            drawStrip(rowText, top, static_cast<float>(kMenuItemH), headerBrush.Get());
        }
    }
    if (showHeader && headerBrush && sepBrush) {
        const float hTop = static_cast<float>(menuH);
        renderTarget_->FillRectangle(
            D2D1::RectF(static_cast<float>(kPadX), hTop + headerH - 1.0f,
                        static_cast<float>(boxW - kPadX), hTop + headerH),
            sepBrush.Get());  // hairline under the header
        // "<chevron> [Auto|Pinned] <pair>" -- the tag tells the user WHY this
        // direction is active (OS keyboard layout vs. a manual override), and the
        // pinned state also gets the distinct amber tint.
        const std::wstring headerText = (dropdownOpen_ ? L"\x25B4  " : L"\x25BE  ") +
                                        std::wstring(state.languagePinned ? L"[Pinned]  "
                                                                          : L"[Auto]  ") +
                                        labels_[state.language];
        ID2D1SolidColorBrush* headerTextBrush =
            (state.languagePinned && pinnedBrush) ? pinnedBrush.Get() : headerBrush.Get();
        drawStrip(headerText, hTop, static_cast<float>(headerH), headerTextBrush);
    }
    const D2D1_POINT_2F bodyOrigin =
        D2D1::Point2F(static_cast<float>(kPadX), static_cast<float>(bodyTop + kPadY));
    // Developer-Mode thermograd: colored rectangles behind the body BEFORE the
    // glyphs, so the text sits on top of its own per-token confidence heatmap.
    if (state.developerMode && !state.translationTokens.empty()) {
        DrawHeatmap(layout.Get(), state.translationTokens, bodyOrigin);
    }
    if (fgBrush) {
        renderTarget_->DrawTextLayout(bodyOrigin, layout.Get(), fgBrush.Get());
    }
    renderTarget_->EndDraw();

    // Position the box per anchor mode.
    POINT dstPos{};
    switch (anchorMode) {
        case AnchorMode::CaretPill:
            // Anchor the body's END (bottom-right, inset by the padding) to the
            // caret; the header/menu extend upward from there.
            dstPos = {state.anchor.x - boxW + kPadX, state.anchor.y - kGapAboveCaret - boxH + kPadY};
            break;
        case AnchorMode::CursorPopup:
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

    // Recompute the interactive SCREEN rects (header + open menu rows) so
    // WM_NCHITTEST / WM_LBUTTONDOWN and the global mouse hook agree on where the
    // clickable chrome is.
    headerRectValid_ = showHeader;
    menuRows_.clear();
    if (showHeader) {
        headerRect_ = {dstPos.x, dstPos.y + menuH, dstPos.x + boxW, dstPos.y + menuH + headerH};
        for (int i = 0; i < menuCount; ++i) {
            const int rowIndex = (i == 0) ? -1 : (i - 1);
            menuRows_.push_back({{dstPos.x, dstPos.y + i * kMenuItemH, dstPos.x + boxW,
                                  dstPos.y + (i + 1) * kMenuItemH},
                                 rowIndex});
        }
    }
    PublishInteractiveRegion();

    // CenterHud with fade: hold at full opacity, then dissolve away.
    if (fade) {
        fadeActive_ = true;
        fadeStart_ = GetTickCount64();
        SetTimer(hwnd_, kFadeTimerId, kFadeTimerMs, nullptr);
    }
}

OverlayWindow::Hit OverlayWindow::HitTest(POINT screenPt, int& outMenuIndex) const {
    outMenuIndex = 0;
    if (!headerRectValid_) {
        return Hit::None;
    }
    if (dropdownOpen_) {
        for (const MenuRow& row : menuRows_) {
            if (PtInRect(&row.rect, screenPt)) {
                outMenuIndex = row.index;
                return Hit::MenuItem;
            }
        }
    }
    if (PtInRect(&headerRect_, screenPt)) {
        return Hit::Header;
    }
    return Hit::None;
}

void OverlayWindow::OnLeftButtonDown(POINT screenPt) {
    int menuIndex = 0;
    switch (HitTest(screenPt, menuIndex)) {
        case Hit::Header:
            dropdownOpen_ = !dropdownOpen_;
            Repaint(lastState_);  // re-render with the menu open/closed
            break;
        case Hit::MenuItem:
            dropdownOpen_ = false;
            if (overrideSink_) {
                overrideSink_(menuIndex);  // -1 = Auto, >=0 = pin that pair
            }
            Repaint(lastState_);  // close the menu immediately (the header updates
                                  // to the new pair on the next snapshot)
            break;
        case Hit::None:
            break;
    }
}

void OverlayWindow::PublishInteractiveRegion() {
    if (!interactiveSink_) {
        return;
    }
    if (!headerRectValid_) {
        interactiveSink_(nullptr);  // nothing clickable right now
        return;
    }
    RECT region = headerRect_;
    for (const MenuRow& row : menuRows_) {  // union in the open menu rows
        region.left = std::min(region.left, row.rect.left);
        region.top = std::min(region.top, row.rect.top);
        region.right = std::max(region.right, row.rect.right);
        region.bottom = std::max(region.bottom, row.rect.bottom);
    }
    interactiveSink_(&region);
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

// ---------------------------------------------------------------------------
// Developer Mode: probability heatmap + UIA/caret debug overlay
// ---------------------------------------------------------------------------
void OverlayWindow::DrawHeatmap(IDWriteTextLayout* layout, const TokenHeatmap& tokens,
                                D2D1_POINT_2F origin) {
    if (!layout || tokens.empty()) {
        return;
    }
    ComPtr<ID2D1SolidColorBrush> cell;
    if (FAILED(renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 0),
                                                    cell.GetAddressOf()))) {
        return;
    }
    // Walk the tokens in text order, mapping each to its [offset, length) range
    // in the layout (which holds exactly the concatenation of tokens[].text) and
    // filling that range's bounding rect(s) with the token's heat color.
    UINT32 offset = 0;
    for (const TokenInfo& tok : tokens) {
        const UINT32 len = static_cast<UINT32>(tok.text.length());
        if (len == 0) {
            continue;
        }
        // A single token rarely spans more than one line box; size for a few and
        // grow only if HitTestTextRange asks for more.
        DWRITE_HIT_TEST_METRICS hit[4];
        UINT32 actual = 0;
        HRESULT hr = layout->HitTestTextRange(offset, len, origin.x, origin.y, hit, 4, &actual);
        std::vector<DWRITE_HIT_TEST_METRICS> dynamic;
        if (hr == E_NOT_SUFFICIENT_BUFFER && actual > 0) {
            dynamic.resize(actual);
            hr = layout->HitTestTextRange(offset, len, origin.x, origin.y, dynamic.data(), actual,
                                          &actual);
        }
        const DWRITE_HIT_TEST_METRICS* metrics = dynamic.empty() ? hit : dynamic.data();
        if (SUCCEEDED(hr)) {
            cell->SetColor(HeatColor(tok.probability));
            for (UINT32 i = 0; i < actual; ++i) {
                const auto& m = metrics[i];
                renderTarget_->FillRectangle(
                    D2D1::RectF(m.left, m.top, m.left + m.width, m.top + m.height), cell.Get());
            }
        }
        offset += len;
    }
}

bool OverlayWindow::EnsureDebugWindow() {
    if (debugHwnd_ && debugRT_ && debugDib_) {
        return true;
    }
    if (!debugHwnd_) {
        static bool registered = false;
        const wchar_t* kDebugClass = L"TypeTranslateOverlayDebugWindow";
        if (!registered) {
            WNDCLASSEXW wc{sizeof(wc)};
            wc.lpfnWndProc = &DefWindowProcW;  // pure output surface; never handles input
            wc.hInstance = reinterpret_cast<HINSTANCE>(GetWindowLongPtrW(hwnd_, GWLP_HINSTANCE));
            wc.lpszClassName = kDebugClass;
            RegisterClassExW(&wc);
            registered = true;
        }
        // WS_EX_TRANSPARENT: fully click-through (unlike the pill, this window has
        // no interactive chrome, so it never needs WM_NCHITTEST). Same layered /
        // no-activate / topmost / tool-window traits otherwise.
        debugHwnd_ = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            kDebugClass, L"", WS_POPUP, 0, 0, 16, 16, nullptr, nullptr,
            reinterpret_cast<HINSTANCE>(GetWindowLongPtrW(hwnd_, GWLP_HINSTANCE)), nullptr);
        if (!debugHwnd_) {
            return false;
        }
    }
    // One DIB sized to the whole virtual screen, so any on-screen box fits from
    // its top-left corner (we blit only the used sub-rect each frame).
    if (!debugDib_) {
        debugDibW_ = std::max(GetSystemMetrics(SM_CXVIRTUALSCREEN), 1);
        debugDibH_ = std::max(GetSystemMetrics(SM_CYVIRTUALSCREEN), 1);
        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = debugDibW_;
        bmi.bmiHeader.biHeight = -debugDibH_;  // top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        debugDC_ = CreateCompatibleDC(nullptr);
        if (!debugDC_) {
            return false;
        }
        void* bits = nullptr;
        debugDib_ = CreateDIBSection(debugDC_, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (!debugDib_) {
            return false;
        }
        SelectObject(debugDC_, debugDib_);
    }
    if (!debugRT_) {
        const D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(
            D2D1_RENDER_TARGET_TYPE_DEFAULT,
            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED));
        if (FAILED(d2dFactory_->CreateDCRenderTarget(&props, debugRT_.GetAddressOf()))) {
            return false;
        }
    }
    return true;
}

void OverlayWindow::HideDebug() {
    if (debugVisible_ && debugHwnd_) {
        ShowWindow(debugHwnd_, SW_HIDE);
        debugVisible_ = false;
    }
}

void OverlayWindow::RenderDebug(const OverlaySnapshot& state) {
    // Only paint when Developer Mode is on and there is at least one thing to
    // draw. Any other state hides the debug surface so stale boxes never linger.
    if (!state.developerMode || (!state.uiaBoundsValid && !state.caretRectValid)) {
        HideDebug();
        return;
    }
    if (!EnsureDebugWindow()) {
        return;
    }

    // Union of the two rects (screen pixels), clamped to the virtual-screen DIB.
    const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
    RECT u{LONG_MAX, LONG_MAX, LONG_MIN, LONG_MIN};
    auto expand = [&u](const RECT& r) {
        u.left = std::min(u.left, r.left);
        u.top = std::min(u.top, r.top);
        u.right = std::max(u.right, r.right);
        u.bottom = std::max(u.bottom, r.bottom);
    };
    if (state.uiaBoundsValid) expand(state.uiaBounds);
    if (state.caretRectValid) expand(state.caretRect);
    // Pad by the stroke width so a border drawn on the union edge is not clipped.
    constexpr int kStroke = 2;
    constexpr int kPad = kStroke + 1;
    u.left -= kPad;
    u.top -= kPad;
    u.right += kPad;
    u.bottom += kPad;

    int boxW = static_cast<int>(u.right - u.left);
    int boxH = static_cast<int>(u.bottom - u.top);
    boxW = std::clamp(boxW, 1, debugDibW_);
    boxH = std::clamp(boxH, 1, debugDibH_);

    const RECT bind{0, 0, boxW, boxH};
    if (FAILED(debugRT_->BindDC(debugDC_, &bind))) {
        return;
    }
    const float ox = static_cast<float>(u.left);  // union origin -> DIB local
    const float oy = static_cast<float>(u.top);

    debugRT_->BeginDraw();
    debugRT_->Clear(D2D1::ColorF(0, 0, 0, 0));  // transparent

    // (a) The UIA element bounding box: 2px semi-transparent RED border.
    if (state.uiaBoundsValid) {
        ComPtr<ID2D1SolidColorBrush> red;
        debugRT_->CreateSolidColorBrush(D2D1::ColorF(1.0f, 0.15f, 0.15f, 0.75f),
                                        red.GetAddressOf());
        if (red) {
            const RECT& b = state.uiaBounds;
            // Inset by half the stroke so the 2px line sits ON the boundary.
            const D2D1_RECT_F r = D2D1::RectF(b.left - ox + kStroke * 0.5f,
                                              b.top - oy + kStroke * 0.5f,
                                              b.right - ox - kStroke * 0.5f,
                                              b.bottom - oy - kStroke * 0.5f);
            debugRT_->DrawRectangle(r, red.Get(), static_cast<float>(kStroke));
        }
    }
    // (b) The exact caret: 2px solid GREEN line at the caret's leading edge,
    // spanning its height (a bare caret has zero width -> a vertical bar).
    if (state.caretRectValid) {
        ComPtr<ID2D1SolidColorBrush> green;
        debugRT_->CreateSolidColorBrush(D2D1::ColorF(0.10f, 0.90f, 0.20f, 0.95f),
                                        green.GetAddressOf());
        if (green) {
            const RECT& c = state.caretRect;
            const float x = c.left - ox;
            const float top = c.top - oy;
            const float bottom = c.bottom - oy;
            const float width = (c.right > c.left) ? static_cast<float>(c.right - c.left)
                                                   : static_cast<float>(kStroke);
            debugRT_->FillRectangle(D2D1::RectF(x, top, x + width, bottom), green.Get());
        }
    }
    if (FAILED(debugRT_->EndDraw())) {
        return;
    }

    POINT dst{static_cast<int>(u.left), static_cast<int>(u.top)};
    // Guard against a union that started off the virtual-screen origin.
    dst.x = std::max(dst.x, static_cast<LONG>(vx));
    dst.y = std::max(dst.y, static_cast<LONG>(vy));
    POINT src{0, 0};
    SIZE size{boxW, boxH};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    HDC screenDC = GetDC(nullptr);
    UpdateLayeredWindow(debugHwnd_, screenDC, &dst, &size, debugDC_, &src, 0, &blend, ULW_ALPHA);
    ReleaseDC(nullptr, screenDC);
    if (!debugVisible_) {
        ShowWindow(debugHwnd_, SW_SHOWNOACTIVATE);
        debugVisible_ = true;
    }
}

LRESULT CALLBACK OverlayWindow::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == kMsgState) {
        std::unique_ptr<OverlaySnapshot> payload(reinterpret_cast<OverlaySnapshot*>(lParam));
        auto* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self) {
            // The debug overlay tracks the focused field independently of the
            // caret pill (which may hide), so paint it from the snapshot too.
            self->RenderDebug(*payload);
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
    // Per-pixel click-through: only the header bar and (when open) the dropdown
    // rows are hittable; the body and everything else fall through to the app
    // beneath, so the overlay never intercepts the user's real clicks.
    if (msg == WM_NCHITTEST) {
        auto* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self) {
            POINT pt{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};  // NCHITTEST is in screen coords
            int ignored = 0;
            if (self->HitTest(pt, ignored) != Hit::None) {
                return HTCLIENT;
            }
        }
        return HTTRANSPARENT;
    }
    if (msg == WM_LBUTTONDOWN) {
        auto* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self) {
            POINT pt{};
            GetCursorPos(&pt);  // reliable screen coords for the just-happened click
            self->OnLeftButtonDown(pt);
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}
