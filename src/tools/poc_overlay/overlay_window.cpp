#include "overlay_window.h"

#include <algorithm>
#include <cmath>
#include <memory>

using Microsoft::WRL::ComPtr;

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
    // Compose the display string + the style split point for this phase.
    // `dimLen` characters from the start render dimmed+italic (the captured
    // source); everything after renders in the phase's foreground color.
    std::wstring text;
    size_t dimLen = 0;
    D2D1_COLOR_F fgColor = D2D1::ColorF(D2D1::ColorF::White);
    switch (state.phase) {
        case OverlayPhase::Typing:
            text = state.source;
            dimLen = text.size();  // everything dimmed: "this is the capture"
            break;
        case OverlayPhase::Translating:
            // Dimmed source, then the streamed partial (or an ellipsis while
            // the first tokens are in flight) in the accent color.
            text = state.source + L"\n" +
                   (state.translation.empty() ? std::wstring(1, L'\x2026')  // "..." spinner
                                              : state.translation);
            dimLen = state.source.size();
            fgColor = D2D1::ColorF(0.55f, 0.75f, 1.0f);  // accent: in progress
            break;
        case OverlayPhase::Ready:
            text = state.translation;  // full-brightness: Ctrl+Enter commits this
            break;
        case OverlayPhase::Hidden:
            break;
    }

    // Nothing to show (hidden phase, cleared field, or no caret to anchor to)
    // -- hide rather than flash an empty pill.
    if (text.empty() || state.phase == OverlayPhase::Hidden || !state.anchorValid) {
        ShowWindow(hwnd_, SW_HIDE);
        return;
    }
    const POINT caretScreenPos = state.anchor;

    // Measure the text to size the pill to its content (supports multiple lines).
    const float maxTextW = static_cast<float>(kMaxWidth - 2 * kPadX);
    const float maxTextH = static_cast<float>(kMaxHeight - 2 * kPadY);
    ComPtr<IDWriteTextLayout> layout;
    if (FAILED(writeFactory_->CreateTextLayout(text.c_str(), static_cast<UINT32>(text.length()),
                                                textFormat_.Get(), maxTextW, maxTextH,
                                                layout.GetAddressOf()))) {
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

    // Pin the layout box to the measured content so TRAILING/FAR alignment hugs
    // the pill's right/bottom inset (each line right-aligns under the caret).
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
    renderTarget_->CreateSolidColorBrush(D2D1::ColorF(0.05f, 0.05f, 0.05f, 0.78f),
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

    // Anchor the string's END (bottom-right of the text, inset by the padding)
    // to the caret: text right edge -> caret.x, text bottom -> kGapAboveCaret px
    // above caret.y. The pill therefore floats up-and-to-the-left of the caret.
    POINT dstPos{caretScreenPos.x - boxW + kPadX,
                 caretScreenPos.y - kGapAboveCaret - boxH + kPadY};
    SIZE size{boxW, boxH};
    POINT srcPos{0, 0};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};

    HDC screenDC = GetDC(nullptr);
    UpdateLayeredWindow(hwnd_, screenDC, &dstPos, &size, memDC_, &srcPos, 0, &blend, ULW_ALPHA);
    ReleaseDC(nullptr, screenDC);

    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
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
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}
