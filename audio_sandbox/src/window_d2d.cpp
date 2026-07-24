// -----------------------------------------------------------------------------
// window_d2d.cpp — see window_d2d.h.
// -----------------------------------------------------------------------------
#include "window_d2d.h"

#include <algorithm>
#include <cmath>

#include "realtime_dsp.h"

namespace rt {
namespace {

constexpr wchar_t kClassName[] = L"WhisperD2DSpectrogram";
constexpr UINT kTimerId = 1;
constexpr UINT kTimerMs = 33;              // ~30 FPS
constexpr float kDynamicRangeDb = 8.0f;    // Whisper's (max - 8) display window

template <class T>
void safe_release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

// Inferno colormap via linear interpolation over 9 evenly-spaced anchors. t in
// [0, 1] -> (b, g, r) bytes (D2D bitmap is BGRA). Perceptually-uniform-ish and
// self-contained (no external LUT file).
void inferno(float t, uint8_t& b, uint8_t& g, uint8_t& r) {
    static const float A[9][3] = {  // R, G, B in [0, 1]
        {0.0010f, 0.0004f, 0.0140f}, {0.1220f, 0.0470f, 0.2830f},
        {0.3340f, 0.0590f, 0.4280f}, {0.5330f, 0.1330f, 0.4160f},
        {0.7290f, 0.2120f, 0.3330f}, {0.8900f, 0.3490f, 0.2010f},
        {0.9760f, 0.5490f, 0.0390f}, {0.9780f, 0.7980f, 0.1980f},
        {0.9880f, 0.9980f, 0.6440f},
    };
    t = std::clamp(t, 0.0f, 1.0f) * 8.0f;   // scale onto [0, 8] anchor span
    const int i = std::min(static_cast<int>(t), 7);
    const float f = t - static_cast<float>(i);
    const float rr = A[i][0] + f * (A[i + 1][0] - A[i][0]);
    const float gg = A[i][1] + f * (A[i + 1][1] - A[i][1]);
    const float bb = A[i][2] + f * (A[i + 1][2] - A[i][2]);
    r = static_cast<uint8_t>(std::lround(rr * 255.0f));
    g = static_cast<uint8_t>(std::lround(gg * 255.0f));
    b = static_cast<uint8_t>(std::lround(bb * 255.0f));
}

}  // namespace

WindowD2D::WindowD2D(SpectrogramBuffer& spec, const wchar_t* title)
    : spec_(spec), title_(title) {
    bmp_w_ = spec_.max_frames();
    bmp_h_ = spec_.n_mels();
    pixels_.assign(static_cast<size_t>(bmp_w_) * bmp_h_ * 4, 0);
}

WindowD2D::~WindowD2D() {
    discard_device_resources();
    safe_release(factory_);
    if (hwnd_) DestroyWindow(hwnd_);
}

bool WindowD2D::create(int client_w, int client_h) {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = &WindowD2D::WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = kClassName;
    RegisterClassExW(&wc);

    RECT r{0, 0, client_w, client_h};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    hwnd_ = CreateWindowExW(0, kClassName, title_.c_str(), WS_OVERLAPPEDWINDOW,
                            CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                            nullptr, nullptr, wc.hInstance, this);
    if (!hwnd_) return false;

    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &factory_))) return false;

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    SetTimer(hwnd_, kTimerId, kTimerMs, nullptr);
    return true;
}

void WindowD2D::run_message_loop() {
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

LRESULT CALLBACK WindowD2D::WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<WindowD2D*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    return self ? self->handle(hwnd, msg, wp, lp) : DefWindowProcW(hwnd, msg, wp, lp);
}

LRESULT WindowD2D::handle(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_TIMER:
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(hwnd, &ps);
            render();
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_SIZE:
            on_resize(LOWORD(lp), HIWORD(lp));
            return 0;
        case WM_DESTROY:
            KillTimer(hwnd, kTimerId);
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

HRESULT WindowD2D::ensure_device_resources() {
    if (target_) return S_OK;

    RECT rc;
    GetClientRect(hwnd_, &rc);
    const D2D1_SIZE_U size = D2D1::SizeU(static_cast<UINT32>(rc.right - rc.left),
                                         static_cast<UINT32>(rc.bottom - rc.top));
    HRESULT hr = factory_->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(),
        D2D1::HwndRenderTargetProperties(hwnd_, size), &target_);
    if (FAILED(hr)) return hr;

    // Bitmap is exactly [max_frames wide x n_mels tall]; DrawBitmap stretches it
    // to the client area, so window resizes cost nothing here.
    const D2D1_BITMAP_PROPERTIES bp = D2D1::BitmapProperties(
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
    hr = target_->CreateBitmap(D2D1::SizeU(static_cast<UINT32>(bmp_w_),
                                           static_cast<UINT32>(bmp_h_)),
                               nullptr, 0, &bp, &bitmap_);
    return hr;
}

void WindowD2D::discard_device_resources() {
    safe_release(bitmap_);
    safe_release(target_);
}

void WindowD2D::on_resize(UINT w, UINT h) {
    if (target_) target_->Resize(D2D1::SizeU(w, h));
}

void WindowD2D::render() {
    if (FAILED(ensure_device_resources())) return;

    spec_.snapshot(scratch_);
    const int n_cols = static_cast<int>(scratch_.size());

    // Rolling normalisation: map [global_max - 8 dB, global_max] -> [0, 1] over
    // the VISIBLE history. This reproduces Whisper's dynamic-range shape without
    // needing a whole-clip max (impossible in a stream); it is a VIZ-ONLY step
    // and does not touch the parity-preserving DSP.
    float vmax = -1e30f;
    for (const auto& col : scratch_)
        for (float v : col) vmax = std::max(vmax, v);
    const float vmin = vmax - kDynamicRangeDb;
    const float inv_range = (vmax > vmin) ? 1.0f / (vmax - vmin) : 0.0f;

    // Background (inferno's near-black floor).
    for (size_t i = 0; i < pixels_.size(); i += 4) {
        pixels_[i + 0] = 3; pixels_[i + 1] = 0; pixels_[i + 2] = 0; pixels_[i + 3] = 255;
    }

    // Newest column pinned to the right edge; older columns scroll left.
    const int start = std::max(0, bmp_w_ - n_cols);
    for (int c = 0; c < n_cols && (start + c) < bmp_w_; ++c) {
        const std::vector<float>& col = scratch_[static_cast<size_t>(c)];
        const int x = start + c;
        const int mels = std::min(bmp_h_, static_cast<int>(col.size()));
        for (int m = 0; m < mels; ++m) {
            const float norm = (col[static_cast<size_t>(m)] - vmin) * inv_range;
            uint8_t b, g, r;
            inferno(norm, b, g, r);
            const int py = (bmp_h_ - 1) - m;  // invert Y: mel 0 -> bottom row
            const size_t idx = (static_cast<size_t>(py) * bmp_w_ + x) * 4;
            pixels_[idx + 0] = b; pixels_[idx + 1] = g; pixels_[idx + 2] = r; pixels_[idx + 3] = 255;
        }
    }

    const D2D1_RECT_U rect = D2D1::RectU(0, 0, static_cast<UINT32>(bmp_w_),
                                         static_cast<UINT32>(bmp_h_));
    bitmap_->CopyFromMemory(&rect, pixels_.data(), static_cast<UINT32>(bmp_w_) * 4);

    target_->BeginDraw();
    target_->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f));
    if (n_cols > 0) {
        const D2D1_SIZE_F sz = target_->GetSize();
        // Source = ONLY the populated columns [start, bmp_w_); destination = a
        // right-aligned slab of the client area whose width is proportional to
        // the fill. Every mel column therefore keeps a constant on-screen width
        // (client_w / bmp_w_) and the 128-tall bitmap stretches to the full
        // client height. LINEAR interpolation smooths both the 128->height
        // vertical scale and the horizontal scale. (HIGH_QUALITY_CUBIC needs an
        // ID2D1DeviceContext; this is an ID2D1HwndRenderTarget, so LINEAR is the
        // best mode available here.)
        const D2D1_RECT_F src = D2D1::RectF(static_cast<float>(start), 0.0f,
                                            static_cast<float>(bmp_w_),
                                            static_cast<float>(bmp_h_));
        const float fill = static_cast<float>(bmp_w_ - start) / static_cast<float>(bmp_w_);
        const D2D1_RECT_F dst = D2D1::RectF(sz.width * (1.0f - fill), 0.0f, sz.width, sz.height);
        target_->DrawBitmap(bitmap_, &dst, 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, &src);
    }
    if (target_->EndDraw() == D2DERR_RECREATE_TARGET) {
        discard_device_resources();  // rebuilt on the next render
    }
}

}  // namespace rt
