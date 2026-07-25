// -----------------------------------------------------------------------------
// window_d2d.cpp — see window_d2d.h. DX11 swap chain shared by a Direct2D
// spectrogram pass and a Dear ImGui (DX11) control-panel pass.
// -----------------------------------------------------------------------------
#include "window_d2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include <d2d1.h>       // D2D1:: helper namespace (RectF, ColorF, ...)
#include <dxgi.h>

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

#include "audio_recorder.h"
#include "realtime_dsp.h"

// ImGui's Win32 message handler (declared in imgui_impl_win32.h).
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace rt {
namespace {

constexpr wchar_t kClassName[] = L"WhisperD2DSpectrogram";
constexpr UINT kTimerId = 1;
constexpr UINT kTimerMs = 16;              // ~60 FPS UI tick
constexpr float kDynamicRangeDb = 8.0f;    // Whisper's (max - 8) display window
constexpr int kColsPerSecond = 100;        // 10 ms hop -> 100 columns / second

template <class T>
void safe_release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

// Inferno colormap via linear interpolation over 9 evenly-spaced anchors. t in
// [0, 1] -> (b, g, r) bytes (D2D bitmap is BGRA).
void inferno(float t, uint8_t& b, uint8_t& g, uint8_t& r) {
    static const float A[9][3] = {  // R, G, B in [0, 1]
        {0.0010f, 0.0004f, 0.0140f}, {0.1220f, 0.0470f, 0.2830f},
        {0.3340f, 0.0590f, 0.4280f}, {0.5330f, 0.1330f, 0.4160f},
        {0.7290f, 0.2120f, 0.3330f}, {0.8900f, 0.3490f, 0.2010f},
        {0.9760f, 0.5490f, 0.0390f}, {0.9780f, 0.7980f, 0.1980f},
        {0.9880f, 0.9980f, 0.6440f},
    };
    t = std::clamp(t, 0.0f, 1.0f) * 8.0f;
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

WindowD2D::WindowD2D(SpectrogramBuffer& spec, AudioRecorder& recorder, const wchar_t* title)
    : spec_(spec), recorder_(recorder), title_(title) {
    bmp_w_ = spec_.max_frames();
    bmp_h_ = spec_.n_mels();
    pixels_.assign(static_cast<size_t>(bmp_w_) * bmp_h_ * 4, 0);
}

WindowD2D::~WindowD2D() {
    if (imgui_ready_) {
        ImGui_ImplDX11_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
    }
    safe_release(spectro_bitmap_);
    release_swapchain_resources();
    safe_release(d2d_context_);
    safe_release(d2d_device_);
    safe_release(d2d_factory_);
    safe_release(swap_chain_);
    safe_release(d3d_context_);
    safe_release(d3d_device_);
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
    if (!init_pipeline()) return false;

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    SetTimer(hwnd_, kTimerId, kTimerMs, nullptr);
    return true;
}

bool WindowD2D::init_pipeline() {
    RECT rc;
    GetClientRect(hwnd_, &rc);

    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = static_cast<UINT>(rc.right - rc.left);
    sd.BufferDesc.Height = static_cast<UINT>(rc.bottom - rc.top);
    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // D2D interop requires BGRA
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd_;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    const UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;  // required for D2D on DXGI
    D3D_FEATURE_LEVEL got_level = D3D_FEATURE_LEVEL_11_0;
    if (FAILED(D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION,
            &sd, &swap_chain_, &d3d_device_, &got_level, &d3d_context_))) {
        return false;
    }

    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &d2d_factory_))) return false;

    IDXGIDevice* dxgi_device = nullptr;
    if (FAILED(d3d_device_->QueryInterface(__uuidof(IDXGIDevice),
                                           reinterpret_cast<void**>(&dxgi_device)))) {
        return false;
    }
    HRESULT hr = d2d_factory_->CreateDevice(dxgi_device, &d2d_device_);
    dxgi_device->Release();
    if (FAILED(hr)) return false;
    if (FAILED(d2d_device_->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &d2d_context_)))
        return false;

    if (!create_swapchain_resources()) return false;

    // CPU-updated mel heatmap bitmap (device-independent of the backbuffer).
    const D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_NONE,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
    if (FAILED(d2d_context_->CreateBitmap(
            D2D1::SizeU(static_cast<UINT32>(bmp_w_), static_cast<UINT32>(bmp_h_)),
            nullptr, 0, &bp, &spectro_bitmap_))) {
        return false;
    }

    // Dear ImGui (Win32 + DX11 backends).
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // no imgui.ini clutter in the sandbox
    ImGui::StyleColorsDark();

    // The default embedded font atlas only covers Basic Latin (0x20-0xFF), so the
    // translator's Russian output renders as '?' boxes. Load a Windows system TTF
    // with the Cyrillic ranges BEFORE the DX11 backend builds the font texture
    // (that upload happens in ImGui_ImplDX11_Init/NewFrame, so it must be set up
    // here first). The ranges array MUST outlive the atlas Build() — ImGui only
    // stores the pointer — hence `static`.
    static const ImWchar cyrillic_ranges[] = {
        0x0020, 0x00FF,  // Basic Latin + Latin-1 Supplement
        0x0400, 0x052F,  // Cyrillic + Cyrillic Supplement
        0x2DE0, 0x2DFF,  // Cyrillic Extended-A
        0xA640, 0xA69F,  // Cyrillic Extended-B
        0,
    };
    const char* const system_fonts[] = {
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    };
    bool font_loaded = false;
    for (const char* path : system_fonts) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec) &&
            io.Fonts->AddFontFromFileTTF(path, 18.0f, nullptr, cyrillic_ranges) != nullptr) {
            font_loaded = true;
            break;
        }
    }
    if (!font_loaded) io.Fonts->AddFontDefault();  // ASCII-only fallback

    if (!ImGui_ImplWin32_Init(hwnd_)) return false;
    if (!ImGui_ImplDX11_Init(d3d_device_, d3d_context_)) return false;
    imgui_ready_ = true;
    return true;
}

bool WindowD2D::create_swapchain_resources() {
    ID3D11Texture2D* backbuffer = nullptr;
    if (FAILED(swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                      reinterpret_cast<void**>(&backbuffer)))) {
        return false;
    }
    HRESULT hr = d3d_device_->CreateRenderTargetView(backbuffer, nullptr, &rtv_);
    if (SUCCEEDED(hr)) {
        IDXGISurface* surface = nullptr;
        hr = backbuffer->QueryInterface(__uuidof(IDXGISurface),
                                        reinterpret_cast<void**>(&surface));
        if (SUCCEEDED(hr)) {
            const D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(
                D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
            hr = d2d_context_->CreateBitmapFromDxgiSurface(surface, &bp, &d2d_target_);
            surface->Release();
            if (SUCCEEDED(hr)) d2d_context_->SetTarget(d2d_target_);
        }
    }
    backbuffer->Release();
    return SUCCEEDED(hr);
}

void WindowD2D::release_swapchain_resources() {
    if (d2d_context_) d2d_context_->SetTarget(nullptr);
    safe_release(d2d_target_);
    safe_release(rtv_);
}

void WindowD2D::on_resize(UINT w, UINT h) {
    if (!swap_chain_ || w == 0 || h == 0) return;
    release_swapchain_resources();
    swap_chain_->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
    create_swapchain_resources();
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
    if (imgui_ready_ && ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp)) return 1;

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
            if (wp != SIZE_MINIMIZED) on_resize(LOWORD(lp), HIWORD(lp));
            return 0;
        case WM_DESTROY:
            KillTimer(hwnd, kTimerId);
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

void WindowD2D::draw_spectrogram() {
    spec_.snapshot(scratch_);
    const int n_cols = static_cast<int>(scratch_.size());

    // Rolling normalisation over the visible history (viz-only; never touches the
    // parity-preserving DSP): map [global_max - 8 dB, global_max] -> [0, 1].
    float vmax = -1e30f;
    for (const auto& col : scratch_)
        for (float v : col) vmax = std::max(vmax, v);
    const float vmin = vmax - kDynamicRangeDb;
    const float inv_range = (vmax > vmin) ? 1.0f / (vmax - vmin) : 0.0f;

    for (size_t i = 0; i < pixels_.size(); i += 4) {  // inferno near-black floor
        pixels_[i + 0] = 3; pixels_[i + 1] = 0; pixels_[i + 2] = 0; pixels_[i + 3] = 255;
    }
    const int start = std::max(0, bmp_w_ - n_cols);  // newest column at the right
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
    spectro_bitmap_->CopyFromMemory(&rect, pixels_.data(), static_cast<UINT32>(bmp_w_) * 4);

    d2d_context_->BeginDraw();
    d2d_context_->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f));
    if (n_cols > 0) {
        const D2D1_SIZE_F sz = d2d_context_->GetSize();
        // Time-scale zoom: show only the most recent `visible_cols` columns across
        // the window. Fewer columns -> faster scroll + wider columns (zoom in);
        // more columns -> slower scroll + compressed time (zoom out).
        const int visible_cols =
            std::clamp(static_cast<int>(visible_time_window_sec_ * kColsPerSecond), 1, bmp_w_);
        const int shown = std::min(visible_cols, n_cols);
        const D2D1_RECT_F src = D2D1::RectF(static_cast<float>(bmp_w_ - shown), 0.0f,
                                            static_cast<float>(bmp_w_), static_cast<float>(bmp_h_));
        const float fill = static_cast<float>(shown) / static_cast<float>(visible_cols);
        const D2D1_RECT_F dst = D2D1::RectF(sz.width * (1.0f - fill), 0.0f, sz.width, sz.height);
        d2d_context_->DrawBitmap(spectro_bitmap_, &dst, 1.0f,
                                 D2D1_INTERPOLATION_MODE_LINEAR, &src);
    }
    d2d_context_->EndDraw();
}

void WindowD2D::draw_ui_panel() {
    const AudioRecorder::Status st = recorder_.snapshot();

    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Audio & Spectrogram Control Panel");

    // --- Audio level meter with a threshold marker line ----------------------
    ImGui::TextUnformatted("Input level");
    const float lvl = std::clamp((st.level_db + 60.0f) / 60.0f, 0.0f, 1.0f);  // -60..0 dB
    char lvl_text[32];
    std::snprintf(lvl_text, sizeof(lvl_text), "%.1f dBFS", st.level_db);
    ImGui::ProgressBar(lvl, ImVec2(-1.0f, 0.0f), lvl_text);
    {
        float thr = recorder_.threshold_db.load();
        const ImVec2 p0 = ImGui::GetItemRectMin();
        const ImVec2 p1 = ImGui::GetItemRectMax();
        const float tx = p0.x + (p1.x - p0.x) * std::clamp((thr + 60.0f) / 60.0f, 0.0f, 1.0f);
        ImGui::GetWindowDrawList()->AddLine(ImVec2(tx, p0.y), ImVec2(tx, p1.y),
                                            IM_COL32(255, 80, 80, 255), 2.0f);
    }

    ImGui::Separator();

    // --- VAD controls --------------------------------------------------------
    bool enabled = recorder_.enabled.load();
    if (ImGui::Checkbox("Auto-Record (VAD)", &enabled)) recorder_.enabled.store(enabled);

    float thr = recorder_.threshold_db.load();
    if (ImGui::SliderFloat("Threshold (dB)", &thr, -60.0f, 0.0f, "%.1f"))
        recorder_.threshold_db.store(thr);

    float hang = recorder_.hangover_sec.load();
    if (ImGui::SliderFloat("Silence hangover (s)", &hang, 0.2f, 3.0f, "%.2f"))
        recorder_.hangover_sec.store(hang);

    ImGui::Separator();

    // --- Recording status ----------------------------------------------------
    switch (st.state) {
        case AudioRecorder::State::Idle:
            ImGui::TextUnformatted("Status: [IDLE]");
            break;
        case AudioRecorder::State::Recording:
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                               "Status: [RECORDING %05.1fs]", st.record_seconds);
            break;
        case AudioRecorder::State::Hangover:
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                               "Status: [HANGOVER %05.1fs]", st.record_seconds);
            break;
    }
    if (!st.last_saved.empty()) ImGui::Text("Last saved: %s", st.last_saved.c_str());

    ImGui::Separator();

    // --- Spectrogram time-scale / speed --------------------------------------
    ImGui::SliderFloat("Time window (s)", &visible_time_window_sec_, 1.0f, 10.0f, "%.1f");
    ImGui::TextDisabled("%d columns (%d ms hop)", static_cast<int>(visible_time_window_sec_ * kColsPerSecond),
                        1000 / kColsPerSecond);

    ImGui::End();
}

void WindowD2D::render() {
    if (!d2d_context_ || !d2d_target_ || !rtv_) return;

    draw_spectrogram();  // Direct2D pass onto the shared backbuffer

    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    draw_ui_panel();
    if (extra_panel_) extra_panel_();  // optional consumer panel (e.g. live transcript)
    ImGui::Render();

    // ImGui (DX11) draws on top of the D2D result; do NOT clear the RTV here.
    d3d_context_->OMSetRenderTargets(1, &rtv_, nullptr);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    swap_chain_->Present(1, 0);
}

}  // namespace rt
