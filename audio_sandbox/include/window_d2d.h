#pragma once
// -----------------------------------------------------------------------------
// window_d2d.h — Win32 window compositing a Direct2D spectrogram with a Dear
// ImGui control panel, both on a shared DX11 / DXGI swap chain.
//
// Pipeline per frame: D2D draws the scrolling mel bitmap onto the DXGI back
// buffer, then ImGui (DX11 backend) draws the "Audio & Spectrogram Control
// Panel" on top, then Present. Runs on the UI thread; it only ever READS a
// snapshot of the spectrogram and the recorder status, and WRITES recorder
// config + the local zoom via atomics/UI-thread-owned state.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <d2d1_1.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rt {

class SpectrogramBuffer;
class AudioRecorder;

class WindowD2D {
public:
    WindowD2D(SpectrogramBuffer& spec, AudioRecorder& recorder, const wchar_t* title);
    ~WindowD2D();
    WindowD2D(const WindowD2D&) = delete;
    WindowD2D& operator=(const WindowD2D&) = delete;

    bool create(int client_w, int client_h);  // false on failure
    void run_message_loop();

private:
    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT handle(HWND, UINT, WPARAM, LPARAM);

    bool init_pipeline();                 // D3D11 + D2D + ImGui
    bool create_swapchain_resources();    // RTV + D2D target bound to backbuffer
    void release_swapchain_resources();
    void render();
    void draw_spectrogram();              // D2D pass
    void draw_ui_panel();                 // ImGui pass content
    void on_resize(UINT w, UINT h);

    SpectrogramBuffer& spec_;
    AudioRecorder& recorder_;
    std::wstring title_;
    HWND hwnd_ = nullptr;

    // DX11 / DXGI.
    ID3D11Device* d3d_device_ = nullptr;
    ID3D11DeviceContext* d3d_context_ = nullptr;
    IDXGISwapChain* swap_chain_ = nullptr;
    ID3D11RenderTargetView* rtv_ = nullptr;

    // Direct2D (1.1) on the shared DXGI surface.
    ID2D1Factory1* d2d_factory_ = nullptr;
    ID2D1Device* d2d_device_ = nullptr;
    ID2D1DeviceContext* d2d_context_ = nullptr;
    ID2D1Bitmap1* d2d_target_ = nullptr;    // wraps the backbuffer (render target)
    ID2D1Bitmap1* spectro_bitmap_ = nullptr; // CPU-updated mel heatmap

    std::vector<uint8_t> pixels_;               // BGRA, bmp_w_ * bmp_h_ * 4
    std::vector<std::vector<float>> scratch_;   // reused snapshot buffer
    int bmp_w_ = 0;                             // == SpectrogramBuffer::max_frames()
    int bmp_h_ = 0;                             // == SpectrogramBuffer::n_mels()

    // Spectrogram time-scale control (UI-thread only): visible history width.
    float visible_time_window_sec_ = 10.0f;    // 1.0 .. 10.0 s  (100 .. 1000 cols)
    bool imgui_ready_ = false;
};

}  // namespace rt
