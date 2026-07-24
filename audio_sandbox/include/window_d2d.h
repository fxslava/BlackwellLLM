#pragma once
// -----------------------------------------------------------------------------
// window_d2d.h — STAGE 3: Win32 + Direct2D scrolling spectrogram view.
//
// Reads the shared SpectrogramBuffer each frame, maps the rolling history to an
// inferno-colormapped ID2D1Bitmap (mel 0 at the BOTTOM), and stretches it over
// the client area. A ~30 FPS WM_TIMER drives continuous redraws. Runs entirely
// on the UI thread; it only ever reads a snapshot of the spectrogram.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d2d1.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rt {

class SpectrogramBuffer;

class WindowD2D {
public:
    WindowD2D(SpectrogramBuffer& spec, const wchar_t* title);
    ~WindowD2D();
    WindowD2D(const WindowD2D&) = delete;
    WindowD2D& operator=(const WindowD2D&) = delete;

    bool create(int client_w, int client_h);  // false on failure
    void run_message_loop();

private:
    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT handle(HWND, UINT, WPARAM, LPARAM);

    HRESULT ensure_device_resources();
    void discard_device_resources();
    void render();
    void on_resize(UINT w, UINT h);

    SpectrogramBuffer& spec_;
    std::wstring title_;
    HWND hwnd_ = nullptr;

    ID2D1Factory* factory_ = nullptr;
    ID2D1HwndRenderTarget* target_ = nullptr;
    ID2D1Bitmap* bitmap_ = nullptr;

    std::vector<uint8_t> pixels_;                 // BGRA, bmp_w_ * bmp_h_ * 4
    std::vector<std::vector<float>> scratch_;     // reused snapshot buffer
    int bmp_w_ = 0;                               // == SpectrogramBuffer::max_frames()
    int bmp_h_ = 0;                               // == SpectrogramBuffer::n_mels()
};

}  // namespace rt
