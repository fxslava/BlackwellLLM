#pragma once
// -----------------------------------------------------------------------------
// splash_screen.hpp — the frameless window that stands in front of a startup
// which takes long enough to look like a hang.
//
// WHY IT OWNS A THREAD, which is the whole design and not an optimisation. The
// work this covers is main()'s: opening the capture device, planning the VRAM
// budget, loading 5.3 GB of AWQ weights, bringing up an ORT session and prefilling
// a system prompt. All of it happens on the main thread, sequentially, inside
// constructors -- so the main thread is NOT pumping messages for the whole of it.
// A splash created on that thread would be a white rectangle that never repaints
// and that Windows would mark "not responding" a few seconds in, which is a worse
// impression than no splash at all.
//
// So the splash runs its own thread with its own message loop and its own timer.
// It repaints, it fades, it shows a status line, and it does all of it while the
// main thread is blocked in cudaMalloc.
//
// WHAT IT MAY TOUCH: nothing. It renders a string it was handed and a progress
// pulse. It reads no app state, holds no reference to any subsystem, and calls
// nothing back. That is what makes it safe to run alongside a startup sequence
// whose ordering is otherwise load-bearing -- see main.cpp's preamble.
//
// COM. The splash thread does pure GDI and initializes no apartment. That is
// deliberate: main() claims STA for WebView2 (see ComApartment), and a second
// thread claiming anything is a race for no benefit.
//
// THE FADE-OUT IS SYNCHRONOUS. dismiss() runs the fade and joins, so when it
// returns the window is gone and the thread is joined -- which is what lets the
// caller sequence "splash down, then tray up" without a handshake.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace rt {

class SplashScreen {
public:
    SplashScreen() = default;
    ~SplashScreen();

    SplashScreen(const SplashScreen&) = delete;
    SplashScreen& operator=(const SplashScreen&) = delete;

    // Spawn the splash thread and show the window. Returns immediately -- the
    // caller goes straight on to the heavy initialization this is covering.
    // Non-fatal on failure: startup proceeds without a splash.
    void show(const std::wstring& title, const std::wstring& subtitle);

    // ANY thread. Replace the status line ("loading weights...", "opening the
    // microphone..."). Cheap: it stores a string and lets the splash's own timer
    // pick it up, so a caller on the critical path never blocks on a repaint.
    void set_status(const std::wstring& text);

    // Fade out, destroy the window, join the thread. Idempotent, and safe to
    // call when show() was never called or failed.
    void dismiss();

    [[nodiscard]] bool active() const noexcept { return running_.load(std::memory_order_acquire); }

private:
    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    void thread_main(std::wstring title, std::wstring subtitle);
    void paint(HWND hwnd);

    std::thread       thread_;
    std::atomic<bool> running_{false};
    // Written by dismiss() on the caller's thread, read by the splash thread's
    // timer. The fade itself runs ON the splash thread -- doing it from the
    // caller would mean touching a window from a thread that does not own it.
    std::atomic<bool> fading_{false};
    std::atomic<HWND> hwnd_{nullptr};

    std::wstring title_;
    std::wstring subtitle_;

    mutable std::mutex mu_;      // guards status_
    std::wstring       status_ = L"starting...";
};

}  // namespace rt
