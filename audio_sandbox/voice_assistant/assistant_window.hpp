#pragma once
// -----------------------------------------------------------------------------
// assistant_window.hpp — the voice_assistant main window: a Win32 frame whose
// entire client area is a Chromium (Edge WebView2) view rendering the messenger.
//
// PORTED FROM poc_overlay's settings window, and deliberately the same shape:
// a virtual host name mapped onto the on-disk `web/` folder next to the exe, so
// https://<host>/index.html serves the UI and its linked .css/.js exactly like a
// real site. The C++ side is a pure state bridge -- it pushes JSON events in and
// takes JSON commands out. No markup, no styling, no layout logic in this file.
//
// LIFETIME. Unlike the poc_overlay dialog, this object does NOT self-delete on
// WM_NCDESTROY. It is owned by main() and outlives the HWND on purpose: the
// engine and dispatcher threads keep calling post_event() until they are joined,
// which happens AFTER run_message_loop() returns. A self-deleting window would
// leave those threads posting into freed memory during a normal shutdown.
//
// THREADING. Everything here except post_event() is UI-thread only -- WebView2 is
// STA and rejects calls from any other thread. post_event() is the ONE crossing
// point: it appends to a mutex-guarded queue and PostMessage()s a wake-up, so a
// producer never blocks and never touches COM. A post to an already-destroyed
// HWND simply fails; the queue is then drained by the destructor.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

#include "settings_store.hpp"  // rt::AssistantSettings

namespace rt {

// What the page can ask the app to do. All invoked on the UI thread, from inside
// the message loop -- so an implementation may touch the settings/producer edges
// freely, but must NOT call into the engine (single-thread doctrine): marshal.
struct AssistantWindowCallbacks {
    // The user sent a typed message. Routed to the same commit gate voice uses.
    std::function<void(const std::string& text)> on_send_text;
    // The mic toggle flipped. `listening` false mutes capture (push-to-talk off).
    std::function<void(bool listening)> on_mic_toggle;
    // Settings were saved. `live_only` is true when nothing restart-tier changed,
    // in which case the app applies them to the running engine and stays up.
    std::function<void(const AssistantSettings& next, bool live_only)> on_settings_apply;
    // The user asked to restart the app so restart-tier settings can take effect.
    std::function<void()> on_restart;
};

class AssistantWindow {
public:
    explicit AssistantWindow(const wchar_t* title);
    ~AssistantWindow();
    AssistantWindow(const AssistantWindow&) = delete;
    AssistantWindow& operator=(const AssistantWindow&) = delete;

    void set_callbacks(AssistantWindowCallbacks cb) { cb_ = std::move(cb); }
    // Seeds the Settings modal. Call before create(); updated on every save.
    void set_settings(const AssistantSettings& s) { settings_ = s; }

    // Creates the frame and starts WebView2 asynchronously. Returns false only on
    // a Win32 failure; a MISSING WebView2 runtime is reported to the user from
    // inside the async callback and closes the window (there is no usable UI
    // without it, and silently falling back to a blank frame would be worse).
    bool create(int client_w, int client_h);

    void run_message_loop();

    // ANY thread. Queues one JSON message for the page. Cheap and non-blocking.
    void post_event(std::string json);

    [[nodiscard]] HWND hwnd() const noexcept { return hwnd_; }

private:
    struct Impl;   // hides <WebView2.h> / <wrl.h> from every consumer of this header

    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    LRESULT handle(HWND, UINT, WPARAM, LPARAM);

    void create_webview();
    bool map_assets_and_navigate();
    void resize_to_client();
    void drain_pending();                     // UI thread: queue -> PostWebMessageAsJson
    void on_web_message(const std::wstring& json);
    void push_settings();                     // seed/refresh the Settings modal
    void browse_for_folder(const std::string& target);
    void report_webview_unavailable();

    std::wstring title_;
    HWND hwnd_ = nullptr;              // UI thread only
    // The HWND post_event() may target, published for the producer threads and
    // cleared on WM_DESTROY. Separate from hwnd_ (and atomic) because it is the
    // one field read off the UI thread: PostMessage to a stale handle is
    // harmless, but a torn read of one is not something to rely on.
    std::atomic<HWND> post_target_{nullptr};
    Impl* impl_ = nullptr;         // owns the WebView2 COM pointers
    bool page_ready_ = false;      // NavigationCompleted fired -> safe to post
    bool com_initialized_ = false;

    AssistantWindowCallbacks cb_;
    AssistantSettings settings_;

    std::mutex q_mu_;
    std::deque<std::string> pending_;
};

}  // namespace rt
