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

// THE HOT AUDIO SET, as its own type rather than a subset of AssistantSettings.
//
// The separation is enforced by the TYPE, not by discipline: there is no field
// here that a restart could depend on, so no code path that takes one of these
// can decide to restart the engine even by mistake. That is the whole design --
// the previous arrangement passed a full AssistantSettings and relied on every
// caller remembering that audio changes are cheap.
//
// Each member is optional-by-sentinel so a slider drag sends one field rather
// than a snapshot of five: -2 means "index not present in this message" (-1 is
// already taken, and means "follow the system default"), and a negative gain
// means "gain not present". A message that carries only what moved cannot
// clobber a device the user selected a moment earlier from a stale copy.
struct AudioHotUpdate {
    static constexpr int kNoIndex = -2;

    int output_device_index = kNoIndex;
    int input_device_index = kNoIndex;
    bool has_output_name = false;
    bool has_input_name = false;
    std::string output_device_name;
    std::string input_device_name;
    float tts_volume = -1.0f;   // <0 = absent
    float mic_gain = -1.0f;     // <0 = absent

    // SPEAKER MUTE (the dock's "deafen"), tri-state for the same reason the
    // gains are sentinel-encoded: a message carrying only what moved cannot
    // clobber the other fields.
    //
    // SEPARATE FROM tts_volume ON PURPOSE. Muting by writing volume 0 destroys
    // the value being muted, so unmuting has to guess -- and the obvious guess
    // (restore to 1.0) silently promotes everyone who listens at 30% to full
    // scale the first time they tap the speaker icon. Keeping mute orthogonal
    // means the slider still shows, and still holds, what the user chose.
    //
    // NOT PERSISTED, unlike every other field here: a mute is a statement about
    // the next few minutes, and an app that starts up silent because of a tap
    // three days ago reads as broken. main.cpp applies it and does not save it.
    enum class Tri { Absent, Off, On };
    Tri tts_muted = Tri::Absent;

    // True when anything requiring a DEVICE reopen is present. The gains alone
    // are atomics and must not cost a device close/open -- dragging a volume
    // slider through 40 values would otherwise reopen WASAPI 40 times. Mute is
    // on the gain side of that line: it is one atomic store, not a reopen.
    bool needs_device_reload() const noexcept {
        return output_device_index != kNoIndex || input_device_index != kNoIndex ||
               has_output_name || has_input_name;
    }
};

// What the page can ask the app to do. All invoked on the UI thread, from inside
// the message loop -- so an implementation may touch the settings/producer edges
// freely, but must NOT call into the engine (single-thread doctrine): marshal.
struct AssistantWindowCallbacks {
    // The user sent a typed message. Routed to the same commit gate voice uses.
    std::function<void(const std::string& text)> on_send_text;
    // The mic toggle flipped. `listening` false mutes capture (push-to-talk off).
    // Raised by the mic button AND by the talk hotkey -- deliberately the same
    // callback, so a hotkey is not a second, differently-behaved path to the
    // microphone.
    std::function<void(bool listening)> on_mic_toggle;
    // Interrupt whatever is generating right now (the cancel hotkey). Equivalent
    // to a barge-in, minus the speech.
    std::function<void()> on_cancel;
    // Settings were saved. `live_only` is true when nothing restart-tier changed,
    // in which case the app applies them to the running engine and stays up.
    std::function<void(const AssistantSettings& next, bool live_only)> on_settings_apply;
    // The user asked to restart the app so restart-tier settings can take effect.
    std::function<void()> on_restart;

    // "Check sound": play a short tone through the LIVE output pipeline, so the
    // thing being tested is the thing that speaks. Fired from the settings
    // panel; runs on the UI thread and must not block it for long.
    std::function<void()> on_test_tone;

    // THE HOT AUDIO PATH, and the reason it is a separate callback from
    // on_settings_apply rather than a flag on it.
    //
    // on_settings_apply receives the WHOLE form and asks requires_restart()
    // whether the engine has to come back. That is correct for the settings
    // modal and catastrophic for a volume slider: the payload carries every
    // restart-tier field too, so one blank model path -- a modal that was never
    // opened, a tab never seeded -- differs from what is running and restarts a
    // process holding 8 GB of weights because somebody dragged a slider.
    //
    // This callback carries ONLY the audio fields. It cannot compute a restart
    // because it is not given anything a restart could depend on, which is a
    // stronger guarantee than remembering not to ask.
    std::function<void(const AudioHotUpdate&)> on_audio_hot_update;
    // The system prompt changed and must be re-frozen: tokenize, prefill, and
    // republish the KV rewind floor. This is ENGINE work and is therefore
    // asynchronous by nature -- the implementation marshals it onto the engine
    // thread (post_engine_task) and reports back through system_prompt_applied().
    // Called only when the text actually differs from what is running.
    std::function<void(const std::string& prompt)> on_system_prompt_apply;
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

    // ANY thread. The engine finished (or failed) a system-prompt rebuild:
    // `tokens` is the new frozen prefix length, `detail` is empty on success and
    // carries the failure text otherwise. Closes the loop the Apply button opened
    // -- a precompute that takes seconds must not look like a button that did
    // nothing.
    void post_system_prompt_applied(bool ok, unsigned tokens, const std::string& detail);

    // Microphone amplitude for the settings meter, [0, 1]. Posted several times
    // a second from the telemetry thread; it is a display value, so it is
    // dropped rather than queued when the page is not up.
    void post_audio_level(float level);

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
    // Native picker for one settings field. FOLDER or FILE is decided from the
    // target name (see the table in the .cpp): almost every path this app takes
    // is a checkpoint DIRECTORY, and the GGML Whisper model is the exception.
    void browse_for_path(const std::string& target);

    // Answers the page's "audio.devices.request" with the live endpoint lists,
    // so the settings dropdowns are built from what the machine actually has
    // rather than from a number the user read off the console. Re-enumerates on
    // every call; see the .cpp for why it is not a fetch endpoint.
    void push_audio_devices();

    // ---- hotkeys (UI thread) -------------------------------------------------
    void register_hotkeys();      // (re)register all three from settings_
    void unregister_hotkeys();
    void on_hotkey(int id);
    void set_listening(bool on, bool tell_page);   // one place that owns mic state
    void poll_push_to_talk();     // hold-to-talk release watchdog (see the .cpp)
    // `stage` names the call that failed and `hr` is its status. Both end up in
    // the dialog AND on stderr: a bare "could not initialize the runtime" sends
    // the user off to reinstall a runtime that is already installed, which is
    // exactly the wrong trail when the real cause is the COM apartment.
    void report_webview_unavailable(const char* stage, HRESULT hr);

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
    // True when this thread was ALREADY in the multi-threaded apartment before we
    // asked for STA (CoInitializeEx -> RPC_E_CHANGED_MODE). WebView2 is STA-only,
    // so this single fact explains an otherwise inscrutable environment-creation
    // failure and is reported as such.
    bool mta_conflict_ = false;

    AssistantWindowCallbacks cb_;
    AssistantSettings settings_;

    // Mic state lives HERE, not in the page: the talk hotkey and the mic button
    // both flip it, and a toggle that reads its previous value out of the DOM
    // would desynchronize the first time a hotkey fired while the window was
    // hidden. The page follows via the `mic` event.
    bool listening_ = true;
    bool ptt_held_ = false;       // a hold-to-talk chord is currently down
    bool hotkeys_registered_ = false;

    std::mutex q_mu_;
    std::deque<std::string> pending_;
};

}  // namespace rt
