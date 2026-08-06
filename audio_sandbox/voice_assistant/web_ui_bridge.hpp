#pragma once
// -----------------------------------------------------------------------------
// web_ui_bridge.hpp — the Chromium messenger's host side: it owns the window,
// binds every page message to the subsystem that answers it, and owns the
// diagnostics poller that feeds the Settings panel.
//
// WHAT A "HOST BINDING" IS HERE. AssistantWindow is a pure state bridge -- JSON
// events in, JSON commands out -- and AssistantWindowCallbacks is the list of
// commands it can raise. Something has to turn each of those into a call on the
// audio pipeline, the router, or the engine, and that fan-out is this class. It
// used to be eleven lambdas capturing thirty locals by reference at the bottom of
// main().
//
// WHY THE SETTINGS HANDLERS LIVE HERE and not in a component of their own: a
// settings save IS a page message, and distributing it is the same act as
// distributing any other. Both handlers are the ones with real policy in them, so
// each carries its argument at the point it applies:
//
//   on_settings_apply     the WHOLE form. Live-tier fields are pushed to the
//                         running system; restart-tier fields are persisted and
//                         picked up by the next process.
//   on_audio_hot_update   ONLY the audio fields, and that is a TYPE-enforced
//                         guarantee rather than a discipline: the payload cannot
//                         express a restart-tier change, so a volume drag
//                         physically cannot restart a process holding 8 GB of
//                         weights. See AudioHotUpdate.
//
// THREADING. Everything here runs on the UI thread (the window message loop),
// with one exception: the diagnostics poller is its own thread and touches only
// the view, the gate counters and the input meter -- all of which document
// cross-thread use. NOTHING here calls into the engine directly; the two paths
// that need to (the persona rebuild, the audio task prefix) marshal through
// post_engine_task inside the router and the settings fan-out respectively.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <atomic>
#include <string>
#include <thread>

#include "assistant_view.hpp"
#include "assistant_window.hpp"
#include "console_overlay.hpp"
#include "log_buffer.hpp"
#include "settings_store.hpp"
#include "settings_validator.hpp"

#include "app_context.hpp"
#include "app_lifecycle.hpp"
#include "audio_pipeline_binder.hpp"
#include "conversation_router.hpp"

namespace rt {

class WebUIBridge {
public:
    // Everything here is BORROWED and must outlive this object, which is what
    // constructing it LAST in main() gives. `settings` is the live copy the app
    // runs on: this class writes it (a save is the only thing that changes it) and
    // main reads it afterwards.
    // `log` is BORROWED and may be null (a build or a launch with no capture);
    // the console then shows only what is posted to it directly.
    WebUIBridge(AssistantSettings& settings, AppContext& ctx, AppLifecycleManager& lifecycle,
                AudioPipelineBinder& audio, ConversationRouter& router, AssistantView& view,
                LogBuffer* log = nullptr);
    ~WebUIBridge();

    WebUIBridge(const WebUIBridge&) = delete;
    WebUIBridge& operator=(const WebUIBridge&) = delete;

    // Create the frame and start WebView2. Throws (INIT tier) on a Win32 failure;
    // a MISSING WebView2 runtime is reported to the user from inside the window's
    // own async callback, because there is no usable UI without it and silently
    // falling back to a blank frame would be worse.
    void create(int client_w, int client_h);

    // Blocks until the window closes.
    void run_message_loop();

    // Start the 1 Hz gate-counter / 12 fps input-meter poller. Its own thread and
    // deliberately NOT on the chat screen -- the messenger carries no inference
    // telemetry; every counter lives behind the Settings modal.
    void start_diagnostics_poller();
    // Join it. Idempotent. Called between the engine thread's join and the
    // dispatcher's stop, which is where main() has always joined it.
    void stop_diagnostics_poller() noexcept;

    // True when the user asked for a restart so restart-tier settings take effect.
    // Read by main AFTER everything has been released -- see relaunch_self().
    bool restart_requested() const noexcept {
        return restart_requested_.load(std::memory_order_acquire);
    }

    // Apply the whole live-settings fan-out once, at startup, with the launch
    // configuration. Same path every later save runs: it publishes on the bus.
    void apply_live_settings(const AssistantSettings& s);

    // The validation context this launch validates against, assembled from what
    // the lifecycle actually brought up rather than from the settings -- so the
    // budget the modal quotes is the one bring-up would reach.
    [[nodiscard]] ValidationContext validation_context() const;

    AssistantWindow& window() noexcept { return window_; }
    SettingsBus&     bus() noexcept { return bus_; }

private:
    void install_callbacks();
    void install_subscribers();
    void on_settings_apply(const AssistantSettings& next, bool live_only);
    void on_audio_hot_update(const AudioHotUpdate& u);
    // The engine-facing half of the fan-out: sampling knobs, context mode, the
    // speech-mode thresholds and the audio-task prefix rebuild. A bus subscriber
    // like every other consumer -- it is only a named method because it is the
    // one with real policy in it.
    void apply_engine_settings(const AssistantSettings& s);
    // Push a validation report to the page so each message lands on the field
    // that produced it. Errors also go to stderr, and therefore to the console.
    void publish_diagnostics(const ValidationReport& report);
    // Relay a residency transition to the page.
    void publish_residency(const EngineResidency::Progress& p);

    AssistantSettings&   settings_;
    AppContext&          ctx_;
    AppLifecycleManager& lifecycle_;
    AudioPipelineBinder& audio_;
    ConversationRouter&  router_;
    AssistantView&       view_;

    AssistantWindow window_{L"Assistant"};

    // The log overlay. Created in create(), on the UI thread, so its messages are
    // dispatched by the same GetMessage loop the main window's are -- which is
    // why it needs no thread and no pump of its own.
    ConsoleOverlay console_;
    LogBuffer*     log_ = nullptr;   // borrowed, may be null

    // THE FAN-OUT. Subscribers are registered once in the constructor and every
    // settings change -- startup, a save, a hot update -- is published on it.
    // See the threading contract in settings_validator.hpp.
    SettingsBus bus_;

    // The persona system prompt is the ONE live setting absent from the fan-out:
    // it is a KV cache rebuild, not a store, and goes through the router. Under
    // session isolation the AUDIO TASK prompt joins it in that category -- it is
    // the transcription sequence's frozen prefix -- so these track the values the
    // ENGINE is actually running, to rebuild only on a real edit.
    //
    // TWO settings compose that ONE prefix (the task text and the forced spoken
    // language), and the retry is a FLAG rather than a poisoned shadow copy:
    // clearing the copies cannot express "retry" when the value the user just
    // chose is itself the empty string, which is exactly what "no forced language"
    // is.
    std::string applied_audio_task_;
    std::string applied_speech_language_;
    bool        audio_prefix_dirty_ = false;

    std::atomic<bool> restart_requested_{false};
    std::atomic<bool> polling_{false};
    std::thread       stats_thread_;
};

}  // namespace rt
