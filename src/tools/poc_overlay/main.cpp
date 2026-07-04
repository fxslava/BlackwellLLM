// poc_overlay: the live overlay translator -- a stateful, debounce-driven
// pipeline from OS input capture to real BlackwellEngine inference and back
// into the focused control:
//
//   [UI thread]      WH_KEYBOARD_LL hook (lightweight TRIGGER + fallback buffer)
//                        |  RequestUpdate(fallbackText)     [condvar signal]
//                        v
//   [COM/UIA thread]  CaretTracker: reads the REAL text from the focused control
//                      (IUIAutomationTextPattern document-up-to-caret, or
//                      IUIAutomationValuePattern) and runs the interaction
//                      state machine:
//                        Typing      capture area (granularity-bounded) shown
//                                    dimmed in the overlay; idle timer re-arms
//                                    AND a speculative background prefill
//                                    fires (TrackUpdate) on every keystroke
//                        Translating idle timer expired -> TriggerGeneration();
//                                    overlay shows a loading/streaming state
//                        Ready       translation shown; Ctrl+Enter surgically
//                                    replaces source_raw via TextInjector --
//                                    HOST-SIDE ONLY, no inference on commit
//                        |  PostState(snapshot)             [PostMessage]
//                        v
//   [tracker thread]  LiveTranslationTracker (inside TranslationService): talks
//                      DIRECTLY to BlackwellLLMAdapter / the engine's prefill
//                      coordinator -- no AgentOrchestrator, no ReAct loop.
//                        |  OnPreviewResult(text, done)     [condvar signal]
//                        v
//   [UI thread]       OverlayWindow: layered, click-through, D2D-rendered popup
//                      whose string END anchors just above the caret.
//
// A system tray icon provides the only way to exit (no visible window/taskbar
// entry otherwise).
#include <windows.h>
#include <objbase.h>

#include <string>

#include "caret_tracker.h"
#include "config.h"
#include "hook_manager.h"
#include "overlay_window.h"
#include "settings_dialog.h"
#include "translation_service.h"
#include "tray_icon.h"

namespace {

constexpr UINT kTrayIconId = 1;
constexpr UINT_PTR kReadyTimerId = 1;  // polls engine readiness after Mode ON

// The live, in-memory application config. Loaded at startup and rewritten by the
// settings window; only ever touched on the UI thread.
Config g_config;

// The tracker owns the capture/debounce settings; the settings window pushes
// changes into it live via ApplyConfig (model path still needs a restart).
CaretTracker* g_caretTracker = nullptr;
// The inference service, for the readiness poll that drives the toggle HUD from
// "Initializing..." to "ACTIVE". Only touched on the UI thread.
TranslationService* g_translator = nullptr;

// Pushes a config into the running app: rebinds hotkeys and capture settings
// in memory immediately. (Model path / context size are engine-construction
// parameters and require a restart.)
void ApplyConfig(const Config& config) {
    g_config = config;
    HookManager::Instance().SetCommitShortcut(config.commitShortcut);
    HookManager::Instance().SetActivationShortcut(config.activationShortcut);
    if (g_caretTracker) {
        g_caretTracker->SetCaptureSettings(config.captureGranularity, config.idleTimerMs);
    }
}

LRESULT CALLBACK ControllerWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* trayIcon = reinterpret_cast<TrayIcon*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
        case TrayIcon::WM_TRAYICON: {
            if (!trayIcon) {
                return 0;
            }
            switch (trayIcon->HandleMessage(wParam, lParam)) {
                case TrayCommand::ShowSettings: {
                    const HINSTANCE hInst =
                        reinterpret_cast<HINSTANCE>(GetWindowLongPtrW(hwnd, GWLP_HINSTANCE));
                    ShowSettingsWindow(hwnd, hInst, g_config,
                                       [](const Config& c) { ApplyConfig(c); });
                    break;
                }
                case TrayCommand::Exit:
                    DestroyWindow(hwnd);
                    break;
                case TrayCommand::None:
                    break;
            }
            return 0;
        }
        case WM_TIMER:
            // Toggle ON started the engine warming up; poll for readiness and
            // flip the HUD banner from "Initializing..." to "ACTIVE" (or an error).
            if (wParam == kReadyTimerId && g_translator && g_caretTracker) {
                switch (g_translator->state()) {
                    case TranslationService::State::Ready:
                        KillTimer(hwnd, kReadyTimerId);
                        g_caretTracker->SetActive(true);
                        g_caretTracker->ShowHud(L"Translation Mode: ACTIVE", /*fade=*/true);
                        break;
                    case TranslationService::State::Error:
                        KillTimer(hwnd, kReadyTimerId);
                        g_caretTracker->ShowHud(L"Engine unavailable - set the model path in Settings",
                                                /*fade=*/true);
                        break;
                    case TranslationService::State::Loading:
                        break;  // keep waiting
                }
            }
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int) {
    // Without this, the process defaults to DPI-unaware and Windows silently
    // virtualizes/rescales screen coordinates for it -- inconsistently across
    // monitors with different scale factors. That breaks the whole pipeline:
    // UIA reports the caret's real physical-pixel position, but our own
    // SetWindowPos/UpdateLayeredWindow calls would land in the *virtualized*
    // coordinate space, so the overlay ends up nowhere near the caret except
    // by coincidence on a single-monitor, 100%-scale setup. Must be set
    // before any window is created.
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    // D2D/DWrite/tray-icon setup on this thread don't themselves require COM,
    // but initialize an apartment anyway since it's harmless and conventional
    // for a GUI thread. UI Automation's own STA lives on CaretTracker's
    // dedicated worker thread -- it is NOT this one.
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    OverlayWindow overlay;
    if (!overlay.Create(hInstance)) {
        MessageBoxW(nullptr, L"Failed to create the overlay window (Direct2D/DirectWrite init failed).",
                    L"poc_overlay", MB_ICONERROR);
        return 1;
    }

    // Config must be loaded BEFORE the TranslationService: it carries the model
    // directory and the sampling parameters the engine is constructed with.
    g_config = ConfigStore::Load();
    ApplyConfig(g_config);

    // The inference stack. Engine construction (DirectStorage weight streaming)
    // happens on the service's own worker thread -- this constructor is instant.
    // The preview sink is late-bound below, once the CaretTracker exists.
    TranslationService::Settings svcSettings;
    svcSettings.modelDir = g_config.modelPath;
    // promptCacheRoot left empty -> %LOCALAPPDATA%\Blackwell\Cache. The service
    // derives the per-model subdirectory from the loaded checkpoint's hash and
    // JIT-compiles the .bkv prompt cache there on first run -- nothing is
    // shipped with (or read from) the model directory itself.
    svcSettings.maxSeqLen = static_cast<size_t>(g_config.contextSize);
    svcSettings.temperature = g_config.temperature;
    svcSettings.topP = g_config.topP;
    // Tiered KV prefix-cache budget -> engine RuntimeConfig knobs. An empty
    // spill path resolves to %LOCALAPPDATA%\Blackwell\spill.bkv.
    svcSettings.vramCacheBlocks = g_config.vramCacheBlocks;
    svcSettings.ramTierBlocks = g_config.ramTierBlocks;
    svcSettings.diskSpillEnabled = g_config.diskSpillEnabled;
    svcSettings.diskSpillBlocks = g_config.diskSpillBlocks;
    svcSettings.spillFilePath = g_config.spillFilePath.empty()
                                    ? ConfigStore::DefaultSpillPath()
                                    : g_config.spillFilePath;
    TranslationService translator(std::move(svcSettings));
    g_translator = &translator;  // for the readiness poll (WM_TIMER)

    // CaretTracker owns the interaction state machine on ITS OWN (COM/UIA STA)
    // thread: capture extraction, the idle-timer debounce, and the surgical
    // commit (so the UIA interfaces never cross apartments). All callbacks fire
    // on that thread and only enqueue.
    CaretTracker::Callbacks trackerCallbacks;
    trackerCallbacks.render = [&overlay](const OverlaySnapshot& snapshot) {
        overlay.PostState(snapshot);  // PostMessage-marshaled to the UI thread
    };
    trackerCallbacks.trackUpdate = [&translator](const std::wstring& source,
                                                 const std::wstring& context) {
        translator.TrackUpdate(source, context);
    };
    trackerCallbacks.triggerGeneration = [&translator](const std::wstring& source,
                                                       const std::wstring& context) {
        translator.TriggerGeneration(source, context);
    };
    trackerCallbacks.cancelGeneration = [&translator]() { translator.Cancel(); };
    trackerCallbacks.injectionGuard = [](bool active) {
        HookManager::Instance().SetInjecting(active);
    };
    CaretTracker caretTracker(std::move(trackerCallbacks), g_config.captureGranularity,
                              g_config.idleTimerMs);
    g_caretTracker = &caretTracker;

    // Late-bound sink: translation deltas/finals flow back into the tracker's
    // state machine (thread-safe enqueue), which decides what the overlay shows.
    // The service's generation counter already suppresses most stale deliveries;
    // the tracker's phase check catches the rest.
    translator.SetPreviewSink(
        [&caretTracker](std::uint64_t /*gen*/, const std::wstring& text, bool done) {
            caretTracker.OnPreviewResult(text, done);
        });

    // Invisible controller window: owns the tray icon, receives its callback
    // messages (WM_TRAYICON), and drives the engine-readiness poll timer. Created
    // BEFORE the hooks so the master-toggle callback can arm its timer on it.
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &ControllerWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"BlackwellPocOverlayController";
    RegisterClassExW(&wc);

    HWND controller = CreateWindowExW(0, wc.lpszClassName, L"poc_overlay controller", WS_OVERLAPPED,
                                       0, 0, 0, 0, nullptr, nullptr, hInstance, nullptr);

    TrayIcon trayIcon(controller, kTrayIconId);
    SetWindowLongPtrW(controller, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&trayIcon));
    trayIcon.Create(L"Blackwell PoC: Translation Mode (Alt+Shift+T)");

    // The hook callbacks run inline in the global hook chain on THIS thread and
    // must stay fast. They only hand work to the caret tracker's queue (an O(1)
    // lock + condvar notify) -- the UIA round-trip, D2D paint, and the whole
    // 3-tier replacement happen off this call stack entirely.
    HookManager::Callbacks callbacks;
    callbacks.onTrigger = [&caretTracker](const std::wstring& fallbackText, bool wordBoundary) {
        // Every keystroke (Mode ACTIVE only): re-capture, re-arm the idle timer,
        // interrupt any in-flight/displayed translation (State 2/3 -> State 1).
        caretTracker.RequestUpdate(fallbackText, wordBoundary);
    };
    callbacks.onCommit = [&caretTracker](const std::wstring& fallbackText) {
        // Ctrl+Enter: surgical host-side replace of the Ready snapshot (typing OR
        // selection popup); a no-op in any other state.
        caretTracker.RequestCommit(fallbackText);
    };
    callbacks.onReset = [&caretTracker]() {
        // Nav key / mouse-button-down: cancel + hide. The tracker renders the
        // hide itself (and leaves a CenterHud banner alone) -- do NOT call
        // overlay.Hide() here or a toggle banner would be wiped.
        caretTracker.RequestReset();
    };
    // Master toggle (Alt+Shift+T): show the HUD, warm the engine, gate tracking.
    callbacks.onActivationToggle = [&caretTracker, &translator, controller](bool active) {
        if (active) {
            if (translator.state() == TranslationService::State::Ready) {
                caretTracker.SetActive(true);
                caretTracker.ShowHud(L"Translation Mode: ACTIVE", /*fade=*/true);
            } else {
                caretTracker.ShowHud(L"Initializing Translation Engine...", /*fade=*/false);
                SetTimer(controller, kReadyTimerId, 150, nullptr);  // poll -> ACTIVE
            }
        } else {
            KillTimer(controller, kReadyTimerId);
            caretTracker.SetActive(false);
            caretTracker.ShowHud(L"Translation Mode: OFF", /*fade=*/true);
        }
    };
    // Left-button-up (Mode ACTIVE only): a selection may now exist -> translate it.
    callbacks.onSelectionCandidate = [&caretTracker]() { caretTracker.RequestSelectionCheck(); };

    const bool hookInstalled = HookManager::Instance().Install(std::move(callbacks));
    if (!hookInstalled) {
        MessageBoxW(nullptr, L"Failed to install the low-level keyboard/mouse hooks.", L"poc_overlay",
                    MB_ICONERROR);
        return 1;
    }

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    HookManager::Instance().Uninstall();
    g_caretTracker = nullptr;
    g_translator = nullptr;

    // Teardown ordering: join the service worker FIRST, so its sink can never
    // fire into the CaretTracker while (or after) the tracker is destroyed
    // during stack unwinding. After Shutdown(), the tracker's own teardown may
    // still call RequestPreview/CancelPending -- both no-op against a stopped
    // service that is destroyed later (declared earlier) in this scope.
    translator.Shutdown();

    CoUninitialize();
    return static_cast<int>(msg.wParam);
}
