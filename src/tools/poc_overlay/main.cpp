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

#include <algorithm>
#include <cwctype>
#include <string>
#include <vector>

#include "caret_tracker.h"
#include "config.h"
#include "hook_manager.h"
#include "overlay_window.h"
#include "settings_dialog.h"
#include "translation_service.h"
#include "tray_icon.h"

namespace {

constexpr UINT kTrayIconId = 1;
constexpr UINT_PTR kReadyTimerId = 1;      // polls engine readiness after Mode ON
constexpr UINT_PTR kLifecycleTimerId = 2;  // coarse inactivity tick (spill/hibernate)
constexpr UINT kLifecycleTickMs = 2'000;   // timeouts have second granularity now; the
                                           // tick is two atomic reads -- 2s is cheap

// The live, in-memory application config. Loaded at startup and rewritten by the
// settings window; only ever touched on the UI thread.
Config g_config;

// The pair set currently WIRED into the app -- the engine's prompt branches,
// the CaretTracker routing, the overlay dropdown labels, and the Alt+<N>
// hotkey bound all derive from it. ApplyConfig() re-derives everything on a
// Settings save, so edited pairs work immediately (hot-reload, NO restart):
// a brand-new direction simply prefills cold on first use, or explicitly via
// the per-pair "Pre-cache" button.
std::vector<LanguagePair> g_sessionPairs;
int g_sessionActive = 0;  // active index among g_sessionPairs (cycle/force)

// The tracker owns the capture/debounce settings; the settings window pushes
// changes into it live via ApplyConfig (model path still needs a restart).
CaretTracker* g_caretTracker = nullptr;
// The inference service, for the readiness poll that drives the toggle HUD from
// "Initializing..." to "ACTIVE". Only touched on the UI thread.
TranslationService* g_translator = nullptr;
// The overlay, for hot-reloading the header/dropdown labels on a Settings save.
OverlayWindow* g_overlay = nullptr;

// Map the configured language pairs onto the CaretTracker's OS-aware routing
// indices. The source language is auto-detected by the model, so we key off each
// pair's TARGET name: a pair targeting English is the "translate away from
// Russian" branch (used when typing on a Cyrillic layout), a pair targeting
// Russian is the "translate away from English" branch (Latin layout) and also
// the asymmetric reading default for selections. Robust fallbacks keep every
// index in range for single-pair or exotic configs.
CaretTracker::LanguageRouting DeriveLanguageRouting(const std::vector<LanguagePair>& pairs) {
    CaretTracker::LanguageRouting routing;
    routing.count = static_cast<int>(pairs.size());

    auto findTarget = [&](const wchar_t* needle) -> int {
        for (size_t i = 0; i < pairs.size(); ++i) {
            std::wstring t = pairs[i].target;
            std::transform(t.begin(), t.end(), t.begin(), ::towlower);
            if (t.find(needle) != std::wstring::npos) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    const int toRussian = findTarget(L"rus");  // EN->RU branch
    const int toEnglish = findTarget(L"eng");  // RU->EN branch
    const int fallback = 0;                    // always in range (count>=1 in practice)

    routing.typingLatin = (toRussian >= 0) ? toRussian : fallback;       // typing EN -> RU
    routing.typingCyrillic = (toEnglish >= 0) ? toEnglish : fallback;    // typing RU -> EN
    routing.selectionReading = (toRussian >= 0) ? toRussian : fallback;  // read foreign -> RU
    return routing;
}

// Pushes a config into the running app: rebinds hotkeys, capture settings AND
// the language pairs in memory immediately -- pairs hot-reload through every
// consumer (hook bound, overlay labels, tracker routing, engine prompts), so
// a just-saved direction translates on the very next keystroke. Only the
// model path / context size / memory budget remain engine-construction
// parameters that need a restart. Runs at startup (before the consumers
// exist -- the null checks skip them; wWinMain wires them explicitly) and on
// every Settings save (UI thread).
void ApplyConfig(const Config& config) {
    g_config = config;
    HookManager::Instance().SetCommitShortcut(config.commitShortcut);
    HookManager::Instance().SetActivationShortcut(config.activationShortcut);
    HookManager::Instance().SetCycleLanguageShortcut(config.cycleLanguageShortcut);
    if (g_caretTracker) {
        g_caretTracker->SetCaptureSettings(config.captureGranularity, config.idleTimerMs);
    }

    g_sessionPairs = config.languagePairs;
    g_sessionActive = (config.activeLanguage >= 0 &&
                       config.activeLanguage < static_cast<int>(g_sessionPairs.size()))
                          ? config.activeLanguage
                          : 0;
    HookManager::Instance().SetLanguagePairCount(static_cast<int>(g_sessionPairs.size()));
    if (g_overlay) {
        std::vector<std::wstring> labels;
        for (const LanguagePair& pair : g_sessionPairs) {
            labels.push_back(pair.label);
        }
        g_overlay->SetLanguageLabels(std::move(labels));
    }
    if (g_caretTracker) {
        // Clears a now-dangling manual pin / relocks a live typing session.
        g_caretTracker->SetLanguageRouting(DeriveLanguageRouting(g_sessionPairs));
    }
    if (g_translator) {
        std::vector<std::string> targets;
        for (const LanguagePair& pair : g_sessionPairs) {
            targets.push_back(ToUtf8(pair.target));
        }
        // Rebuilds the per-direction system prompts and swaps them into the
        // live tracker -- the engine serves the new set with no restart.
        g_translator->UpdateLanguagePairs(std::move(targets), g_sessionActive);
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
                    ShowSettingsWindow(
                        hwnd, hInst, g_config, [](const Config& c) { ApplyConfig(c); },
                        // Pre-cache: compile one direction's .bkv branch in the
                        // background (tracker worker); `done` may fire on that
                        // worker -- the settings window marshals it itself.
                        [](const std::wstring& target, std::function<void(bool, int)> done) {
                            if (g_translator) {
                                g_translator->PrecacheLanguage(ToUtf8(target), std::move(done));
                            } else if (done) {
                                done(false, 0);
                            }
                        });
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
                    case TranslationService::State::Idle:  // EnsureLoaded is imminent
                    case TranslationService::State::Loading: {
                        // Keep waiting, folding the weight-load progress into the
                        // sticky center HUD. Re-shown only when the percent moves
                        // (the poll runs at 150ms; the HUD repaint need not).
                        static int lastShownPct = -1;
                        const int pct = g_translator->LoadProgressPercent();
                        if (pct != lastShownPct) {
                            lastShownPct = pct;
                            g_caretTracker->ShowHud(L"Initializing Translation Engine... (" +
                                                        std::to_wstring(pct) + L"%)",
                                                    /*fade=*/false);
                        }
                        break;
                    }
                }
            }
            // Coarse inactivity tick: the service compares idle time against the
            // configured thresholds and hands the due stage (KV disk spill / soft
            // hibernation) to the engine-owning worker -- nothing heavy runs here.
            if (wParam == kLifecycleTimerId && g_translator) {
                g_translator->LifecycleTick(g_config.kvSpillTimeoutSec,
                                            g_config.hibernateTimeoutSec);
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
    // ApplyConfig also seeds g_sessionPairs/g_sessionActive -- the overlay,
    // tracker and service consumers below wire from that seed explicitly (they
    // do not exist yet, so ApplyConfig's own hot-reload pushes are skipped).
    g_config = ConfigStore::Load();
    ApplyConfig(g_config);

    // The inference stack. Engine construction (DirectStorage weight streaming)
    // happens on the service's own worker thread -- this constructor is instant.
    // The preview sink is late-bound below, once the CaretTracker exists.
    TranslationService::Settings svcSettings;
    svcSettings.modelDir = g_config.modelPath;
    // Each configured direction becomes a system prompt + a JIT-compiled .bkv
    // branch (the radix tree dedups the shared prefix). Only the target language
    // name reaches the service; the "RU -> EN" label is a UI concern.
    svcSettings.targetLanguages.clear();  // replace the {"English"} default
    for (const LanguagePair& pair : g_sessionPairs) {
        svcSettings.targetLanguages.push_back(ToUtf8(pair.target));
    }
    svcSettings.activeLanguage = g_sessionActive;
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
    // Launch behavior: activate-on-startup loads the engine immediately (the
    // HUD flow below tracks it); otherwise the backend is constructed but the
    // heavy VRAM weight load is deferred until the first activation toggle.
    svcSettings.deferLoad = !g_config.activateOnStartup;
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
    // The tracker owns the effective translation direction (OS-aware typing lock,
    // asymmetric selection reading, or a manual override) and pushes it to the
    // engine right before each request so the correct .bkv branch is addressed.
    trackerCallbacks.setActiveLanguage = [&translator](int index) {
        translator.SetActiveLanguage(index);
    };
    CaretTracker caretTracker(std::move(trackerCallbacks), g_config.captureGranularity,
                              g_config.idleTimerMs, DeriveLanguageRouting(g_sessionPairs));
    g_caretTracker = &caretTracker;

    // The overlay's header/dropdown chrome: give it the pair labels to display,
    // a sink to apply a manual override (dropdown or the Auto row), and a sink to
    // publish its clickable region so the global mouse hook lets those clicks
    // through instead of hiding the overlay. Labels hot-reload on a Settings
    // save via ApplyConfig (g_overlay).
    g_overlay = &overlay;
    {
        std::vector<std::wstring> labels;
        for (const LanguagePair& pair : g_sessionPairs) {
            labels.push_back(pair.label);
        }
        overlay.SetLanguageLabels(std::move(labels));
    }
    overlay.SetOverrideSink([&caretTracker](int index) {
        // -1 = Auto (clear the override, back to OS-aware routing); >=0 pins a pair.
        caretTracker.SetLanguageOverride(index);
    });
    overlay.SetInteractiveRegionSink(
        [](const RECT* rect) { HookManager::Instance().SetInteractiveRect(rect); });

    // Bound the Alt+<N> force-override hotkeys to the configured pairs (kept in
    // step by ApplyConfig on every Settings save) -- Alt+N never fires for a
    // pair index the routing has no entry for.
    HookManager::Instance().SetLanguagePairCount(static_cast<int>(g_sessionPairs.size()));

    // Late-bound sink: translation deltas/finals flow back into the tracker's
    // state machine (thread-safe enqueue), which decides what the overlay shows.
    // The service's generation counter already suppresses most stale deliveries;
    // the tracker's phase check catches the rest.
    translator.SetPreviewSink(
        [&caretTracker](std::uint64_t /*gen*/, const std::wstring& text, bool done) {
            caretTracker.OnPreviewResult(text, done);
        });

    // Lifecycle HUDs. Fired on the tracker's worker thread; ShowHud is a
    // lock+notify enqueue, so this stays marshaling-only like the preview sink.
    // The KV spill (stage 1) is deliberately silent -- it is invisible to the
    // user by design (pages fault back in on demand).
    translator.SetLifecycleSink([&caretTracker](TranslationService::LifecycleEvent event) {
        switch (event) {
            case TranslationService::LifecycleEvent::KvSpilled:
                break;
            case TranslationService::LifecycleEvent::Hibernated:
                caretTracker.ShowHud(L"Translation Engine: hibernated (VRAM freed)",
                                     /*fade=*/true);
                break;
            case TranslationService::LifecycleEvent::WakingUp:
                // Sticky banner: it covers the whole PCIe DMA restore and is
                // replaced by the Awake HUD (or the translation overlay) below.
                caretTracker.ShowHud(L"Waking up...", /*fade=*/false);
                break;
            case TranslationService::LifecycleEvent::Awake:
                caretTracker.ShowHud(L"Translation Mode: ACTIVE", /*fade=*/true);
                break;
        }
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
            translator.EnsureLoaded();  // deferred-launch path: first toggle loads the model
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
    // Language switcher (Alt+Shift+L, Mode ACTIVE only): advance the active
    // direction, push it to the tracker, and flash the new pair in a HUD. The
    // next typing/selection translation uses it (the user can flip direction
    // right before checking a reverse translation).
    callbacks.onCycleLanguage = [&caretTracker]() {
        const int count = static_cast<int>(g_config.languagePairs.size());
        if (count <= 0) {
            return;
        }
        g_config.activeLanguage = (g_config.activeLanguage + 1) % count;
        // Cycling pins a manual override (like the overlay dropdown): the tracker
        // owns pushing it to the engine, so OS-aware auto-routing is suspended
        // until the user picks the "Auto" row in the dropdown.
        caretTracker.SetLanguageOverride(g_config.activeLanguage);
        caretTracker.ShowHud(L"Language: " + g_config.languagePairs[g_config.activeLanguage].label,
                             /*fade=*/true);
    };
    // Force-override hotkeys (Alt+1, Alt+2, ...): pin a specific direction,
    // suppressing OS-layout auto-routing. The hook already bounds the index to a
    // configured pair; SetLanguageOverride pins it (the header shows "[Pinned]").
    callbacks.onForceLanguage = [&caretTracker](int index) {
        if (index < 0 || index >= static_cast<int>(g_config.languagePairs.size())) {
            return;
        }
        g_config.activeLanguage = index;
        caretTracker.SetLanguageOverride(index);
        caretTracker.ShowHud(L"Pinned: " + g_config.languagePairs[index].label, /*fade=*/true);
    };

    const bool hookInstalled = HookManager::Instance().Install(std::move(callbacks));
    if (!hookInstalled) {
        MessageBoxW(nullptr, L"Failed to install the low-level keyboard/mouse hooks.", L"poc_overlay",
                    MB_ICONERROR);
        return 1;
    }

    // Launch behavior. activateOnStartup: the engine load is already running
    // (deferLoad=false); enter Translation Mode now, hold the sticky
    // "Initializing..." center HUD, and let the readiness poll flip it to
    // ACTIVE (fading out) exactly like a manual toggle. Otherwise: start
    // inactive with a flashed OFF banner -- the backend is alive, but no model
    // weights touch VRAM until the first Alt+Shift+T.
    if (g_config.activateOnStartup) {
        HookManager::Instance().SetTranslationMode(true);
        caretTracker.ShowHud(L"Initializing Translation Engine...", /*fade=*/false);
        SetTimer(controller, kReadyTimerId, 150, nullptr);  // -> SetActive + ACTIVE fade
    } else {
        caretTracker.ShowHud(L"Translation Mode: OFF", /*fade=*/true);
    }
    // The dual-stage inactivity state machine's heartbeat (a no-op until the
    // engine is Ready; see LifecycleTick).
    SetTimer(controller, kLifecycleTimerId, kLifecycleTickMs, nullptr);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    HookManager::Instance().Uninstall();
    g_caretTracker = nullptr;
    g_translator = nullptr;
    g_overlay = nullptr;

    // Teardown ordering: join the service worker FIRST, so its sink can never
    // fire into the CaretTracker while (or after) the tracker is destroyed
    // during stack unwinding. After Shutdown(), the tracker's own teardown may
    // still call RequestPreview/CancelPending -- both no-op against a stopped
    // service that is destroyed later (declared earlier) in this scope.
    translator.Shutdown();

    CoUninitialize();
    return static_cast<int>(msg.wParam);
}
