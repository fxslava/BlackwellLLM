// poc_overlay: the live overlay translator -- the full pipeline from OS input
// capture to real BlackwellEngine inference and back into the focused control:
//
//   [UI thread]      WH_KEYBOARD_LL hook (lightweight TRIGGER + fallback buffer)
//                        |  RequestUpdate(fallbackText)     [condvar signal]
//                        v
//   [COM/UIA thread]  CaretTracker: reads the REAL text from the focused control
//                      (IUIAutomationTextPattern document-up-to-caret, or
//                      IUIAutomationValuePattern), plus the caret rectangle.
//                        |  PostUpdate(text, caretPos)      [PostMessage]
//                        |  word boundary -> RequestPreview [condvar signal]
//                        |  Ctrl+Enter    -> TranslateBlocking (bounded wait)
//                        v
//   [agent worker]    TranslationService: BlackwellLLMAdapter (CUDA engine +
//                      tokenizer) driven by the 1-step AgentOrchestrator loops.
//                        |  PostTranslation(text)           [PostMessage]
//                        v
//   [UI thread]       OverlayWindow: layered, click-through, D2D-rendered popup
//                      whose string END anchors just above the caret.
//
// A system tray icon provides the only way to exit (no visible window/taskbar
// entry otherwise).
#include <windows.h>
#include <objbase.h>

#include <chrono>
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

// How long a Ctrl+Enter commit may block the CaretTracker STA thread waiting
// for the engine before falling back to injecting the source unchanged.
constexpr std::chrono::seconds kCommitTimeout{5};

// The live, in-memory application config. Loaded at startup and rewritten by the
// settings window; only ever touched on the UI thread.
Config g_config;

// Pushes a config into the running app: rebinds hotkeys in memory immediately.
// (Model path / context size will be handed to the inference engine here later.)
void ApplyConfig(const Config& config) {
    g_config = config;
    HookManager::Instance().SetCommitShortcut(config.commitShortcut);
    HookManager::Instance().SetActivationShortcut(config.activationShortcut);
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
    // (A model-path change in the settings window currently requires a restart;
    // the engine is built once, on the service's worker thread.)
    g_config = ConfigStore::Load();
    ApplyConfig(g_config);

    // The inference stack. Constructed FIRST so it is destroyed LAST: the
    // CaretTracker STA thread calls into it, so the service must outlive the
    // tracker. Engine construction (DirectStorage weight streaming) happens on
    // the service's own worker thread -- this constructor is instant.
    TranslationService::Settings svcSettings;
    svcSettings.modelDir = g_config.modelPath;
    svcSettings.promptCacheDir = g_config.modelPath.empty()
                                     ? std::wstring()
                                     : g_config.modelPath + L"\\prompt_cache";
    svcSettings.maxSeqLen = static_cast<size_t>(g_config.contextSize);
    svcSettings.temperature = g_config.temperature;
    svcSettings.topP = g_config.topP;
    svcSettings.commitMaxNewTokens = g_config.maxTokens;
    TranslationService translator(
        std::move(svcSettings),
        // PreviewSink: runs on the agent worker thread; PostTranslation is
        // PostMessage-based, so forwarding straight through is thread-safe.
        [&overlay](std::uint64_t /*gen*/, const std::wstring& text, bool /*done*/) {
            overlay.PostTranslation(text);
        });

    // Real translation applied at commit: blocks the STA thread (bounded by
    // kCommitTimeout) while the agent worker runs the 2-iteration commit loop.
    // On timeout / engine error the segment is injected unchanged -- Ctrl+Enter
    // must never destroy what the user typed.
    auto commitTransform = [&translator](const std::wstring& segment,
                                         const std::wstring& context) -> std::wstring {
        return translator.TranslateBlocking(segment, context, kCommitTimeout)
            .value_or(segment);
    };
    // Brackets the injection window on the hook so it flags synthetic input.
    auto injectionGuard = [](bool active) { HookManager::Instance().SetInjecting(active); };

    // CaretTracker resolves the authoritative text + caret position on ITS OWN
    // (COM/UIA STA) thread, and also performs the commit replacement there (so the
    // UIA interfaces never cross apartments). The update callback runs on that
    // thread and must not touch the overlay directly -- PostUpdate() marshals the
    // result back to the overlay window's owning (UI) thread via PostMessage.
    CaretTracker caretTracker(
        [&](const CaretUpdate& update) {
            if (update.caretFound) {
                overlay.PostUpdate(update.text, update.caretScreenPos);
            }
            // Word boundary on the authoritative UIA text: the preview gate.
            // Latest-wins coalescing + the generation counter inside the
            // service make bursts of boundaries cheap and self-cancelling.
            if (update.wordBoundary && !update.text.empty()) {
                translator.RequestPreview(update.text, update.inferenceContext);
            }
        },
        commitTransform, injectionGuard);

    // The hook callbacks run inline in the global hook chain on THIS thread and
    // must stay fast. They only hand work to the caret tracker's queue (an O(1)
    // lock + condvar notify) -- the UIA round-trip, D2D paint, and the whole
    // 3-tier replacement happen off this call stack entirely.
    HookManager::Callbacks callbacks;
    callbacks.onTrigger = [&caretTracker](const std::wstring& fallbackText, bool wordBoundary) {
        caretTracker.RequestUpdate(fallbackText, wordBoundary);  // "field changed -> poll UIA"
    };
    callbacks.onCommit = [&](const std::wstring& fallbackText) {
        overlay.Hide();
        caretTracker.RequestCommit(fallbackText);  // re-resolve + replace on the STA thread
    };
    callbacks.onReset = [&]() {
        overlay.Hide();
        translator.CancelPending();  // focus/caret moved: a preview result could
                                     // never be shown, so stop paying for it
    };

    const bool hookInstalled = HookManager::Instance().Install(std::move(callbacks));
    if (!hookInstalled) {
        MessageBoxW(nullptr, L"Failed to install the low-level keyboard/mouse hooks.", L"poc_overlay",
                    MB_ICONERROR);
        return 1;
    }

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &ControllerWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"BlackwellPocOverlayController";
    RegisterClassExW(&wc);

    // Invisible controller window: exists solely to own the tray icon and
    // receive its callback messages (WM_TRAYICON).
    HWND controller = CreateWindowExW(0, wc.lpszClassName, L"poc_overlay controller", WS_OVERLAPPED,
                                       0, 0, 0, 0, nullptr, nullptr, hInstance, nullptr);

    TrayIcon trayIcon(controller, kTrayIconId);
    SetWindowLongPtrW(controller, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&trayIcon));
    trayIcon.Create(L"Blackwell PoC: Input Hook -> UIA Caret -> Overlay");

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    HookManager::Instance().Uninstall();
    CoUninitialize();
    return static_cast<int>(msg.wParam);
}
