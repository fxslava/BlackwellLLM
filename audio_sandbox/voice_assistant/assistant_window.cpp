// -----------------------------------------------------------------------------
// assistant_window.cpp — WebView2 host for the voice_assistant messenger UI.
// See assistant_window.hpp for the lifetime and threading contract.
// -----------------------------------------------------------------------------
#include "assistant_window.hpp"

#include <objbase.h>
#include <shobjidl.h>  // IFileOpenDialog (FOS_PICKFOLDERS)
#include <wrl.h>
#include <WebView2.h>

#include <nlohmann/json.hpp>

#include <cstdio>    // fprintf (hotkey registration warnings)
#include <cstdlib>   // atoi (F-key parsing)
#include <string>
#include <vector>

#include "audio_devices.h"   // enumerate_audio_devices, for GET /api/audio-devices
#include "hotkey_spec.hpp"   // rt::Hotkey, parse_hotkey, hotkey_is_down (one table)

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
using nlohmann::json;

namespace rt {
namespace {

constexpr wchar_t kClassName[] = L"BlackwellVoiceAssistantWindow";

// Virtual host mapped onto <exe dir>\web. Serving the UI over https://<host>/
// rather than file:// keeps it a normal same-origin document, so relative
// fetches, modules and localStorage all behave the way they would on a real
// site -- the same arrangement poc_overlay's settings window uses.
constexpr wchar_t kAssetHost[] = L"appassets.voiceassistant";
constexpr wchar_t kAssetUrl[]  = L"https://appassets.voiceassistant/index.html";

// UI-thread wake-up: "the cross-thread queue has something in it". Carries no
// payload, so a post that outlives the window leaks nothing.
constexpr UINT kMsgDrain = WM_APP + 1;

// RegisterHotKey ids. Small and window-scoped, so they cannot collide with
// another app's global hotkeys (the OS namespaces them per window).
constexpr int kHotkeyTalk   = 1;
constexpr int kHotkeyCancel = 2;
constexpr int kHotkeyShow   = 3;

// Hold-to-talk release watchdog. WM_HOTKEY is a key-DOWN notification and there
// is no matching key-up message, so a hold has to be closed by polling the chord.
// 60 ms is comfortably below the point where a released key still feels held, and
// the timer only runs while a chord is actually down.
constexpr UINT_PTR kPttTimerId = 1;
constexpr UINT     kPttPollMs  = 60;

// Hotkey, parse_hotkey, vk_from_name and hotkey_is_down now live in
// hotkey_spec.hpp: the settings validator and the console overlay parse the same
// chords, and a second copy of the key table is a second thing to keep in step.
using rt::Hotkey;
using rt::hotkey_is_down;
using rt::parse_hotkey;

// Uppercase ASCII compare; the page sends what the user typed and case is not a
// meaningful distinction in a key name.
bool iequals(const std::string& a, const char* b) { return rt::hotkey_iequals(a, b); }

std::string to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()),
                                      nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), out.data(), n,
                        nullptr, nullptr);
    return out;
}

std::wstring from_utf8(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                                      nullptr, 0);
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::wstring assets_dir() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir(exe);
    if (const size_t slash = dir.find_last_of(L"\\/"); slash != std::wstring::npos) {
        dir.resize(slash + 1);
    }
    return dir + L"web";
}

bool file_exists(const std::wstring& path) {
    const DWORD attr = GetFileAttributesW(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

// =============================================================================
// The audio-device list the settings dropdowns are built from.
// =============================================================================
// DELIVERED OVER THE postMessage CHANNEL, not as a fetch endpoint, and that is
// forced rather than chosen. There is no HTTP server in this process: the UI is
// a WebView2 document on a VIRTUAL host mapped to <exe dir>\web, and requests
// the mapping satisfies are served off disk WITHOUT raising
// WebResourceRequested -- measured, see the handler in create_webview(). So a
// same-origin `fetch("/api/audio-devices")` is precisely the request shape that
// cannot be intercepted; it would 404 against the folder.
//
// The alternative that would have preserved a fetch() call site is an unmapped
// hostname, which DOES raise the event. Rejected: it means inventing a domain
// we do not own and relying on interception to stop a real DNS lookup from
// leaving the machine. A request-id round trip over the channel the app already
// uses for settings, mic and browse costs one message type and leaks nothing.
//
// The shape the page consumes, and it is deliberately NOT the C++ struct:
//
//   { "outputs": ["(system default)", "Speakers (...)", ...],
//     "inputs":  ["(system default)", "Microphone (...)", ...] }
//
// Entry 0 of each list is the SYNTHETIC "(system default)" row, so a select's
// option positions are `device index + 1` and the sentinel -1 lands at position
// 0. Building that offset into the payload rather than the page keeps the two
// definitions of "which row means default" from drifting: the page never does
// arithmetic on a device index it did not receive.
//
// NAMES TRAVEL VERBATIM, with no "* default" decoration folded in. The page
// saves the string it was given straight back as output_device_name, so any
// decoration would have to be stripped client-side by pattern -- and a device
// genuinely named like the pattern would then be unsaveable. Which endpoint
// Windows currently prefers is already expressed by row 0 meaning "follow it",
// and the startup console listing still marks it for diagnostics.
json audio_devices_payload() {
    const auto build = [](AudioDeviceKind kind) {
        json list = json::array();
        list.push_back("(system default)");
        for (const AudioDeviceInfo& d : enumerate_audio_devices(kind)) {
            list.push_back(d.name);
        }
        return list;
    };
    json out;
    out["outputs"] = build(AudioDeviceKind::Playback);
    out["inputs"]  = build(AudioDeviceKind::Capture);
    // One line per serve, and it earns its place: "the dropdown is empty" has
    // two very different causes -- the page never asked (a transport problem)
    // and the machine reported nothing (a backend problem) -- and they are
    // indistinguishable from the UI. A missing line means the former; a line
    // reading 0/0 means the latter. The counts exclude the synthetic
    // "(system default)" row so they match what the startup listing printed.
    std::printf("[webview] audio.devices -> page (%zu playback, %zu capture)\n",
                out["outputs"].size() - 1, out["inputs"].size() - 1);
    std::fflush(stdout);
    return out;
}

}  // namespace

// The COM pointers live here so <WebView2.h> and <wrl.h> stay out of the header.
struct AssistantWindow::Impl {
    ComPtr<ICoreWebView2Controller> controller;
    ComPtr<ICoreWebView2> webview;
};

AssistantWindow::AssistantWindow(const wchar_t* title)
    : title_(title != nullptr ? title : L"Assistant"), impl_(new Impl()) {}

AssistantWindow::~AssistantWindow() {
    if (impl_ != nullptr) {
        if (impl_->controller) impl_->controller->Close();
        delete impl_;
        impl_ = nullptr;
    }
    if (com_initialized_) CoUninitialize();
}

bool AssistantWindow::create(int client_w, int client_h) {
    // WebView2 is STA-only. main() claims the apartment before anything else
    // touches COM, so this normally returns S_FALSE (already STA); asking again
    // here keeps the class usable on its own. RPC_E_CHANGED_MODE means someone
    // got in first with COINIT_MULTITHREADED and the thread is stuck in the MTA
    // -- environment creation WILL fail, and this is the only place that knows
    // why, so record it for the error message.
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    com_initialized_ = SUCCEEDED(co);
    mta_conflict_ = (co == RPC_E_CHANGED_MODE);
    if (mta_conflict_) {
        std::fprintf(stderr,
                     "[webview] FATAL: this thread is in the multi-threaded COM apartment; "
                     "WebView2 requires STA. Something initialized COM as MTA before the "
                     "window was created.\n");
    }

    static bool registered = false;
    if (!registered) {
        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = &AssistantWindow::WndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = kClassName;
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        // Dark neutral: the page paints its own background, and a white flash
        // before the first frame reads as a bug on a dark UI.
        wc.hbrBackground = CreateSolidBrush(RGB(23, 24, 28));
        RegisterClassExW(&wc);
        registered = true;
    }

    RECT rc{0, 0, client_w, client_h};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    hwnd_ = CreateWindowExW(0, kClassName, title_.c_str(), WS_OVERLAPPEDWINDOW,
                            CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left,
                            rc.bottom - rc.top, nullptr, nullptr,
                            GetModuleHandleW(nullptr), this);
    if (hwnd_ == nullptr) return false;
    post_target_.store(hwnd_, std::memory_order_release);

    // Show the frame BEFORE creating the WebView so it attaches to a window that
    // already has a real (non-zero) client area.
    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);

    // The tray, before the page loads and for the same reason the hotkeys are:
    // it needs only the HWND, and it is the thing that makes the app reachable
    // if WebView2 never comes up at all.
    tray_.set_callbacks([this] {
        TrayCallbacks tc;
        tc.on_show = [this] { summon(); };
        tc.on_hide = [this] { hide_to_tray_now(); };
        tc.on_toggle_console = [this] { if (cb_.on_toggle_console) cb_.on_toggle_console(); };
        // The real quit path. DestroyWindow rather than a hide, and NOT
        // PostQuitMessage directly: WM_DESTROY is where the hotkeys are released,
        // the WebView2 controller is closed and the tray icon is removed, and
        // skipping it would leak all three.
        tc.on_exit = [this] {
            hide_to_tray_ = false;   // so the WM_CLOSE below is not intercepted
            if (cb_.on_exit) cb_.on_exit();
            if (hwnd_ != nullptr) DestroyWindow(hwnd_);
        };
        return tc;
    }());
    tray_.create(hwnd_, L"Assistant");
    apply_window_settings(settings_);

    // Hotkeys need only the HWND, so they are live before the page has loaded --
    // the window is usable by keyboard while Chromium is still starting up.
    register_hotkeys();
    // Push-to-talk means the chord opens the mic; anything else would leave it
    // open until the first press, which is the opposite of what the mode is for.
    // Through set_listening, NOT by assigning listening_: the pipeline has to be
    // muted for real, and only the callback does that. The page has not loaded
    // yet, so the event it would post is redundant -- push_settings() re-asserts
    // the state on NavigationCompleted.
    if (settings_.hotkey_push_to_talk) set_listening(false, /*tell_page=*/false);

    create_webview();

    // AFTER the WebView has been asked for, not before. The frame was shown
    // above so Chromium attaches to a window with a real (non-zero) client area;
    // hiding it now keeps that sizing while putting the app straight into the
    // tray, which is what a shortcut in the Startup folder wants. Hiding FIRST
    // would attach the WebView to a zero-area window and the first summon would
    // show a blank frame.
    if (settings_.start_minimized && tray_.alive()) {
        ShowWindow(hwnd_, SW_HIDE);
    }
    return true;
}

void AssistantWindow::create_webview() {
    std::wstring user_data;
    wchar_t local[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) > 0) {
        user_data = std::wstring(local) + L"\\BlackwellVoiceAssistant\\WebView2";
    }

    const HRESULT hr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, user_data.empty() ? nullptr : user_data.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result) || env == nullptr) {
                    report_webview_unavailable("CreateCoreWebView2Environment (callback)",
                                               FAILED(result) ? result : E_POINTER);
                    return S_OK;
                }
                env->CreateCoreWebView2Controller(
                    hwnd_,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this](HRESULT r2, ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(r2) || controller == nullptr) {
                                report_webview_unavailable("CreateCoreWebView2Controller",
                                                           FAILED(r2) ? r2 : E_POINTER);
                                return S_OK;
                            }
                            impl_->controller = controller;
                            const HRESULT gw =
                                impl_->controller->get_CoreWebView2(&impl_->webview);
                            if (!impl_->webview) {
                                report_webview_unavailable("get_CoreWebView2", gw);
                                return S_OK;
                            }

                            // Register handlers BEFORE navigating.
                            EventRegistrationToken token{};
                            impl_->webview->add_WebMessageReceived(
                                Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                    [this](ICoreWebView2*,
                                           ICoreWebView2WebMessageReceivedEventArgs* args)
                                        -> HRESULT {
                                        LPWSTR raw = nullptr;
                                        if (SUCCEEDED(args->get_WebMessageAsJson(&raw)) &&
                                            raw != nullptr) {
                                            on_web_message(raw);
                                            CoTaskMemFree(raw);
                                        }
                                        return S_OK;
                                    })
                                    .Get(),
                                &token);

                            impl_->webview->add_NavigationCompleted(
                                Callback<ICoreWebView2NavigationCompletedEventHandler>(
                                    [this](ICoreWebView2*,
                                           ICoreWebView2NavigationCompletedEventArgs* args)
                                        -> HRESULT {
                                        BOOL ok = FALSE;
                                        if (args != nullptr) args->get_IsSuccess(&ok);
                                        if (!ok) {
                                            MessageBoxW(hwnd_,
                                                        L"The assistant UI failed to load.",
                                                        L"Voice Assistant",
                                                        MB_ICONWARNING | MB_OK);
                                            return S_OK;
                                        }
                                        page_ready_ = true;
                                        push_settings();
                                        drain_pending();  // flush everything queued
                                                          // while the page loaded
                                        return S_OK;
                                    })
                                    .Get(),
                                &token);

                            // ---- revalidation for the REMOTE assets -----------
                            // The build REDEPLOYS web/ next to the exe on every
                            // compile, but the WebView keeps a persistent HTTP
                            // cache -- so a rebuilt style.css could lose to the
                            // copy cached by the previous run, rendering a NEW
                            // index.html against an OLD stylesheet.
                            //
                            // THIS HANDLER DOES NOT ACTUALLY COVER THAT CASE, and
                            // the comment used to claim it did. Measured
                            // 2026-08-04 by logging every URI that reaches here:
                            // the ONLY requests raised are the external ones
                            // (cdn.jsdelivr.net for marked/highlight.js).
                            // index.html, app.js and style.css never appear --
                            // requests satisfied by SetVirtualHostNameToFolderMapping
                            // are served straight off disk and do not raise
                            // WebResourceRequested at all.
                            //
                            // Which also means it cannot host an /api/... route:
                            // a same-origin fetch to the virtual host is exactly
                            // the kind of request that never gets here. That is
                            // why the audio-device list travels over the
                            // postMessage channel instead (see on_web_message).
                            //
                            // Kept because the CDN assets are genuinely cached and
                            // genuinely worth revalidating. If local-asset
                            // staleness ever bites, the fix is a cache-busting
                            // query string in index.html, not this handler.
                            ComPtr<ICoreWebView2> wv = impl_->webview;
                            wv->AddWebResourceRequestedFilter(
                                L"*", COREWEBVIEW2_WEB_RESOURCE_CONTEXT_ALL);
                            EventRegistrationToken res_token{};
                            wv->add_WebResourceRequested(
                                Callback<ICoreWebView2WebResourceRequestedEventHandler>(
                                    [this](ICoreWebView2*,
                                           ICoreWebView2WebResourceRequestedEventArgs* args)
                                        -> HRESULT {
                                        if (args == nullptr) return S_OK;
                                        ComPtr<ICoreWebView2WebResourceRequest> req;
                                        if (FAILED(args->get_Request(&req)) || !req) return S_OK;
                                        ComPtr<ICoreWebView2HttpRequestHeaders> headers;
                                        if (FAILED(req->get_Headers(&headers)) || !headers) {
                                            return S_OK;
                                        }
                                        // no-cache, not no-store: a 304 is still
                                        // allowed, so an unchanged asset costs a
                                        // stat rather than a re-read.
                                        headers->SetHeader(L"Cache-Control", L"no-cache");
                                        return S_OK;
                                    })
                                    .Get(),
                                &res_token);

                            // A shipping messenger is not a browser: no context
                            // menu, no dev-tools chrome, no zoom-by-scroll.
                            ComPtr<ICoreWebView2Settings> settings;
                            if (SUCCEEDED(impl_->webview->get_Settings(&settings)) && settings) {
                                settings->put_AreDefaultContextMenusEnabled(FALSE);
                                settings->put_IsZoomControlEnabled(FALSE);
                                settings->put_AreDevToolsEnabled(FALSE);
                                settings->put_IsStatusBarEnabled(FALSE);
                            }

                            impl_->controller->put_IsVisible(TRUE);
                            resize_to_client();

                            if (!map_assets_and_navigate()) {
                                MessageBoxW(hwnd_,
                                            L"UI assets were not found next to the "
                                            L"executable (expected a 'web' folder with "
                                            L"index.html). Rebuild or reinstall.",
                                            L"Voice Assistant", MB_ICONWARNING | MB_OK);
                                DestroyWindow(hwnd_);
                            }
                            return S_OK;
                        })
                        .Get());
                return S_OK;
            })
            .Get());

    // Synchronous failure: the loader could not even reach the runtime, or this
    // thread is in the wrong apartment.
    if (FAILED(hr)) {
        report_webview_unavailable("CreateCoreWebView2EnvironmentWithOptions", hr);
    }
}

bool AssistantWindow::map_assets_and_navigate() {
    const std::wstring dir = assets_dir();
    if (!file_exists(dir + L"\\index.html")) return false;

    ComPtr<ICoreWebView2_3> wv3;
    if (SUCCEEDED(impl_->webview.As(&wv3)) && wv3) {
        wv3->SetVirtualHostNameToFolderMapping(
            kAssetHost, dir.c_str(), COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW);
        return SUCCEEDED(impl_->webview->Navigate(kAssetUrl));
    }
    // Ancient runtime without ICoreWebView2_3: fall back to file://.
    std::wstring url = L"file:///" + dir + L"\\index.html";
    for (wchar_t& c : url) {
        if (c == L'\\') c = L'/';
    }
    return SUCCEEDED(impl_->webview->Navigate(url.c_str()));
}

void AssistantWindow::report_webview_unavailable(const char* stage, HRESULT hr) {
    std::fprintf(stderr, "[webview] %s failed: hr=0x%08lX\n", stage,
                 static_cast<unsigned long>(hr));
    std::fflush(stderr);

    std::wstring msg;
    if (mta_conflict_) {
        // The cause we can actually name. Telling this user to install a runtime
        // they already have is worse than useless -- it hides a code bug.
        msg = L"The assistant's interface could not start because COM was already "
              L"initialized in the multi-threaded apartment on this thread.\n\n"
              L"WebView2 requires a single-threaded apartment (STA). This is a bug in "
              L"the app's startup order, not a problem with your machine.";
    } else {
        msg = L"Could not initialize the Edge WebView2 runtime, which this app's "
              L"interface is built on.\n\nIf the Microsoft Edge WebView2 Runtime is not "
              L"installed, install it and start the assistant again.";
    }
    wchar_t detail[160];
    wsprintfW(detail, L"\n\nStage: %hs\nError: 0x%08lX", stage,
              static_cast<unsigned long>(hr));
    msg += detail;

    MessageBoxW(hwnd_, msg.c_str(), L"Voice Assistant", MB_ICONERROR | MB_OK);
    if (hwnd_ != nullptr) DestroyWindow(hwnd_);
}

// ---- hotkeys ------------------------------------------------------------------
// GLOBAL, by design: the point of an activation keybind on a voice assistant is
// that it works while another application has focus. RegisterHotKey is the right
// primitive for that -- unlike a WH_KEYBOARD_LL hook it neither sees nor can
// swallow anything but the chord it asked for, which is what keeps this app off
// the list of things that quietly observe everything you type.
//
// A chord another process already owns simply fails to register. That is
// reported, not retried: silently rebinding to something the user did not choose
// would be worse than the hotkey not working.

void AssistantWindow::register_hotkeys() {
    if (hwnd_ == nullptr) return;
    unregister_hotkeys();

    const struct { int id; const std::string& spec; const char* label; } binds[] = {
        {kHotkeyTalk,   settings_.hotkey_talk,   "talk"},
        {kHotkeyCancel, settings_.hotkey_cancel, "cancel"},
        {kHotkeyShow,   settings_.hotkey_show,   "show"},
    };
    for (const auto& b : binds) {
        const Hotkey hk = parse_hotkey(b.spec);
        if (hk.vk == 0) continue;   // "" or unparseable -> this binding is off
        // MOD_NOREPEAT: auto-repeat while a chord is held would otherwise fire
        // the toggle dozens of times a second. Hold-to-talk gets its release from
        // the watchdog timer, never from repeats.
        if (!RegisterHotKey(hwnd_, b.id, hk.mods | MOD_NOREPEAT, hk.vk)) {
            std::fprintf(stderr,
                         "[hotkey] WARN: could not register %s hotkey '%s' -- another "
                         "application already owns that combination.\n",
                         b.label, b.spec.c_str());
        }
    }
    hotkeys_registered_ = true;
}

void AssistantWindow::unregister_hotkeys() {
    if (hwnd_ == nullptr || !hotkeys_registered_) return;
    UnregisterHotKey(hwnd_, kHotkeyTalk);
    UnregisterHotKey(hwnd_, kHotkeyCancel);
    UnregisterHotKey(hwnd_, kHotkeyShow);
    hotkeys_registered_ = false;
}

void AssistantWindow::set_listening(bool on, bool tell_page) {
    if (listening_ == on) return;
    listening_ = on;
    if (cb_.on_mic_toggle) cb_.on_mic_toggle(on);
    if (!tell_page) return;
    json out;
    out["type"] = "mic";
    out["on"] = on;
    post_event(out.dump());
}

void AssistantWindow::on_hotkey(int id) {
    switch (id) {
        case kHotkeyTalk:
            if (settings_.hotkey_push_to_talk) {
                // HOLD: open the mic now and let the watchdog close it. Re-arming
                // the timer on a repeat press is harmless and keeps the hold alive
                // if the OS did deliver one.
                ptt_held_ = true;
                set_listening(true, /*tell_page=*/true);
                SetTimer(hwnd_, kPttTimerId, kPttPollMs, nullptr);
            } else {
                set_listening(!listening_, /*tell_page=*/true);
            }
            return;
        case kHotkeyCancel:
            if (cb_.on_cancel) cb_.on_cancel();
            return;
        case kHotkeyShow:
            // ONE definition of "bring it back", shared with the tray's left
            // click and its Show item -- see summon().
            summon();
            return;
        default:
            return;
    }
}

// ---- summoning, hiding, and the window flags --------------------------------
void AssistantWindow::summon() {
    if (hwnd_ == nullptr) return;
    // THE ORDER MATTERS. A window hidden with SW_HIDE is not iconic, so the
    // IsIconic branch alone would leave it hidden; and SW_RESTORE on a hidden
    // window does not un-hide it. Doing the un-hide first and the restore second
    // is what makes this work from BOTH states -- hidden-to-tray and minimized.
    ShowWindow(hwnd_, SW_SHOW);
    if (IsIconic(hwnd_)) ShowWindow(hwnd_, SW_RESTORE);

    // SetForegroundWindow is subject to the foreground lock, so it can
    // legitimately do nothing when another application is actively being typed
    // into. BringWindowToTop raises it in the Z-order regardless, which is the
    // visible half of what the user asked for and is why both are called rather
    // than either alone.
    BringWindowToTop(hwnd_);
    SetForegroundWindow(hwnd_);
    // Re-asserted on every summon: a window that was hidden while topmost can
    // come back below the Z-order position it had, and re-stating the flag is
    // cheaper than reasoning about when that happens.
    SetWindowPos(hwnd_, always_on_top_ ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
}

void AssistantWindow::hide_to_tray_now() {
    if (hwnd_ == nullptr) return;
    ShowWindow(hwnd_, SW_HIDE);
    // ONCE. The balloon exists so the first hide does not read as a crash; an
    // app that says it every time is worse than one that never did.
    if (!tray_hint_shown_ && tray_.alive()) {
        tray_hint_shown_ = true;
        tray_.notify(L"Assistant is still running",
                     L"The window was hidden, not closed. Click the tray icon to bring it "
                     L"back, or use Exit to quit.");
    }
}

void AssistantWindow::apply_window_settings(const AssistantSettings& s) {
    hide_to_tray_ = s.minimize_to_tray;
    always_on_top_ = s.always_on_top;
    if (hwnd_ == nullptr) return;
    // SWP_NOACTIVATE: applying a setting must not steal focus, and this runs on
    // every save.
    SetWindowPos(hwnd_, always_on_top_ ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
}

void AssistantWindow::poll_push_to_talk() {
    if (!ptt_held_) {
        KillTimer(hwnd_, kPttTimerId);
        return;
    }
    if (hotkey_is_down(parse_hotkey(settings_.hotkey_talk))) return;   // still held
    ptt_held_ = false;
    KillTimer(hwnd_, kPttTimerId);
    set_listening(false, /*tell_page=*/true);
}

// ---- cross-thread event queue ----------------------------------------------

void AssistantWindow::post_event(std::string json) {
    {
        std::lock_guard<std::mutex> lk(q_mu_);
        pending_.push_back(std::move(json));
    }
    // Fails harmlessly once the window is gone; the queue dies with the object.
    if (HWND h = post_target_.load(std::memory_order_acquire); h != nullptr) {
        PostMessageW(h, kMsgDrain, 0, 0);
    }
}

void AssistantWindow::drain_pending() {
    if (!page_ready_ || !impl_->webview) return;   // hold until the page can receive
    for (;;) {
        std::string msg;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            if (pending_.empty()) return;
            msg = std::move(pending_.front());
            pending_.pop_front();
        }
        impl_->webview->PostWebMessageAsJson(from_utf8(msg).c_str());
    }
}

// ---- page -> app -------------------------------------------------------------

void AssistantWindow::on_web_message(const std::wstring& message_json) {
    try {
        const json j = json::parse(to_utf8(message_json));
        const std::string type = j.value("type", std::string());

        if (type == "ready") {
            push_settings();
            return;
        }
        // Deliberately NOT also pushed on "ready": the page asks for itself, at
        // load and on every settings open. One trigger means one enumeration per
        // request and a page-side watchdog that is always armed -- pushing here
        // too would enumerate twice at startup and leave the unsolicited copy
        // unaccounted for by that watchdog.
        if (type == "audio.devices.request") {
            push_audio_devices();
            return;
        }
        if (type == "audio.test_tone") {
            if (cb_.on_test_tone) cb_.on_test_tone();
            return;
        }
        // ---- the session sidebar --------------------------------------------
        // Pure routing: this class does not own the store, so every one of these
        // is forwarded and answered asynchronously by the app (see the callback
        // declarations). Note there is no "session.rename" -- a conversation is
        // identified by how it opened, which needs no UI to maintain.
        if (type == "session.list_request") {
            if (cb_.on_session_list_request) cb_.on_session_list_request();
            return;
        }
        if (type == "session.select") {
            const std::string id = j.value("id", std::string());
            if (!id.empty() && cb_.on_session_select) cb_.on_session_select(id);
            return;
        }
        if (type == "session.new") {
            if (cb_.on_session_new) cb_.on_session_new();
            return;
        }
        if (type == "session.delete") {
            const std::string id = j.value("id", std::string());
            if (!id.empty() && cb_.on_session_delete) cb_.on_session_delete(id);
            return;
        }
        // THE HOT PATH. Note what is absent: no AssistantSettings is built, no
        // requires_restart() is consulted, and control never reaches the
        // settings.save branch below. A message that cannot express a
        // restart-tier field cannot cause a restart.
        if (type == "audio.hot_update") {
            AudioHotUpdate u;
            if (j.contains("output_device_index") && j["output_device_index"].is_number()) {
                u.output_device_index = j["output_device_index"].get<int>();
            }
            if (j.contains("input_device_index") && j["input_device_index"].is_number()) {
                u.input_device_index = j["input_device_index"].get<int>();
            }
            if (j.contains("output_device_name") && j["output_device_name"].is_string()) {
                u.output_device_name = j["output_device_name"].get<std::string>();
                u.has_output_name = true;
            }
            if (j.contains("input_device_name") && j["input_device_name"].is_string()) {
                u.input_device_name = j["input_device_name"].get<std::string>();
                u.has_input_name = true;
            }
            if (j.contains("tts_volume") && j["tts_volume"].is_number()) {
                u.tts_volume = j["tts_volume"].get<float>();
            }
            if (j.contains("mic_gain") && j["mic_gain"].is_number()) {
                u.mic_gain = j["mic_gain"].get<float>();
            }
            if (j.contains("tts_muted") && j["tts_muted"].is_boolean()) {
                u.tts_muted = j["tts_muted"].get<bool>() ? rt::AudioHotUpdate::Tri::On
                                                         : rt::AudioHotUpdate::Tri::Off;
            }
            if (cb_.on_audio_hot_update) cb_.on_audio_hot_update(u);
            return;
        }
        if (type == "send") {
            const std::string text = j.value("text", std::string());
            if (!text.empty() && cb_.on_send_text) cb_.on_send_text(text);
            return;
        }
        // The Stop button in the composer. Routed to the SAME callback the cancel
        // hotkey raises -- see AssistantWindowCallbacks::on_cancel. The page is
        // told nothing back: it returns to its Send state when the answer stream
        // ends (remote.final), which is the app's word for "this turn is over"
        // and is what arrives whether the stop was honoured, raced, or landed
        // after the last token.
        if (type == "interrupt_generation") {
            if (cb_.on_cancel) cb_.on_cancel();
            return;
        }
        if (type == "mic") {
            // The page reports what the BUTTON now shows; it is already painted,
            // so the echo back is suppressed.
            set_listening(j.value("on", true), /*tell_page=*/false);
            return;
        }
        if (type == "browse") {
            browse_for_path(j.value("target", std::string("model_dir")));
            return;
        }
        if (type == "restart") {
            if (cb_.on_restart) cb_.on_restart();
            return;
        }
        if (type != "settings.save") return;

        // Start from what is RUNNING and overlay whatever the payload carries, so
        // a tab the user never opened cannot blank the fields it owns. from_json
        // clamps, so nothing out of range gets past this line.
        const json& p = j.contains("payload") && j["payload"].is_object() ? j["payload"] : j;
        AssistantSettings next = settings_;
        from_json(p, next);

        // THE restart question, answered in one place (settings_store.hpp) and
        // handed to both the app and the page -- so the "needs restart" badge in
        // the modal can never disagree with what the app actually did.
        const bool live_only = !requires_restart(settings_, next);
        // The system prompt is live but NOT free: it is a KV cache rebuild on the
        // engine thread. Fire it only on a real change -- re-prefilling an
        // identical prompt would throw away a warm prefix cache and several
        // seconds for nothing.
        const bool prompt_changed = settings_.system_prompt != next.system_prompt;

        settings_ = next;
        if (cb_.on_settings_apply) cb_.on_settings_apply(next, live_only);
        // AFTER on_settings_apply: that call is what publishes the new hotkey
        // strings to the app, and re-registering reads them from settings_.
        register_hotkeys();
        // Entering push-to-talk closes the mic; the chord is what opens it. Doing
        // this on the flip (rather than continuously) leaves a user who toggled
        // the mic back on by hand alone.
        if (settings_.hotkey_push_to_talk && !ptt_held_) {
            set_listening(false, /*tell_page=*/true);
        }
        if (prompt_changed && cb_.on_system_prompt_apply) {
            cb_.on_system_prompt_apply(settings_.system_prompt);
        }
        push_settings();

        json out;
        out["type"] = "settings.saved";
        out["live_only"] = live_only;
        // The modal keeps its "precomputing…" state until the engine answers.
        out["prompt_rebuilding"] = prompt_changed;
        post_event(out.dump());
    } catch (const std::exception&) {
        // Ignore malformed messages rather than take down the UI thread.
    }
}

void AssistantWindow::post_system_prompt_applied(bool ok, unsigned tokens,
                                                 const std::string& detail) {
    json out;
    out["type"] = "system_prompt.applied";
    out["ok"] = ok;
    out["tokens"] = tokens;
    out["detail"] = detail;
    post_event(out.dump());
}

void AssistantWindow::push_settings() {
    json out;
    out["type"] = "settings";
    out["payload"] = to_json(settings_);
    // The restart-tier field names travel WITH the payload: the page computes its
    // "needs restart" banner from this list instead of a parallel set of markup
    // attributes, so the banner cannot disagree with requires_restart().
    out["restart_fields"] = restart_fields();
    post_event(out.dump());
    // The mic button follows C++ state (the talk hotkey can flip it while the
    // page is not even visible), so re-assert it whenever settings are pushed.
    json mic;
    mic["type"] = "mic";
    mic["on"] = listening_;
    post_event(mic.dump());
}

void AssistantWindow::post_audio_level(float level) {
    // Dropped, not queued, when the page is not ready: a meter reading is only
    // interesting while it is current, and a backlog of stale amplitudes would
    // animate the bar through history the moment the modal opened.
    if (!page_ready_) return;
    json j;
    j["type"] = "audio.level";
    j["level"] = level;
    post_event(j.dump());
}

void AssistantWindow::push_audio_devices() {
    // ENUMERATION RUNS HERE, ON THE UI THREAD, and that is deliberate: it opens
    // a fresh ma_context per call (~tens of ms on WASAPI), so the list reflects
    // hardware plugged in since launch rather than a snapshot taken at startup.
    // The cost is paid when the page asks -- page load and every settings open.
    json out = audio_devices_payload();
    out["type"] = "audio.devices";
    post_event(out.dump());
}

void AssistantWindow::browse_for_path(const std::string& target) {
    // The native picker, on this UI thread. A checkpoint path typed by hand into
    // a web input is the single most common way to mis-configure this app.
    ComPtr<IFileDialog> dialog;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dialog)))) {
        return;
    }
    // FOLDER unless the field names a single file. Everything else this app
    // browses for is a checkpoint DIRECTORY, so folder is the default and the
    // exceptions are listed rather than the rule being restated at each site.
    const bool pick_file = (target == "whisper_model_path");

    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST |
                       (pick_file ? DWORD{0} : DWORD{FOS_PICKFOLDERS}));
    if (pick_file) {
        // A GGML model is a .bin, and pointing this at a .safetensors checkpoint
        // is the predictable mistake -- the filter is what makes it hard to make.
        // "All files" stays available because the extension is a convention, not
        // a format requirement (.gguf exists in the wild too).
        static const COMDLG_FILTERSPEC kGgmlFilters[] = {
            {L"GGML/GGUF Whisper model", L"*.bin;*.gguf"},
            {L"All files", L"*.*"},
        };
        dialog->SetFileTypes(ARRAYSIZE(kGgmlFilters), kGgmlFilters);
        dialog->SetFileTypeIndex(1);
    }
    if (FAILED(dialog->Show(hwnd_))) return;   // cancelled

    ComPtr<IShellItem> item;
    if (FAILED(dialog->GetResult(&item)) || !item) return;
    PWSTR path = nullptr;
    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path != nullptr) {
        json out;
        out["type"] = "browsed";
        out["target"] = target;
        out["path"] = to_utf8(path);
        CoTaskMemFree(path);
        post_event(out.dump());
    }
}

// ---- window plumbing ---------------------------------------------------------

void AssistantWindow::resize_to_client() {
    if (!impl_->controller) return;
    RECT rc{};
    GetClientRect(hwnd_, &rc);
    impl_->controller->put_Bounds(rc);
}

void AssistantWindow::run_message_loop() {
    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

LRESULT CALLBACK AssistantWindow::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }
    auto* self = reinterpret_cast<AssistantWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (self == nullptr) return DefWindowProcW(hwnd, msg, wParam, lParam);
    return self->handle(hwnd, msg, wParam, lParam);
}

LRESULT AssistantWindow::handle(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_SIZE:
            resize_to_client();
            return 0;
        case kMsgDrain:
            drain_pending();
            return 0;
        case WM_HOTKEY:
            on_hotkey(static_cast<int>(wParam));
            return 0;

        // ---- the tray, and the [X] that no longer quits ----------------------
        // ONE forwarded message is the entire tray integration; TrayIcon answers
        // everything else itself (see its header).
        case kTrayCallbackMessage:
            if (tray_.handle_message(wParam, lParam)) return 0;
            break;

        case WM_CLOSE:
            // THE INTERCEPT. [X] hides to the notification area instead of
            // terminating, because this is a background utility a hotkey
            // summons -- an assistant that has to be relaunched every time its
            // window is dismissed is not one.
            //
            // GATED ON THE TRAY ACTUALLY EXISTING, and that is not defensive
            // padding: if Shell_NotifyIcon refused the icon (a notification area
            // that has not started yet at login), hiding here would make the app
            // unreachable by every route at once -- no window, no icon, and only
            // Task Manager left. Falling through to a real close is the correct
            // behaviour in exactly that case.
            if (hide_to_tray_ && tray_.alive()) {
                hide_to_tray_now();
                return 0;
            }
            break;   // DefWindowProc -> WM_DESTROY -> PostQuitMessage
        case WM_TIMER:
            // Only OUR timer is claimed; anything else belongs to a component that
            // set it on this HWND and must reach the default handler.
            if (wParam == kPttTimerId) {
                poll_push_to_talk();
                return 0;
            }
            break;
        case WM_GETMINMAXINFO: {
            // Below this the composer and the bubbles start fighting for room.
            auto* mmi = reinterpret_cast<MINMAXINFO*>(lParam);
            mmi->ptMinTrackSize.x = 420;
            mmi->ptMinTrackSize.y = 480;
            return 0;
        }
        case WM_DESTROY:
            page_ready_ = false;
            // BEFORE the window dies. An icon whose owner is destroyed without a
            // NIM_DELETE leaves a ghost in the notification area until the user
            // hovers over it -- the most recognisable symptom of an app that did
            // not clean up after itself.
            tray_.destroy();
            // Global hotkeys are a PROCESS-WIDE registration: releasing them here
            // (rather than leaving it to the window dying) is what lets the
            // relaunched instance re-register the same chords immediately after a
            // Save & Restart.
            KillTimer(hwnd, kPttTimerId);
            unregister_hotkeys();
            // Stop the producers posting BEFORE the view goes away, so nothing
            // races the controller teardown below.
            post_target_.store(nullptr, std::memory_order_release);
            if (impl_ != nullptr && impl_->controller) {
                impl_->controller->Close();
                impl_->controller.Reset();
                impl_->webview.Reset();
            }
            hwnd_ = nullptr;
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace rt
