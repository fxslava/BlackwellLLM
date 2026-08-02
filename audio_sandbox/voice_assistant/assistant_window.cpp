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

#include <string>
#include <vector>

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
    // WebView2 is STA. Initialising here (rather than in main) keeps the
    // apartment requirement with the only code that actually has one.
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    com_initialized_ = SUCCEEDED(co);

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
    create_webview();
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
                    report_webview_unavailable();
                    return S_OK;
                }
                env->CreateCoreWebView2Controller(
                    hwnd_,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this](HRESULT r2, ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(r2) || controller == nullptr) {
                                report_webview_unavailable();
                                return S_OK;
                            }
                            impl_->controller = controller;
                            impl_->controller->get_CoreWebView2(&impl_->webview);
                            if (!impl_->webview) {
                                report_webview_unavailable();
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

    if (FAILED(hr)) report_webview_unavailable();
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

void AssistantWindow::report_webview_unavailable() {
    MessageBoxW(hwnd_,
                L"Could not initialize the Edge WebView2 runtime, which this app's "
                L"interface is built on.\n\nInstall the Microsoft Edge WebView2 Runtime "
                L"and start the assistant again.",
                L"Voice Assistant", MB_ICONERROR | MB_OK);
    if (hwnd_ != nullptr) DestroyWindow(hwnd_);
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
        if (type == "send") {
            const std::string text = j.value("text", std::string());
            if (!text.empty() && cb_.on_send_text) cb_.on_send_text(text);
            return;
        }
        if (type == "mic") {
            if (cb_.on_mic_toggle) cb_.on_mic_toggle(j.value("on", true));
            return;
        }
        if (type == "browse") {
            browse_for_folder(j.value("target", std::string("model_dir")));
            return;
        }
        if (type == "restart") {
            if (cb_.on_restart) cb_.on_restart();
            return;
        }
        if (type != "settings.save") return;

        const json& p = j.contains("payload") && j["payload"].is_object() ? j["payload"] : j;
        AssistantSettings next = settings_;
        next.model_dir      = p.value("model_dir", next.model_dir);
        next.audio_head     = p.value("audio_head", next.audio_head);
        next.projector_path = p.value("projector_path", next.projector_path);
        next.data_dir       = p.value("data_dir", next.data_dir);
        next.system_prompt  = p.value("system_prompt", next.system_prompt);
        next.device_id      = p.value("device_id", next.device_id);
        next.max_context    = p.value("max_context", next.max_context);
        next.simulated      = p.value("simulated", next.simulated);
        next.neural_vad     = p.value("neural_vad", next.neural_vad);
        next.loopback_capture = p.value("loopback_capture", next.loopback_capture);
        next.vad_threshold  = p.value("vad_threshold", next.vad_threshold);
        next.context_mode   = p.value("context_mode", next.context_mode);
        next.history_budget_tokens =
            p.value("history_budget_tokens", next.history_budget_tokens);
        next.live_streaming = p.value("live_streaming", next.live_streaming);
        clamp_settings(next);

        // THE restart question, answered in one place (settings_store.hpp) and
        // handed to both the app and the page -- so the "needs restart" badge in
        // the modal can never disagree with what the app actually did.
        const bool live_only = !settings_.requires_restart(next);
        settings_ = next;
        if (cb_.on_settings_apply) cb_.on_settings_apply(next, live_only);
        push_settings();

        json out;
        out["type"] = "settings.saved";
        out["live_only"] = live_only;
        post_event(out.dump());
    } catch (const std::exception&) {
        // Ignore malformed messages rather than take down the UI thread.
    }
}

void AssistantWindow::push_settings() {
    json p;
    p["model_dir"]      = settings_.model_dir;
    p["audio_head"]     = settings_.audio_head;
    p["projector_path"] = settings_.projector_path;
    p["data_dir"]       = settings_.data_dir;
    p["system_prompt"]  = settings_.system_prompt;
    p["device_id"]      = settings_.device_id;
    p["max_context"]    = settings_.max_context;
    p["simulated"]      = settings_.simulated;
    p["neural_vad"]     = settings_.neural_vad;
    p["loopback_capture"] = settings_.loopback_capture;
    p["vad_threshold"]  = settings_.vad_threshold;
    p["context_mode"]   = settings_.context_mode;
    p["history_budget_tokens"] = settings_.history_budget_tokens;
    p["live_streaming"] = settings_.live_streaming;

    json out;
    out["type"] = "settings";
    out["payload"] = std::move(p);
    post_event(out.dump());
}

void AssistantWindow::browse_for_folder(const std::string& target) {
    // The native picker, on this UI thread. A checkpoint path typed by hand into
    // a web input is the single most common way to mis-configure this app.
    ComPtr<IFileDialog> dialog;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dialog)))) {
        return;
    }
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
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
        case WM_GETMINMAXINFO: {
            // Below this the composer and the bubbles start fighting for room.
            auto* mmi = reinterpret_cast<MINMAXINFO*>(lParam);
            mmi->ptMinTrackSize.x = 420;
            mmi->ptMinTrackSize.y = 480;
            return 0;
        }
        case WM_DESTROY:
            page_ready_ = false;
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
