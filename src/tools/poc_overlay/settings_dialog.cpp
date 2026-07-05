#include "settings_dialog.h"

#include <objbase.h>
#include <shlobj.h>     // SHCreateDirectoryExW
#include <shobjidl.h>   // IFileDialog folder picker
#include <wrl.h>
#include <WebView2.h>

#include <nlohmann/json.hpp>

#include <cstdlib>  // _wtoi
#include <string>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
using nlohmann::json;

namespace {

constexpr wchar_t kClassName[] = L"BlackwellPocSettingsWindow";
HWND g_openWindow = nullptr;  // single-instance guard

// Virtual host the WebView2 maps onto the on-disk assets folder (web/ next to
// the exe). Navigating https://<host>/settings.html serves settings.html and its
// linked settings.css / settings.js exactly like a real site, so the UI markup,
// styling, and logic all live in editable web files -- NOT in this C++ source.
// The C++ side is now a pure state bridge: it pushes the Config as JSON on load
// and applies the JSON the page posts back on Save.
constexpr wchar_t kAssetHost[] = L"appassets.blackwell";
constexpr wchar_t kAssetUrl[] = L"https://appassets.blackwell/settings.html";

// <exe dir>\web -- where CMake deploys settings.html/.css/.js next to the binary.
std::wstring AssetsDir() {
    wchar_t exePath[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring dir(exePath);
    const size_t slash = dir.find_last_of(L"\\/");
    if (slash != std::wstring::npos) {
        dir.resize(slash + 1);
    }
    return dir + L"web";
}

bool FileExists(const std::wstring& path) {
    const DWORD attr = GetFileAttributesW(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

// Owns the settings window + its WebView2. Heap-allocated; self-deletes on
// WM_NCDESTROY.
class SettingsWindow {
public:
    SettingsWindow(Config config, std::function<void(const Config&)> onApply)
        : config_(std::move(config)), onApply_(std::move(onApply)) {}

    bool Create(HWND owner, HINSTANCE hInstance) {
        // Initial size is a close guess; the page reports its real height on load
        // and ResizeToContentHeight() fits the window exactly (no scrollbars).
        hwnd_ = CreateWindowExW(0, kClassName, L"Blackwell PoC - Settings",
                                 WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                 CW_USEDEFAULT, CW_USEDEFAULT, 620, 680, owner, nullptr, hInstance,
                                 this);
        if (!hwnd_) {
            return false;
        }
        g_openWindow = hwnd_;
        // Show the host frame immediately so the WebView2 attaches to a window
        // that already has a real (non-zero) client area.
        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
        CreateWebView();
        return true;
    }

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
    void CreateWebView();
    void OnControllerCreated(ICoreWebView2Controller* controller);
    bool MapAssetsAndNavigate();  // virtual-host map -> Navigate; false = assets missing
    void OnWebMessage(const std::wstring& messageJson);
    void PushConfigToJs();
    void ResizeToClient();
    void ResizeToContentHeight(int cssHeight);  // fit the window to the page (no scrollbars)
    void FitWindowToContent();                  // measure the page via ExecuteScript, then fit
    void ReportWebViewUnavailable();
    void BrowseForModelFolder();                // native folder picker -> JS

    HWND hwnd_ = nullptr;
    ComPtr<ICoreWebView2Controller> controller_;
    ComPtr<ICoreWebView2> webview_;
    Config config_;
    std::function<void(const Config&)> onApply_;
};

void SettingsWindow::CreateWebView() {
    // WebView2 needs a writable user-data folder; keep it under %LOCALAPPDATA%.
    std::wstring userData;
    wchar_t local[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) > 0) {
        userData = std::wstring(local) + L"\\BlackwellPocOverlay\\WebView2";
    }

    const HRESULT hr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, userData.empty() ? nullptr : userData.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result) || !env) {
                    ReportWebViewUnavailable();
                    return S_OK;
                }
                env->CreateCoreWebView2Controller(
                    hwnd_,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this](HRESULT r2, ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(r2) || !controller) {
                                ReportWebViewUnavailable();
                                return S_OK;
                            }
                            OnControllerCreated(controller);
                            return S_OK;
                        })
                        .Get());
                return S_OK;
            })
            .Get());

    // Synchronous failure (e.g. loader can't locate the runtime at all).
    if (FAILED(hr)) {
        ReportWebViewUnavailable();
    }
}

void SettingsWindow::OnControllerCreated(ICoreWebView2Controller* controller) {
    controller_ = controller;
    controller_->get_CoreWebView2(&webview_);
    if (!webview_) {
        ReportWebViewUnavailable();
        return;
    }

    // Register handlers BEFORE navigating so NavigationCompleted / messages fire.
    EventRegistrationToken token{};
    webview_->add_WebMessageReceived(
        Callback<ICoreWebView2WebMessageReceivedEventHandler>(
            [this](ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                LPWSTR raw = nullptr;
                if (SUCCEEDED(args->get_WebMessageAsJson(&raw)) && raw) {
                    OnWebMessage(raw);
                    CoTaskMemFree(raw);
                }
                return S_OK;
            })
            .Get(),
        &token);

    // Push the config when the page loads; surface a concrete error if the
    // navigation fails instead of leaving a silent blank page.
    webview_->add_NavigationCompleted(
        Callback<ICoreWebView2NavigationCompletedEventHandler>(
            [this](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs* args) -> HRESULT {
                BOOL ok = FALSE;
                if (args) {
                    args->get_IsSuccess(&ok);
                }
                if (ok) {
                    PushConfigToJs();
                    FitWindowToContent();
                } else {
                    COREWEBVIEW2_WEB_ERROR_STATUS status =
                        COREWEBVIEW2_WEB_ERROR_STATUS_UNKNOWN;
                    if (args) {
                        args->get_WebErrorStatus(&status);
                    }
                    wchar_t msg[160];
                    wsprintfW(msg,
                              L"WebView2 failed to load the settings page (error status %d).",
                              static_cast<int>(status));
                    MessageBoxW(hwnd_, msg, L"Settings", MB_ICONWARNING | MB_OK);
                }
                return S_OK;
            })
            .Get(),
        &token);

    controller_->put_IsVisible(TRUE);
    ResizeToClient();
    SetForegroundWindow(hwnd_);

    if (!MapAssetsAndNavigate()) {
        MessageBoxW(hwnd_,
                    L"Settings UI assets were not found next to the executable "
                    L"(expected a 'web' folder with settings.html). Reinstall or rebuild.",
                    L"Settings", MB_ICONWARNING | MB_OK);
        DestroyWindow(hwnd_);
    }
}

bool SettingsWindow::MapAssetsAndNavigate() {
    const std::wstring dir = AssetsDir();
    if (!FileExists(dir + L"\\settings.html")) {
        return false;  // assets not deployed -- caller reports it
    }
    // Map the virtual host onto the assets folder so https://<host>/settings.html
    // serves the file (and its linked .css/.js) with normal same-origin fetches.
    // Requires ICoreWebView2_3 (WebView2 SDK 1.0.774+); on an ancient runtime we
    // simply fall back to a file:// navigation below.
    ComPtr<ICoreWebView2_3> wv3;
    if (SUCCEEDED(webview_.As(&wv3)) && wv3) {
        wv3->SetVirtualHostNameToFolderMapping(
            kAssetHost, dir.c_str(), COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW);
        return SUCCEEDED(webview_->Navigate(kAssetUrl));
    }
    std::wstring fileUrl = L"file:///" + dir + L"\\settings.html";
    for (wchar_t& c : fileUrl) {
        if (c == L'\\') c = L'/';
    }
    return SUCCEEDED(webview_->Navigate(fileUrl.c_str()));
}

void SettingsWindow::PushConfigToJs() {
    json j;
    j["type"] = "load";
    j["activation"] = {{"modifiers", config_.activationShortcut.modifiers},
                       {"vk", config_.activationShortcut.vk}};
    j["commit"] = {{"modifiers", config_.commitShortcut.modifiers},
                   {"vk", config_.commitShortcut.vk}};
    j["cycle"] = {{"modifiers", config_.cycleLanguageShortcut.modifiers},
                  {"vk", config_.cycleLanguageShortcut.vk}};
    // Language pairs (index-aligned with the Alt+<N> force hotkeys and the engine
    // branches). label = HUD/overlay text, target = the language name the model
    // translates INTO (source is auto-detected).
    json pairs = json::array();
    for (const LanguagePair& p : config_.languagePairs) {
        pairs.push_back({{"label", ToUtf8(p.label)}, {"target", ToUtf8(p.target)}});
    }
    j["languagePairs"] = std::move(pairs);
    j["activeLanguage"] = config_.activeLanguage;
    j["modelPath"] = ToUtf8(config_.modelPath);
    j["contextSize"] = config_.contextSize;
    j["temperature"] = config_.temperature;
    j["topP"] = config_.topP;
    j["maxTokens"] = config_.maxTokens;
    j["captureGranularity"] = ToString(config_.captureGranularity);
    j["idleTimerMs"] = config_.idleTimerMs;
    j["vramCacheBlocks"] = config_.vramCacheBlocks;
    j["ramTierBlocks"] = config_.ramTierBlocks;
    j["diskSpillEnabled"] = config_.diskSpillEnabled;
    j["diskSpillBlocks"] = config_.diskSpillBlocks;
    // Show the resolved default so the user sees where the file actually goes.
    j["spillFilePath"] = ToUtf8(config_.spillFilePath.empty()
                                    ? ConfigStore::DefaultSpillPath()
                                    : config_.spillFilePath);
    if (webview_) {
        webview_->PostWebMessageAsJson(FromUtf8(j.dump()).c_str());
    }
}

void SettingsWindow::OnWebMessage(const std::wstring& messageJson) {
    try {
        const json j = json::parse(ToUtf8(messageJson));
        const std::string type = j.value("type", std::string());

        if (type == "resize") {
            ResizeToContentHeight(j.value("height", 0));
            return;
        }
        if (type == "browse") {
            BrowseForModelFolder();
            return;
        }
        if (type != "save") {
            return;
        }
        if (j.contains("activation")) {
            config_.activationShortcut.modifiers = j["activation"].value("modifiers", 0u);
            config_.activationShortcut.vk = j["activation"].value("vk", 0u);
        }
        if (j.contains("commit")) {
            config_.commitShortcut.modifiers = j["commit"].value("modifiers", 0u);
            config_.commitShortcut.vk = j["commit"].value("vk", 0u);
        }
        if (j.contains("cycle")) {
            config_.cycleLanguageShortcut.modifiers = j["cycle"].value("modifiers", 0u);
            config_.cycleLanguageShortcut.vk = j["cycle"].value("vk", 0u);
        }
        // Language pairs: replace the whole list (add/remove is done in the UI).
        // Drop rows with no target; keep the old set if the UI sent nothing usable
        // so a stray message can never wipe the user's directions.
        if (j.contains("languagePairs") && j["languagePairs"].is_array()) {
            std::vector<LanguagePair> pairs;
            for (const auto& e : j["languagePairs"]) {
                LanguagePair p;
                p.label = FromUtf8(e.value("label", std::string()));
                p.target = FromUtf8(e.value("target", std::string()));
                if (!p.target.empty()) {
                    if (p.label.empty()) p.label = p.target;  // fall back to the target name
                    pairs.push_back(std::move(p));
                }
            }
            if (!pairs.empty()) config_.languagePairs = std::move(pairs);
        }
        config_.activeLanguage = j.value("activeLanguage", config_.activeLanguage);
        if (config_.activeLanguage < 0 ||
            config_.activeLanguage >= static_cast<int>(config_.languagePairs.size())) {
            config_.activeLanguage = 0;
        }
        config_.modelPath = FromUtf8(j.value("modelPath", std::string()));
        config_.contextSize = j.value("contextSize", config_.contextSize);
        config_.temperature = j.value("temperature", config_.temperature);
        config_.topP = j.value("topP", config_.topP);
        config_.maxTokens = j.value("maxTokens", config_.maxTokens);
        config_.captureGranularity = CaptureGranularityFromString(
            j.value("captureGranularity", std::string()), config_.captureGranularity);
        config_.idleTimerMs = j.value("idleTimerMs", config_.idleTimerMs);
        if (config_.idleTimerMs < 100) config_.idleTimerMs = 100;
        config_.vramCacheBlocks = j.value("vramCacheBlocks", config_.vramCacheBlocks);
        config_.ramTierBlocks = j.value("ramTierBlocks", config_.ramTierBlocks);
        config_.diskSpillEnabled = j.value("diskSpillEnabled", config_.diskSpillEnabled);
        config_.diskSpillBlocks = j.value("diskSpillBlocks", config_.diskSpillBlocks);
        config_.spillFilePath = FromUtf8(j.value("spillFilePath", std::string()));
        if (config_.vramCacheBlocks < 0) config_.vramCacheBlocks = 0;
        if (config_.ramTierBlocks < 0) config_.ramTierBlocks = 0;
        if (config_.diskSpillBlocks < 0) config_.diskSpillBlocks = 0;

        ConfigStore::Save(config_);  // persist to config.json
        if (onApply_) {
            onApply_(config_);  // rebind hotkeys / model in memory immediately
        }
        if (webview_) {
            webview_->PostWebMessageAsJson(LR"({"type":"saved"})");
        }
    } catch (const std::exception&) {
        // Ignore malformed messages rather than crash the UI thread.
    }
}

void SettingsWindow::ResizeToClient() {
    if (!controller_) {
        return;
    }
    RECT rc{};
    GetClientRect(hwnd_, &rc);
    controller_->put_Bounds(rc);
}

void SettingsWindow::FitWindowToContent() {
    if (!webview_) {
        return;
    }
    // Measure the laid-out page height directly from C++ (no dependency on a JS
    // postMessage round-trip firing). scrollHeight comes back as a JSON number.
    webview_->ExecuteScript(
        L"document.body.scrollHeight",
        Callback<ICoreWebView2ExecuteScriptCompletedHandler>(
            [this](HRESULT error, LPCWSTR resultJson) -> HRESULT {
                if (SUCCEEDED(error) && resultJson) {
                    const int cssHeight = _wtoi(resultJson);
                    if (cssHeight > 0) {
                        ResizeToContentHeight(cssHeight + 2);  // +2: DPI rounding guard
                    }
                }
                return S_OK;
            })
            .Get());
}

void SettingsWindow::ResizeToContentHeight(int cssHeight) {
    if (cssHeight <= 0) {
        return;
    }
    // The page reports its height in CSS pixels; WebView2 maps 1 CSS px to
    // dpi/96 physical px, so scale to physical to get an exact, scrollbar-free fit.
    const UINT dpi = GetDpiForWindow(hwnd_);
    const int clientHeight = MulDiv(cssHeight, static_cast<int>(dpi), 96);

    RECT client{};
    GetClientRect(hwnd_, &client);
    const int clientWidth = client.right - client.left;

    RECT rect{0, 0, clientWidth, clientHeight};
    const DWORD style = static_cast<DWORD>(GetWindowLongPtrW(hwnd_, GWL_STYLE));
    const DWORD exStyle = static_cast<DWORD>(GetWindowLongPtrW(hwnd_, GWL_EXSTYLE));
    AdjustWindowRectExForDpi(&rect, style, FALSE, exStyle, dpi);

    SetWindowPos(hwnd_, nullptr, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    ResizeToClient();  // WM_SIZE also does this, but keep the WebView in lockstep
}

void SettingsWindow::BrowseForModelFolder() {
    ComPtr<IFileDialog> dialog;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dialog)))) {
        return;
    }
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    if (FAILED(dialog->Show(hwnd_))) {
        return;  // user cancelled
    }
    ComPtr<IShellItem> item;
    if (FAILED(dialog->GetResult(&item)) || !item) {
        return;
    }
    PWSTR path = nullptr;
    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) {
        json out;
        out["type"] = "modelPath";
        out["path"] = ToUtf8(path);
        if (webview_) {
            webview_->PostWebMessageAsJson(FromUtf8(out.dump()).c_str());
        }
        CoTaskMemFree(path);
    }
}

void SettingsWindow::ReportWebViewUnavailable() {
    MessageBoxW(hwnd_,
                L"Could not initialize WebView2.\n\nInstall the Microsoft Edge WebView2 Runtime "
                L"and reopen Settings.",
                L"Settings", MB_ICONWARNING | MB_OK);
    DestroyWindow(hwnd_);
}

LRESULT CALLBACK SettingsWindow::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }
    auto* self = reinterpret_cast<SettingsWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
        case WM_SIZE:
            if (self) {
                self->ResizeToClient();
            }
            return 0;
        case WM_DESTROY:
            if (self && self->controller_) {
                self->controller_->Close();
            }
            g_openWindow = nullptr;
            return 0;
        case WM_NCDESTROY:
            delete self;  // matches the `new` in ShowSettingsWindow
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

void ShowSettingsWindow(HWND owner, HINSTANCE hInstance, const Config& current,
                        std::function<void(const Config&)> onApply) {
    // Single instance: just refocus if it is already open.
    if (g_openWindow && IsWindow(g_openWindow)) {
        SetForegroundWindow(g_openWindow);
        return;
    }

    static bool registered = false;
    if (!registered) {
        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = &SettingsWindow::WndProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = kClassName;
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        RegisterClassExW(&wc);
        registered = true;
    }

    auto* window = new SettingsWindow(current, std::move(onApply));
    if (!window->Create(owner, hInstance)) {
        delete window;
    }
    // Ownership now lives with the HWND (freed on WM_NCDESTROY).
}
