#include "settings_dialog.h"

#include <objbase.h>
#include <shlobj.h>     // SHCreateDirectoryExW
#include <shobjidl.h>   // IFileDialog folder picker
#include <wrl.h>
#include <WebView2.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
using nlohmann::json;

namespace {

constexpr wchar_t kClassName[] = L"BlackwellPocSettingsWindow";
HWND g_openWindow = nullptr;  // single-instance guard

// Settings UI. Stored as a NARROW UTF-8 raw string literal (R"HTML(...)HTML") so
// there is zero backslash/quote escaping to get wrong, then converted to UTF-16
// for NavigateToString. The hotkey fields are readonly <input> boxes that capture
// keydown and record { modifiers, vk } (event.keyCode maps to Windows VK_*). All
// IPC handlers are wrapped in try/catch so a single bad message can't blank the UI.
constexpr char kHtml[] = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    color-scheme: dark;
    --bg:#191a1f; --card:#232530; --card2:#1d1f27; --line:#33363f;
    --fg:#e9eaee; --muted:#9aa0ad; --accent:#5b8cff; --accent2:#3f6fe0;
    --ok:#4ade80; --field:#14151a;
  }
  *{ box-sizing:border-box; }
  html,body{ margin:0; height:100%; }
  body{
    background:var(--bg); color:var(--fg);
    font-family:'Segoe UI',system-ui,-apple-system,sans-serif; font-size:13px;
    padding:20px; -webkit-user-select:none; user-select:none;
  }
  h1{ font-size:18px; margin:0; font-weight:650; letter-spacing:.2px; }
  .subtitle{ color:var(--muted); font-size:12px; margin:2px 0 18px; }
  .card{ background:var(--card); border:1px solid var(--line);
    border-radius:12px; padding:16px 16px 18px; margin-bottom:14px; }
  .card > h2{ font-size:12px; text-transform:uppercase; letter-spacing:.6px;
    color:var(--muted); margin:0 0 14px; font-weight:600; }
  .field{ margin-bottom:12px; }
  .field:last-child{ margin-bottom:0; }
  label{ display:block; color:var(--muted); font-size:11.5px; margin-bottom:6px; }
  input{
    width:100%; background:var(--field); color:var(--fg);
    border:1px solid var(--line); border-radius:8px; padding:9px 11px;
    font-size:13px; font-family:inherit; outline:none; transition:border-color .12s, box-shadow .12s;
  }
  input:focus{ border-color:var(--accent); box-shadow:0 0 0 3px rgba(91,140,255,.22); }
  input[readonly]{ cursor:pointer; }
  input[readonly]:focus{ border-color:var(--accent); box-shadow:0 0 0 3px rgba(91,140,255,.28); }
  .grid{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .footer{ display:flex; align-items:center; gap:14px; margin-top:4px; }
  button{
    background:var(--accent); color:#fff; border:none; border-radius:8px;
    padding:10px 22px; font-size:13px; font-weight:600; cursor:pointer;
    font-family:inherit; transition:background .12s;
  }
  button:hover{ background:var(--accent2); }
  button:active{ transform:translateY(1px); }
  #toast{ color:var(--ok); font-size:12.5px; font-weight:600; opacity:0;
    transform:translateX(-6px); transition:opacity .2s, transform .2s; }
  #toast.show{ opacity:1; transform:translateX(0); }
  .hint{ color:var(--muted); font-size:11px; margin-top:6px; }
  .pathrow{ display:flex; gap:8px; align-items:stretch; }
  .pathrow input{ flex:1 1 auto; min-width:0; }
  button.secondary{ background:#2b2e39; color:var(--fg); border:1px solid var(--line);
    padding:9px 14px; font-weight:600; white-space:nowrap; flex:0 0 auto; }
  button.secondary:hover{ background:#333747; }
  html,body{ overflow-x:hidden; }
</style>
</head>
<body>
  <h1>Blackwell Overlay</h1>
  <div class="subtitle">Live translation assistant &mdash; settings</div>

  <div class="card">
    <h2>Shortcuts</h2>
    <div class="field">
      <label for="activation">Activation shortcut (toggle assistant on / off)</label>
      <input type="text" id="activation" readonly placeholder="Click, then press keys">
    </div>
    <div class="field">
      <label for="commit">Commit shortcut (replace typed text with translation)</label>
      <input type="text" id="commit" readonly placeholder="Click, then press keys">
    </div>
  </div>

  <div class="card">
    <h2>Model &amp; inference</h2>
    <div class="field">
      <label for="modelPath">Model path (weights directory)</label>
      <div class="pathrow">
        <input type="text" id="modelPath" spellcheck="false" placeholder="C:\models\my-model">
        <button type="button" id="browse" class="secondary">Browse&hellip;</button>
      </div>
    </div>
    <div class="grid">
      <div class="field">
        <label for="contextSize">Context size (tokens)</label>
        <input type="number" id="contextSize" min="512" max="1048576" step="512">
      </div>
      <div class="field">
        <label for="maxTokens">Max tokens</label>
        <input type="number" id="maxTokens" min="1" max="1048576" step="1">
      </div>
      <div class="field">
        <label for="temperature">Temperature (0.0 &ndash; 2.0)</label>
        <input type="number" id="temperature" min="0" max="2" step="0.1">
      </div>
      <div class="field">
        <label for="topP">Top P (0.0 &ndash; 1.0)</label>
        <input type="number" id="topP" min="0" max="1" step="0.05">
      </div>
    </div>
  </div>

  <div class="footer">
    <button id="save" type="button">Save</button>
    <span id="toast">Saved</span>
  </div>

<script>
  "use strict";
  const HK = { SHIFT:1, CONTROL:2, ALT:4 };
  const MOD_VK = [16, 17, 18, 91, 92]; // Shift/Ctrl/Alt/Win left+right

  const state = {
    activation: { modifiers:0, vk:0 },
    commit:     { modifiers:0, vk:0 }
  };

  function modsFromEvent(e){
    return (e.ctrlKey ? HK.CONTROL : 0)
         | (e.shiftKey ? HK.SHIFT : 0)
         | (e.altKey ? HK.ALT : 0);
  }

  function vkName(vk){
    const map = {
      8:'Backspace', 9:'Tab', 13:'Enter', 27:'Esc', 32:'Space',
      33:'PageUp', 34:'PageDown', 35:'End', 36:'Home',
      37:'Left', 38:'Up', 39:'Right', 40:'Down', 45:'Insert', 46:'Delete',
      186:';', 187:'=', 188:',', 189:'-', 190:'.', 191:'/', 192:'`',
      219:'[', 220:'\\', 221:']', 222:"'"
    };
    if (map[vk]) return map[vk];
    if (vk >= 65 && vk <= 90) return String.fromCharCode(vk);       // A-Z
    if (vk >= 48 && vk <= 57) return String.fromCharCode(vk);       // 0-9
    if (vk >= 96 && vk <= 105) return 'Num' + (vk - 96);            // numpad 0-9
    if (vk >= 112 && vk <= 123) return 'F' + (vk - 111);            // F1-F12
    return 'Key' + vk;
  }

  function label(sc){
    if (!sc || !sc.vk) return '';
    const parts = [];
    if (sc.modifiers & HK.CONTROL) parts.push('Ctrl');
    if (sc.modifiers & HK.SHIFT)   parts.push('Shift');
    if (sc.modifiers & HK.ALT)     parts.push('Alt');
    parts.push(vkName(sc.vk));
    return parts.join(' + ');
  }

  function bindHotkey(id, key){
    const el = document.getElementById(id);
    el.addEventListener('keydown', function(e){
      e.preventDefault();
      e.stopPropagation();
      if (MOD_VK.indexOf(e.keyCode) !== -1) return;  // wait for a real key
      state[key] = { modifiers: modsFromEvent(e), vk: e.keyCode };
      el.value = label(state[key]);
    });
  }
  bindHotkey('activation', 'activation');
  bindHotkey('commit', 'commit');

  function applyConfig(cfg){
    state.activation = cfg.activation || { modifiers:0, vk:0 };
    state.commit     = cfg.commit     || { modifiers:0, vk:0 };
    document.getElementById('activation').value  = label(state.activation);
    document.getElementById('commit').value      = label(state.commit);
    document.getElementById('modelPath').value   = cfg.modelPath || '';
    document.getElementById('contextSize').value = cfg.contextSize != null ? cfg.contextSize : 4096;
    document.getElementById('temperature').value = cfg.temperature != null ? cfg.temperature : 0.7;
    document.getElementById('topP').value        = cfg.topP != null ? cfg.topP : 0.95;
    document.getElementById('maxTokens').value   = cfg.maxTokens != null ? cfg.maxTokens : 1024;
  }

  function showToast(){
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(function(){ t.classList.remove('show'); }, 1600);
  }

  const bridge = (window.chrome && window.chrome.webview) ? window.chrome.webview : null;

  // Ask the host to size its window to the content so there are no scrollbars.
  // The +2 guards against sub-pixel DPI rounding reintroducing a vertical bar.
  function reportSize(){
    try {
      const h = Math.ceil(document.documentElement.scrollHeight) + 2;
      if (bridge) bridge.postMessage({ type:'resize', height:h });
    } catch (err) { /* ignore */ }
  }

  if (bridge){
    bridge.addEventListener('message', function(event){
      try {
        const msg = event.data;
        if (!msg || typeof msg !== 'object') return;
        if (msg.type === 'load') {
          applyConfig(msg);
          reportSize();
        } else if (msg.type === 'modelPath') {
          if (msg.path) document.getElementById('modelPath').value = msg.path;
        } else if (msg.type === 'saved') {
          showToast();
        }
      } catch (err) {
        console.error('settings: failed to handle host message', err);
      }
    });
  }

  document.getElementById('browse').addEventListener('click', function(){
    try { if (bridge) bridge.postMessage({ type:'browse' }); }
    catch (err) { console.error('settings: failed to request folder picker', err); }
  });

  window.addEventListener('load', reportSize);

  document.getElementById('save').addEventListener('click', function(){
    try {
      const payload = {
        type: 'save',
        activation: state.activation,
        commit: state.commit,
        modelPath: document.getElementById('modelPath').value,
        contextSize: parseInt(document.getElementById('contextSize').value, 10) || 4096,
        temperature: parseFloat(document.getElementById('temperature').value) || 0.0,
        topP: parseFloat(document.getElementById('topP').value) || 0.0,
        maxTokens: parseInt(document.getElementById('maxTokens').value, 10) || 1024
      };
      if (bridge) bridge.postMessage(payload);
    } catch (err) {
      console.error('settings: failed to post save message', err);
    }
  });
</script>
</body>
</html>)HTML";

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
                                 CW_USEDEFAULT, CW_USEDEFAULT, 580, 660, owner, nullptr, hInstance,
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
    void OnWebMessage(const std::wstring& messageJson);
    void PushConfigToJs();
    void ResizeToClient();
    void ResizeToContentHeight(int cssHeight);  // fit the window to the page (no scrollbars)
    void ReportWebViewUnavailable();
    bool WriteHtmlFile(std::wstring& outUrl);  // returns a file:// URL to the UI
    void BrowseForModelFolder();               // native folder picker -> JS

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

    // Prefer navigating to a real file (most reliable, and you can open the same
    // file in a browser to confirm the HTML renders). Fall back to NavigateToString.
    std::wstring url;
    if (WriteHtmlFile(url)) {
        webview_->Navigate(url.c_str());
    } else {
        webview_->NavigateToString(FromUtf8(kHtml).c_str());
    }
}

bool SettingsWindow::WriteHtmlFile(std::wstring& outUrl) {
    wchar_t local[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) == 0) {
        return false;
    }
    const std::wstring dir = std::wstring(local) + L"\\BlackwellPocOverlay";
    SHCreateDirectoryExW(nullptr, dir.c_str(), nullptr);  // ok if it already exists

    const std::wstring path = dir + L"\\settings.html";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(kHtml, static_cast<std::streamsize>(sizeof(kHtml) - 1));  // raw UTF-8 bytes
    out.close();

    std::wstring url = L"file:///" + path;
    for (wchar_t& c : url) {
        if (c == L'\\') {
            c = L'/';
        }
    }
    outUrl = url;
    return true;
}

void SettingsWindow::PushConfigToJs() {
    json j;
    j["type"] = "load";
    j["activation"] = {{"modifiers", config_.activationShortcut.modifiers},
                       {"vk", config_.activationShortcut.vk}};
    j["commit"] = {{"modifiers", config_.commitShortcut.modifiers},
                   {"vk", config_.commitShortcut.vk}};
    j["modelPath"] = ToUtf8(config_.modelPath);
    j["contextSize"] = config_.contextSize;
    j["temperature"] = config_.temperature;
    j["topP"] = config_.topP;
    j["maxTokens"] = config_.maxTokens;
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
        config_.modelPath = FromUtf8(j.value("modelPath", std::string()));
        config_.contextSize = j.value("contextSize", config_.contextSize);
        config_.temperature = j.value("temperature", config_.temperature);
        config_.topP = j.value("topP", config_.topP);
        config_.maxTokens = j.value("maxTokens", config_.maxTokens);

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
