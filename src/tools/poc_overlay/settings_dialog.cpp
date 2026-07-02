#include "settings_dialog.h"

#include <objbase.h>
#include <wrl.h>
#include <WebView2.h>

#include <nlohmann/json.hpp>

#include <string>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
using nlohmann::json;

namespace {

constexpr wchar_t kClassName[] = L"BlackwellPocSettingsWindow";
HWND g_openWindow = nullptr;  // single-instance guard

// Modern settings UI. Shortcut fields are "key capture" boxes: click, press the
// combo, and JS records the modifiers + Windows virtual-key (event.keyCode maps
// to VK_* for the keys we care about). ASCII-only so it is safe to embed as a
// wide raw string literal regardless of source encoding.
constexpr wchar_t kHtml[] = LR"HTML(<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root{color-scheme:dark;}
  *{box-sizing:border-box;}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#1e1e22;color:#eaeaea;margin:0;padding:22px;}
  h1{font-size:17px;margin:0 0 4px;}
  p.sub{margin:0 0 18px;color:#8a9099;font-size:12px;}
  .card{background:#26262c;border:1px solid #34343c;border-radius:10px;padding:16px;margin-bottom:14px;}
  .card h2{font-size:13px;margin:0 0 12px;color:#c7ccd1;font-weight:600;}
  label{display:block;font-size:12px;color:#9aa0a6;margin:10px 0 6px;}
  label:first-child{margin-top:0;}
  input[type=text],input[type=number]{width:100%;background:#17171b;border:1px solid #3a3a42;border-radius:6px;color:#eaeaea;padding:8px 10px;font-size:13px;}
  .capture{width:100%;background:#17171b;border:1px solid #3a3a42;border-radius:6px;color:#eaeaea;padding:9px 10px;font-size:13px;cursor:pointer;user-select:none;}
  .capture:focus,.capture.active{outline:none;border-color:#4c8bf5;box-shadow:0 0 0 2px rgba(76,139,245,.28);}
  .row{display:flex;gap:12px;}
  .row>div{flex:1;}
  .bar{display:flex;align-items:center;margin-top:4px;}
  button{background:#4c8bf5;border:none;border-radius:6px;color:#fff;padding:10px 20px;font-size:13px;font-weight:600;cursor:pointer;}
  button:hover{background:#3f7ae0;}
  #status{margin-left:14px;font-size:12px;color:#57d97e;opacity:0;transition:opacity .2s;}
  #status.show{opacity:1;}
</style></head>
<body>
  <h1>Blackwell Overlay</h1>
  <p class="sub">Live translation assistant settings</p>

  <div class="card">
    <h2>Shortcuts</h2>
    <label>Activation shortcut (toggle assistant on/off)</label>
    <div class="capture" id="capActivation" tabindex="0">Click, then press keys</div>
    <label>Commit shortcut (replace typed text with translation)</label>
    <div class="capture" id="capCommit" tabindex="0">Click, then press keys</div>
  </div>

  <div class="card">
    <h2>Model configuration</h2>
    <label>Model path</label>
    <input type="text" id="modelPath" placeholder="C:\models\model.gguf" spellcheck="false">
    <div class="row">
      <div>
        <label>Context size (tokens)</label>
        <input type="number" id="contextSize" min="512" max="1048576" step="512">
      </div>
    </div>
  </div>

  <div class="bar">
    <button id="save">Save</button>
    <span id="status">Saved</span>
  </div>

<script>
  const HK = { SHIFT:1, CONTROL:2, ALT:4 };
  const state = { activation:{modifiers:0,vk:0}, commit:{modifiers:0,vk:0} };

  function keyName(vk){
    const m = {8:'Backspace',9:'Tab',13:'Enter',27:'Esc',32:'Space',
               37:'Left',38:'Up',39:'Right',40:'Down',46:'Delete'};
    if (m[vk]) return m[vk];
    if (vk>=65 && vk<=90) return String.fromCharCode(vk);
    if (vk>=48 && vk<=57) return String.fromCharCode(vk);
    if (vk>=112 && vk<=123) return 'F'+(vk-111);
    return 'VK'+vk;
  }
  function label(sc){
    if (!sc.vk) return 'Click, then press keys';
    const p = [];
    if (sc.modifiers & HK.CONTROL) p.push('Ctrl');
    if (sc.modifiers & HK.ALT)     p.push('Alt');
    if (sc.modifiers & HK.SHIFT)   p.push('Shift');
    p.push(keyName(sc.vk));
    return p.join(' + ');
  }
  function bindCapture(id, key){
    const el = document.getElementById(id);
    el.addEventListener('click', () => { el.classList.add('active'); el.focus(); });
    el.addEventListener('blur',  () => el.classList.remove('active'));
    el.addEventListener('keydown', e => {
      e.preventDefault();
      if ([16,17,18,91,92].includes(e.keyCode)) return;  // lone modifier
      state[key] = {
        modifiers:(e.ctrlKey?HK.CONTROL:0)|(e.altKey?HK.ALT:0)|(e.shiftKey?HK.SHIFT:0),
        vk:e.keyCode
      };
      el.textContent = label(state[key]);
      el.blur();
    });
  }
  bindCapture('capActivation','activation');
  bindCapture('capCommit','commit');

  window.chrome.webview.addEventListener('message', e => {
    const d = e.data;
    if (d.type === 'load') {
      state.activation = d.activation; state.commit = d.commit;
      document.getElementById('capActivation').textContent = label(state.activation);
      document.getElementById('capCommit').textContent = label(state.commit);
      document.getElementById('modelPath').value = d.modelPath || '';
      document.getElementById('contextSize').value = d.contextSize || 4096;
    } else if (d.type === 'saved') {
      const s = document.getElementById('status');
      s.classList.add('show');
      setTimeout(() => s.classList.remove('show'), 1500);
    }
  });

  document.getElementById('save').addEventListener('click', () => {
    window.chrome.webview.postMessage({
      type:'save',
      activation: state.activation,
      commit: state.commit,
      modelPath: document.getElementById('modelPath').value,
      contextSize: parseInt(document.getElementById('contextSize').value, 10) || 4096
    });
  });
</script></body></html>)HTML";

// Owns the settings window + its WebView2. Heap-allocated; self-deletes on
// WM_NCDESTROY.
class SettingsWindow {
public:
    SettingsWindow(Config config, std::function<void(const Config&)> onApply)
        : config_(std::move(config)), onApply_(std::move(onApply)) {}

    bool Create(HWND owner, HINSTANCE hInstance) {
        hwnd_ = CreateWindowExW(0, kClassName, L"Blackwell PoC - Settings",
                                 WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                 CW_USEDEFAULT, CW_USEDEFAULT, 560, 520, owner, nullptr, hInstance,
                                 this);
        if (!hwnd_) {
            return false;
        }
        g_openWindow = hwnd_;
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
    void ReportWebViewUnavailable();

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

void SettingsWindow::ReportWebViewUnavailable() {
    MessageBoxW(hwnd_,
                L"Could not initialize WebView2.\n\nInstall the Microsoft Edge WebView2 Runtime "
                L"and reopen Settings.",
                L"Settings", MB_ICONWARNING | MB_OK);
    DestroyWindow(hwnd_);
}

void SettingsWindow::OnControllerCreated(ICoreWebView2Controller* controller) {
    controller_ = controller;
    controller_->get_CoreWebView2(&webview_);
    ResizeToClient();

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

    // Push the current config once the page's script is ready to receive it.
    webview_->add_NavigationCompleted(
        Callback<ICoreWebView2NavigationCompletedEventHandler>(
            [this](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs*) -> HRESULT {
                PushConfigToJs();
                return S_OK;
            })
            .Get(),
        &token);

    webview_->NavigateToString(kHtml);
    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
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
    if (webview_) {
        webview_->PostWebMessageAsJson(FromUtf8(j.dump()).c_str());
    }
}

void SettingsWindow::OnWebMessage(const std::wstring& messageJson) {
    try {
        const json j = json::parse(ToUtf8(messageJson));
        if (j.value("type", std::string()) != "save") {
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

        ConfigStore::Save(config_);  // persist to config.json
        if (onApply_) {
            onApply_(config_);  // rebind hotkeys / model in memory immediately
        }
        if (webview_) {
            webview_->PostWebMessageAsJson(LR"({"type":"saved"})");
        }
    } catch (const std::exception&) {
        // Ignore malformed messages.
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
