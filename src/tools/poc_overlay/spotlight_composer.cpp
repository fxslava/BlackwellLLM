#define NOMINMAX
#include "spotlight_composer.h"

#include <windowsx.h>  // GET_X_LPARAM (unused today, kept for hit-testing parity)

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "hook_manager.h"              // SetInjecting guard around our synthetic paste
#include "live_translation_tracker.h"  // the "spinal cord" this composer drives
#include "text_injector.h"             // kInjectedSignature: mark our own input so the hook ignores it
#include "token_info.h"                // TokenHeatmap (tracker StreamCallback signature)

using Microsoft::WRL::ComPtr;

namespace {

// Clipboard/paste settle times -- lifted from text_injector.cpp's tuned values.
// Without them the target app can miss the freshly-set clipboard, or we restore
// the previous clipboard before the paste consumes ours.
constexpr DWORD kFocusSettleMs = 20;      // after foreground restore, before touching input
constexpr DWORD kClipboardSettleMs = 20;  // after SetClipboardData, before Ctrl+V
constexpr DWORD kPasteConsumeMs = 60;     // after Ctrl+V, before restoring the clipboard

const wchar_t kWindowClass[] = L"TypeTranslateSpotlightComposer";
const wchar_t kPlaceholder[] = L"Type to translate\x2026";  // shown when the input is empty
const wchar_t kSpinner[] = L"\x2026";                       // "..." while the first tokens are in flight

// --- theme ------------------------------------------------------------------
// Honor the OS "Apps use light theme" setting so the squircle matches the shell.
// A missing key (older Windows) reads as dark, which is the composer's default.
bool SystemUsesLightTheme() {
    DWORD value = 0;
    DWORD size = sizeof(value);
    if (RegGetValueW(HKEY_CURRENT_USER,
                     L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
                     L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr, &value, &size) == ERROR_SUCCESS) {
        return value != 0;
    }
    return false;
}

// --- SendInput helpers (every event tagged so our own WH_KEYBOARD_LL ignores it) --
void AppendKey(std::vector<INPUT>& events, WORD vk, bool keyUp) {
    INPUT input{};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = vk;
    input.ki.dwFlags = keyUp ? KEYEVENTF_KEYUP : 0;
    input.ki.dwExtraInfo = TextInjector::kInjectedSignature;
    events.push_back(input);
}

// Release any physically-held modifier before we synthesize the paste. The
// summon chord is Alt+Space, and although the user has since typed, we cannot
// assume Alt/Ctrl/Shift are up -- a stray held modifier would turn our Ctrl+V
// into Ctrl+Alt+V (or worse) in the target app. Mirrors text_injector.cpp.
void SendModifierRelease() {
    static constexpr WORD kModifiers[] = {VK_LCONTROL, VK_RCONTROL, VK_LSHIFT,
                                          VK_RSHIFT,   VK_LMENU,    VK_RMENU};
    std::vector<INPUT> events;
    events.reserve(6);
    for (WORD vk : kModifiers) {
        AppendKey(events, vk, /*keyUp=*/true);
    }
    SendInput(static_cast<UINT>(events.size()), events.data(), sizeof(INPUT));
}

// The escape hatch's core: a precise 4-event Ctrl+V (Ctrl down, V down, V up,
// Ctrl up). Sent as ONE atomic SendInput batch so nothing interleaves.
void SendCtrlV() {
    std::vector<INPUT> events;
    events.reserve(4);
    AppendKey(events, VK_CONTROL, /*keyUp=*/false);
    AppendKey(events, 'V', /*keyUp=*/false);
    AppendKey(events, 'V', /*keyUp=*/true);
    AppendKey(events, VK_CONTROL, /*keyUp=*/true);
    SendInput(static_cast<UINT>(events.size()), events.data(), sizeof(INPUT));
}

// --- clipboard RAII ---------------------------------------------------------
// Opens the clipboard (with retry -- another app may briefly own it) and closes
// it on scope exit. `ok()` reports whether the open succeeded.
class ClipboardGuard {
public:
    explicit ClipboardGuard(HWND owner) {
        for (int attempt = 0; attempt < 10 && !open_; ++attempt) {
            if (OpenClipboard(owner)) {
                open_ = true;
            } else {
                Sleep(10);
            }
        }
    }
    ~ClipboardGuard() {
        if (open_) {
            CloseClipboard();
        }
    }
    ClipboardGuard(const ClipboardGuard&) = delete;
    ClipboardGuard& operator=(const ClipboardGuard&) = delete;
    bool ok() const { return open_; }

private:
    bool open_ = false;
};

bool GetClipboardText(HWND owner, std::wstring& out) {
    ClipboardGuard clip(owner);
    if (!clip.ok()) {
        return false;
    }
    HANDLE handle = GetClipboardData(CF_UNICODETEXT);
    if (!handle) {
        return false;
    }
    if (const auto* text = static_cast<const wchar_t*>(GlobalLock(handle))) {
        out.assign(text);
        GlobalUnlock(handle);
        return true;
    }
    return false;
}

bool SetClipboardText(HWND owner, const std::wstring& text) {
    ClipboardGuard clip(owner);
    if (!clip.ok()) {
        return false;
    }
    if (!EmptyClipboard()) {
        return false;
    }
    const size_t bytes = (text.size() + 1) * sizeof(wchar_t);
    HGLOBAL mem = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (!mem) {
        return false;
    }
    bool ok = false;
    if (void* dst = GlobalLock(mem)) {
        memcpy(dst, text.c_str(), bytes);
        GlobalUnlock(mem);
        if (SetClipboardData(CF_UNICODETEXT, mem)) {
            ok = true;  // the clipboard now OWNS `mem`; do not free it
        }
    }
    if (!ok) {
        GlobalFree(mem);  // ownership was not transferred
    }
    return ok;
}

}  // namespace

// ============================================================================
// Construction / teardown
// ============================================================================

bool SpotlightComposer::Create(HINSTANCE hInstance) {
    // tracker_ stays null until BindTracker() (the engine loads asynchronously).
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = &SpotlightComposer::WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = kWindowClass;
    wc.hCursor = LoadCursorW(nullptr, IDC_IBEAM);
    RegisterClassExW(&wc);  // benign ERROR_CLASS_ALREADY_EXISTS on a second Create

    // WS_EX_LAYERED: per-pixel alpha via UpdateLayeredWindow (rounded squircle
    // over the desktop). WS_EX_TOPMOST: above the target app. WS_EX_TOOLWINDOW:
    // no taskbar/Alt-Tab entry. WS_EX_NOACTIVATE (initial): the window exists but
    // never steals focus until we deliberately summon it -- Summon() strips this
    // bit so the box can own the keyboard while composing.
    hwnd_ = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, kWindowClass, L"",
        WS_POPUP, 0, 0, kBaseWidth, kBaseHeight, nullptr, nullptr, hInstance, this);
    if (!hwnd_) {
        return false;
    }
    // WndProc's WM_NCCREATE stashes `this`; set it here too so the very first
    // messages (if any preceded NCCREATE routing) resolve.
    SetWindowLongPtrW(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

    if (!InitDirect2D()) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
        return false;
    }
    return true;
}

SpotlightComposer::~SpotlightComposer() {
    // Drop any in-flight tracker work BEFORE the window dies so a late worker
    // callback cannot PostMessage into a destroyed HWND (PostMessage to a dead
    // window merely fails, but this also stops needless decoding).
    if (tracker_) {
        tracker_->Cancel();
    }
    render_target_.Reset();
    if (dib_) {
        DeleteObject(dib_);
    }
    if (mem_dc_) {
        DeleteDC(mem_dc_);
    }
    if (hwnd_) {
        DestroyWindow(hwnd_);
    }
}

bool SpotlightComposer::InitDirect2D() {
    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2d_factory_.GetAddressOf()))) {
        return false;
    }
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                                   reinterpret_cast<IUnknown**>(dwrite_factory_.GetAddressOf())))) {
        return false;
    }

    // Text formats are created at BASE point sizes; the render target's DPI is
    // set per paint (SetDpi) so DirectWrite scales glyphs to the target monitor.
    if (FAILED(dwrite_factory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, static_cast<float>(kBaseInputFont), L"en-us",
            input_format_.GetAddressOf()))) {
        return false;
    }
    input_format_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    input_format_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    input_format_->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);

    if (FAILED(dwrite_factory_->CreateTextFormat(
            L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL, static_cast<float>(kBaseTransFont), L"en-us",
            trans_format_.GetAddressOf()))) {
        return false;
    }
    trans_format_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
    trans_format_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    trans_format_->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);

    const D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED));
    return SUCCEEDED(d2d_factory_->CreateDCRenderTarget(&props, render_target_.GetAddressOf()));
}

bool SpotlightComposer::EnsureBackingBitmap() {
    // Reuse the existing DIB unless the scaled size changed since the last build.
    if (dib_ && dib_scaled_w_ == size_.cx && dib_scaled_h_ == size_.cy) {
        return true;
    }
    if (mem_dc_ == nullptr) {
        mem_dc_ = CreateCompatibleDC(nullptr);
        if (!mem_dc_) {
            return false;
        }
    }
    if (dib_) {
        DeleteObject(dib_);
        dib_ = nullptr;
    }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = size_.cx;
    bmi.bmiHeader.biHeight = -size_.cy;  // negative = top-down (matches D2D's origin)
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    dib_ = CreateDIBSection(mem_dc_, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!dib_) {
        return false;
    }
    SelectObject(mem_dc_, dib_);
    dib_scaled_w_ = static_cast<int>(size_.cx);
    dib_scaled_h_ = static_cast<int>(size_.cy);
    return true;
}

// ============================================================================
// Window plumbing
// ============================================================================

LRESULT CALLBACK SpotlightComposer::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        auto* self = static_cast<SpotlightComposer*>(
            reinterpret_cast<CREATESTRUCTW*>(lParam)->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        self->hwnd_ = hwnd;
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
    auto* self = reinterpret_cast<SpotlightComposer*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (!self) {
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
    return self->HandleMessage(msg, wParam, lParam);
}

LRESULT SpotlightComposer::HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case kMsgSummon:
            DoSummon();
            return 0;

        case kMsgTranslation:
            OnTranslation(reinterpret_cast<StreamPayload*>(lParam));
            return 0;

        case WM_CHAR:
            OnChar(static_cast<wchar_t>(wParam));
            return 0;

        case WM_KEYDOWN:
            if (OnKeyDown(wParam)) {
                return 0;
            }
            break;

        case WM_TIMER:
            if (wParam == kDebounceTimerId) {
                KillTimer(hwnd_, kDebounceTimerId);
                IssueGeneration();
            } else if (wParam == kCaretTimerId) {
                caret_on_ = !caret_on_;
                OnPaint();
            }
            return 0;

        case WM_ACTIVATE:
            // Focus left the composer (the user clicked into another app or hit
            // some global chord). Spotlight semantics: dismiss WITHOUT injecting.
            // Suppressed during the summon handshake, where our own focus-steal
            // briefly toggles activation.
            if (LOWORD(wParam) == WA_INACTIVE && visible_ && !summoning_) {
                Dismiss(/*should_inject=*/false);
            }
            return 0;

        case WM_KILLFOCUS:
            if (visible_ && !summoning_) {
                Dismiss(/*should_inject=*/false);
            }
            return 0;

        case WM_DESTROY:
            KillTimer(hwnd_, kDebounceTimerId);
            KillTimer(hwnd_, kCaretTimerId);
            return 0;

        default:
            break;
    }
    return DefWindowProcW(hwnd_, msg, wParam, lParam);
}

// ============================================================================
// Summon / positioning / focus
// ============================================================================

void SpotlightComposer::Summon() {
    // Thread-safe entry. If we are not on the window's thread, marshal; otherwise
    // act inline. GetWindowThreadProcessId(hwnd_) == our creator thread.
    if (hwnd_ && GetCurrentThreadId() != GetWindowThreadProcessId(hwnd_, nullptr)) {
        PostMessageW(hwnd_, kMsgSummon, 0, 0);
        return;
    }
    DoSummon();
}

void SpotlightComposer::DoSummon() {
    if (!hwnd_) {
        return;
    }
    // Capture the paste target BEFORE we take focus -- once we are foreground,
    // GetForegroundWindow would just return us. Ignore our own window if somehow
    // already frontmost.
    HWND fg = GetForegroundWindow();
    if (fg != hwnd_) {
        prev_foreground_hwnd_ = fg;
    }

    ResetComposition();

    // Position over the paste target's monitor (fallback: the monitor under the
    // cursor), then size the backing bitmap to that monitor's DPI.
    PositionForMonitor(prev_foreground_hwnd_);
    if (!EnsureBackingBitmap()) {
        return;
    }

    // Allow activation now: strip WS_EX_NOACTIVATE so SetForegroundWindow/SetFocus
    // actually grant us the keyboard. (It is re-added on Dismiss.)
    LONG_PTR ex = GetWindowLongPtrW(hwnd_, GWL_EXSTYLE);
    SetWindowLongPtrW(hwnd_, GWL_EXSTYLE, ex & ~WS_EX_NOACTIVATE);

    summoning_ = true;  // suppress the WM_ACTIVATE self-dismiss during the handshake
    ShowWindow(hwnd_, SW_SHOWNA);
    OnPaint();  // populate the layered surface before it becomes visible-with-focus

    // Robust foreground steal. SetForegroundWindow alone is subject to the OS
    // foreground lock; attaching to the previous foreground thread's input queue
    // lets us legitimately take focus, exactly as documented for this scenario.
    const DWORD ourThread = GetCurrentThreadId();
    const DWORD fgThread =
        prev_foreground_hwnd_ ? GetWindowThreadProcessId(prev_foreground_hwnd_, nullptr) : 0;
    if (fgThread && fgThread != ourThread) {
        AttachThreadInput(fgThread, ourThread, TRUE);
        SetForegroundWindow(hwnd_);
        SetFocus(hwnd_);
        AttachThreadInput(fgThread, ourThread, FALSE);
    } else {
        SetForegroundWindow(hwnd_);
        SetFocus(hwnd_);
    }
    summoning_ = false;

    visible_ = true;
    // Blink the input caret on the OS caret cadence.
    SetTimer(hwnd_, kCaretTimerId, GetCaretBlinkTime(), nullptr);
    caret_on_ = true;
    OnPaint();

    if (visibility_observer_) {
        visibility_observer_(true);
    }
}

void SpotlightComposer::PositionForMonitor(HWND anchor) {
    HMONITOR mon = anchor ? MonitorFromWindow(anchor, MONITOR_DEFAULTTONEAREST)
                          : MonitorFromPoint(POINT{}, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi{sizeof(mi)};
    RECT work{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
    if (GetMonitorInfoW(mon, &mi)) {
        work = mi.rcWork;
    }

    // Per-monitor DPI: the process is PerMonitorV2 (set in wWinMain), so we scale
    // our base layout to the monitor we are about to appear on. GetDpiForWindow
    // (user32, already linked) reports the anchor's monitor DPI; fall back to the
    // system DPI when there is no anchor.
    const UINT dpi = anchor ? GetDpiForWindow(anchor) : GetDpiForSystem();
    dpi_scale_ = (dpi ? static_cast<float>(dpi) : 96.0f) / 96.0f;

    const int sw = std::min(static_cast<int>(std::lround(kBaseWidth * dpi_scale_)), kMaxDibWidth);
    const int sh = std::min(static_cast<int>(std::lround(kBaseHeight * dpi_scale_)), kMaxDibHeight);
    size_.cx = sw;
    size_.cy = sh;

    const int workW = static_cast<int>(work.right - work.left);
    const int workH = static_cast<int>(work.bottom - work.top);
    origin_.x = work.left + std::max(0, (workW - sw) / 2);
    // Sit ~15% down from the top of the work area (Spotlight-like), clamped so the
    // box never runs off a short work area.
    origin_.y = work.top + std::min(std::max(0, workH - sh), workH * 15 / 100);
}

// ============================================================================
// Composition + engine wiring
// ============================================================================

void SpotlightComposer::ResetComposition() {
    if (tracker_) {
        tracker_->Cancel();  // drop any pending/in-flight decode from a prior session
    }
    KillTimer(hwnd_, kDebounceTimerId);
    input_buffer_.clear();
    translation_buffer_.clear();
    gen_.fetch_add(1, std::memory_order_relaxed);  // invalidate late deliveries
}

void SpotlightComposer::OnInputChanged() {
    // Speculative reconcile on EVERY edit (growth OR deletion). The tracker's
    // update_sequence re-tokenizes the whole buffer, diffs it against the
    // session's token mirror (LCP), truncates the KV cache to the divergence
    // point via Copy-on-Write page rewind, and delta-computes only the new
    // suffix. This is the one call that makes "t" -> "th" -> "the" cheap: no
    // keystroke ever runs generation here -- that is strictly debounce-gated
    // below, so a backspace truncates without decoding, exactly as required.
    if (tracker_) {
        tracker_->TrackUpdate(input_buffer_, /*context=*/std::wstring());
    }

    // (Re)arm the debounce: a decode fires only after the user pauses. Restarting
    // the timer on each edit coalesces a burst into a single generation.
    if (input_buffer_.empty()) {
        KillTimer(hwnd_, kDebounceTimerId);
        translation_buffer_.clear();
        gen_.fetch_add(1, std::memory_order_relaxed);  // drop any stragglers
    } else {
        SetTimer(hwnd_, kDebounceTimerId, kDebounceMs, nullptr);
    }
    OnPaint();
}

void SpotlightComposer::IssueGeneration() {
    if (!tracker_ || input_buffer_.empty()) {
        return;
    }
    // Stamp this decode so a delayed callback for a superseded buffer is dropped.
    const std::uint64_t gen = gen_.fetch_add(1, std::memory_order_relaxed) + 1;
    const HWND hwnd = hwnd_;  // stable; captured by value for the worker thread

    // The StreamCallback runs on the TRACKER'S worker thread. Contract: marshal
    // only -- so we heap a payload and PostMessage it to our window. No engine,
    // COM, or GDI state is touched here.
    tracker_->TriggerGeneration(
        input_buffer_, /*context=*/std::wstring(),
        [gen, hwnd](const std::wstring& text, const TokenHeatmap& /*tokens*/, bool done) {
            auto* payload = new StreamPayload{gen, text, done};
            if (!PostMessageW(hwnd, kMsgTranslation, 0, reinterpret_cast<LPARAM>(payload))) {
                delete payload;  // window gone / queue full -- drop it
            }
        });
}

void SpotlightComposer::OnTranslation(StreamPayload* payload) {
    std::unique_ptr<StreamPayload> owned(payload);  // own it however we exit
    if (!owned) {
        return;
    }
    // Drop deliveries superseded by a newer edit/decode.
    if (owned->gen != gen_.load(std::memory_order_relaxed)) {
        return;
    }
    translation_buffer_ = owned->text;  // growing partial, or the final string
    OnPaint();
}

void SpotlightComposer::SetActiveLanguage(int index) {
    if (tracker_) {
        tracker_->SetActiveLanguage(index);
    }
}

// ============================================================================
// Input handling
// ============================================================================

void SpotlightComposer::OnChar(wchar_t ch) {
    // Ignore control characters: Enter/Backspace/Escape are handled in WM_KEYDOWN,
    // and we never want a literal tab/bell in the buffer.
    if (ch < 0x20) {
        return;
    }
    input_buffer_.push_back(ch);
    OnInputChanged();
}

bool SpotlightComposer::OnKeyDown(WPARAM vk) {
    switch (vk) {
        case VK_RETURN:
            // Commit: hide, restore focus, paste the translation into the target.
            Dismiss(/*should_inject=*/true);
            return true;

        case VK_ESCAPE:
            Dismiss(/*should_inject=*/false);
            return true;

        case VK_BACK:
            if (!input_buffer_.empty()) {
                input_buffer_.pop_back();
                OnInputChanged();  // TrackUpdate here TRUNCATES the KV cache; no decode
            }
            return true;

        case 'V':
            // Ctrl+V typed INTO the composer: paste text to translate. (This is
            // inbound; the Tier-4 escape hatch's Ctrl+V is a separate, outbound
            // SendInput in InjectTranslation.)
            if (GetKeyState(VK_CONTROL) & 0x8000) {
                PasteFromClipboardIntoBuffer();
                return true;
            }
            return false;

        default:
            return false;
    }
}

void SpotlightComposer::PasteFromClipboardIntoBuffer() {
    std::wstring clip;
    if (!GetClipboardText(hwnd_, clip) || clip.empty()) {
        return;
    }
    // Collapse newlines to spaces: the composer is a single-line input box.
    std::replace(clip.begin(), clip.end(), L'\r', L' ');
    std::replace(clip.begin(), clip.end(), L'\n', L' ');
    input_buffer_ += clip;
    OnInputChanged();
}

// ============================================================================
// Dismiss + Tier-4 injection
// ============================================================================

void SpotlightComposer::Dismiss(bool should_inject) {
    if (!hwnd_ || !visible_) {
        return;
    }
    visible_ = false;
    KillTimer(hwnd_, kDebounceTimerId);
    KillTimer(hwnd_, kCaretTimerId);
    if (tracker_) {
        tracker_->Cancel();  // stop any decode still streaming for this session
    }

    // (a) Instantly hide so the target app is visually unobstructed during paste.
    ShowWindow(hwnd_, SW_HIDE);
    // Restore the non-activating style for the next dormant period.
    LONG_PTR ex = GetWindowLongPtrW(hwnd_, GWL_EXSTYLE);
    SetWindowLongPtrW(hwnd_, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE);

    const std::wstring toInject = translation_buffer_;
    if (should_inject && !toInject.empty()) {
        InjectTranslation();
    } else if (prev_foreground_hwnd_ && IsWindow(prev_foreground_hwnd_)) {
        // Even on a non-injecting dismiss, hand focus back where it came from.
        SetForegroundWindow(prev_foreground_hwnd_);
    }

    if (visibility_observer_) {
        visibility_observer_(false);
    }
}

void SpotlightComposer::InjectTranslation() {
    const std::wstring text = translation_buffer_;
    if (text.empty() || !prev_foreground_hwnd_ || !IsWindow(prev_foreground_hwnd_)) {
        return;
    }

    // (b) Give focus back to the app that owned it when we were summoned.
    SetForegroundWindow(prev_foreground_hwnd_);
    // (c) Let the message queues settle and the foreground actually change hands
    // before we synthesize input -- without this the paste can land in the void.
    Sleep(kFocusSettleMs);

    // Guard our own synthetic keystrokes: the app's WH_KEYBOARD_LL hook (ours)
    // must pass them straight through rather than re-triggering a translation.
    // The kInjectedSignature stamp is the robust check; SetInjecting is the belt.
    HookManager::Instance().SetInjecting(true);

    // (d) Tier-4 clipboard swap: back up whatever is on the clipboard, put our
    // translation there as CF_UNICODETEXT, and restore the original afterward so
    // the user's clipboard is left untouched.
    std::wstring backup;
    const bool hadBackup = GetClipboardText(hwnd_, backup);
    if (SetClipboardText(hwnd_, text)) {
        Sleep(kClipboardSettleMs);
        // (e) Force the target to accept the text: clear stray modifiers, then a
        // precise 4-event synthetic Ctrl+V.
        SendModifierRelease();
        SendCtrlV();
        Sleep(kPasteConsumeMs);  // let the foreground app consume the paste
        SetClipboardText(hwnd_, hadBackup ? backup : std::wstring());
    }

    HookManager::Instance().SetInjecting(false);
}

// ============================================================================
// Rendering (layered, per-pixel alpha)
// ============================================================================

void SpotlightComposer::OnPaint() {
    if (!hwnd_ || !render_target_ || !EnsureBackingBitmap()) {
        return;
    }

    const bool light = SystemUsesLightTheme();
    // Palette. Alpha < 1 on the background so the squircle reads as frosted glass.
    const D2D1_COLOR_F bg = light ? D2D1::ColorF(0.97f, 0.97f, 0.98f, 0.96f)
                                  : D2D1::ColorF(0.10f, 0.11f, 0.13f, 0.94f);
    const D2D1_COLOR_F border = light ? D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.10f)
                                      : D2D1::ColorF(1.0f, 1.0f, 1.0f, 0.10f);
    const D2D1_COLOR_F inputColor = light ? D2D1::ColorF(0.08f, 0.09f, 0.11f, 1.0f)
                                          : D2D1::ColorF(0.97f, 0.98f, 1.0f, 1.0f);
    const D2D1_COLOR_F placeholderColor = light ? D2D1::ColorF(0.45f, 0.47f, 0.52f, 1.0f)
                                                : D2D1::ColorF(0.55f, 0.58f, 0.64f, 1.0f);
    const D2D1_COLOR_F transColor = D2D1::ColorF(0.42f, 0.66f, 1.0f, 1.0f);  // accent, both themes
    const D2D1_COLOR_F sepColor = light ? D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.08f)
                                        : D2D1::ColorF(1.0f, 1.0f, 1.0f, 0.08f);

    // BindDC bounds are in PHYSICAL pixels (the DIB size); everything drawn below
    // is in DIPs (base-unit coordinates). SetDpi(96*scale) is the single bridge:
    // a DIP of value V lands at V*scale physical pixels, so base-unit geometry AND
    // the base-point fonts both scale uniformly to the monitor. Do NOT mix
    // physical pixels into the drawing coordinates.
    const RECT bounds{0, 0, static_cast<LONG>(size_.cx), static_cast<LONG>(size_.cy)};
    if (FAILED(render_target_->BindDC(mem_dc_, &bounds))) {
        return;
    }
    render_target_->SetDpi(96.0f * dpi_scale_, 96.0f * dpi_scale_);

    render_target_->BeginDraw();
    render_target_->Clear(D2D1::ColorF(0, 0, 0, 0));  // fully transparent outside the squircle

    // All geometry in DIPs == base units.
    const float w = static_cast<float>(kBaseWidth);
    const float h = static_cast<float>(kBaseHeight);
    const float padX = static_cast<float>(kBasePadX);
    const float padY = static_cast<float>(kBasePadY);
    const float corner = static_cast<float>(kBaseCorner);

    // Squircle body (a generously rounded rect), inset by 1 DIP so the stroke sits
    // fully inside the surface.
    const D2D1_ROUNDED_RECT squircle{D2D1::RectF(1.0f, 1.0f, w - 1.0f, h - 1.0f), corner, corner};
    ComPtr<ID2D1SolidColorBrush> bgBrush;
    render_target_->CreateSolidColorBrush(bg, bgBrush.GetAddressOf());
    if (bgBrush) {
        render_target_->FillRoundedRectangle(squircle, bgBrush.Get());
    }
    ComPtr<ID2D1SolidColorBrush> borderBrush;
    render_target_->CreateSolidColorBrush(border, borderBrush.GetAddressOf());
    if (borderBrush) {
        render_target_->DrawRoundedRectangle(squircle, borderBrush.Get(), 1.0f);
    }

    // Two stacked text blocks: raw input (top half), translation (bottom half).
    const float midY = h * 0.5f;
    const float textLeft = padX;

    // --- input line -----------------------------------------------------------
    const bool empty = input_buffer_.empty();
    const std::wstring inputShown = empty ? kPlaceholder : input_buffer_;
    const float inputTop = padY;
    const float inputBottom = midY - (padY * 0.25f);

    ComPtr<ID2D1SolidColorBrush> inputBrush;
    render_target_->CreateSolidColorBrush(empty ? placeholderColor : inputColor,
                                          inputBrush.GetAddressOf());

    // A text layout lets us both draw the input AND measure its width (in DIPs) to
    // place the caret exactly at the end of the string.
    ComPtr<IDWriteTextLayout> inputLayout;
    float caretX = textLeft;
    if (SUCCEEDED(dwrite_factory_->CreateTextLayout(
            inputShown.c_str(), static_cast<UINT32>(inputShown.length()), input_format_.Get(),
            w - 2.0f * padX, inputBottom - inputTop, inputLayout.GetAddressOf()))) {
        if (inputBrush) {
            render_target_->DrawTextLayout(D2D1::Point2F(textLeft, inputTop), inputLayout.Get(),
                                           inputBrush.Get(), D2D1_DRAW_TEXT_OPTIONS_CLIP);
        }
        if (!empty) {
            DWRITE_TEXT_METRICS m{};
            inputLayout->GetMetrics(&m);
            // Clamp so a very long line keeps the caret inside the box.
            caretX = std::min(textLeft + m.widthIncludingTrailingWhitespace, w - padX);
        }
    }

    // Blinking input caret (a thin vertical bar) right after the typed text.
    if (caret_on_ && inputBrush) {
        render_target_->FillRectangle(
            D2D1::RectF(caretX, inputTop + 2.0f, caretX + 1.5f, inputBottom - 2.0f),
            inputBrush.Get());
    }

    // --- separator ------------------------------------------------------------
    ComPtr<ID2D1SolidColorBrush> sepBrush;
    render_target_->CreateSolidColorBrush(sepColor, sepBrush.GetAddressOf());
    if (sepBrush) {
        render_target_->DrawLine(D2D1::Point2F(padX, midY), D2D1::Point2F(w - padX, midY),
                                 sepBrush.Get(), 1.0f);
    }

    // --- translation line -----------------------------------------------------
    // While a decode is in flight (input present, no translation yet) show the
    // spinner glyph; otherwise the streamed/final translation.
    std::wstring transShown;
    if (!input_buffer_.empty()) {
        transShown = translation_buffer_.empty() ? std::wstring(kSpinner) : translation_buffer_;
    }
    if (!transShown.empty()) {
        ComPtr<ID2D1SolidColorBrush> transBrush;
        render_target_->CreateSolidColorBrush(transColor, transBrush.GetAddressOf());
        const D2D1_RECT_F transRect =
            D2D1::RectF(textLeft, midY + padY * 0.25f, w - padX, h - padY);
        if (transBrush) {
            render_target_->DrawTextW(transShown.c_str(), static_cast<UINT32>(transShown.length()),
                                      trans_format_.Get(), transRect, transBrush.Get(),
                                      D2D1_DRAW_TEXT_OPTIONS_CLIP);
        }
    }

    if (FAILED(render_target_->EndDraw())) {
        return;  // device loss: the next OnPaint rebuilds via BindDC
    }

    // Push the DIB to the layered window. Premultiplied source (D2D produced it)
    // + AC_SRC_ALPHA is what gives us the anti-aliased rounded corners over the
    // live desktop. UpdateLayeredWindow also moves/sizes the window in one shot.
    POINT ptSrc{0, 0};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    UpdateLayeredWindow(hwnd_, nullptr, &origin_, &size_, mem_dc_, &ptSrc, 0, &blend, ULW_ALPHA);
}
