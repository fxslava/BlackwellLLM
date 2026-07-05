#include "text_injector.h"

#include <UIAutomation.h>

#include <vector>

namespace TextInjector {

namespace {

// Small settle times: without them the target app can miss the freshly-set
// clipboard, or we can restore the old clipboard before the paste consumes it.
constexpr DWORD kClipboardSettleMs = 20;  // after SetClipboardData, before Ctrl+V
constexpr DWORD kPasteConsumeMs = 60;     // after Ctrl+V, before restoring clipboard
constexpr DWORD kSelectSettleMs = 20;     // after Select()/Shift+Left, before paste

// ---- SendInput helpers (every event tagged so our own hook ignores it) ------

void AppendKey(std::vector<INPUT>& events, WORD vk, bool keyUp, bool extended = false) {
    INPUT input{};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = vk;
    input.ki.dwFlags = (keyUp ? KEYEVENTF_KEYUP : 0) | (extended ? KEYEVENTF_EXTENDEDKEY : 0);
    input.ki.dwExtraInfo = kInjectedSignature;
    events.push_back(input);
}

// Release any physically-held modifier so synthesized chords are clean. Critical
// because the commit shortcut (e.g. Ctrl+Enter) may still have Ctrl down, which
// would turn Shift+Left into Ctrl+Shift+Left (select-word) or corrupt the paste.
void AppendModifierRelease(std::vector<INPUT>& events) {
    for (WORD vk : {VK_LCONTROL, VK_RCONTROL, VK_LSHIFT, VK_RSHIFT, VK_LMENU, VK_RMENU}) {
        AppendKey(events, vk, /*keyUp=*/true);
    }
}

void SendCtrlV() {
    std::vector<INPUT> events;
    AppendModifierRelease(events);
    AppendKey(events, VK_CONTROL, /*keyUp=*/false);
    AppendKey(events, 'V', /*keyUp=*/false);
    AppendKey(events, 'V', /*keyUp=*/true);
    AppendKey(events, VK_CONTROL, /*keyUp=*/true);
    SendInput(static_cast<UINT>(events.size()), events.data(), sizeof(INPUT));
}

void SendShiftLeft(size_t count) {
    std::vector<INPUT> events;
    events.reserve(count * 2 + 8);
    AppendModifierRelease(events);
    AppendKey(events, VK_SHIFT, /*keyUp=*/false);
    for (size_t i = 0; i < count; ++i) {
        AppendKey(events, VK_LEFT, /*keyUp=*/false, /*extended=*/true);
        AppendKey(events, VK_LEFT, /*keyUp=*/true, /*extended=*/true);
    }
    AppendKey(events, VK_SHIFT, /*keyUp=*/true);
    SendInput(static_cast<UINT>(events.size()), events.data(), sizeof(INPUT));
}

// Ctrl+End -- moves the caret to the very end of the control. Used after a Tier-1
// SetValue, which otherwise leaves the caret parked at the start of the field.
void SendCaretToEnd() {
    std::vector<INPUT> events;
    AppendModifierRelease(events);
    AppendKey(events, VK_CONTROL, /*keyUp=*/false);
    AppendKey(events, VK_END, /*keyUp=*/false, /*extended=*/true);
    AppendKey(events, VK_END, /*keyUp=*/true, /*extended=*/true);
    AppendKey(events, VK_CONTROL, /*keyUp=*/true);
    SendInput(static_cast<UINT>(events.size()), events.data(), sizeof(INPUT));
}

// ---- Clipboard helpers ------------------------------------------------------

bool OpenClipboardWithRetry() {
    for (int attempt = 0; attempt < 10; ++attempt) {
        if (OpenClipboard(nullptr)) {
            return true;
        }
        Sleep(10);  // another app briefly holds the clipboard
    }
    return false;
}

// Reads CF_UNICODETEXT into `out`. NOTE: only unicode text is backed up/restored
// -- non-text clipboard payloads (images/files) are not preserved. Adequate for
// this PoC's paste-swap.
bool GetClipboardText(std::wstring& out) {
    if (!OpenClipboardWithRetry()) {
        return false;
    }
    bool ok = false;
    if (HANDLE handle = GetClipboardData(CF_UNICODETEXT)) {
        if (const auto* text = static_cast<const wchar_t*>(GlobalLock(handle))) {
            out.assign(text);
            GlobalUnlock(handle);
            ok = true;
        }
    }
    CloseClipboard();
    return ok;
}

bool SetClipboardText(const std::wstring& text) {
    if (!OpenClipboardWithRetry()) {
        return false;
    }
    bool ok = false;
    EmptyClipboard();
    const size_t bytes = (text.size() + 1) * sizeof(wchar_t);
    if (HGLOBAL mem = GlobalAlloc(GMEM_MOVEABLE, bytes)) {
        if (void* dst = GlobalLock(mem)) {
            memcpy(dst, text.c_str(), bytes);
            GlobalUnlock(mem);
            if (SetClipboardData(CF_UNICODETEXT, mem)) {
                ok = true;  // clipboard now owns `mem`
            }
        }
        if (!ok) {
            GlobalFree(mem);  // ownership not transferred
        }
    }
    CloseClipboard();
    return ok;
}

// Backup clipboard -> set our text -> Ctrl+V -> restore. Used by tiers 2 and 3.
bool PasteViaClipboard(const std::wstring& text) {
    std::wstring backup;
    const bool hadBackup = GetClipboardText(backup);

    if (!SetClipboardText(text)) {
        return false;
    }
    Sleep(kClipboardSettleMs);
    SendCtrlV();
    Sleep(kPasteConsumeMs);  // let the foreground app consume the paste

    // Restore the original clipboard (best effort). If there was no unicode text,
    // clear ours rather than leaving the translation behind.
    SetClipboardText(hadBackup ? backup : std::wstring());
    return true;
}

}  // namespace

Tier Replace(const Request& request) {
    if (request.source.empty() && request.replacement.empty() && !request.selectionActive) {
        return Tier::None;
    }

    // Tier 1: ValuePattern SetValue -- only when the whole control value equals
    // the text we replace, so we never clobber surrounding/other-line context.
    if (request.valuePattern) {
        BOOL readOnly = TRUE;
        if (SUCCEEDED(request.valuePattern->get_CurrentIsReadOnly(&readOnly)) && !readOnly) {
            BSTR current = nullptr;
            if (SUCCEEDED(request.valuePattern->get_CurrentValue(&current)) && current) {
                const std::wstring value(current, SysStringLen(current));
                SysFreeString(current);
                if (value == request.source) {
                    BSTR replacement = SysAllocString(request.replacement.c_str());
                    const HRESULT hr = request.valuePattern->SetValue(replacement);
                    SysFreeString(replacement);
                    if (SUCCEEDED(hr)) {
                        // SetValue drops the caret at the field start -- move it to
                        // the end of the new text so the user keeps typing there.
                        Sleep(kSelectSettleMs);
                        SendCaretToEnd();
                        return Tier::ValuePattern;
                    }
                }
            }
        }
    }

    // Tier 2: select the exact UIA range in the OS, then paste over it.
    if (request.selectionRange) {
        if (SUCCEEDED(request.selectionRange->Select())) {
            Sleep(kSelectSettleMs);
            if (PasteViaClipboard(request.replacement)) {
                return Tier::RangeSelectPaste;
            }
        }
    }

    // Tier 4: selection-commit fallback -- the OS/user already has the target
    // highlighted, so paste straight over it. MUST come before the Shift+Left
    // tier: re-selecting a live drag-selection by keystroke would move or clear
    // the very range we mean to overwrite.
    if (request.selectionActive) {
        if (PasteViaClipboard(request.replacement)) {
            return Tier::SelectionPaste;
        }
        return Tier::None;
    }

    // Tier 3: UIA-opaque typing commit -- select N characters left of the caret,
    // then paste. Chromium collapses synthetic VK_BACK spam, so we select-then-
    // paste (Shift+Left x N, then Ctrl+V) instead of backspacing.
    if (!request.source.empty()) {
        SendShiftLeft(request.source.size());
        Sleep(kSelectSettleMs);
        if (PasteViaClipboard(request.replacement)) {
            return Tier::KeystrokeSelectPaste;
        }
    }

    return Tier::None;
}

}  // namespace TextInjector
