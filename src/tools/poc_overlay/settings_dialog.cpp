#include "settings_dialog.h"

#include <commctrl.h>

#include <vector>

namespace {

constexpr WORD kIdHotkey = 1001;

// Predefined control-class atoms used inside a DLGITEMTEMPLATE (see the
// "Predefined System Classes" the dialog manager understands).
constexpr WORD kAtomButton = 0x0080;
constexpr WORD kAtomStatic = 0x0082;

// --- In-memory DLGTEMPLATE builder -----------------------------------------
// A dialog template is a DLGTEMPLATE header followed by menu/class/title (and,
// with DS_SETFONT, a font) as WORD-aligned UTF-16 arrays, then one
// DWORD-aligned DLGITEMTEMPLATE per control. We assemble the raw bytes by hand
// so no .rc file has to be added to the build.

template <typename T>
void Append(std::vector<BYTE>& buf, T value) {
    const auto* p = reinterpret_cast<const BYTE*>(&value);
    buf.insert(buf.end(), p, p + sizeof(T));
}

// Appends a null-terminated UTF-16 string (each code unit as a WORD).
void AppendString(std::vector<BYTE>& buf, const wchar_t* s) {
    for (const wchar_t* p = s;; ++p) {
        Append<WORD>(buf, static_cast<WORD>(*p));
        if (*p == L'\0') {
            break;
        }
    }
}

void AlignToDword(std::vector<BYTE>& buf) {
    while (buf.size() % 4 != 0) {
        buf.push_back(0);
    }
}

// Emits one DLGITEMTEMPLATE. `className` may be either a predefined atom (pass
// its value in `classAtom`, leaving `className` null) or a class-name string.
void AppendControl(std::vector<BYTE>& buf, DWORD style, short x, short y, short cx, short cy,
                   WORD id, WORD classAtom, const wchar_t* className, const wchar_t* text) {
    AlignToDword(buf);
    Append<DWORD>(buf, style);
    Append<DWORD>(buf, 0);  // extended style
    Append<short>(buf, x);
    Append<short>(buf, y);
    Append<short>(buf, cx);
    Append<short>(buf, cy);
    Append<WORD>(buf, id);

    if (className) {
        AppendString(buf, className);
    } else {
        Append<WORD>(buf, 0xFFFF);
        Append<WORD>(buf, classAtom);
    }
    AppendString(buf, text ? text : L"");
    Append<WORD>(buf, 0);  // no creation-data
}

std::vector<BYTE> BuildTemplate() {
    std::vector<BYTE> buf;

    // DLGTEMPLATE header.
    Append<DWORD>(buf, DS_SETFONT | DS_MODALFRAME | DS_CENTER | WS_POPUP | WS_CAPTION | WS_SYSMENU);
    Append<DWORD>(buf, 0);   // extended style
    Append<WORD>(buf, 4);    // control count
    Append<short>(buf, 0);   // x
    Append<short>(buf, 0);   // y
    Append<short>(buf, 210); // cx (dialog units)
    Append<short>(buf, 84);  // cy
    Append<WORD>(buf, 0);    // no menu
    Append<WORD>(buf, 0);    // default dialog class
    AppendString(buf, L"Blackwell PoC - Settings");
    Append<WORD>(buf, 9);    // font point size (DS_SETFONT)
    AppendString(buf, L"Segoe UI");

    const DWORD kVisibleChild = WS_CHILD | WS_VISIBLE;

    // Static label.
    AppendControl(buf, kVisibleChild | SS_LEFT, 12, 12, 186, 10, 0xFFFF, kAtomStatic, nullptr,
                  L"Commit / trigger shortcut:");
    // Hotkey control (needs ICC_HOTKEY_CLASS registered before creation).
    AppendControl(buf, kVisibleChild | WS_TABSTOP | WS_BORDER, 12, 26, 186, 14, kIdHotkey, 0,
                  L"msctls_hotkey32", nullptr);
    // OK (default) + Cancel buttons.
    AppendControl(buf, kVisibleChild | WS_TABSTOP | BS_DEFPUSHBUTTON, 96, 58, 48, 14, IDOK,
                  kAtomButton, nullptr, L"OK");
    AppendControl(buf, kVisibleChild | WS_TABSTOP | BS_PUSHBUTTON, 150, 58, 48, 14, IDCANCEL,
                  kAtomButton, nullptr, L"Cancel");

    return buf;
}

INT_PTR CALLBACK SettingsProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_INITDIALOG: {
            // lParam is the Shortcut* passed to DialogBoxIndirectParam.
            auto* shortcut = reinterpret_cast<Shortcut*>(lParam);
            SetWindowLongPtrW(hwnd, DWLP_USER, static_cast<LONG_PTR>(lParam));
            const WORD packed = MAKEWORD(static_cast<BYTE>(shortcut->vk),
                                         static_cast<BYTE>(shortcut->modifiers));
            SendDlgItemMessageW(hwnd, kIdHotkey, HKM_SETHOTKEY, packed, 0);
            return TRUE;
        }
        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case IDOK: {
                    auto* shortcut =
                        reinterpret_cast<Shortcut*>(GetWindowLongPtrW(hwnd, DWLP_USER));
                    if (shortcut) {
                        // HKM_GETHOTKEY: low byte = vk, high byte = HOTKEYF_* flags.
                        const WORD packed = static_cast<WORD>(
                            SendDlgItemMessageW(hwnd, kIdHotkey, HKM_GETHOTKEY, 0, 0));
                        shortcut->vk = LOBYTE(packed);
                        shortcut->modifiers = HIBYTE(packed);
                    }
                    EndDialog(hwnd, IDOK);
                    return TRUE;
                }
                case IDCANCEL:
                    EndDialog(hwnd, IDCANCEL);
                    return TRUE;
            }
            break;
        case WM_CLOSE:
            EndDialog(hwnd, IDCANCEL);
            return TRUE;
    }
    return FALSE;
}

}  // namespace

bool ShowSettingsDialog(HWND owner, HINSTANCE hInstance, Shortcut& shortcut) {
    // The hotkey control lives in comctl32; register its class before the
    // dialog manager tries to instantiate it from the template.
    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_HOTKEY_CLASS};
    InitCommonControlsEx(&icc);

    const std::vector<BYTE> tmpl = BuildTemplate();
    const INT_PTR result = DialogBoxIndirectParamW(
        hInstance, reinterpret_cast<const DLGTEMPLATE*>(tmpl.data()), owner, &SettingsProc,
        reinterpret_cast<LPARAM>(&shortcut));
    return result == IDOK;
}
