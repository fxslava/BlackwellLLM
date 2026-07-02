#pragma once
#include <windows.h>
#include <commctrl.h>  // HOTKEYF_*

#include <string>

#include "hook_manager.h"  // Shortcut

// Persisted application configuration. Shortcuts use the same HOTKEYF_* / vk
// encoding as everything else in this PoC (see Shortcut).
struct Config {
    Shortcut activationShortcut{HOTKEYF_CONTROL | HOTKEYF_SHIFT, VK_SPACE};
    Shortcut commitShortcut{HOTKEYF_CONTROL, VK_RETURN};
    std::wstring modelPath;
    int contextSize = 4096;
};

namespace ConfigStore {
// <exe directory>\config.json
std::wstring DefaultPath();
// Loads config.json; returns defaults if it is missing or malformed.
Config Load();
// Rewrites config.json. Returns false on I/O failure.
bool Save(const Config& config);
}  // namespace ConfigStore

// UTF-8 <-> UTF-16 helpers -- JSON (and the WebView2 bridge) is UTF-8, Win32 is
// UTF-16. Kept inline so both config.cpp and settings_dialog.cpp can share them.
inline std::string ToUtf8(const std::wstring& w) {
    if (w.empty()) {
        return {};
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                       nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr,
                        nullptr);
    return s;
}

inline std::wstring FromUtf8(const std::string& s) {
    if (s.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}
