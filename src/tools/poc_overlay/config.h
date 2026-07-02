#pragma once
#include <windows.h>
#include <commctrl.h>  // HOTKEYF_*

#include <string>

#include "hook_manager.h"  // Shortcut

// Persisted application configuration. Shortcuts use the same HOTKEYF_* / vk
// encoding as everything else in this PoC (see Shortcut). The inference fields
// mirror the playground server's generation settings.
struct Config {
    Shortcut activationShortcut{HOTKEYF_CONTROL | HOTKEYF_SHIFT, 'T'};  // Ctrl+Shift+T
    Shortcut commitShortcut{HOTKEYF_CONTROL, VK_RETURN};                // Ctrl+Enter

    std::wstring modelPath;    // weights directory / file
    int contextSize = 4096;    // KV-cache / context window (tokens)
    float temperature = 0.7f;  // sampling temperature   [0.0 .. 2.0]
    float topP = 0.95f;        // nucleus sampling cutoff [0.0 .. 1.0]
    int maxTokens = 1024;      // max tokens to generate
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
