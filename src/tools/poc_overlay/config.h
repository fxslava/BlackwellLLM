#pragma once
#include <windows.h>
#include <commctrl.h>  // HOTKEYF_*

#include <string>
#include <vector>

#include "hook_manager.h"  // Shortcut

// A translation direction cycled by the language switcher. The source language
// is auto-detected by the model, so only the TARGET drives the system prompt
// (and thus which cached radix branch is hit); `label` is purely for the HUD.
struct LanguagePair {
    std::wstring label;   // "EN -> RU" -- shown in the switcher HUD
    std::wstring target;  // "Russian" -- used verbatim in the system prompt
};

// How far back from the caret the live capture area extends. Boundaries nest:
// a newline always ends the capture; Sentence additionally stops at . ! ?;
// Clause additionally stops at , ; : (so Clause is the tightest capture).
enum class CaptureGranularity { Clause, Sentence, Paragraph };

inline const char* ToString(CaptureGranularity g) {
    switch (g) {
        case CaptureGranularity::Clause:    return "clause";
        case CaptureGranularity::Sentence:  return "sentence";
        case CaptureGranularity::Paragraph: return "paragraph";
    }
    return "sentence";
}

inline CaptureGranularity CaptureGranularityFromString(const std::string& s,
                                                       CaptureGranularity fallback) {
    if (s == "clause") return CaptureGranularity::Clause;
    if (s == "sentence") return CaptureGranularity::Sentence;
    if (s == "paragraph") return CaptureGranularity::Paragraph;
    return fallback;
}

// Persisted application configuration. Shortcuts use the same HOTKEYF_* / vk
// encoding as everything else in this PoC (see Shortcut). The inference fields
// mirror the playground server's generation settings.
struct Config {
    Shortcut activationShortcut{HOTKEYF_ALT | HOTKEYF_SHIFT, 'T'};      // Alt+Shift+T (master toggle)
    Shortcut commitShortcut{HOTKEYF_CONTROL, VK_RETURN};               // Ctrl+Enter
    Shortcut cycleLanguageShortcut{HOTKEYF_ALT | HOTKEYF_SHIFT, 'L'};  // Alt+Shift+L (cycle direction)

    // Translation directions. Each compiles its own .bkv branch on first run
    // (the radix tree dedups the shared prefix) and gets a global Alt+<N> force-
    // override hotkey (Alt+1 = index 0, Alt+2 = index 1, ...). Editable in the
    // Settings UI; engine/branch changes take effect on the next app start.
    std::vector<LanguagePair> languagePairs{
        {L"RU -> EN", L"English"},
        {L"RU -> ZH", L"Chinese"},
    };
    int activeLanguage = 0;  // index into languagePairs

    std::wstring modelPath;    // weights directory / file
    int contextSize = 4096;    // KV-cache / context window (tokens)
    float temperature = 0.7f;  // sampling temperature   [0.0 .. 2.0]
    float topP = 0.95f;        // nucleus sampling cutoff [0.0 .. 1.0]
    int maxTokens = 1024;      // max tokens to generate

    // Capture / debounce UX (the stateful overlay pipeline).
    CaptureGranularity captureGranularity = CaptureGranularity::Sentence;
    int idleTimerMs = 700;  // typing-idle debounce before preview inference starts

    // Developer Mode: overlay debug instrumentation. When true the overlay draws
    // what the OS hooks + UIA actually see (the focused element's bounding box in
    // red, the exact caret rect in green) and paints a per-token probability
    // heatmap ("thermograd") behind the translation. Off by default -- it adds a
    // full-vocab logits read per decoded token, so it is a debugging aid, not a
    // steady-state path.
    bool developerMode = false;

    // Startup & inactivity lifecycle (the dual-stage memory state machine).
    // activateOnStartup: true = warm the engine and enter Translation Mode at
    // launch; false = start inactive and defer loading model weights into VRAM
    // until the first activation. The timeouts are SECONDS of typing/selection
    // inactivity (the Settings UI offers a minutes/seconds unit selector and
    // always persists seconds; legacy *Min keys are migrated on load): stage 1
    // spills the KV prefix cache (radix tree) down the tier waterfall to disk;
    // stage 2 soft-hibernates -- weights leave GPU VRAM for pinned host RAM
    // (the engine object survives; wakeup is a PCIe DMA burst).
    bool activateOnStartup = false;
    int kvSpillTimeoutSec = 600;     // stage 1: KV disk spill
    int hibernateTimeoutSec = 1800;  // stage 2: soft hibernation (weights -> RAM)

    // Tiered KV prefix-cache memory budget (engine RuntimeConfig knobs; one
    // block = one KV page = 16 tokens across all layers).
    int vramCacheBlocks = 1024;    // device-pool floor kept for cached prefixes
    int ramTierBlocks = 2048;      // pinned host-RAM demotion tier capacity
    bool diskSpillEnabled = true;  // NVMe spill tier on/off
    int diskSpillBlocks = 8192;    // spill file capacity, in blocks
    std::wstring spillFilePath;    // empty = ConfigStore::DefaultSpillPath()
};

namespace ConfigStore {
// <exe directory>\config.json
std::wstring DefaultPath();
// %LOCALAPPDATA%\Blackwell\spill.bkv -- the default NVMe spill-tier backing
// file (used when Config::spillFilePath is empty). The directory is created
// on demand by the caller that opens the file.
std::wstring DefaultSpillPath();
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
