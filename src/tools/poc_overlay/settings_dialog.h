#pragma once
#include <windows.h>

#include <functional>

#include "config.h"

// On-demand JIT pre-cache request from the settings UI. `target` is the pair's
// target-language name (e.g. L"Chinese"); the receiver compiles that branch in
// the background and MUST invoke `done(ok, prefilledTokens)` exactly once --
// from any thread (the settings window marshals the result back to its own UI
// thread and forwards it to the page).
using PrecacheHandler =
    std::function<void(const std::wstring& target, std::function<void(bool, int)> done)>;

// Opens the WebView2-based settings window. It is modeless -- pumped by the main
// message loop -- and single-instance (a second call just refocuses the existing
// window). `current` seeds the form. `onApply` is invoked on the UI thread each
// time the user saves, with the just-persisted Config, so the caller can rebind
// hotkeys / model settings / language pairs in memory immediately (pairs
// hot-reload -- no restart). `onPrecache` may be empty (the button then fails
// gracefully).
//
// Requires the Microsoft Edge WebView2 Runtime to be installed; if it is missing
// the function shows a message box and returns without opening a window.
void ShowSettingsWindow(HWND owner, HINSTANCE hInstance, const Config& current,
                        std::function<void(const Config&)> onApply,
                        PrecacheHandler onPrecache = {});
