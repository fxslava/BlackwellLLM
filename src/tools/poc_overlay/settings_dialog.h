#pragma once
#include <windows.h>

#include <functional>

#include "config.h"

// Opens the WebView2-based settings window. It is modeless -- pumped by the main
// message loop -- and single-instance (a second call just refocuses the existing
// window). `current` seeds the form. `onApply` is invoked on the UI thread each
// time the user saves, with the just-persisted Config, so the caller can rebind
// hotkeys / model settings in memory immediately.
//
// Requires the Microsoft Edge WebView2 Runtime to be installed; if it is missing
// the function shows a message box and returns without opening a window.
void ShowSettingsWindow(HWND owner, HINSTANCE hInstance, const Config& current,
                        std::function<void(const Config&)> onApply);
