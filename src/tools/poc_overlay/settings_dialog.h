#pragma once
#include <windows.h>

#include "hook_manager.h"  // Shortcut

// Shows a modal Win32 settings dialog letting the user rebind the commit/trigger
// shortcut via a standard msctls_hotkey32 control. The dialog is built from an
// in-memory DLGTEMPLATE (no .rc resource is compiled for this PoC).
//
// `shortcut` is used to seed the control and, on OK, is overwritten with the new
// binding. Returns true if the user accepted (OK), false on Cancel/close.
bool ShowSettingsDialog(HWND owner, HINSTANCE hInstance, Shortcut& shortcut);
