#pragma once
// -----------------------------------------------------------------------------
// hotkey_spec.hpp — "Ctrl+Alt+Space" <-> {MOD_*, VK_*}, in ONE place.
//
// WHY IT MOVED OUT OF assistant_window.cpp. The parser was an anonymous-namespace
// detail of the window, which was right while the window was the only thing that
// needed it. Two more callers now do: the settings VALIDATOR has to know whether
// a chord parses and whether it collides with something Windows owns, and the
// console overlay registers a chord of its own. A second copy of vk_from_name()
// is a second table to keep in step with the first, and the first is the one that
// gets fixed -- so there is one, here.
//
// WHAT A CHORD MEANS. vk == 0 is "this hotkey is off": an empty string, a chord
// with no non-modifier key, and an unknown key name all produce it. There is
// deliberately no partially-registered outcome and no error return -- a hotkey
// that cannot be honoured is simply absent, and the validator is what tells the
// user so.
//
// Windows only; this is Win32 vocabulary end to end.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdlib>
#include <string>

namespace rt {

struct Hotkey {
    UINT mods = 0;   // MOD_* flags
    UINT vk = 0;     // 0 == "no hotkey" (empty or unparseable)

    [[nodiscard]] bool armed() const noexcept { return vk != 0; }
    // Identity for conflict detection. Two specs that differ only in spelling
    // ("ctrl+alt+a" vs "Ctrl+Alt+A") must compare EQUAL, which is exactly what
    // comparing the parsed form rather than the string gives.
    [[nodiscard]] bool operator==(const Hotkey& o) const noexcept {
        return mods == o.mods && vk == o.vk;
    }
};

// Uppercase ASCII compare; the page sends what the user typed and case is not a
// meaningful distinction in a key name.
inline bool hotkey_iequals(const std::string& a, const char* b) {
    size_t i = 0;
    for (; i < a.size() && b[i] != '\0'; ++i) {
        const char ca = (a[i] >= 'a' && a[i] <= 'z') ? static_cast<char>(a[i] - 32) : a[i];
        const char cb = (b[i] >= 'a' && b[i] <= 'z') ? static_cast<char>(b[i] - 32) : b[i];
        if (ca != cb) return false;
    }
    return i == a.size() && b[i] == '\0';
}

// One key NAME (no modifiers) -> virtual-key code. 0 means "not a key we accept",
// which is what makes an unparseable chord register nothing rather than register
// something the user did not ask for.
inline UINT vk_from_name(const std::string& name) {
    if (name.size() == 1) {
        const char c = name[0];
        if (c >= 'a' && c <= 'z') return static_cast<UINT>(c - 32);
        if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return static_cast<UINT>(c);
    }
    if (name.size() >= 2 && (name[0] == 'F' || name[0] == 'f')) {
        const int n = std::atoi(name.c_str() + 1);
        if (n >= 1 && n <= 24) return static_cast<UINT>(VK_F1 + n - 1);
    }
    static const struct { const char* name; UINT vk; } kNamed[] = {
        {"Space", VK_SPACE},   {"Enter", VK_RETURN},  {"Return", VK_RETURN},
        {"Tab", VK_TAB},       {"Escape", VK_ESCAPE}, {"Esc", VK_ESCAPE},
        {"Backspace", VK_BACK},{"Delete", VK_DELETE}, {"Insert", VK_INSERT},
        {"Home", VK_HOME},     {"End", VK_END},       {"PageUp", VK_PRIOR},
        {"PageDown", VK_NEXT}, {"Left", VK_LEFT},     {"Right", VK_RIGHT},
        {"Up", VK_UP},         {"Down", VK_DOWN},     {"Pause", VK_PAUSE},
        // THE CONSOLE KEY. `~` and the backtick share one physical key, and its
        // virtual-key code is layout-dependent -- VK_OEM_3 on US layouts, which
        // is what a Quake-style console is defined against. Both spellings and
        // both names resolve to it so a chord typed as "Ctrl+`" and one typed as
        // "Ctrl+~" are the same hotkey.
        {"Tilde", VK_OEM_3},   {"~", VK_OEM_3},       {"`", VK_OEM_3},
        {"Backquote", VK_OEM_3}, {"Grave", VK_OEM_3},
    };
    for (const auto& k : kNamed) {
        if (hotkey_iequals(name, k.name)) return k.vk;
    }
    return 0;
}

// "Ctrl+Alt+Space" -> {MOD_CONTROL|MOD_ALT, VK_SPACE}.
//
// '+' and ' ' separate tokens. '-' does NOT, and that is a fix rather than a
// simplification: it used to, which made "Ctrl+-" (and every chord naming a
// literal minus) parse as a modifier followed by nothing.
inline Hotkey parse_hotkey(const std::string& spec) {
    Hotkey hk;
    std::string token;
    auto take = [&] {
        if (token.empty()) return;
        if (hotkey_iequals(token, "Ctrl") || hotkey_iequals(token, "Control")) {
            hk.mods |= MOD_CONTROL;
        } else if (hotkey_iequals(token, "Alt")) {
            hk.mods |= MOD_ALT;
        } else if (hotkey_iequals(token, "Shift")) {
            hk.mods |= MOD_SHIFT;
        } else if (hotkey_iequals(token, "Win") || hotkey_iequals(token, "Super")) {
            hk.mods |= MOD_WIN;
        } else {
            hk.vk = vk_from_name(token);   // last non-modifier token wins
        }
        token.clear();
    };
    for (const char c : spec) {
        if (c == '+' || c == ' ') take();
        else token.push_back(c);
    }
    take();
    return hk;
}

// Is the chord still physically down? Used by the hold-to-talk watchdog. The
// modifiers are checked too, so releasing Ctrl while keeping Space held ends the
// hold -- which is what a user who let go of the chord means.
inline bool hotkey_is_down(const Hotkey& hk) {
    if (hk.vk == 0) return false;
    auto down = [](int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
    if (!down(static_cast<int>(hk.vk))) return false;
    if ((hk.mods & MOD_CONTROL) && !down(VK_CONTROL)) return false;
    if ((hk.mods & MOD_ALT) && !down(VK_MENU)) return false;
    if ((hk.mods & MOD_SHIFT) && !down(VK_SHIFT)) return false;
    if ((hk.mods & MOD_WIN) && !down(VK_LWIN) && !down(VK_RWIN)) return false;
    return true;
}

}  // namespace rt
