// asset_paths.cpp — see asset_paths.hpp. Moved verbatim out of main.cpp.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "asset_paths.hpp"

namespace rt {

std::string exe_dir() {
    wchar_t buf[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, buf, MAX_PATH) == 0) return {};
    std::wstring w(buf);
    if (const size_t slash = w.find_last_of(L"\\/"); slash != std::wstring::npos) {
        w.resize(slash + 1);
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr,
                                      0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), out.data(), n, nullptr,
                        nullptr);
    return out;
}

bool path_exists(const std::string& p) {
    return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES;
}

std::string resolve_asset_dir(const std::string& dir, const char* probe) {
    if (path_exists(dir + "/" + probe)) return dir;
    const std::string beside = exe_dir() + dir;
    if (path_exists(beside + "/" + probe)) return beside;
    return dir;
}

std::string resolve_asset_file(const std::string& path, const char* fallback_name) {
    if (!path.empty() && path_exists(path)) return path;
    const std::string beside = exe_dir() + fallback_name;
    if (path_exists(beside)) return beside;
    return path;
}

bool relaunch_self() {
    wchar_t exe[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, exe, MAX_PATH) == 0) return false;
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + std::wstring(exe) + L"\"";
    if (!CreateProcessW(exe, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

}  // namespace rt
