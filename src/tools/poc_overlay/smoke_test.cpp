// smoke_test: headless post-deploy validation. See smoke_test.h for the contract.
//
// Why each check exists (deployment failure modes this catches on a clean box):
//   [1] CUDA        -- wrong/absent driver, no CUDA device, or a fatbinary with
//                      no SASS/PTX the installed GPU can run. Proven by actually
//                      standing up a runtime context, not just querying the count.
//   [2] WH_KEYBOARD_LL -- Antivirus / Windows Defender / EDR frequently SILENTLY
//                      block global low-level hooks. SetWindowsHookExW returning
//                      NULL is the only signal, and it never surfaces to the user
//                      at runtime (the app just "does nothing on hotkeys").
//   [3] UI Automation -- the accessibility COM stack can be disabled or broken on
//                      hardened/stripped SKUs; the caret tracker is dead without it.
//   [4] config/model -- the installer writes config.json + an AI-data path; verify
//                      it parses and the model directory is actually READABLE by
//                      this user (ACLs, moved drive, etc.).
//
// The routine is deliberately self-contained: it initializes and tears down its
// OWN COM apartment and never constructs the engine / overlay / real hook chain.

#include "smoke_test.h"

#include <windows.h>
#include <objbase.h>
#include <shellapi.h>  // CommandLineToArgvW
#include <uiautomation.h>

#include <cuda_runtime.h>

#include <cstdio>  // swprintf_s
#include <string>

#include "config.h"

namespace smoke {
namespace {

// Exit codes -- kept stable; documented in deploy/README.md and consumed by CI.
enum SmokeStage : int {
    kOk           = 0,
    kFailCuda     = 1,
    kFailHook     = 2,
    kFailUia      = 3,
    kFailConfig   = 4,
};

// Diagnostics sink. A WIN32-subsystem process has no console of its own, so we
// attach to the PARENT console when there is one (interactive run / CI) and
// always mirror to the debugger. Absent a console, checks still run and the
// exit code is authoritative -- the operator loses only the human-readable log.
bool g_haveConsole = false;

void ReportInit() {
    // ATTACH_PARENT_PROCESS: -1. Succeeds only when launched from a console.
    g_haveConsole = (AttachConsole(ATTACH_PARENT_PROCESS) != FALSE);
}

void Report(const std::wstring& line) {
    const std::wstring text = line + L"\r\n";
    OutputDebugStringW(text.c_str());
    if (g_haveConsole) {
        const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        if (out != INVALID_HANDLE_VALUE && out != nullptr) {
            DWORD written = 0;
            WriteConsoleW(out, text.c_str(), static_cast<DWORD>(text.size()), &written, nullptr);
        }
    }
}

// ----------------------------------------------------------------------------
// [1] CUDA runtime + device availability.
// ----------------------------------------------------------------------------
bool CheckCuda() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        Report(L"[FAIL] CUDA: cudaGetDeviceCount -> " +
               std::wstring(FromUtf8(cudaGetErrorString(err))));
        return false;
    }
    if (deviceCount <= 0) {
        Report(L"[FAIL] CUDA: no CUDA-capable device visible to the runtime.");
        return false;
    }

    // Bind + force actual context creation: this is what fails when the shipped
    // fatbinary has no code image the installed GPU can execute, or the driver
    // is too old for the runtime we link. A pure query would miss both.
    err = cudaSetDevice(0);
    if (err == cudaSuccess) {
        err = cudaFree(nullptr);  // no-op alloc call that materializes the context
    }
    if (err != cudaSuccess) {
        Report(L"[FAIL] CUDA: context init on device 0 -> " +
               std::wstring(FromUtf8(cudaGetErrorString(err))));
        return false;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        Report(L"[PASS] CUDA: " + FromUtf8(prop.name) + L" (sm_" +
               std::to_wstring(prop.major) + std::to_wstring(prop.minor) + L"), " +
               std::to_wstring(deviceCount) + L" device(s).");
    } else {
        Report(L"[PASS] CUDA: " + std::to_wstring(deviceCount) + L" device(s).");
    }
    // Leave no lingering context around before the process may go on to real work.
    cudaDeviceReset();
    return true;
}

// ----------------------------------------------------------------------------
// [2] WH_KEYBOARD_LL dry-run register / unregister.
// ----------------------------------------------------------------------------
LRESULT CALLBACK NoOpLowLevelKeyboardProc(int code, WPARAM wParam, LPARAM lParam) {
    // Pure pass-through: the smoke test installs and immediately removes this; it
    // must never swallow or mutate input even in the microseconds it is live.
    return CallNextHookEx(nullptr, code, wParam, lParam);
}

bool CheckKeyboardHook() {
    const HHOOK hook = SetWindowsHookExW(WH_KEYBOARD_LL, &NoOpLowLevelKeyboardProc,
                                         GetModuleHandleW(nullptr), 0);
    if (hook == nullptr) {
        const DWORD gle = GetLastError();
        Report(L"[FAIL] WH_KEYBOARD_LL: SetWindowsHookExW blocked (GetLastError=" +
               std::to_wstring(gle) +
               L"). Antivirus/EDR is likely intercepting global hooks -- whitelist "
               L"the executable.");
        return false;
    }
    UnhookWindowsHookEx(hook);
    Report(L"[PASS] WH_KEYBOARD_LL: install/uninstall succeeded (no silent block).");
    return true;
}

// ----------------------------------------------------------------------------
// [3] UI Automation COM stack.  Caller owns the COM apartment (see RunSmokeTest).
// ----------------------------------------------------------------------------
bool CheckUiAutomation() {
    IUIAutomation* automation = nullptr;
    const HRESULT hr = CoCreateInstance(CLSID_CUIAutomation, nullptr, CLSCTX_INPROC_SERVER,
                                        IID_IUIAutomation, reinterpret_cast<void**>(&automation));
    if (FAILED(hr) || automation == nullptr) {
        wchar_t buf[32];
        swprintf_s(buf, L"0x%08lX", static_cast<unsigned long>(hr));
        Report(L"[FAIL] UIAutomation: CoCreateInstance(CLSID_CUIAutomation) -> HRESULT " + std::wstring(buf));
        return false;
    }
    automation->Release();
    Report(L"[PASS] UIAutomation: IUIAutomation instantiated.");
    return true;
}

// ----------------------------------------------------------------------------
// [4] config.json parse + model-path read permission.
// ----------------------------------------------------------------------------
bool CheckConfigAndModelPath() {
    // ConfigStore::Load() reads <exe dir>\config.json and returns defaults on a
    // missing/malformed file, so a parse failure here surfaces as an empty path.
    const Config cfg = ConfigStore::Load();
    if (cfg.modelPath.empty()) {
        Report(L"[FAIL] config.json: model_path is empty (missing or unparseable config).");
        return false;
    }

    const DWORD attr = GetFileAttributesW(cfg.modelPath.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) {
        Report(L"[FAIL] model path: '" + cfg.modelPath +
               L"' does not exist or is not accessible.");
        return false;
    }

    // Prove READ permission, not just existence: enumerate the directory (or stat
    // the file). ACL denial surfaces here as ERROR_ACCESS_DENIED.
    if ((attr & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        WIN32_FIND_DATAW fd{};
        const HANDLE find = FindFirstFileW((cfg.modelPath + L"\\*").c_str(), &fd);
        if (find == INVALID_HANDLE_VALUE) {
            Report(L"[FAIL] model path: '" + cfg.modelPath +
                   L"' is not readable (GetLastError=" + std::to_wstring(GetLastError()) + L").");
            return false;
        }
        FindClose(find);
    } else {
        const HANDLE file = CreateFileW(cfg.modelPath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            Report(L"[FAIL] model path: '" + cfg.modelPath +
                   L"' is not readable (GetLastError=" + std::to_wstring(GetLastError()) + L").");
            return false;
        }
        CloseHandle(file);
    }

    Report(L"[PASS] config.json parsed; model path readable: " + cfg.modelPath);
    return true;
}

}  // namespace

bool WantsSmokeTest() {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv == nullptr) {
        return false;
    }
    bool wants = false;
    for (int i = 1; i < argc; ++i) {
        if (lstrcmpiW(argv[i], L"--smoke-test") == 0) {
            wants = true;
            break;
        }
    }
    LocalFree(argv);
    return wants;
}

int RunSmokeTest() {
    ReportInit();
    Report(L"=== Blackwell Overlay :: post-deploy smoke test ===");

    // Single-threaded apartment for the UIA client, torn down before we return so
    // the (aborted) normal startup path never inherits our COM state.
    const HRESULT coHr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool ownsCom = SUCCEEDED(coHr);

    int result = kOk;

    // Fail fast, most-fundamental first: without CUDA nothing else matters, and a
    // distinct exit code per stage lets the installer/CI pinpoint the fault.
    if (!CheckCuda()) {
        result = kFailCuda;
    } else if (!CheckKeyboardHook()) {
        result = kFailHook;
    } else if (!CheckUiAutomation()) {
        result = kFailUia;
    } else if (!CheckConfigAndModelPath()) {
        result = kFailConfig;
    }

    if (ownsCom) {
        CoUninitialize();
    }

    Report(result == kOk ? L"=== SMOKE TEST PASSED (exit 0) ==="
                         : L"=== SMOKE TEST FAILED (exit " + std::to_wstring(result) + L") ===");
    return result;
}

}  // namespace smoke
