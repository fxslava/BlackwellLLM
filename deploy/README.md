# Blackwell Deployment Suite

Production packaging for **Blackwell Overlay (Live Translation Prosthetic)** — a
native C++/CUDA Win32 application. This directory is the entire release pipeline:
a staging orchestrator, an Inno Setup installer with a hardware-audit gate, and
this guide.

| File | Role |
|---|---|
| [`make_staging.py`](make_staging.py) | Resolves and assembles `build_staging/` from the Release build tree. |
| [`installer.iss`](installer.iss) | Inno Setup script: hardware audit + `config.json` synthesis + WebView2 deployment. |
| `README.md` | This document. |

---

## Prerequisites (build machine)

- **Visual Studio 2022** (MSVC v143) + Windows 10/11 SDK.
- **CUDA Toolkit 12.8+** (13.x recommended — required for native `sm_120`
  Blackwell SASS). `%CUDA_PATH%` must point at the toolkit root.
- **Python 3.9+** (standard library only; no third-party packages).
- **Inno Setup 6.3+** — provides `ISCC.exe` (the command-line compiler).
- **Windows SDK signing tools** — `signtool.exe` (see [Code Signing](#4-code-signing-mandatory)).
- Optional: `deps/MicrosoftEdgeWebview2Setup.exe` (Evergreen bootstrapper). If
  absent, the installer relies on the runtime already shipping with modern Windows.

---

## The three-step pipeline

```text
  ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
  │ 1. CMake Release      │──▶│ 2. make_staging.py    │──▶│ 3. ISCC installer.iss │
  │    (build the .exe +  │   │    (assemble a clean  │   │    (audit host + emit │
  │     core.dll fatbin)  │   │     build_staging/)   │   │     Setup.exe)        │
  └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

### 1. CMake Release build

Build the shipping artifacts with the release preset. The root `CMakeLists.txt`
enforces the enterprise profile automatically: **static CRT (`/MT`)**, the
**AOT fatbinary matrix** (`sm_75/86/89/90` + `90-virtual` PTX + native `sm_120`),
and **aggressive Release optimization** (`/O2 /Ob3 /GL` + `/LTCG /OPT:REF,ICF`).

```bat
:: from a VS Developer / vcvars64 shell, at the repo root
cmake --preset x64-release
cmake --build --preset x64-release --target poc_overlay blackwell_core
```

Outputs land under `out/build/x64-Release/…` (the MSVC multi-config generator
scatters them into per-config subfolders — step 2 resolves them by search, so no
exact path is hardcoded).

> **Smoke-test the binary before packaging it.** The executable ships a headless
> self-check (`--smoke-test`) that validates CUDA init + device availability, a
> `WH_KEYBOARD_LL` dry-run register/unregister (catches silent AV/EDR hook
> blocks), the UI Automation COM stack, and `config.json` + model-path read
> access. It returns `0` on success or a stage-specific code (`1`=CUDA,
> `2`=hook, `3`=UIAutomation, `4`=config/model) within ~2 s:
>
> ```bat
> blackwell_overlay.exe --smoke-test  &  echo exit=%ERRORLEVEL%
> ```
>
> Wire this into CI and into a post-install verification step.

### 2. Stage the payload

`make_staging.py` builds a clean, relocatable `build_staging/` and resolves every
dependency dynamically — the executable + `blackwell_core.dll`, exactly one
`cudart64_*.dll` isolated from `%CUDA_PATH%` (never the whole toolkit), the
WebView2 frontend under `ui/`, and the optional bootstrapper from `deps/`.

```bat
python deploy\make_staging.py            :: default: harvest x64-Release
python deploy\make_staging.py --strict   :: promote soft warnings to hard errors
```

Resulting tree:

```text
build_staging/
├── blackwell_overlay.exe
├── blackwell_core.dll        (embedded CUDA fatbinaries)
├── cudart64_1X.dll           (portable CUDA runtime, version auto-detected)
├── cublas64_*.dll            (staged only if the engine links them)
├── ui/                       (WebView2 HTML/CSS/JS)
└── MicrosoftEdgeWebview2Setup.exe   (optional, if present in deps/)
```

### 3. Compile the installer

```bat
ISCC.exe deploy\installer.iss
```

`installer.iss` runs a **multi-layer hardware audit** before writing any file —
64-bit Windows 10/11 only, an NVIDIA GPU is mandatory (WMI
`Win32_VideoController`), a ≥6 GB VRAM advisory, and an SSD-vs-HDD latency check
(WMI `MSFT_PhysicalDisk.MediaType`) on the chosen AI-data volume. It then writes
`{app}\config.json` with the C++-escaped model path and silently deploys the
WebView2 runtime if it is missing. Output: `Output\BlackwellOverlaySetup-<ver>.exe`.

---

## 4. Code Signing (MANDATORY)

**An unsigned build that installs a global keyboard hook will be quarantined.**
SmartScreen, Defender, and third-party AV treat unsigned executables that call
`SetWindowsHookEx(WH_KEYBOARD_LL, …)` as keyloggers. Sign **both** the application
binaries *and* the finished installer, using an **EV** or OV **Authenticode**
certificate (an EV cert clears SmartScreen reputation immediately; an OV cert
accrues reputation over time and installs).

Sign the payload **after staging, before compiling the installer** — then sign
the installer itself:

```bat
:: (a) sign the shipped binaries in build_staging\
signtool sign ^
    /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 ^
    /n "Your Company, Inc." ^
    "build_staging\blackwell_overlay.exe" "build_staging\blackwell_core.dll"

:: (b) compile the installer (see step 3), then sign the installer executable
signtool sign ^
    /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 ^
    /n "Your Company, Inc." ^
    "Output\BlackwellOverlaySetup-1.0.0.exe"

:: (c) verify the Authenticode chain + timestamp
signtool verify /pa /v "Output\BlackwellOverlaySetup-1.0.0.exe"
```

- **`/tr` + `/td` (RFC 3161 timestamp)** are non-negotiable: without a timestamp
  the signature expires with the certificate and every already-installed copy
  goes "unverified."
- **Sign inside-out**: binaries first, installer last, so the installer's
  signature covers already-signed contents.
- Inno Setup can drive `signtool` automatically via a `SignTool` directive if you
  prefer signing as part of the ISCC run — but the two-phase manual flow above is
  explicit and CI-friendly.

---

## 5. Antivirus / Defender Whitelisting

A low-level keyboard hook is **inherently** a false-positive magnet. Even
correctly signed, expect first-week detections. Mitigate proactively:

### 5.1 Submit to Microsoft (Defender / SmartScreen)

Submit the **signed installer and the signed `blackwell_overlay.exe`** to the
**Microsoft Security Intelligence** portal for analysis and false-positive
clearance:

- **Portal:** <https://www.microsoft.com/en-us/wdsi/filesubmission>
- Choose **"Software developer"** and **"I believe this file is incorrectly
  detected (false positive)."**
- Attach the signed binaries, state the certificate subject, and describe the
  legitimate use of `WH_KEYBOARD_LL` (accessibility / live-translation caret
  tracking — it observes keystroke *timing* to debounce, it does not exfiltrate
  input).
- Re-submit **every** newly signed release; reputation is keyed to the exact hash.

### 5.2 Broaden coverage

- **VirusTotal** — upload post-signing to see which engines flag it; contact the
  specific vendors' false-positive forms for any stragglers.
- **Third-party AV FP portals** — the large vendors (e.g. Kaspersky, ESET,
  Avast/AVG, Bitdefender, Norton) each run their own "report a false positive"
  submission; file with the ones your user base runs.
- **Enterprise deployments** — ship a documented **Defender exclusion** (path or
  publisher/certificate exclusion) for IT admins, and provide the signing
  certificate thumbprint so they can whitelist by publisher rather than by path.

### 5.3 Reduce the trigger surface

- Keep the hook callback minimal and non-blocking (the app already offloads all
  work off the hook thread — see `main.cpp`), which avoids the behavioral
  heuristics that flag keyloggers (bulk buffering, network sends from the hook).
- Never write captured keystrokes to disk or the network from the hook path.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `make_staging.py`: *"%CUDA_PATH% is not set"* | Install the CUDA Toolkit or set `CUDA_PATH` to its root. |
| `make_staging.py`: *"no 'cudart64_*.dll' found"* | CUDA 13 relocated the redist to `bin\x64\`; the script already probes both — check the toolkit install is complete. |
| `make_staging.py` WARN: *"dynamic VC CRT"* | The build was `/MD`, not `/MT`. Reconfigure with the updated root `CMakeLists.txt` and rebuild Release. |
| Installer aborts: *"No NVIDIA GPU detected"* | Non-NVIDIA host, or WMI `Win32_VideoController` is stripped. Expected on unsupported hardware. |
| `--smoke-test` returns `1` | CUDA driver too old for the runtime, or the fatbinary has no image for the installed GPU. Update the driver / verify the arch matrix. |
| `--smoke-test` returns `2` | AV/EDR is silently blocking `WH_KEYBOARD_LL` — apply the whitelisting steps above. |
