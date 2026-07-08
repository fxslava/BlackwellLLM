#!/usr/bin/env python3
# =============================================================================
#  Blackwell Deployment Suite  --  make_staging.py
# -----------------------------------------------------------------------------
#  Stage-1 of the deployment pipeline: assemble a self-contained, relocatable
#  `build_staging/` tree that the Inno Setup compiler (installer.iss) packs into
#  the final installer executable.
#
#  Design contract (the "linker" phase of deployment):
#    * Resolve every artifact DYNAMICALLY -- no absolute, machine-specific paths
#      are ever baked in. The MSVC multi-config generator scatters outputs across
#      per-config subfolders (`.../Release/foo.exe`), so binaries are located by a
#      recursive best-candidate search rather than a fixed relative path.
#    * Isolate exactly ONE CUDA runtime redistributable (`cudart64_*.dll`) next to
#      the executable -- the C++ engine links the CUDA Runtime dynamically but we
#      refuse to ship a multi-gigabyte toolkit. The AOT fatbinaries (sm_75/86/89/90
#      + sm_89 virtual PTX) live inside blackwell_core.dll and need no toolkit.
#    * The C++ runtime itself is statically linked (/MT), so no VC redist DLLs are
#      staged -- this is asserted, not assumed (see verify_static_crt()).
#
#  The script is idempotent: it wipes and rebuilds `build_staging/` on each run.
#
#  Usage:
#      python deploy/make_staging.py                 # defaults: x64-Release
#      python deploy/make_staging.py --config Debug  # stage a debug drop
#      python deploy/make_staging.py --strict        # fail hard on soft warnings
# =============================================================================

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
#  Deployment manifest -- the single source of truth for artifact naming.
#  These names are the *product* (shipping) identities. The current dev target is
#  historically `poc_overlay`; we accept it as a fallback so the pipeline works
#  before the target is renamed, without hardcoding the transitional name.
# -----------------------------------------------------------------------------
PRIMARY_EXE_NAME   = "blackwell_overlay.exe"
FALLBACK_EXE_NAMES = ("poc_overlay.exe",)          # transitional dev-target name
CORE_DLL_NAME      = "blackwell_core.dll"          # carries the CUDA fatbinaries
WEBVIEW2_BOOTSTRAP = "MicrosoftEdgeWebview2Setup.exe"

# Frontend asset sources, tried in order. `src/ui/dist/` is the target layout;
# `src/tools/poc_overlay/web/` is where the current WebView2 assets actually live.
UI_SOURCE_CANDIDATES = (
    Path("src") / "ui" / "dist",
    Path("src") / "tools" / "poc_overlay" / "web",
)

# CUDA runtime DLLs the engine may pull in. cudart is mandatory; the rest are
# staged opportunistically only if the engine was linked against them.
CUDART_GLOB           = "cudart64_*.dll"
OPTIONAL_CUDA_GLOBS   = ("cublas64_*.dll", "cublasLt64_*.dll", "cudnn64_*.dll")

# DirectStorage runtime DLLs. When USE_DIRECT_STORAGE=ON, CMake's
# blackwell_copy_dstorage_dlls() copies these next to every consumer exe as a
# POST_BUILD step, so the shipped exe hard-depends on them at load time. When DS
# is OFF they are simply absent next to the exe and genuinely not needed.
DSTORAGE_DLLS         = ("dstorage.dll", "dstoragecore.dll")

# Log symbols -- ASCII-only so this survives a non-UTF-8 legacy console codepage.
OK, WARN, ERR, INFO = "[ OK ]", "[WARN]", "[FAIL]", "[ .. ]"


# -----------------------------------------------------------------------------
#  Diagnostics
# -----------------------------------------------------------------------------
class StagingError(RuntimeError):
    """Fatal, unrecoverable staging failure -- aborts the pipeline."""


def log(symbol: str, msg: str) -> None:
    print(f"  {symbol}  {msg}", flush=True)


def die(msg: str) -> "None":
    raise StagingError(msg)


# -----------------------------------------------------------------------------
#  Path resolution primitives
# -----------------------------------------------------------------------------
def repo_root() -> Path:
    """Repo root == parent of this script's `deploy/` directory.

    Anchored to __file__, never to the CWD, so the pipeline runs identically from
    a CI runner, an IDE build step, or an interactive shell.
    """
    return Path(__file__).resolve().parent.parent


def find_binary(search_root: Path, names: tuple[str, ...], config: str) -> Path:
    """Recursively locate a build artifact, preferring the requested config.

    The MSVC generator emits `<target>/<Config>/<name>` with per-config folders,
    so we cannot address a single deterministic path. Candidate ranking:
        1. path contains the requested config segment (e.g. '/Release/')  -> +2
        2. newest mtime wins ties (freshest build)
    Returns the highest-ranked existing file, or raises if none match.
    """
    if not search_root.is_dir():
        die(f"build output directory not found: {search_root}\n"
            f"         Build the Release target first "
            f"(cmake --build --preset x64-release --target poc_overlay).")

    cfg_token = os.sep + config.lower() + os.sep
    candidates: list[Path] = []
    for name in names:
        candidates.extend(p for p in search_root.rglob(name) if p.is_file())

    if not candidates:
        pretty = " / ".join(names)
        die(f"could not locate '{pretty}' anywhere under {search_root}")

    def rank(p: Path) -> tuple[int, float]:
        score = 2 if cfg_token in (os.sep + str(p).lower() + os.sep) else 0
        try:
            return (score, p.stat().st_mtime)
        except OSError:
            return (score, 0.0)

    best = max(candidates, key=rank)
    if cfg_token not in (os.sep + str(best).lower() + os.sep):
        log(WARN, f"no '{config}' build of {best.name} found; "
                  f"using closest match: {best}")
    return best


def copy_file(src: Path, dst_dir: Path, rename_to: str | None = None) -> Path:
    """Copy a single file into dst_dir, preserving metadata; returns the dest."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (rename_to or src.name)
    shutil.copy2(src, dst)
    return dst


# -----------------------------------------------------------------------------
#  Static-CRT assertion (/MT contract)
# -----------------------------------------------------------------------------
def verify_static_crt(exe: Path, strict: bool) -> None:
    """Best-effort check that the binary is NOT dynamically linked to the VC CRT.

    With /MT there must be no `VCRUNTIME140*.dll` / `MSVCP140*.dll` import. We scan
    the raw image for those import strings -- a heuristic, but enough to catch a
    misconfigured /MD build that would otherwise ship a broken-on-clean-machine
    installer. On a clean box a missing VC redist is a classic launch failure.
    """
    try:
        blob = exe.read_bytes()
    except OSError as exc:                       # unreadable -> can't assert, skip
        log(WARN, f"could not read {exe.name} for CRT check: {exc}")
        return

    markers = (b"VCRUNTIME140", b"MSVCP140", b"api-ms-win-crt")
    hits = sorted({m.decode() for m in markers if m in blob})
    if hits:
        msg = (f"{exe.name} appears to import the dynamic VC CRT ({', '.join(hits)}). "
               f"The deployment contract requires static linking (/MT).")
        if strict:
            die(msg)
        log(WARN, msg + " -- staging anyway (pass --strict to enforce).")
    else:
        log(OK, f"{exe.name}: static CRT confirmed (no dynamic VC runtime imports).")


# -----------------------------------------------------------------------------
#  Staging steps
# -----------------------------------------------------------------------------
def stage_application_binaries(root: Path, build_dir: Path, staging: Path,
                               config: str, strict: bool) -> Path:
    """Task 1: the executable + the core engine DLL (with embedded fatbinaries).

    Returns the directory the executable was resolved from -- the authoritative
    location for the runtime DLLs CMake copies next to the exe (DirectStorage).
    """
    log(INFO, "Resolving application binaries...")
    exe = find_binary(build_dir, (PRIMARY_EXE_NAME, *FALLBACK_EXE_NAMES), config)
    dll = find_binary(build_dir, (CORE_DLL_NAME,), config)

    # Normalize the shipped exe name to the product identity regardless of the
    # transitional dev-target name it was built under.
    staged_exe = copy_file(exe, staging, rename_to=PRIMARY_EXE_NAME)
    staged_dll = copy_file(dll, staging)
    log(OK, f"{exe.relative_to(root)}  ->  {staged_exe.name}")
    log(OK, f"{dll.relative_to(root)}  ->  {staged_dll.name}")

    verify_static_crt(staged_exe, strict)
    return exe.parent


def stage_directstorage_runtime(root: Path, exe_dir: Path, build_dir: Path,
                                staging: Path) -> None:
    """Stage the DirectStorage runtime DLLs (dstorage.dll, dstoragecore.dll).

    These are a hard load-time dependency of the engine whenever it was built with
    USE_DIRECT_STORAGE=ON -- CMake stages them next to every consumer exe, so the
    copies beside the resolved executable are the authoritative, arch-correct
    (x64) ones. Falling back to a tree search deliberately EXCLUDES the SDK's
    _deps/native/bin/{ARM64,x86} folders. If the DLLs are absent next to the exe
    the build simply did not enable DirectStorage, so they are not needed and
    their absence is not an error (even under --strict).
    """
    log(INFO, "Staging DirectStorage runtime...")
    missing: list[str] = []
    for name in DSTORAGE_DLLS:
        candidate = exe_dir / name
        if not candidate.is_file():
            # Fallback: an x64 copy elsewhere in the build tree, never ARM64/x86.
            hits = [p for p in build_dir.rglob(name)
                    if p.is_file() and "arm64" not in str(p).lower()
                    and f"{os.sep}x86{os.sep}" not in str(p).lower()]
            # Prefer a copy CMake placed next to an exe over the raw SDK payload.
            hits.sort(key=lambda p: ("_deps" in p.parts, len(p.parts)))
            candidate = hits[0] if hits else None
        if candidate is None:
            missing.append(name)
            continue
        copy_file(candidate, staging)
        log(OK, f"DirectStorage: {candidate.relative_to(root)}  ->  {name}")

    if len(missing) == len(DSTORAGE_DLLS):
        log(INFO, "DirectStorage DLLs not present next to the exe -- this build "
                  "did not enable USE_DIRECT_STORAGE; nothing to stage.")
    elif missing:
        # A partial set is a genuine packaging fault -- one half of a hard dep.
        log(WARN, f"DirectStorage partially staged; missing: {', '.join(missing)}. "
                  f"The app will fail to load if it imports DirectStorage.")


def stage_cuda_runtime(staging: Path) -> None:
    """Task 2: isolate exactly one cudart64_*.dll from %CUDA_PATH%\\bin.

    We ship the runtime redistributable ONLY -- never the toolkit. The version
    (12.x) is discovered from the filename, not assumed, so a toolkit bump needs
    no script change.
    """
    log(INFO, "Isolating portable CUDA runtime...")
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        die("%CUDA_PATH% is not set -- cannot locate the CUDA runtime.\n"
            "         Install the CUDA Toolkit or point CUDA_PATH at its root.")

    cuda_bin = Path(cuda_path) / "bin"
    if not cuda_bin.is_dir():
        die(f"CUDA bin directory does not exist: {cuda_bin}")

    # The redist DLL location moved across toolkit majors: CUDA <=12 ships it in
    # bin/, CUDA 13+ ships it in bin/x64/. Probe both, then fall back to a bounded
    # recursive search so a future relayout does not break the pipeline.
    search_dirs = [cuda_bin, cuda_bin / "x64"]
    cudart: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            cudart.extend(d.glob(CUDART_GLOB))
    if not cudart:
        cudart = [p for p in cuda_bin.rglob(CUDART_GLOB) if p.is_file()]
    if not cudart:
        die(f"no '{CUDART_GLOB}' found under {cuda_bin} (checked bin/ and bin/x64/) "
            f"-- broken or unexpected CUDA installation.")
    if len({p.name for p in cudart}) > 1:
        log(WARN, f"multiple CUDA runtime versions present, choosing newest: "
                  f"{sorted({p.name for p in cudart})}")
    chosen = max(cudart, key=lambda p: p.stat().st_mtime)
    copy_file(chosen, staging)
    log(OK, f"CUDA runtime: {chosen.name}  (from {chosen.parent})")

    # Opportunistically stage optional CUDA math libs IF the engine links them.
    # Search the same directory the runtime came from (co-located redists).
    staged_optional: set[str] = set()
    for d in (chosen.parent, *search_dirs):
        if not d.is_dir():
            continue
        for pattern in OPTIONAL_CUDA_GLOBS:
            for lib in d.glob(pattern):
                if lib.name not in staged_optional:
                    copy_file(lib, staging)
                    staged_optional.add(lib.name)
                    log(OK, f"optional CUDA lib: {lib.name}")


def stage_frontend_assets(root: Path, staging: Path, strict: bool) -> None:
    """Task 3: copy the compiled WebView2 frontend into build_staging/ui/."""
    log(INFO, "Staging WebView2 frontend assets...")
    for candidate in UI_SOURCE_CANDIDATES:
        src = root / candidate
        if src.is_dir() and any(src.iterdir()):
            dst = staging / "ui"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            n = sum(1 for _ in dst.rglob("*") if _.is_file())
            log(OK, f"frontend: {candidate}  ->  ui/  ({n} files)")
            return

    msg = (f"no frontend assets found (looked in: "
           f"{', '.join(str(c) for c in UI_SOURCE_CANDIDATES)}). "
           f"Build the UI (e.g. `npm run build`) before staging.")
    if strict:
        die(msg)
    log(WARN, msg + " -- shipping without bundled UI.")


def stage_webview2_bootstrapper(root: Path, staging: Path) -> None:
    """Task 4: bundle the Evergreen WebView2 bootstrapper if present in deps/.

    Optional by design: the installer falls back to an online fetch when the
    bootstrapper is absent, so a missing file is a soft warning, never fatal.
    """
    log(INFO, "Staging WebView2 Evergreen bootstrapper...")
    src = root / "deps" / WEBVIEW2_BOOTSTRAP
    if src.is_file():
        copy_file(src, staging)
        log(OK, f"WebView2 bootstrapper: deps/{WEBVIEW2_BOOTSTRAP}")
    else:
        log(WARN, f"deps/{WEBVIEW2_BOOTSTRAP} not present -- the installer will "
                  f"rely on its online WebView2 fallback.")


# -----------------------------------------------------------------------------
#  Orchestration
# -----------------------------------------------------------------------------
def reset_staging_dir(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blackwell Deployment Suite -- stage the installer payload.")
    p.add_argument("--config", default="Release",
                   help="Build configuration to harvest (default: Release).")
    p.add_argument("--build-dir", default=None,
                   help="Override the CMake build tree "
                        "(default: out/build/x64-<config>).")
    p.add_argument("--staging", default="build_staging",
                   help="Output staging directory (default: build_staging).")
    p.add_argument("--strict", action="store_true",
                   help="Promote soft warnings (missing UI, dynamic CRT) to errors.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()

    # Default build tree mirrors the CMake preset layout: out/build/x64-<Config>.
    build_dir = (Path(args.build_dir) if args.build_dir
                 else root / "out" / "build" / f"x64-{args.config}")
    if not build_dir.is_absolute():
        build_dir = (root / build_dir).resolve()

    staging = Path(args.staging)
    if not staging.is_absolute():
        staging = (root / staging).resolve()

    print("=" * 70)
    print("  Blackwell Deployment Suite  --  staging pipeline")
    print("=" * 70)
    log(INFO, f"repo root   : {root}")
    log(INFO, f"build tree  : {build_dir}")
    log(INFO, f"staging out : {staging}")
    log(INFO, f"config      : {args.config}  |  strict: {args.strict}")
    print("-" * 70)

    try:
        reset_staging_dir(staging)
        exe_dir = stage_application_binaries(root, build_dir, staging,
                                             args.config, args.strict)
        stage_directstorage_runtime(root, exe_dir, build_dir, staging)
        stage_cuda_runtime(staging)
        stage_frontend_assets(root, staging, args.strict)
        stage_webview2_bootstrapper(root, staging)
    except StagingError as exc:
        print("-" * 70)
        log(ERR, str(exc))
        print("=" * 70)
        return 1
    except OSError as exc:                        # disk full, ACL denial, etc.
        print("-" * 70)
        log(ERR, f"filesystem error during staging: {exc}")
        print("=" * 70)
        return 1

    print("-" * 70)
    log(OK, f"Staging complete -> {staging}")
    log(INFO, "Next: compile deploy/installer.iss with the Inno Setup compiler "
              "(ISCC.exe).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
