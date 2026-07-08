---
name: compiler-hygiene
description: The zero-warning policy for BlackwellLLM's MSVC build. Use when adding or changing C++ code, when a build fails with warning-as-error (C4xxx treated as error / C2220), when touching the warning flags in CMakeLists.txt, or when deciding how to silence vs. fix a compiler diagnostic.
---

# Compiler Hygiene — the zero-warning policy

First-party C++ compiles under **`/W4 /WX`** (warnings are errors). A warning is a bug
report from the compiler; the policy is to **fix it at the source**, not suppress it.

## How the policy is wired (root `CMakeLists.txt`)

The flags are set once, as a directory property placed **after** the `cmake/` dependency
includes so already-defined FetchContent targets (googletest, …) keep their own level:

```cmake
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
        "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>"   # <>-includes => external
        "$<$<COMPILE_LANGUAGE:CXX>:/external:W0>"              # ...at W0 (no warnings)
        "$<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>"
    )
endif()
```

Three deliberate scoping decisions — understand them before changing the block:

1. **CXX only** (`$<$<COMPILE_LANGUAGE:CXX>:…>`). CUDA (`.cu`) compilation is *untouched*:
   nvcc host-passthrough of `/W4 /WX` over the CUDA toolkit / CCCL headers is not viable
   (they are not `/W4`-clean). Kernel correctness is guarded by tests, not `/W4`.
2. **`/external:anglebrackets` + `/external:W0`.** `#include <...>` (CUDA toolkit, STL,
   Win32, third-party) is treated as *external* and drops to W0, so `/W4 /WX` fires **only
   on our own `"..."`-included code**. This is what makes `/WX` survivable on a
   CUDA + Win32 codebase — do not remove it.
3. **`_CRT_SECURE_NO_WARNINGS`.** C4996 deprecates *standard C* functions
   (`getenv`/`fopen`/…) in favour of MSVC's non-portable `*_s` variants. This code is
   cross-platform-standard, so this one non-warning is silenced globally rather than
   rewritten un-portably. This is the **only** blanket suppression that is sanctioned.

## Fixing warnings — the sanctioned moves

Fix the real thing; reach for a suppression only in the documented cases above.

| Warning | Meaning | Clean fix |
|---|---|---|
| C4100 | unreferenced formal parameter | If genuinely unused, `(void)param;` with a one-line comment saying **why** it stays in the signature. If it's a leftover, delete it (and update the declaration + callers). |
| C4244 / C4267 | narrowing conversion (e.g. `size_t`→`int`, `int`→`WORD`) | Make the types agree. For a braced-init list of `int` macros iterated as a narrower type, use a typed `static constexpr T[]` (constants that fit don't narrow). Add an explicit `static_cast<T>()` only where the truncation is intended and safe. |
| C4189 | local initialized but unused | Delete it, or `(void)` if it exists for a side effect / RAII. |
| C4459 / C4456 | declaration hides outer scope | Rename the inner variable. |
| C4996 | deprecated (usually the CRT `_s` nag) | Already handled globally by `_CRT_SECURE_NO_WARNINGS`. If it's a *real* deprecation (an actually-removed API), migrate off it. |

**Never** blanket `#pragma warning(disable: ...)` a first-party file to dodge a real
warning, and never add per-target `/W0`/`/W3` to lower the bar. If a specific third-party
header slips past `/external` and warns, wrap only that `#include` in a
`#pragma warning(push/disable/pop)` — never your own code.

## Workflow when a change trips `/WX`

1. Read the exact `warning Cxxxx` line (MSVC prints it right under the `C2220` "treated as
   an error"). It names the file, line, and cause.
2. Apply the matching clean fix above.
3. To enumerate *all* warnings at once (rather than one `/WX` stop at a time), build with
   ninja's keep-going: `cmake --build --preset x64-debug -- -k 0`, from a shell where the
   MSVC + CUDA include env is set correctly (a `build_target.bat`-style **batch file** — an
   inline `cmd /c "call vcvars && set INCLUDE=%INCLUDE%;… && cmake …"` clobbers `INCLUDE`
   because `%INCLUDE%` expands before `vcvars` runs).
4. A green full build (`EXIT 0`) is the gate. Then `ctest --preset validation`.

## The bar for new code

Any new `.cpp`/`.h` you add compiles clean at `/W4 /WX` **before** you consider it done —
the same as the existing tree, which carries zero first-party warnings (verified 2026-07).
