#pragma once

// Post-deploy smoke test: a headless, ~2-second validation of the four runtime
// preconditions the overlay silently depends on -- CUDA, the low-level keyboard
// hook, the UI Automation COM stack, and a readable model path in config.json.
//
// Contract: invoked at the very top of wWinMain, BEFORE any window/COM/engine
// state is created, so a failing check never leaves half-initialized global
// state behind. Returns a process exit code (0 = all green, >0 = the first
// failed stage's code) suitable for `installer.iss` / CI to branch on.
namespace smoke {

// True iff the process command line contains the `--smoke-test` switch. Cheap;
// parses GetCommandLineW() and does not touch any subsystem.
bool WantsSmokeTest();

// Runs the full validation suite headlessly and returns the exit code. Distinct
// non-zero codes per stage (see SmokeStage) so an operator can tell WHICH
// precondition failed from %ERRORLEVEL% alone.
int RunSmokeTest();

}  // namespace smoke
