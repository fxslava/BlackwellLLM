#pragma once
// -----------------------------------------------------------------------------
// asset_paths.hpp — resolve the app's assets against the EXE's own directory,
// not the caller's CWD, and relaunch the process when a restart-tier setting
// changes.
//
// THE BUG THIS FIXES. A relative path like "data" is resolved against whoever
// launched us, so the app used to start under F5 (where CMake anchors the
// debugger's working directory) and die from Explorer with
// `cannot open: data/mel_filters.bin`. The build deploys every asset next to the
// binary, so these prefer THAT copy and fall back to the CWD-relative one --
// which keeps an explicitly passed --data-dir, and a developer running out of the
// source tree, working exactly as before.
// -----------------------------------------------------------------------------
#include <string>

namespace rt {

// The directory this executable lives in, with a trailing separator. Empty on a
// Win32 failure, which every caller below degrades through rather than treats as
// fatal.
std::string exe_dir();

bool path_exists(const std::string& p);

// `dir` is usable if it holds `probe`; otherwise try <exe dir>/<dir>. Returns
// `dir` UNCHANGED when neither works, so the caller still reports the original
// path in its error rather than a rewritten one the user never typed.
std::string resolve_asset_dir(const std::string& dir, const char* probe);

// Same idea for a single file (the VAD model, whose default is a configure-time
// absolute path that does not survive being copied to another machine).
std::string resolve_asset_file(const std::string& path, const char* fallback_name);

// Relaunch this executable with NO arguments and let the current process exit.
//
// Restart-tier settings choose what gets ALLOCATED at bring-up, so applying them
// means a new engine -- and the honest way to get one is a new process, not a
// hot-swap of a 5.3 GB weight set under a live decode loop. Deliberately argv-
// free: every setting is persisted by the time this runs, and replaying the old
// CLI flags would re-override the very values the user just saved.
//
// MUST be called LAST, after this process has released the CUDA context, the
// capture device and the WebView2 user-data folder -- starting the new instance
// any earlier has two processes fighting over all three.
bool relaunch_self();

}  // namespace rt
