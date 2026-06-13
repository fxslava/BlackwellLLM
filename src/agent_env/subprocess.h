// Subprocess: the agent's only sanctioned door to process execution.
//
// Architectural choice (see src/agent_env/CMakeLists.txt for the full rationale):
// a thin, dependency-free wrapper over the native process API -- Win32
// CreateProcess on Windows, fork/exec on POSIX -- rather than a heavy library
// such as Boost.Process or reproc. Every external tool the agent drives (git,
// ninja, cmake, the compiled test binaries) goes through run_process(), which:
//   * takes an argv vector with NO shell parsing, so the agent cannot smuggle
//     `&& rm -rf` or pipe redirection through an argument;
//   * captures stdout and stderr fully and *separately*;
//   * returns the real process exit code;
//   * closes the child's stdin (no accidental hangs waiting on input);
//   * supports an optional wall-clock timeout with forced termination.
#ifndef BLACKWELL_AGENT_ENV_SUBPROCESS_H
#define BLACKWELL_AGENT_ENV_SUBPROCESS_H

#include <filesystem>
#include <string>
#include <vector>

namespace agent::env {

struct ProcessResult {
    bool launched = false;   // false if the executable could not even be started
    bool timed_out = false;  // true if killed because it exceeded the timeout
    int exit_code = -1;      // child exit status (process-specific on timeout)
    std::string out;         // full captured stdout
    std::string err;         // full captured stderr

    // Convenience: a clean, completed run.
    bool ok() const { return launched && !timed_out && exit_code == 0; }
};

struct ProcessOptions {
    std::filesystem::path cwd;   // working directory; empty = inherit caller's
    unsigned timeout_ms = 0;     // 0 = wait indefinitely
};

// Run `exe` (resolved against PATH) with `args` as a literal argv. No shell is
// involved. Blocks until the child exits or the timeout fires.
ProcessResult run_process(const std::string& exe,
                          const std::vector<std::string>& args,
                          const ProcessOptions& opts = {});

}  // namespace agent::env

#endif  // BLACKWELL_AGENT_ENV_SUBPROCESS_H
