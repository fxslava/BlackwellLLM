// Workspace: provisions and tears down an isolated, disposable sandbox for one
// agent session.
//
// The preferred backing is `git worktree add`: it gives the agent a real,
// fully-buildable checkout of the repository under out/agent_sessions/<id>/, on
// its own throwaway branch, completely detached from the user's working tree and
// index. The agent can edit, compile, run tests and even commit inside it; the
// main repo's HEAD, working tree and stash are never touched. A plain
// (non-git) fallback is offered for sandboxes that do not need version control.
//
// Lifetime is RAII: the destructor -- including during stack unwinding from an
// exception or a crash handler -- removes the worktree and deletes the directory,
// so a session that blows up mid-flight never leaves orphaned checkouts or stray
// branches behind. Move-only.
#ifndef BLACKWELL_AGENT_ENV_WORKSPACE_H
#define BLACKWELL_AGENT_ENV_WORKSPACE_H

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "sandbox_fs.h"
#include "subprocess.h"

namespace agent::env {

class Workspace {
public:
    // Create a git-worktree sandbox of `repo` (which must be a git work tree) on
    // a fresh `agent/session-<id>` branch. `sessions_root` defaults to
    // <repo>/out/agent_sessions. Returns nullopt if git is unavailable, `repo`
    // is not a git repository, or the worktree could not be created.
    static std::optional<Workspace> create(
        const std::filesystem::path& repo,
        std::filesystem::path sessions_root = {});

    // Create a plain (version-control-free) sandbox directory under
    // `sessions_root`. Useful for scratch work or when no repo exists yet.
    static std::optional<Workspace> create_plain(
        const std::filesystem::path& sessions_root);

    ~Workspace();
    Workspace(Workspace&&) noexcept;
    Workspace& operator=(Workspace&&) noexcept;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    const std::filesystem::path& path() const { return path_; }
    const std::string& id() const { return id_; }
    bool git_backed() const { return git_backed_; }
    bool active() const { return active_; }

    // A path-confined filesystem facade rooted at this sandbox.
    SandboxFs fs() const { return SandboxFs(path_); }

    // Run an arbitrary command with the working directory defaulted to the
    // sandbox root.
    ProcessResult run(const std::string& exe,
                      const std::vector<std::string>& args,
                      unsigned timeout_ms = 0) const;

    // Convenience git wrapper executed inside the sandbox (`git -C <path> ...`).
    ProcessResult git(const std::vector<std::string>& args,
                      unsigned timeout_ms = 0) const;

    // Explicit teardown; also invoked by the destructor. Idempotent and safe to
    // call after a move. Removes the worktree registration (if any) and deletes
    // the directory tree.
    void destroy();

private:
    Workspace() = default;

    std::filesystem::path path_;    // the sandbox directory
    std::filesystem::path repo_;    // origin repo (for `git worktree remove`)
    std::string id_;                // session id
    std::string branch_;            // throwaway branch name (git-backed only)
    bool git_backed_ = false;
    bool active_ = false;
};

// Returns true if a usable `git` is on PATH. Exposed so tests / callers can skip
// git-dependent paths cleanly.
bool git_available();

}  // namespace agent::env

#endif  // BLACKWELL_AGENT_ENV_WORKSPACE_H
