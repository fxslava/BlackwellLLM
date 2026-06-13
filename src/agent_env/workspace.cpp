#include "workspace.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace agent::env {

namespace fs = std::filesystem;

namespace {

unsigned long current_pid() {
#ifdef _WIN32
    return static_cast<unsigned long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

// A filesystem- and git-ref-safe session id: YYYYMMDD-HHMMSS-<pid>-<counter>.
// The monotonic counter disambiguates sessions created within the same second.
std::string make_session_id() {
    static std::atomic<unsigned> counter{0};
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);

    return std::string(stamp) + "-" + std::to_string(current_pid()) + "-" +
           std::to_string(counter.fetch_add(1));
}

}  // namespace

bool git_available() {
    return run_process("git", {"--version"}).ok();
}

std::optional<Workspace> Workspace::create(const fs::path& repo,
                                           fs::path sessions_root) {
    if (!git_available()) return std::nullopt;

    // Confirm `repo` really is a git work tree before we touch anything.
    const auto check = run_process(
        "git", {"-C", repo.string(), "rev-parse", "--is-inside-work-tree"});
    if (!check.ok()) return std::nullopt;

    if (sessions_root.empty()) sessions_root = repo / "out" / "agent_sessions";

    std::error_code ec;
    fs::create_directories(sessions_root, ec);
    if (ec) return std::nullopt;

    Workspace ws;
    ws.id_ = make_session_id();
    ws.repo_ = repo;
    ws.path_ = sessions_root / ws.id_;
    ws.branch_ = "agent/session-" + ws.id_;
    ws.git_backed_ = true;

    // Create the worktree on a brand-new branch pointed at the current HEAD.
    const auto add = run_process(
        "git", {"-C", repo.string(), "worktree", "add", "-b", ws.branch_,
                ws.path_.string()});
    if (!add.ok()) return std::nullopt;

    ws.active_ = true;
    return ws;
}

std::optional<Workspace> Workspace::create_plain(const fs::path& sessions_root) {
    std::error_code ec;
    fs::create_directories(sessions_root, ec);
    if (ec) return std::nullopt;

    Workspace ws;
    ws.id_ = make_session_id();
    ws.path_ = sessions_root / ws.id_;
    ws.git_backed_ = false;

    fs::create_directories(ws.path_, ec);
    if (ec) return std::nullopt;

    ws.active_ = true;
    return ws;
}

Workspace::~Workspace() { destroy(); }

Workspace::Workspace(Workspace&& other) noexcept
    : path_(std::move(other.path_)),
      repo_(std::move(other.repo_)),
      id_(std::move(other.id_)),
      branch_(std::move(other.branch_)),
      git_backed_(other.git_backed_),
      active_(other.active_) {
    other.active_ = false;  // the moved-from husk must not tear anything down
}

Workspace& Workspace::operator=(Workspace&& other) noexcept {
    if (this != &other) {
        destroy();  // release whatever we currently own first
        path_ = std::move(other.path_);
        repo_ = std::move(other.repo_);
        id_ = std::move(other.id_);
        branch_ = std::move(other.branch_);
        git_backed_ = other.git_backed_;
        active_ = other.active_;
        other.active_ = false;
    }
    return *this;
}

ProcessResult Workspace::run(const std::string& exe,
                             const std::vector<std::string>& args,
                             unsigned timeout_ms) const {
    ProcessOptions opts;
    opts.cwd = path_;
    opts.timeout_ms = timeout_ms;
    return run_process(exe, args, opts);
}

ProcessResult Workspace::git(const std::vector<std::string>& args,
                             unsigned timeout_ms) const {
    std::vector<std::string> full{"-C", path_.string()};
    full.insert(full.end(), args.begin(), args.end());
    return run_process("git", full, {{}, timeout_ms});
}

void Workspace::destroy() {
    if (!active_) return;
    active_ = false;  // make destroy() idempotent / re-entrancy-safe up front

    if (git_backed_) {
        // Unregister the worktree from the origin repo. --force because the
        // sandbox may carry uncommitted edits we are deliberately discarding.
        run_process("git", {"-C", repo_.string(), "worktree", "remove",
                            "--force", path_.string()});
        // Drop the throwaway branch so sessions don't accrete dead refs.
        run_process("git", {"-C", repo_.string(), "branch", "-D", branch_});
        // Belt-and-braces: clear any stale administrative entries.
        run_process("git", {"-C", repo_.string(), "worktree", "prune"});
    }

    // Whatever git did or didn't manage, ensure the directory is gone.
    std::error_code ec;
    fs::remove_all(path_, ec);
}

}  // namespace agent::env
