// Verification suite for the agent_env sandbox layer.
//
// Covers the three pillars the OS-integration layer promises:
//   * SandboxFs  -- safe read/write/patch AND hard rejection of path traversal
//                   (`..`, absolute, drive-relative) so the host never leaks.
//   * Subprocess -- a real child process is launched, its stdout/stderr captured
//                   separately and its exit code surfaced faithfully.
//   * Workspace  -- a sterile sandbox is provisioned and then fully torn down,
//                   in both the plain and the git-worktree backings.
#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "sandbox_fs.h"
#include "subprocess.h"
#include "workspace.h"

namespace fs = std::filesystem;
using namespace agent::env;

namespace {

// A throwaway directory under the OS temp area, unique per test, deleted on
// destruction regardless of how the test exits.
class TempDir {
public:
    TempDir() {
        const auto base = fs::temp_directory_path();
        for (int i = 0;; ++i) {
            auto cand = base / ("agent_env_test_" +
                                std::to_string(::testing::UnitTest::GetInstance()
                                                   ->random_seed()) +
                                "_" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
                                "_" + std::to_string(i));
            std::error_code ec;
            if (fs::create_directory(cand, ec)) {
                path_ = cand;
                return;
            }
        }
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

std::string trimmed(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s;
}

// Cross-platform helpers so the subprocess assertions read the same everywhere.
ProcessResult run_echo(const std::string& text) {
#ifdef _WIN32
    return run_process("cmd.exe", {"/c", "echo", text});
#else
    return run_process("/bin/echo", {text});
#endif
}
ProcessResult run_exit(int code) {
#ifdef _WIN32
    return run_process("cmd.exe", {"/c", "exit", std::to_string(code)});
#else
    return run_process("/bin/sh", {"-c", "exit " + std::to_string(code)});
#endif
}
ProcessResult run_to_stderr(const std::string& text) {
#ifdef _WIN32
    return run_process("cmd.exe", {"/c", "echo " + text + " 1>&2"});
#else
    return run_process("/bin/sh", {"-c", "echo " + text + " 1>&2"});
#endif
}

}  // namespace

// --- SandboxFs: I/O ----------------------------------------------------------

TEST(SandboxFs, WriteCreatesParentsAndReadsBack) {
    TempDir tmp;
    SandboxFs sb(tmp.path());

    ASSERT_TRUE(sb.write("src/nested/hello.txt", "Hello, sandbox"));
    EXPECT_TRUE(sb.exists("src/nested/hello.txt"));

    auto got = sb.read("src/nested/hello.txt");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, "Hello, sandbox");

    // It physically landed under the root, nowhere else.
    EXPECT_TRUE(fs::exists(tmp.path() / "src" / "nested" / "hello.txt"));
}

TEST(SandboxFs, PatchReplacesFirstMatchAndReportsMisses) {
    TempDir tmp;
    SandboxFs sb(tmp.path());
    ASSERT_TRUE(sb.write("code.c", "int x = 1; int x_again = 1;"));

    ASSERT_TRUE(sb.patch("code.c", "1", "42"));
    EXPECT_EQ(*sb.read("code.c"), "int x = 42; int x_again = 1;");

    // A needle that is not present is a reported failure, not a silent no-op.
    EXPECT_FALSE(sb.patch("code.c", "no_such_text", "x"));
    // ... and the file is untouched by the failed patch.
    EXPECT_EQ(*sb.read("code.c"), "int x = 42; int x_again = 1;");
}

TEST(SandboxFs, RemoveAndList) {
    TempDir tmp;
    SandboxFs sb(tmp.path());
    ASSERT_TRUE(sb.write("a.txt", "a"));
    ASSERT_TRUE(sb.write("b/c.txt", "c"));

    auto entries = sb.list(".");
    EXPECT_EQ(entries.size(), 2u);  // a.txt and b/

    ASSERT_TRUE(sb.remove("b"));
    EXPECT_FALSE(sb.exists("b/c.txt"));
    EXPECT_FALSE(sb.exists("b"));
}

// --- SandboxFs: the security guarantee --------------------------------------

TEST(SandboxFs, RejectsPathTraversal) {
    TempDir tmp;
    // Give the sandbox a subdirectory so "../" has somewhere real to escape to.
    const auto root = tmp.path() / "box";
    fs::create_directories(root);
    SandboxFs sb(root);

    // resolve() refuses anything that climbs out or is rooted elsewhere.
    EXPECT_FALSE(sb.resolve("../escape.txt").has_value());
    EXPECT_FALSE(sb.resolve("../../etc/passwd").has_value());
    EXPECT_FALSE(sb.resolve("a/b/../../../escape").has_value());
    EXPECT_FALSE(sb.resolve("/etc/passwd").has_value());
#ifdef _WIN32
    EXPECT_FALSE(sb.resolve("C:\\Windows\\System32\\x").has_value());
    EXPECT_FALSE(sb.resolve("\\\\server\\share\\x").has_value());
#endif

    // And the mutating ops refuse too -- nothing is written outside the root.
    EXPECT_FALSE(sb.write("../evil.txt", "pwned"));
    EXPECT_FALSE(fs::exists(tmp.path() / "evil.txt"));

    // A legitimate interior path that merely *contains* ".." components but
    // stays inside is still allowed.
    EXPECT_TRUE(sb.resolve("a/b/../c.txt").has_value());
}

TEST(SandboxFs, RefusesToDeleteItsOwnRoot) {
    TempDir tmp;
    SandboxFs sb(tmp.path());
    EXPECT_FALSE(sb.remove("."));
    EXPECT_TRUE(fs::exists(tmp.path()));
}

// --- Subprocess --------------------------------------------------------------

TEST(Subprocess, CapturesStdoutAndExitZero) {
    auto r = run_echo("Hello");
    ASSERT_TRUE(r.launched);
    EXPECT_EQ(r.exit_code, 0);
    EXPECT_FALSE(r.timed_out);
    EXPECT_EQ(trimmed(r.out), "Hello");
    EXPECT_TRUE(r.ok());
}

TEST(Subprocess, SurfacesNonZeroExitCode) {
    auto r = run_exit(3);
    ASSERT_TRUE(r.launched);
    EXPECT_EQ(r.exit_code, 3);
    EXPECT_FALSE(r.ok());
}

TEST(Subprocess, SeparatesStderrFromStdout) {
    auto r = run_to_stderr("Boom");
    ASSERT_TRUE(r.launched);
    EXPECT_EQ(r.exit_code, 0);
    EXPECT_NE(r.err.find("Boom"), std::string::npos);
    EXPECT_EQ(r.out.find("Boom"), std::string::npos);  // did NOT leak to stdout
}

TEST(Subprocess, ReportsFailureToLaunchMissingExe) {
    auto r = run_process("definitely_not_a_real_program_xyzzy", {});
    EXPECT_FALSE(r.launched);
}

// --- Workspace: plain backing ------------------------------------------------

TEST(Workspace, PlainSandboxInitWriteTeardown) {
    TempDir tmp;
    fs::path created;
    {
        auto ws = Workspace::create_plain(tmp.path() / "sessions");
        ASSERT_TRUE(ws.has_value());
        EXPECT_TRUE(ws->active());
        EXPECT_FALSE(ws->git_backed());
        created = ws->path();
        EXPECT_TRUE(fs::exists(created));

        // The agent works through the confined FS rooted at the sandbox.
        auto sb = ws->fs();
        ASSERT_TRUE(sb.write("notes/todo.md", "plan"));
        EXPECT_TRUE(fs::exists(created / "notes" / "todo.md"));
    }  // RAII teardown here

    EXPECT_FALSE(fs::exists(created));  // sandbox fully removed
}

// --- Workspace: git worktree backing ----------------------------------------

TEST(Workspace, GitWorktreeSandboxLifecycle) {
    if (!git_available()) GTEST_SKIP() << "git not on PATH";

    TempDir tmp;
    const auto repo = tmp.path() / "repo";
    fs::create_directories(repo);

    // Stand up a minimal, self-contained git repo with one commit.
    auto in_repo = [&](std::vector<std::string> args) {
        std::vector<std::string> full{"-C", repo.string()};
        full.insert(full.end(), args.begin(), args.end());
        return run_process("git", full);
    };
    ASSERT_TRUE(in_repo({"init"}).ok());
    ASSERT_TRUE(in_repo({"config", "user.email", "test@example.com"}).ok());
    ASSERT_TRUE(in_repo({"config", "user.name", "Test"}).ok());
    {
        SandboxFs rs(repo);
        ASSERT_TRUE(rs.write("README.md", "# seed\n"));
    }
    ASSERT_TRUE(in_repo({"add", "."}).ok());
    ASSERT_TRUE(in_repo({"commit", "-m", "seed"}).ok());

    fs::path session_path;
    {
        auto ws = Workspace::create(repo, tmp.path() / "sessions");
        ASSERT_TRUE(ws.has_value()) << "git worktree add failed";
        EXPECT_TRUE(ws->git_backed());
        session_path = ws->path();

        // The worktree is a real checkout: the seeded file is present.
        EXPECT_TRUE(fs::exists(session_path / "README.md"));

        // git runs cleanly inside the sandbox.
        auto status = ws->git({"status", "--porcelain"});
        EXPECT_TRUE(status.ok());
        EXPECT_TRUE(trimmed(status.out).empty());  // clean tree

        // A new file shows up as untracked in the sandbox -- and crucially does
        // NOT appear in the origin repo's working tree.
        ASSERT_TRUE(ws->fs().write("scratch.txt", "agent work"));
        auto status2 = ws->git({"status", "--porcelain"});
        EXPECT_NE(status2.out.find("scratch.txt"), std::string::npos);
        EXPECT_FALSE(fs::exists(repo / "scratch.txt"));
    }  // RAII teardown: worktree removed, branch deleted, dir gone

    EXPECT_FALSE(fs::exists(session_path));

    // The origin repo no longer lists the worktree.
    auto wt = run_process("git", {"-C", repo.string(), "worktree", "list"});
    ASSERT_TRUE(wt.ok());
    EXPECT_EQ(wt.out.find("sessions"), std::string::npos);
}
