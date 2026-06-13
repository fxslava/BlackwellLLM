#include "subprocess.h"

#include <array>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
extern char** environ;
#endif

namespace agent::env {

#ifdef _WIN32

namespace {

// Quote a single argument per the CommandLineToArgvW parsing rules so that an
// argv vector round-trips through the single command-line string Win32 demands.
// Backslashes only matter when they precede a quote; this is the canonical MSDN
// algorithm ("Everyone quotes command line arguments the wrong way").
void append_quoted(std::string& cmd, const std::string& arg) {
    if (!arg.empty() &&
        arg.find_first_of(" \t\n\v\"") == std::string::npos) {
        cmd += arg;
        return;
    }
    cmd += '"';
    for (auto it = arg.begin();; ++it) {
        unsigned backslashes = 0;
        while (it != arg.end() && *it == '\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            cmd.append(backslashes * 2, '\\');
            break;
        } else if (*it == '"') {
            cmd.append(backslashes * 2 + 1, '\\');
            cmd += '"';
        } else {
            cmd.append(backslashes, '\\');
            cmd += *it;
        }
    }
    cmd += '"';
}

// Drain a pipe read-handle to EOF. Runs on its own thread per stream so that a
// child filling both stdout and stderr cannot deadlock against a single reader.
void drain_pipe(HANDLE read_end, std::string& sink) {
    std::array<char, 4096> buf;
    DWORD got = 0;
    while (ReadFile(read_end, buf.data(), static_cast<DWORD>(buf.size()), &got,
                    nullptr) &&
           got > 0) {
        sink.append(buf.data(), got);
    }
}

struct Handle {
    HANDLE h = nullptr;
    ~Handle() {
        if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
};

}  // namespace

ProcessResult run_process(const std::string& exe,
                          const std::vector<std::string>& args,
                          const ProcessOptions& opts) {
    ProcessResult result;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    Handle out_r, out_w, err_r, err_w, in_r;
    if (!CreatePipe(&out_r.h, &out_w.h, &sa, 0)) return result;
    if (!CreatePipe(&err_r.h, &err_w.h, &sa, 0)) return result;
    // The parent's read ends must NOT be inherited by the child.
    SetHandleInformation(out_r.h, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_r.h, HANDLE_FLAG_INHERIT, 0);

    // Feed the child an empty stdin from the NUL device so anything that reads
    // input sees a clean EOF instead of blocking forever.
    in_r.h = CreateFileA("NUL", GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
                         0, nullptr);

    std::string cmd;
    append_quoted(cmd, exe);
    for (const auto& a : args) {
        cmd += ' ';
        append_quoted(cmd, a);
    }
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r.h;
    si.hStdOutput = out_w.h;
    si.hStdError = err_w.h;

    PROCESS_INFORMATION pi{};
    const std::string cwd = opts.cwd.empty() ? std::string() : opts.cwd.string();

    const BOOL ok = CreateProcessA(
        /*lpApplicationName=*/nullptr, cmd_buf.data(),
        /*proc attrs=*/nullptr, /*thread attrs=*/nullptr,
        /*bInheritHandles=*/TRUE, CREATE_NO_WINDOW,
        /*env=*/nullptr, cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    if (!ok) return result;
    result.launched = true;

    // Close the child's ends in the parent so our reads see EOF when the child
    // exits and its handles drop.
    CloseHandle(out_w.h);
    out_w.h = nullptr;
    CloseHandle(err_w.h);
    err_w.h = nullptr;
    CloseHandle(in_r.h);
    in_r.h = nullptr;

    std::thread t_out(drain_pipe, out_r.h, std::ref(result.out));
    std::thread t_err(drain_pipe, err_r.h, std::ref(result.err));

    const DWORD wait =
        WaitForSingleObject(pi.hProcess, opts.timeout_ms ? opts.timeout_ms : INFINITE);
    if (wait == WAIT_TIMEOUT) {
        result.timed_out = true;
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, INFINITE);
    }

    t_out.join();
    t_err.join();

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    result.exit_code = static_cast<int>(code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return result;
}

#else  // ---------------------------------------------------------------- POSIX

namespace {

void drain_fd(int fd, std::string& sink) {
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            sink.append(buf, static_cast<size_t>(n));
        } else if (n == 0) {
            return;
        } else if (errno != EINTR) {
            return;
        }
    }
}

}  // namespace

ProcessResult run_process(const std::string& exe,
                          const std::vector<std::string>& args,
                          const ProcessOptions& opts) {
    ProcessResult result;

    int out_pipe[2], err_pipe[2];
    if (pipe(out_pipe) != 0) return result;
    if (pipe(err_pipe) != 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        return result;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        return result;
    }

    if (pid == 0) {
        // Child.
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, STDIN_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        if (!opts.cwd.empty()) {
            if (chdir(opts.cwd.c_str()) != 0) _exit(127);
        }
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(exe.c_str()));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(exe.c_str(), argv.data());
        _exit(127);  // exec failed
    }

    // Parent.
    result.launched = true;
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    std::thread t_out(drain_fd, out_pipe[0], std::ref(result.out));
    std::thread t_err(drain_fd, err_pipe[0], std::ref(result.err));

    int status = 0;
    if (opts.timeout_ms == 0) {
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    } else {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(opts.timeout_ms);
        for (;;) {
            const pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                result.timed_out = true;
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    t_out.join();
    t_err.join();
    ::close(out_pipe[0]);
    ::close(err_pipe[0]);

    if (WIFEXITED(status))
        result.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status))
        result.exit_code = 128 + WTERMSIG(status);
    return result;
}

#endif

}  // namespace agent::env
