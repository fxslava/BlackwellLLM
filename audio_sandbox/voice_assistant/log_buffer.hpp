#pragma once
// -----------------------------------------------------------------------------
// log_buffer.hpp — the app's log as a bounded ring of lines, plus the tee that
// fills it from stdout and stderr without taking either away.
//
// WHY A TEE AND NOT A REDIRECT. The task says "redirect stdout/stderr ... to this
// console". Redirecting -- _dup2 the pipe over fd 1 and be done -- would work and
// would be wrong: every diagnostic this app prints is currently read from a
// terminal (the [vram] budget lines, the commit-gate trace, the cascade summary,
// and the three CLI self-tests whose entire output IS their result). Moving them
// into an overlay that only exists while the app has a window would silently
// break `--list-audio-devices`, `--check-volume` and `--say`.
//
// So fd 1 and fd 2 are duplicated FIRST, the pipe goes over the originals, and
// the reader thread writes every byte back out to the duplicate as well as into
// the ring. The terminal behaves exactly as it did; the console gains a copy.
//
// HOW THE CAPTURE WORKS, and the one subtlety worth stating. Three things write
// to "stdout" and they do not share a mechanism:
//
//   printf / fputs        the CRT's fd 1
//   std::cout             a streambuf that eventually reaches the same fd
//   WriteConsole / Win32  the STD_OUTPUT_HANDLE
//
// _dup2 fixes the first two. SetStdHandle is needed for the third, and doing
// only one of them is how a log ends up half-captured. Both are done here.
//
// LINE BUFFERING IS FORCED. The CRT gives a non-tty stdout FULL buffering, so
// the moment fd 1 becomes a pipe, printf output stops appearing until 4 KB have
// accumulated -- which for this app means the console is empty for the first
// minute and then shows everything at once. setvbuf to unbuffered is the fix and
// it is not optional.
//
// THREADING. append() and snapshot() are safe from any thread; the reader thread
// is the only producer in practice but the ring does not care. The renderer polls
// `revision()` and only re-lays-out when it has moved, so a console that is open
// but idle costs nothing.
//
// LIFETIME. LogTee is RAII and restores both descriptors in its destructor. It
// must outlive nothing -- but it must be destroyed BEFORE the CRT tears down, so
// it belongs in main() rather than in a static.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rt {

// Where a line came from. The console colours by this, and it is the only
// classification worth making: a user scanning a wall of text is looking for the
// red one.
enum class LogStream { Out, Err };

struct LogLine {
    LogStream   stream = LogStream::Out;
    std::string text;
};

// The ring. One instance, owned by main and reached by the console through a
// pointer -- it is deliberately NOT a singleton, because a global log buffer is
// exactly the sort of thing that ends up constructed after its first writer.
class LogBuffer {
public:
    explicit LogBuffer(std::size_t max_lines) : max_lines_(max_lines < 1 ? 1 : max_lines) {}

    // ANY thread. `chunk` need not be line-aligned: partial lines are held until
    // their newline arrives, which is what makes a pipe read of arbitrary size
    // produce correct lines rather than ragged ones.
    void append(LogStream stream, const char* data, std::size_t count);

    // ANY thread. A copy of the whole ring, oldest first. The console calls this
    // only when revision() has moved.
    [[nodiscard]] std::vector<LogLine> snapshot() const;

    // Monotone; bumped on every completed line. The renderer's "is a repaint
    // worth it" test.
    [[nodiscard]] std::uint64_t revision() const noexcept {
        return revision_.load(std::memory_order_acquire);
    }

    void clear();

private:
    void push_line(LogStream stream, std::string text);

    mutable std::mutex    mu_;
    std::deque<LogLine>   lines_;
    // The not-yet-terminated tail of each stream, kept separately: stdout and
    // stderr interleave at arbitrary byte boundaries, and a shared partial-line
    // buffer would splice half of one into the other.
    std::string           partial_out_;
    std::string           partial_err_;
    std::size_t           max_lines_;
    std::atomic<std::uint64_t> revision_{0};
};

// The capture itself. Constructing one starts the redirect and the reader
// threads; destroying one restores the descriptors and joins them.
class LogTee {
public:
    // `sink` is BORROWED and must outlive this object.
    explicit LogTee(LogBuffer* sink);
    ~LogTee();

    LogTee(const LogTee&) = delete;
    LogTee& operator=(const LogTee&) = delete;

    // False when the redirect could not be installed (a pipe that would not
    // open, a descriptor that would not duplicate). NOT a fatal condition and
    // deliberately not a throw: the app logs to the terminal exactly as before
    // and the console shows only what is posted to it directly.
    [[nodiscard]] bool active() const noexcept { return active_; }

    // Post a line straight into the ring without going through the pipe. For the
    // app's own announcements ("console ready"), which should appear in the
    // overlay whether or not the capture came up.
    void note(const std::string& text);

private:
    struct Channel {
        HANDLE      read = nullptr;
        HANDLE      write = nullptr;
        int         saved_fd = -1;    // dup of the original, kept for the tee
        int         target_fd = -1;   // 1 or 2
        DWORD       std_handle = 0;   // STD_OUTPUT_HANDLE / STD_ERROR_HANDLE
        HANDLE      saved_std = nullptr;
        LogStream   stream = LogStream::Out;
        std::thread reader;
    };

    bool open_channel(Channel& c, int target_fd, DWORD std_handle, LogStream stream);
    void close_channel(Channel& c);
    void pump(Channel& c);

    LogBuffer* sink_ = nullptr;
    Channel    out_;
    Channel    err_;
    std::atomic<bool> running_{true};
    bool       active_ = false;
};

}  // namespace rt
