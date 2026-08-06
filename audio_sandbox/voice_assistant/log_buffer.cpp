// -----------------------------------------------------------------------------
// log_buffer.cpp — the ring, and the stdout/stderr tee. See the header for why
// this duplicates the descriptors rather than simply redirecting them.
// -----------------------------------------------------------------------------
#include "log_buffer.hpp"

#include <fcntl.h>
#include <io.h>
#include <stdio.h>

#include <cstring>
#include <utility>

namespace rt {
namespace {

// One pipe read. Big enough that a burst of log lines is one syscall, small
// enough that it is a stack buffer.
constexpr DWORD kReadChunk = 4096;

}  // namespace

// -----------------------------------------------------------------------------
// LogBuffer
// -----------------------------------------------------------------------------
void LogBuffer::push_line(LogStream stream, std::string text) {
    // Carriage returns come through on every line the CRT wrote in text mode and
    // would render as a box glyph in DirectWrite.
    if (!text.empty() && text.back() == '\r') text.pop_back();
    lines_.push_back({stream, std::move(text)});
    while (lines_.size() > max_lines_) lines_.pop_front();
    revision_.fetch_add(1, std::memory_order_release);
}

void LogBuffer::append(LogStream stream, const char* data, std::size_t count) {
    if (data == nullptr || count == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    std::string& partial = (stream == LogStream::Out) ? partial_out_ : partial_err_;
    for (std::size_t i = 0; i < count; ++i) {
        const char c = data[i];
        if (c == '\n') {
            push_line(stream, std::move(partial));
            partial.clear();
        } else {
            partial.push_back(c);
            // A single "line" of unbounded length would grow this string until
            // the process died. Anything past a sane terminal width is a binary
            // blob or a runaway loop, and breaking it is better than holding it.
            if (partial.size() >= 4096) {
                push_line(stream, std::move(partial));
                partial.clear();
            }
        }
    }
}

std::vector<LogLine> LogBuffer::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return {lines_.begin(), lines_.end()};
}

void LogBuffer::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    lines_.clear();
    partial_out_.clear();
    partial_err_.clear();
    revision_.fetch_add(1, std::memory_order_release);
}

// -----------------------------------------------------------------------------
// LogTee
// -----------------------------------------------------------------------------
LogTee::LogTee(LogBuffer* sink) : sink_(sink) {
    if (sink_ == nullptr) return;
    const bool a = open_channel(out_, _fileno(stdout), STD_OUTPUT_HANDLE, LogStream::Out);
    const bool b = open_channel(err_, _fileno(stderr), STD_ERROR_HANDLE, LogStream::Err);
    active_ = a || b;

    // FORCED UNBUFFERED, and this is the difference between a console that works
    // and one that appears broken. The CRT gives a non-tty stdout FULL buffering,
    // so the instant fd 1 became a pipe above, printf stopped reaching the reader
    // until 4 KB had accumulated -- which for this app is minutes.
    //
    // stderr is already unbuffered by the standard; setting it again is free and
    // says so at the one place someone would look.
    if (a) setvbuf(stdout, nullptr, _IONBF, 0);
    if (b) setvbuf(stderr, nullptr, _IONBF, 0);
}

bool LogTee::open_channel(Channel& c, int target_fd, DWORD std_handle, LogStream stream) {
    c.target_fd = target_fd;
    c.std_handle = std_handle;
    c.stream = stream;
    if (target_fd < 0) return false;

    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    if (!CreatePipe(&c.read, &c.write, &sa, 0)) return false;

    // THE DUPLICATE IS TAKEN FIRST. It is the terminal, and it is what makes this
    // a tee rather than a redirect -- once _dup2 has overwritten fd 1 there is no
    // way back to the console.
    c.saved_fd = _dup(target_fd);
    c.saved_std = GetStdHandle(std_handle);

    const int pipe_fd = _open_osfhandle(reinterpret_cast<intptr_t>(c.write), _O_WRONLY);
    if (pipe_fd == -1) {
        close_channel(c);
        return false;
    }
    if (_dup2(pipe_fd, target_fd) != 0) {
        _close(pipe_fd);
        close_channel(c);
        return false;
    }
    // _dup2 duplicated it onto the target; this one is surplus. The HANDLE is now
    // owned by the target descriptor, so c.write must NOT be closed separately --
    // it is cleared here to say so.
    _close(pipe_fd);
    c.write = nullptr;

    // The third writer: Win32 calls that go to STD_OUTPUT_HANDLE rather than
    // through the CRT. Fixing only the descriptors is how a log ends up
    // half-captured.
    SetStdHandle(std_handle, reinterpret_cast<HANDLE>(_get_osfhandle(target_fd)));

    c.reader = std::thread([this, &c] { pump(c); });
    return true;
}

void LogTee::pump(Channel& c) {
    char buf[kReadChunk];
    for (;;) {
        DWORD got = 0;
        // Blocks until something is written or every write end is closed, at
        // which point ReadFile fails with ERROR_BROKEN_PIPE and the thread ends.
        if (!ReadFile(c.read, buf, kReadChunk, &got, nullptr) || got == 0) break;
        sink_->append(c.stream, buf, got);
        // BACK OUT TO THE TERMINAL. _write on the saved descriptor rather than
        // fwrite on a FILE*, because the FILE* for this stream is the pipe now --
        // writing to it here would feed the reader its own output forever.
        if (c.saved_fd >= 0) {
            _write(c.saved_fd, buf, got);
        }
        if (!running_.load(std::memory_order_acquire)) break;
    }
}

void LogTee::note(const std::string& text) {
    if (sink_ == nullptr) return;
    sink_->append(LogStream::Out, text.data(), text.size());
    static const char nl = '\n';
    sink_->append(LogStream::Out, &nl, 1);
}

void LogTee::close_channel(Channel& c) {
    if (c.read != nullptr) {
        CloseHandle(c.read);
        c.read = nullptr;
    }
    if (c.write != nullptr) {
        CloseHandle(c.write);
        c.write = nullptr;
    }
    if (c.saved_fd >= 0) {
        _close(c.saved_fd);
        c.saved_fd = -1;
    }
}

LogTee::~LogTee() {
    running_.store(false, std::memory_order_release);

    // RESTORE BEFORE JOINING, and the order is the whole teardown. Putting the
    // saved descriptor back over fd 1/2 closes the pipe's last write end, which
    // is what makes the blocking ReadFile in each reader return so the thread can
    // be joined. Joining first would deadlock: the reader is parked on a pipe
    // nobody is going to write to and nobody has closed.
    auto restore = [](Channel& c) {
        if (c.saved_fd >= 0 && c.target_fd >= 0) {
            _dup2(c.saved_fd, c.target_fd);
            if (c.std_handle != 0 && c.saved_std != nullptr) {
                SetStdHandle(c.std_handle, c.saved_std);
            }
        }
    };
    restore(out_);
    restore(err_);

    if (out_.reader.joinable()) out_.reader.join();
    if (err_.reader.joinable()) err_.reader.join();

    close_channel(out_);
    close_channel(err_);
}

}  // namespace rt
