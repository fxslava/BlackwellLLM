#pragma once
// -----------------------------------------------------------------------------
// SpscRing<T> — wait-free single-producer/single-consumer ring over FIXED,
// pre-allocated storage. The real-time primitive the audio graph is built on.
//
// WHY THIS EXISTS AT ALL. audio_sandbox already has SampleRing (a mutex-guarded
// std::deque). On the CAPTURE path that is defensible: push() runs on the
// miniaudio callback, and an allocation-induced overrun costs 10 ms of mel that
// nobody sees. It is NOT defensible anywhere audio flows the other way or an
// adaptive filter shares the callback:
//   * on the PLAYBACK callback a stall is an audible click, and clicks are the
//     entire perceived quality of a synthesis feature;
//   * once AEC runs in the callback, that callback may not take a lock or
//     allocate at all, because a missed deadline desynchronises the echo
//     canceller from its reference stream.
// So: no locks, no allocation, no syscalls on either side. write() and read()
// are wait-free, bounded by the copy itself.
//
// THREADING CONTRACT — this is a hard requirement, not a recommendation.
// EXACTLY ONE producer thread calls write()/space(); EXACTLY ONE consumer thread
// calls read()/read_or_silence()/available(). Two producers corrupt the buffer
// silently. The observer methods (capacity/overruns/underruns) are safe from any
// thread. reset() is safe ONLY when neither side is running.
//
// HOW IT STAYS CORRECT. Two monotone 64-bit counters of TOTAL samples written /
// read, never wrapped by hand — `written - read` IS the fill level, so the
// classic "is it full or empty?" ambiguity of head/tail indices cannot arise.
// (At 48 kHz a uint64 sample counter wraps after ~12 million years.) Each side
// owns its counter and only ever reads the other's with acquire, publishing its
// own with release: that pairing is what makes the copied samples visible to the
// other thread, and it is the only synchronisation in the class.
//
// The two counters sit on separate cache lines. Without that padding the
// producer's store to `written_` invalidates the line the consumer is reading
// `read_` from on every single call — true sharing of the buffer is unavoidable,
// but false sharing of the counters is pure waste on the one thread that has a
// deadline.
//
// CAPACITY is rounded UP to a power of two so the wrap is a mask, not a modulo.
// Ask for 24000 and you get 32768; capacity() reports what you actually got, and
// that — not the requested value — is the number to reason about.
//
// FAULT POLICY BELONGS TO THE CALLER, NOT THE RING — on BOTH sides. A short
// transfer means opposite things depending on who is asking, so the plain
// operations report a count and judge nothing, and the *_or_* variants are the
// ones that declare "short is a fault here" and account for it:
//
//   read()            short = "nothing has arrived yet". The DSP worker draining
//                     the capture ring hits this constantly; it is the idle case.
//   read_or_silence() short = the sink STARVED. A playback callback must return a
//                     full buffer, so the shortfall is padded with silence and
//                     counted as an underrun.
//   write()           short = "no room right now". A producer that can wait (the
//                     TTS worker runs ~10x realtime and must NOT discard audio
//                     mid-sentence) retries; nothing was lost, so nothing is
//                     counted. Counting here would make the metric meaningless,
//                     since one retried block would register as many drops.
//   write_or_drop()   short = samples LOST. A producer on a real-time callback
//                     cannot wait, so the excess is discarded and counted as an
//                     overrun.
//
// Picking the wrong one is not a correctness bug — the data movement is identical
// — but it makes the health counters lie, which on an audio path is how a real
// fault stays invisible.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace blackwell::audio_rt {

// Padding between the two counters. std::hardware_destructive_interference_size
// would be the standard spelling, but it is inconsistently available across the
// compilers this repo targets and its value is a compile-time guess anyway; 64 is
// correct for every x86-64 part we run on.
inline constexpr std::size_t kCacheLineBytes = 64;

// C4324 ("structure was padded due to alignment specifier") reports EXACTLY the
// thing the alignas below is asking for -- the padding IS the feature, not an
// accident. Scoped to this class only, matching the same suppression and the same
// reasoning in src/bridge/engine_control_bridge.hpp.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

template <typename T>
class SpscRing {
public:
    // memcpy is the whole point of the copy loops below, and a ring that silently
    // ran a copy ctor on the audio thread would defeat the class.
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscRing<T> copies with memcpy; T must be trivially copyable");

    // Allocates ONCE, here. This is the only allocation the class ever performs —
    // INIT tier, so a throw from the vector is correct and expected. After the
    // ctor returns, no method allocates.
    //
    // `min_capacity` is a FLOOR: the real capacity is the next power of two at or
    // above it (and at least 2). Size it from the deadline you must survive, not
    // from the steady state — for a playback sink that is "how long may synthesis
    // stall before the speaker starves", typically 300-500 ms.
    explicit SpscRing(std::size_t min_capacity)
        : capacity_(round_up_pow2(min_capacity < 2 ? 2 : min_capacity)),
          mask_(capacity_ - 1),
          buf_(capacity_) {}

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    // ---- producer side ------------------------------------------------------
    // Copies up to `count` samples in. Returns how many were ACCEPTED; a short
    // return means the ring was full and the remainder was NOT buffered. Counts
    // nothing — the caller decides whether a short write is a fault (see the
    // policy block in the header). Wait-free.
    //
    // The excess is rejected rather than overwriting the oldest queued samples.
    // That is the opposite of SampleRing's policy and it is deliberate: a
    // consumer mid-drain must never have the ground move under it, and on a
    // playback sink the already-queued audio is the audio about to be heard.
    std::size_t write(const T* src, std::size_t count) noexcept {
        if (src == nullptr || count == 0) return 0;

        const std::uint64_t w = written_.load(std::memory_order_relaxed);  // we own it
        const std::uint64_t r = read_.load(std::memory_order_acquire);     // pairs with read()
        const std::size_t space = capacity_ - static_cast<std::size_t>(w - r);
        const std::size_t n = count < space ? count : space;

        if (n != 0) {
            const std::size_t head = static_cast<std::size_t>(w & mask_);
            const std::size_t first = (capacity_ - head) < n ? (capacity_ - head) : n;
            std::memcpy(buf_.data() + head, src, first * sizeof(T));
            if (n > first) std::memcpy(buf_.data(), src + first, (n - first) * sizeof(T));
            // Release: everything memcpy'd above is visible to the consumer once it
            // acquires this value. Publishing the count MUST come last.
            written_.store(w + n, std::memory_order_release);
        }
        return n;
    }

    // write(), but a short write is declared a FAULT: the rejected samples are
    // LOST and overruns() is advanced by them. This is what a real-time capture
    // callback wants — it cannot wait for space, so audio that does not fit is
    // gone and must be visibly accounted for. Mirror of read_or_silence().
    std::size_t write_or_drop(const T* src, std::size_t count) noexcept {
        // A malformed call is a no-op, not lost audio: counting it would make a
        // caller-side null bug look like a starving consumer.
        if (src == nullptr || count == 0) return 0;
        const std::size_t n = write(src, count);
        if (n < count) {
            overruns_.fetch_add(static_cast<std::uint64_t>(count - n), std::memory_order_relaxed);
        }
        return n;
    }

    // Samples the producer may write right now without dropping. Producer thread.
    std::size_t space() const noexcept {
        const std::uint64_t w = written_.load(std::memory_order_relaxed);
        const std::uint64_t r = read_.load(std::memory_order_acquire);
        return capacity_ - static_cast<std::size_t>(w - r);
    }

    // ---- consumer side ------------------------------------------------------
    // Copies up to `count` samples out. Returns how many were produced; a short
    // return is NOT counted as an underrun (see the header block — that policy
    // belongs to the caller). Wait-free.
    std::size_t read(T* dst, std::size_t count) noexcept {
        if (dst == nullptr || count == 0) return 0;

        const std::uint64_t r = read_.load(std::memory_order_relaxed);      // we own it
        const std::uint64_t w = written_.load(std::memory_order_acquire);   // pairs with write()
        const std::size_t avail = static_cast<std::size_t>(w - r);
        const std::size_t n = count < avail ? count : avail;

        if (n != 0) {
            const std::size_t tail = static_cast<std::size_t>(r & mask_);
            const std::size_t first = (capacity_ - tail) < n ? (capacity_ - tail) : n;
            std::memcpy(dst, buf_.data() + tail, first * sizeof(T));
            if (n > first) std::memcpy(dst + first, buf_.data(), (n - first) * sizeof(T));
            // Release: frees the slots for the producer. Must come after the copy,
            // or the producer may overwrite samples we have not read yet.
            read_.store(r + n, std::memory_order_release);
        }
        return n;
    }

    // read(), but a short read is declared a FAULT: the shortfall is filled with
    // value-initialised T (silence for float/int16 PCM) and underruns() is
    // advanced by it. This is what a playback callback wants — it must hand the
    // device a full buffer on every invocation, and a starved sink should be
    // audibly silent and visibly counted, never stale or short.
    void read_or_silence(T* dst, std::size_t count) noexcept {
        if (dst == nullptr || count == 0) return;
        const std::size_t got = read(dst, count);
        if (got < count) {
            // memset over trivially-copyable T: value-initialisation for the
            // arithmetic sample types this ring carries.
            std::memset(dst + got, 0, (count - got) * sizeof(T));
            underruns_.fetch_add(static_cast<std::uint64_t>(count - got),
                                 std::memory_order_relaxed);
        }
    }

    // Samples the consumer may read right now. Consumer thread.
    std::size_t available() const noexcept {
        const std::uint64_t r = read_.load(std::memory_order_relaxed);
        const std::uint64_t w = written_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(w - r);
    }

    // ---- observers (any thread) ---------------------------------------------
    std::size_t capacity() const noexcept { return capacity_; }

    // Samples DROPPED by write_or_drop() because the ring was full. Monotone; any
    // nonzero value means the consumer is not keeping up and audio was lost.
    // Plain write() never advances this — a producer that retries lost nothing.
    std::uint64_t overruns() const noexcept {
        return overruns_.load(std::memory_order_relaxed);
    }

    // Samples of silence injected by read_or_silence() because the ring was
    // empty. Monotone; any nonzero value means the producer starved the sink.
    std::uint64_t underruns() const noexcept {
        return underruns_.load(std::memory_order_relaxed);
    }

    // Total samples accepted / consumed since construction. Diagnostics only —
    // do NOT derive fill level from these across threads; use space()/available()
    // from the side that owns the corresponding counter.
    std::uint64_t total_written() const noexcept {
        return written_.load(std::memory_order_relaxed);
    }
    std::uint64_t total_read() const noexcept {
        return read_.load(std::memory_order_relaxed);
    }

    // ---- teardown / reconfiguration -----------------------------------------
    // Discards buffered samples and zeroes the fault counters. NOT SAFE while
    // either side is running — this is for a device restart or a test, where the
    // producer and consumer are both known to be stopped. A live "drop what is
    // queued" (barge-in) is NOT this: that is an epoch bump at the layer above,
    // which lets the consumer skip stale spans without racing the producer.
    void reset() noexcept {
        written_.store(0, std::memory_order_relaxed);
        read_.store(0, std::memory_order_relaxed);
        overruns_.store(0, std::memory_order_relaxed);
        underruns_.store(0, std::memory_order_relaxed);
    }

private:
    static std::size_t round_up_pow2(std::size_t v) noexcept {
        std::size_t p = 1;
        while (p < v) p <<= 1;
        return p;
    }

    const std::size_t capacity_;   // power of two
    const std::size_t mask_;       // capacity_ - 1
    std::vector<T> buf_;           // allocated once in the ctor; never resized

    // Separate cache lines: the producer stores to written_ on every call and the
    // consumer stores to read_ on every call, so sharing a line would make each
    // side's fast path invalidate the other's.
    alignas(kCacheLineBytes) std::atomic<std::uint64_t> written_{0};
    alignas(kCacheLineBytes) std::atomic<std::uint64_t> read_{0};
    alignas(kCacheLineBytes) std::atomic<std::uint64_t> overruns_{0};
    std::atomic<std::uint64_t> underruns_{0};
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// The whole design assumes these are real atomics, not a libstdc++/MSVC fallback
// to a hidden mutex — a lock here would silently reintroduce exactly the RT
// hazard the class exists to remove.
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "SpscRing requires lock-free 64-bit atomics");

}  // namespace blackwell::audio_rt
