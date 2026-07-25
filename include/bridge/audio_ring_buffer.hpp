#pragma once
// =============================================================================
// bridge/audio_ring_buffer.hpp
//
// Single-Producer / Single-Consumer (SPSC) lock-free ring buffer for float32
// PCM (16 kHz mono), backed by CUDA pinned + mapped host memory.
//
// ROLE IN THE ARCHITECTURE
//   This is the *only* handoff point between the application's audio ingress
//   thread (gRPC/WebSocket callback, arbitrary OS thread) and the single engine
//   thread that owns the CUDA inference loop. It is the concrete realisation of
//   the engine's single-thread doctrine at the audio boundary: the producer
//   NEVER touches the engine, KV cache, or any CUDA state — it only appends
//   samples here; the one engine-owning thread drains them. No OS mutex ever
//   sits on the path the inference loop walks (Architectural Rule #1).
//
//   PRODUCER (audio thread)          CONSUMER (engine / CUDA thread)
//   push_samples() ──────────▶  [ pinned+mapped ring ]  ──────────▶ read_samples()
//                                        │ device_ptr()
//                                        ▼
//                                 log-mel DSP / projector kernels (zero-copy)
//
// CONCURRENCY CONTRACT (violating any of these is undefined behaviour)
//   * EXACTLY ONE thread may ever call the producer API (push_samples,
//     free_samples). EXACTLY ONE thread may ever call the consumer API
//     (read_samples, available_samples, device_ptr consumption). The two roles
//     may be different threads and run fully concurrently.
//   * Both APIs are WAIT-FREE and `noexcept`: bounded, allocation-free work with
//     no locks, no CAS loops, no syscalls. They can be called from a realtime
//     audio callback and from the decode hot loop without priority inversion.
//   * Zero allocation after construction (Architectural Rule #2): the entire
//     ring is allocated once in the ctor (INIT tier). push/read never allocate.
//
// MEMORY-ORDERING PROTOCOL (the heart of the lock-free design)
//   Two monotonically increasing 64-bit cursors, each written by exactly one
//   role and read by the other:
//     write_pos_  — producer publishes with std::memory_order_release AFTER the
//                   sample bytes are in the buffer; consumer reads with acquire,
//                   which makes those bytes visible before it copies them out.
//     read_pos_   — consumer publishes with release AFTER it has finished
//                   reading a region (freeing it); producer reads with acquire
//                   before deciding a region is free to overwrite.
//   64-bit counters are free-running (never wrap in any realistic run: 2^64
//   samples at 16 kHz is ~36 million years); the physical slot is `pos & mask_`.
//   Each cursor is alignas(64) on its own cache line to avoid false sharing
//   between the two threads.
//
// PINNED + MAPPED MEMORY
//   Storage is one cudaHostAlloc(cudaHostAllocMapped [| cudaHostAllocPortable])
//   block. cudaHostGetDevicePointer() yields device_ptr_, a device-visible view
//   of the SAME bytes — enabling zero-copy: the consumer can either drain to a
//   linear staging buffer via read_samples(), or feed the DSP kernels directly
//   from device_ptr() (respecting wrap; see below). It is NOT write-combined:
//   read_samples() reads on the host, and WC memory is slow to read host-side.
//   Requires a device with canMapHostMemory and the mapping flag on the context.
// =============================================================================
#include <atomic>
#include <cstddef>
#include <cstdint>

// cudaHostAlloc / cudaHostGetDevicePointer / cudaStream_t live behind the .cpp;
// the header stays free of CUDA symbols in its signatures so pure-host producers
// (the gRPC layer) can include it without a CUDA toolchain. Only <cuda_runtime.h>
// forward needs — kept out on purpose; the impl TU includes it.

namespace blackwell::bridge {

// The two SPSC cursors are alignas(64) onto their own cache lines to prevent
// false sharing between the producer and consumer threads — that padding is the
// design, so silence MSVC's C4324 (which /W4 /WX would otherwise reject).
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)  // structure padded due to alignment specifier (intentional)
#endif

class AudioRingBuffer {
public:
    // Allocates the pinned+mapped ring. `min_capacity_samples` is rounded UP to
    // the next power of two (the mask trick needs it). One usable slot is NOT
    // reserved — capacity() samples are all storable, because full/empty is
    // disambiguated by the 64-bit cursor delta, not by a gap.
    //
    // INIT tier (Hybrid Error Doctrine): THROWS blackwell::cuda_error if pinned
    // allocation or device-pointer mapping fails. RAII: a throwing ctor frees
    // any partial allocation before unwinding.
    explicit AudioRingBuffer(size_t min_capacity_samples);
    ~AudioRingBuffer();

    // Non-copyable, non-movable: it owns a pinned allocation and two cursors
    // that other threads may be actively touching. Its address is its identity.
    AudioRingBuffer(const AudioRingBuffer&) = delete;
    AudioRingBuffer& operator=(const AudioRingBuffer&) = delete;
    AudioRingBuffer(AudioRingBuffer&&) = delete;
    AudioRingBuffer& operator=(AudioRingBuffer&&) = delete;

    // ---- PRODUCER API (audio thread only) -----------------------------------

    // Append `count` samples. ALL-OR-NOTHING: if fewer than `count` free slots
    // are available it writes nothing and returns false (backpressure signal —
    // the caller decides to drop, retry, or grow its own staging). On success it
    // copies the bytes (handling the ring wrap as two spans) and publishes
    // write_pos_ with release. Wait-free, noexcept, no allocation.
    bool push_samples(const float* pcm, size_t count) noexcept;

    // Free slots currently available to the producer (a lower-bound snapshot:
    // the consumer can only ever make this larger concurrently, so a `true`
    // capacity check from push_samples is never falsely optimistic).
    size_t free_samples() const noexcept;

    // ---- CONSUMER API (engine / CUDA thread only) ---------------------------

    // Copy up to `max_count` samples into `dst`, advancing the read cursor.
    // Returns the number actually copied (0..max_count). Partial reads are
    // normal. Publishes read_pos_ with release so the producer may reuse the
    // freed region. Wait-free, noexcept, no allocation.
    size_t read_samples(float* dst, size_t max_count) noexcept;

    // Readable samples currently in the ring (a lower-bound snapshot: the
    // producer can only make this larger concurrently). Intended for the
    // consumer to decide whether a full DSP hop/window is available yet.
    size_t available_samples() const noexcept;

    // ---- ZERO-COPY DEVICE ACCESS (advanced consumer path) -------------------
    //
    // device_ptr() is the CUDA-visible base of the ring; a kernel/cudaMemcpyAsync
    // may read directly from device_ptr() + index_of(read_pos) WITHOUT a host
    // bounce. Because the region can wrap, a direct consumer must issue the copy
    // as up to two contiguous spans ([read_index, capacity) then [0, remainder))
    // and only then advance the read cursor (see consume_begin/consume_commit).
    // host_ptr() is the same bytes for host-side inspection.
    const float* host_ptr() const noexcept { return host_ptr_; }
    const float* device_ptr() const noexcept { return device_ptr_; }
    size_t capacity() const noexcept { return capacity_; }

    // Physical slot for a monotonic cursor value.
    size_t index_of(uint64_t pos) const noexcept { return static_cast<size_t>(pos) & mask_; }

    // Two-phase zero-copy drain for the direct-device path:
    //   consume_begin()  -> snapshot {read cursor, contiguous run length before
    //                       wrap, total available}; the consumer launches its
    //                       async copies/kernels from device_ptr()+index.
    //   consume_commit(n) -> after those ops are QUEUED on the stream, publish
    //                       the freed samples (release). NB: committing frees the
    //                       slots for the producer; the consumer must ensure its
    //                       async reads are ordered on the same stream before any
    //                       later overwrite is observable (documented invariant).
    struct ConsumeView {
        uint64_t read_pos;         // monotonic cursor at snapshot time
        size_t   start_index;      // read_pos & mask_
        size_t   contiguous_run;   // samples before the ring wraps
        size_t   available;        // total readable (contiguous_run + wrapped tail)
    };
    ConsumeView consume_begin() const noexcept;
    void        consume_commit(size_t n) noexcept;

private:
    // ---- Storage: allocated once, never resized (zero-alloc invariant) ------
    float* host_ptr_ = nullptr;    // cudaHostAlloc(cudaHostAllocMapped) base
    float* device_ptr_ = nullptr;  // cudaHostGetDevicePointer(host_ptr_)
    size_t capacity_ = 0;          // power of two, in samples
    size_t mask_ = 0;              // capacity_ - 1

    // ---- SPSC cursors: monotonic, single-writer each, separate cache lines --
    // write_pos_: PRODUCER stores (release) after copying; CONSUMER loads (acquire).
    alignas(64) std::atomic<uint64_t> write_pos_{0};
    // read_pos_: CONSUMER stores (release) after draining; PRODUCER loads (acquire).
    alignas(64) std::atomic<uint64_t> read_pos_{0};

    // Private per-role mirrors so each role avoids re-loading its OWN atomic and
    // only pays an acquire-load on the peer's cursor when its cached view is
    // exhausted. Each is touched by a single role => no synchronisation needed.
    alignas(64) uint64_t producer_cached_read_pos_ = 0;   // producer thread only
    alignas(64) uint64_t consumer_cached_write_pos_ = 0;  // consumer thread only
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace blackwell::bridge
