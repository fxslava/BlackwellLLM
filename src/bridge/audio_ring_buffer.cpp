// =============================================================================
// bridge/audio_ring_buffer.cpp — SPSC lock-free ring, pinned+mapped storage.
//
// See include/bridge/audio_ring_buffer.hpp for the full concurrency contract.
// Correctness rests on one rule per cursor:
//   * The producer is the SOLE writer of write_pos_. It reads its own cursor
//     RELAXED (no peer writes it) and publishes RELEASE *after* the sample bytes
//     land, so a consumer that reads write_pos_ ACQUIRE sees those bytes.
//   * The consumer is the SOLE writer of read_pos_. It reads its own cursor
//     RELAXED and publishes RELEASE *after* it is done reading, so a producer
//     that reads read_pos_ ACQUIRE never overwrites bytes still being read.
// Each role also keeps a private cached copy of the PEER cursor and only pays an
// acquire-load when that cache says the ring is (empty|full); this keeps the
// common case free of cross-thread cache-line traffic.
// =============================================================================
#include "bridge/audio_ring_buffer.hpp"

#include <cstring>  // std::memcpy

#include <cuda_runtime.h>

#include "common.h"  // CUDA_CHECK_THROW, blackwell::cuda_error

namespace blackwell::bridge {
namespace {

// Smallest power of two >= n, with a floor of 2 slots. 64-bit safe.
size_t round_up_pow2(size_t n) {
    if (n <= 2) return 2;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

}  // namespace

AudioRingBuffer::AudioRingBuffer(size_t min_capacity_samples) {
    capacity_ = round_up_pow2(min_capacity_samples);
    mask_ = capacity_ - 1;

    // Pinned + device-mapped so the consumer can either drain to host (read_
    // samples) or feed CUDA kernels directly (device_ptr). Portable across the
    // engine's contexts; NOT write-combined (read_samples reads it host-side).
    // INIT tier: a failure throws blackwell::cuda_error and this ctor unwinds
    // with nothing allocated.
    void* host = nullptr;
    CUDA_CHECK_THROW(cudaHostAlloc(&host, capacity_ * sizeof(float),
                                   cudaHostAllocMapped | cudaHostAllocPortable));
    host_ptr_ = static_cast<float*>(host);

    void* dev = nullptr;
    cudaError_t map_err = cudaHostGetDevicePointer(&dev, host_ptr_, 0);
    if (map_err != cudaSuccess) {
        cudaFreeHost(host_ptr_);  // don't leak the pinned block if mapping failed
        host_ptr_ = nullptr;
        throw blackwell::cuda_error(map_err,
            std::string("AudioRingBuffer: cudaHostGetDevicePointer failed: ") +
            cudaGetErrorString(map_err));
    }
    device_ptr_ = static_cast<float*>(dev);
    // write_pos_/read_pos_/caches are zero-initialised in the header.
}

AudioRingBuffer::~AudioRingBuffer() {
    // noexcept teardown: ignore the free result (destructors must not throw).
    if (host_ptr_) cudaFreeHost(host_ptr_);
    host_ptr_ = nullptr;
    device_ptr_ = nullptr;
}

// ---- PRODUCER (audio thread) ------------------------------------------------
bool AudioRingBuffer::push_samples(const float* pcm, size_t count) noexcept {
    if (count == 0) return true;
    if (count > capacity_) return false;  // never fits; not a transient backpressure

    // Producer owns write_pos_: relaxed load of its own cursor is correct.
    const uint64_t w = write_pos_.load(std::memory_order_relaxed);

    // Free = capacity - in_flight. Try the cached read cursor first; only touch
    // the peer's atomic (acquire) if the cache says there is not enough room.
    if (capacity_ - static_cast<size_t>(w - producer_cached_read_pos_) < count) {
        producer_cached_read_pos_ = read_pos_.load(std::memory_order_acquire);
        if (capacity_ - static_cast<size_t>(w - producer_cached_read_pos_) < count) {
            return false;  // genuine backpressure: consumer has not drained enough
        }
    }

    // Copy into [w, w+count), splitting at the physical wrap boundary.
    const size_t idx = static_cast<size_t>(w) & mask_;
    const size_t first = (count < capacity_ - idx) ? count : capacity_ - idx;
    std::memcpy(host_ptr_ + idx, pcm, first * sizeof(float));
    if (count > first) {
        std::memcpy(host_ptr_, pcm + first, (count - first) * sizeof(float));
    }

    // Publish AFTER the bytes are in place: release pairs with the consumer's
    // acquire load of write_pos_, making the samples visible before it reads.
    write_pos_.store(w + count, std::memory_order_release);
    return true;
}

size_t AudioRingBuffer::free_samples() const noexcept {
    const uint64_t w = write_pos_.load(std::memory_order_relaxed);
    const uint64_t r = read_pos_.load(std::memory_order_acquire);
    return capacity_ - static_cast<size_t>(w - r);
}

// ---- CONSUMER (engine / CUDA thread) ----------------------------------------
size_t AudioRingBuffer::read_samples(float* dst, size_t max_count) noexcept {
    if (max_count == 0) return 0;

    // Consumer owns read_pos_: relaxed load of its own cursor is correct.
    const uint64_t r = read_pos_.load(std::memory_order_relaxed);

    size_t avail = static_cast<size_t>(consumer_cached_write_pos_ - r);
    if (avail < max_count) {  // cache may be stale-low; refresh from the producer
        consumer_cached_write_pos_ = write_pos_.load(std::memory_order_acquire);
        avail = static_cast<size_t>(consumer_cached_write_pos_ - r);
    }

    const size_t n = (avail < max_count) ? avail : max_count;
    if (n == 0) return 0;

    const size_t idx = static_cast<size_t>(r) & mask_;
    const size_t first = (n < capacity_ - idx) ? n : capacity_ - idx;
    std::memcpy(dst, host_ptr_ + idx, first * sizeof(float));
    if (n > first) {
        std::memcpy(dst + first, host_ptr_, (n - first) * sizeof(float));
    }

    // Publish the freed region AFTER copying out: release pairs with the
    // producer's acquire load of read_pos_.
    read_pos_.store(r + n, std::memory_order_release);
    return n;
}

size_t AudioRingBuffer::available_samples() const noexcept {
    const uint64_t w = write_pos_.load(std::memory_order_acquire);
    const uint64_t r = read_pos_.load(std::memory_order_relaxed);
    return static_cast<size_t>(w - r);
}

// ---- Two-phase zero-copy device drain (consumer) ----------------------------
AudioRingBuffer::ConsumeView AudioRingBuffer::consume_begin() const noexcept {
    const uint64_t r = read_pos_.load(std::memory_order_relaxed);
    const uint64_t w = write_pos_.load(std::memory_order_acquire);
    const size_t avail = static_cast<size_t>(w - r);
    const size_t idx = static_cast<size_t>(r) & mask_;
    const size_t to_wrap = capacity_ - idx;
    const size_t contiguous = (avail < to_wrap) ? avail : to_wrap;
    return ConsumeView{r, idx, contiguous, avail};
}

void AudioRingBuffer::consume_commit(size_t n) noexcept {
    // The caller has QUEUED its device reads of [read_pos_, read_pos_+n) on its
    // stream; publishing the advance frees those slots for the producer. Release
    // pairs with the producer's acquire load of read_pos_. Stream-ordering of the
    // queued reads vs. any later producer overwrite is the caller's invariant
    // (documented in the header): commit only what the stream will read first.
    const uint64_t r = read_pos_.load(std::memory_order_relaxed);
    read_pos_.store(r + n, std::memory_order_release);
}

}  // namespace blackwell::bridge
