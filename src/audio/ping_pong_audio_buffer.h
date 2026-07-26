#pragma once
// -----------------------------------------------------------------------------
// PingPongAudioBuffer — double-buffered (A/B) device staging that hands audio
// embeddings from the audio-pipeline stream (PRODUCER: Whisper encoder ->
// projector) to the LLM stream (CONSUMER: embedding-prefill) fully asynchronously
// and race-free, with NO host-side cudaDeviceSynchronize on the hot path.
//
// Two slots + TWO event pairs cover both cross-stream hazards:
//   ready_[i]    : producer finished WRITING slot i  (producer records, consumer waits)
//                  -> read-after-write: consumer never reads a half-written slot.
//   consumed_[i] : consumer finished READING slot i  (consumer records, producer waits)
//                  -> write-after-read: with only two slots, frame N reuses the
//                     slot of frame N-2, so the producer must not overwrite it
//                     until the frame N-2 consumer has drained it.
//
// A freshly-created CUDA event is treated as already-complete by
// cudaStreamWaitEvent, so the producer_acquire() calls for the first two frames
// are no-ops (nothing to wait for yet).
//
// DOCTRINE: not thread-safe. The frame index and both edges are driven from the
// single engine-owning thread; only the GPU work overlaps (two streams).
// -----------------------------------------------------------------------------
#include <cstddef>

#include <cuda_runtime.h>

#include "common.h"          // CUDA_CHECK_THROW
#include "device_buffer.h"   // blackwell::DeviceBuffer

namespace blackwell::audio {

class PingPongAudioBuffer {
public:
    explicit PingPongAudioBuffer(std::size_t elems_per_slot) : elems_(elems_per_slot) {
        for (int i = 0; i < 2; ++i) {
            buf_[i].allocate(elems_per_slot);
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&ready_[i], cudaEventDisableTiming));
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&consumed_[i], cudaEventDisableTiming));
        }
    }

    ~PingPongAudioBuffer() {
        for (int i = 0; i < 2; ++i) {
            if (ready_[i]) cudaEventDestroy(ready_[i]);
            if (consumed_[i]) cudaEventDestroy(consumed_[i]);
        }
    }

    PingPongAudioBuffer(const PingPongAudioBuffer&) = delete;
    PingPongAudioBuffer& operator=(const PingPongAudioBuffer&) = delete;

    // Slot backing frame `frame` (A/B alternation on the low bit).
    float* slot(long long frame) noexcept { return buf_[frame & 1].get(); }
    const float* slot(long long frame) const noexcept { return buf_[frame & 1].get(); }

    std::size_t elems() const noexcept { return elems_; }
    std::size_t bytes() const noexcept { return elems_ * sizeof(float); }

    // --- Producer edge (audio stream) ---
    // Wait until slot(frame)'s previous user (frame-2) has finished reading it,
    // before overwriting it. No-op for the first two frames.
    void producer_acquire(long long frame, cudaStream_t producer) {
        CUDA_CHECK_THROW(cudaStreamWaitEvent(producer, consumed_[frame & 1], 0));
    }
    // Mark slot(frame) fully written -> the consumer may now read it.
    void producer_publish(long long frame, cudaStream_t producer) {
        CUDA_CHECK_THROW(cudaEventRecord(ready_[frame & 1], producer));
    }

    // --- Consumer edge (LLM stream) ---
    // Make `consumer` wait until slot(frame) is fully written by the producer.
    void consumer_acquire(long long frame, cudaStream_t consumer) {
        CUDA_CHECK_THROW(cudaStreamWaitEvent(consumer, ready_[frame & 1], 0));
    }
    // Mark slot(frame) fully read -> the producer may reuse it (for frame+2).
    void consumer_release(long long frame, cudaStream_t consumer) {
        CUDA_CHECK_THROW(cudaEventRecord(consumed_[frame & 1], consumer));
    }

private:
    std::size_t elems_;
    DeviceBuffer<float> buf_[2];
    cudaEvent_t ready_[2] = {nullptr, nullptr};
    cudaEvent_t consumed_[2] = {nullptr, nullptr};
};

}  // namespace blackwell::audio
