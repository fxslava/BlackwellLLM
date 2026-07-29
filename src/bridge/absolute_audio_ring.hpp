#pragma once
// -----------------------------------------------------------------------------
// AbsoluteAudioRing — the N-second FIFO the continuous pipeline drafts from
// (docs/CONTINUOUS_STREAMING.md §2).
//
// WHY "ABSOLUTE". SpeechSegmenter names its segments in absolute sample indices
// since the stream began, and a redraft asks for [begin_sample, end_sample) of a
// growing utterance over and over. So the ring is addressed the same way: the
// caller never converts to a ring offset, and the two components cannot drift
// into disagreeing about where an utterance starts. The ring simply reports what
// it still holds.
//
// EVICTION IS THE INTERESTING PART. A fixed-capacity ring silently forgets its
// oldest audio, and "silently" is exactly wrong here: a redraft that asks for an
// utterance whose head has already been overwritten must be told so, not handed a
// truncated window that would produce a confidently wrong translation of a
// sentence missing its first word. read() therefore FAILS rather than clamping,
// and available_from() lets a caller check first.
//
// SIZING. The segmenter's max_utterance_ms bounds the longest window a redraft
// can ask for, so a ring of at least that plus the pre-roll can always satisfy
// one. The app sizes it from those knobs rather than guessing.
//
// THREADING: one producer (audio/DSP thread) calling write(); one consumer
// (engine thread) calling read(). The mutex is held for a memcpy — the same
// trade the SegmentJobQueue documents next door, and for the same reason: the
// producer is the DSP worker, not the driver ISR.
//
// DEPENDENCIES: STL only. No CUDA, no engine.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace blackwell::bridge {

class AbsoluteAudioRing {
public:
    explicit AbsoluteAudioRing(std::size_t capacity_samples)
        : buf_(capacity_samples == 0 ? 1 : capacity_samples) {}

    // Producer. Appends `count` samples at the current write head; the oldest
    // audio is overwritten once capacity is reached.
    void write(const float* samples, std::size_t count) {
        if (samples == nullptr || count == 0) return;
        const std::lock_guard<std::mutex> lock(m_);
        for (std::size_t i = 0; i < count; ++i) {
            buf_[static_cast<std::size_t>((written_ + i) % buf_.size())] = samples[i];
        }
        written_ += count;
    }

    // Consumer. Copies [begin, end) into `out`. Returns false — WITHOUT touching
    // `out` — when the range is not entirely resident, which is the honest answer
    // to "translate audio I no longer have".
    bool read(std::uint64_t begin, std::uint64_t end, std::vector<float>* out) const {
        if (out == nullptr || end <= begin) return false;
        const std::lock_guard<std::mutex> lock(m_);
        if (end > written_ || begin < oldest_locked()) return false;

        const std::size_t n = static_cast<std::size_t>(end - begin);
        out->resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            (*out)[i] = buf_[static_cast<std::size_t>((begin + i) % buf_.size())];
        }
        return true;
    }

    // Total samples ever written — also the absolute index one past the newest.
    std::uint64_t written() const {
        const std::lock_guard<std::mutex> lock(m_);
        return written_;
    }

    // Absolute index of the oldest sample still resident.
    std::uint64_t available_from() const {
        const std::lock_guard<std::mutex> lock(m_);
        return oldest_locked();
    }

    bool holds(std::uint64_t begin, std::uint64_t end) const {
        const std::lock_guard<std::mutex> lock(m_);
        return end > begin && end <= written_ && begin >= oldest_locked();
    }

    std::size_t capacity() const { return buf_.size(); }

    // Discontinuity (device change, mode flip). Keeps the absolute clock running:
    // rewinding it would make previously issued segments name the wrong audio.
    void clear() {
        const std::lock_guard<std::mutex> lock(m_);
        dropped_before_ = written_;
    }

private:
    std::uint64_t oldest_locked() const {
        const std::uint64_t by_capacity =
            written_ > buf_.size() ? written_ - buf_.size() : 0;
        return by_capacity > dropped_before_ ? by_capacity : dropped_before_;
    }

    mutable std::mutex m_;
    std::vector<float> buf_;
    std::uint64_t written_ = 0;
    std::uint64_t dropped_before_ = 0;   // raised by clear(); never lowered
};

}  // namespace blackwell::bridge
