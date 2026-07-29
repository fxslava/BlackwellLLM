#pragma once
// -----------------------------------------------------------------------------
// SegmentJobQueue — the audio-thread -> engine-thread handoff for segmenter
// events (docs/CONTINUOUS_STREAMING.md).
//
// THE ASYMMETRY THAT DEFINES IT: a Partial is a REDRAFT REQUEST and is worth
// exactly as much as its recency; a Final is a COMMIT and is worth everything.
//
//   * Consecutive Partials of the SAME utterance COLLAPSE — the newer one covers
//     a strictly longer prefix of the same audio from the same first sample, so
//     drafting the older one afterwards would be drafting stale audio and then
//     immediately throwing it away. When the engine falls behind, collapsing is
//     not a loss, it is the correct behaviour.
//   * A Final is NEVER collapsed, never superseded and never dropped. Losing one
//     would leave the commit pointer behind forever: the utterance would stay in
//     the draft zone and the next rewind would delete it.
//
// ORDER IS PRESERVED across kinds. Final(N) followed by Partial(N+1) must reach
// the engine in that order — running the Partial first would draft utterance N+1
// on top of a commit pointer that has not yet advanced past N, so N's translation
// would be discarded by that very redraft.
//
// LOCKING. A std::mutex, deliberately, and it is not a real-time hazard: the
// producer is the DSP worker (not the driver's ISR), it posts at most one event
// per 10 ms VAD block, and the critical section is a handful of vector
// operations. A hand-rolled lock-free ring with a collapse rule would be
// materially harder to get right than the thing it protects.
//
// DEPENDENCIES: STL + SpeechSegment. No CUDA, no engine.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>

#include "speech_segmenter.hpp"

namespace blackwell::bridge {

class SegmentJobQueue {
public:
    // `soft_capacity` bounds only the PARTIALS that may pile up. Finals are
    // exempt: see the header.
    explicit SegmentJobQueue(std::size_t soft_capacity = 8) noexcept
        : soft_capacity_(soft_capacity == 0 ? 1 : soft_capacity) {}

    // Producer (audio/DSP thread).
    void post(const vad::SpeechSegment& seg) {
        const std::lock_guard<std::mutex> lock(m_);

        // Collapse: only against the TAIL, and only within one utterance. A
        // Partial may never absorb anything that sits behind a Final, because the
        // Final has to be processed first for the commit pointer to advance.
        if (seg.kind == vad::SegmentKind::Partial && !q_.empty() &&
            q_.back().kind == vad::SegmentKind::Partial &&
            q_.back().utterance_id == seg.utterance_id) {
            q_.back() = seg;
            ++collapsed_;
            return;
        }

        q_.push_back(seg);

        // Overflow: shed the OLDEST Partial, which is the most stale redraft
        // request present. Finals are skipped over, so a backlog degrades by
        // losing draft freshness and never by losing a commit.
        while (q_.size() > soft_capacity_) {
            bool shed = false;
            for (auto it = q_.begin(); it != q_.end(); ++it) {
                if (it->kind == vad::SegmentKind::Partial) {
                    q_.erase(it);
                    ++dropped_;
                    shed = true;
                    break;
                }
            }
            // Nothing but Finals: let the queue grow rather than drop a commit.
            // The engine is catastrophically behind and losing translations would
            // be strictly worse than the memory.
            if (!shed) break;
        }
    }

    // Consumer (engine thread). Returns false when there is nothing to do.
    bool take(vad::SpeechSegment* out) {
        const std::lock_guard<std::mutex> lock(m_);
        if (q_.empty()) return false;
        if (out != nullptr) *out = q_.front();
        q_.pop_front();
        return true;
    }

    // Drops everything pending. For a hard reset (device change, mode flip) where
    // the audio those segments name is no longer in the ring.
    void clear() {
        const std::lock_guard<std::mutex> lock(m_);
        q_.clear();
    }

    std::size_t   pending() const { const std::lock_guard<std::mutex> l(m_); return q_.size(); }
    std::uint64_t collapsed() const { const std::lock_guard<std::mutex> l(m_); return collapsed_; }
    std::uint64_t dropped() const { const std::lock_guard<std::mutex> l(m_); return dropped_; }

private:
    mutable std::mutex m_;
    std::deque<vad::SpeechSegment> q_;
    std::size_t   soft_capacity_ = 8;
    std::uint64_t collapsed_ = 0;
    std::uint64_t dropped_ = 0;
};

}  // namespace blackwell::bridge
