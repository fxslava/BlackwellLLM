#pragma once
// -----------------------------------------------------------------------------
// audio_recorder.h — threshold-based Voice Activity Detection (VAD) recorder.
//
// Fed contiguous 16 kHz mono f32 PCM from the DSP worker thread (process()); the
// UI thread reads its config (atomics) and status (snapshot). On a detected
// onset it flushes a 500 ms pre-roll so word beginnings are never clipped; after
// `hangover_sec` of continuous sub-threshold audio it writes the clip to
// recordings/rec_YYYYMMDD_HHMMSS.wav via dr_wav and returns to IDLE.
//
// Timing is SAMPLE-accurate (silence measured in samples, not wall-clock), so it
// is immune to worker-thread scheduling jitter.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace rt {

class AudioRecorder {
public:
    enum class State { Idle, Recording, Hangover };

    struct Status {
        State state = State::Idle;
        float level_db = -120.0f;     // most-recent block RMS level (dBFS)
        double record_seconds = 0.0;  // length of the in-progress clip
        std::string last_saved;       // path of the most recently written clip
    };

    AudioRecorder(int sample_rate, std::string out_dir);

    // --- config: written by the UI thread, read lock-free by the worker --------
    std::atomic<bool>  enabled{false};
    std::atomic<float> threshold_db{-40.0f};  // onset threshold (dBFS)
    std::atomic<float> hangover_sec{1.0f};    // trailing silence before stop

    // Worker thread: consume `n` contiguous mono samples.
    void process(const float* pcm, size_t n);

    // UI thread: current state/level/length/last-saved.
    Status snapshot() const;

private:
    void evaluate_block();          // run VAD on the filled block_
    void start_recording();         // clear + flush pre-roll into record_
    void finalize_and_save();       // write record_ to a timestamped .wav
    void push_preroll_block();      // append block_ into the circular pre-roll
    float block_rms_db() const;     // RMS of block_ in dBFS

    const int sr_;
    const std::string out_dir_;
    const size_t block_size_;       // VAD evaluation block (160 = 10 ms)

    // Worker-thread-only state (no locking; single producer).
    std::vector<float> block_;      // fixed size block_size_
    size_t block_fill_ = 0;
    std::vector<float> preroll_;    // circular, capacity = 500 ms
    size_t preroll_pos_ = 0;
    bool preroll_full_ = false;
    std::vector<float> record_;     // active clip accumulator
    State state_ = State::Idle;
    size_t silence_samples_ = 0;    // consecutive sub-threshold samples

    // Shared status mirror (worker writes, UI reads).
    mutable std::mutex status_mutex_;
    Status status_;
};

}  // namespace rt
