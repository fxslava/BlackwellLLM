#pragma once
// -----------------------------------------------------------------------------
// realtime_dsp.h — STAGE 2: drains the capture ring, slides a 400-sample window
// at a 160-sample hop, and calls WhisperDSP::compute_log_mel_column() per window
// (the parity-preserving core). Resulting 128-float columns are pushed into a
// fixed-width rolling SpectrogramBuffer shared with the UI (STAGE 3).
//
// NOTE on parity: the streaming path takes windows directly off the live signal
// (NO center reflect-padding, NO drop-last, NO global (max-8) compression) — the
// offline-only steps that cannot exist in a stream. The per-frame DSP math is
// identical to the verified offline path; the global normalisation is applied by
// the viz layer over the visible history instead.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "whisper_dsp.h"

namespace rt {

class SampleRing;
class AudioRecorder;

// Fixed-width rolling spectrogram: mel-major columns, newest at the back, capped
// at `max_frames`. Mutex-guarded; written by the DSP worker, read by the UI.
class SpectrogramBuffer {
public:
    SpectrogramBuffer(int n_mels, int max_frames)
        : n_mels_(n_mels), max_frames_(max_frames) {}

    // Push one mel column tagged with its EXACT audio timestamp (seconds), i.e.
    // the sample-accurate position of the frame on the 10 ms hop grid.
    void push_column(std::vector<float>&& col, double t_seconds);
    // Copies the current mel columns (oldest -> newest) for lock-free rendering.
    void snapshot(std::vector<std::vector<float>>& out) const;
    // Exact audio timestamp (seconds) of the newest column; 0 if empty.
    double latest_timestamp() const;

    int n_mels() const { return n_mels_; }
    int max_frames() const { return max_frames_; }

private:
    mutable std::mutex m_;
    std::deque<std::vector<float>> cols_;
    std::deque<double> times_;   // parallel to cols_: each column's audio timestamp
    int n_mels_;
    int max_frames_;
};

class RealTimeDSP {
public:
    // `recorder` is optional (nullable): when present, the worker forwards every
    // popped sample to it for VAD recording, off the audio-callback thread.
    RealTimeDSP(const whisper::WhisperDSP& dsp, SampleRing& ring, SpectrogramBuffer& out,
                AudioRecorder* recorder = nullptr);
    ~RealTimeDSP();
    RealTimeDSP(const RealTimeDSP&) = delete;
    RealTimeDSP& operator=(const RealTimeDSP&) = delete;

    void start();
    void stop();

    // Optional PCM tap: invoked on the worker thread with every contiguous block
    // of samples popped from the capture ring, in order, BEFORE windowing — the
    // single-producer feed the speech pipeline's push_pcm() expects. Set once
    // before start(); null (the default) means the visualiser ignores it. Must be
    // wait-free (it runs inline in the capture-drain loop): it should only forward
    // to a lock-free ring / atomic, never block on I/O or a lock.
    using PcmTap = std::function<void(const float*, size_t)>;
    void set_pcm_tap(PcmTap tap) { pcm_tap_ = std::move(tap); }

private:
    void run();  // worker-thread loop

    const whisper::WhisperDSP& dsp_;
    SampleRing& ring_;
    SpectrogramBuffer& out_;
    AudioRecorder* recorder_;    // optional VAD recorder fed from the worker
    PcmTap pcm_tap_;             // optional speech-pipeline feed (null in the visualiser)
    std::vector<float> window_;  // sliding analysis buffer (owned by the worker)
    // Exact count of samples advanced on the hop grid (worker-thread only, no
    // sync needed). Drives the per-column timestamp; immune to wall-clock jitter.
    uint64_t total_samples_processed_ = 0;
    std::thread worker_;
    std::atomic<bool> running_{false};
};

}  // namespace rt
