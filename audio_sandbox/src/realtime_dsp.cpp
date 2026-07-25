// -----------------------------------------------------------------------------
// realtime_dsp.cpp — see realtime_dsp.h.
// -----------------------------------------------------------------------------
#include "realtime_dsp.h"

#include <chrono>

#include "audio_capture.h"
#include "audio_recorder.h"

namespace rt {

// --- SpectrogramBuffer -------------------------------------------------------
void SpectrogramBuffer::push_column(std::vector<float>&& col, double t_seconds) {
    std::lock_guard<std::mutex> lk(m_);
    cols_.push_back(std::move(col));
    times_.push_back(t_seconds);
    if (static_cast<int>(cols_.size()) > max_frames_) {
        cols_.pop_front();
        times_.pop_front();
    }
}

void SpectrogramBuffer::snapshot(std::vector<std::vector<float>>& out) const {
    std::lock_guard<std::mutex> lk(m_);
    out.assign(cols_.begin(), cols_.end());
}

double SpectrogramBuffer::latest_timestamp() const {
    std::lock_guard<std::mutex> lk(m_);
    return times_.empty() ? 0.0 : times_.back();
}

// --- RealTimeDSP -------------------------------------------------------------
RealTimeDSP::RealTimeDSP(const whisper::WhisperDSP& dsp, SampleRing& ring, SpectrogramBuffer& out,
                         AudioRecorder* recorder)
    : dsp_(dsp), ring_(ring), out_(out), recorder_(recorder) {}

RealTimeDSP::~RealTimeDSP() { stop(); }

void RealTimeDSP::start() {
    if (running_.exchange(true)) return;
    worker_ = std::thread(&RealTimeDSP::run, this);
}

void RealTimeDSP::stop() {
    if (!running_.exchange(false)) return;
    if (worker_.joinable()) worker_.join();
}

void RealTimeDSP::run() {
    const int n_fft = dsp_.config().n_fft;      // 400 (25 ms @ 16 kHz)
    const int hop = dsp_.config().hop_length;   // 160 (10 ms) — the exact hop grid
    const double sr = static_cast<double>(dsp_.config().sample_rate);  // 16000.0

    std::vector<float> chunk(4096);
    window_.clear();

    while (running_.load()) {
        const size_t got = ring_.pop(chunk.data(), chunk.size());
        if (got == 0) {
            // Ring empty: WAIT for the audio callback to deliver more. This poll
            // sleep gates only WHEN we look for data — it never sets the column
            // cadence, which is fixed by the sample-accurate hop grid below.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        // Feed the VAD recorder every popped sample, in order, off the audio
        // callback thread (WAV I/O here never stalls the realtime capture).
        if (recorder_ != nullptr) recorder_->process(chunk.data(), got);

        // Same in-order feed to the speech pipeline (push_pcm runs its own VAD and
        // marshals engine work). This worker is that pipeline's single producer.
        if (pcm_tap_) pcm_tap_(chunk.data(), got);

        window_.insert(window_.end(), chunk.begin(), chunk.begin() + got);

        // Emit one column per FULL hop only. We advance while a full 400-sample
        // window is present, stepping by exactly `hop` (=160) so consecutive
        // columns are spaced 10 ms apart with the standard 400/160 STFT overlap.
        // Fewer than n_fft samples -> no column (we loop back and wait for more),
        // so no partial hop can ever distort the Whisper frame spacing.
        size_t pos = 0;
        while (window_.size() - pos >= static_cast<size_t>(n_fft)) {
            std::vector<float> frame(window_.begin() + pos, window_.begin() + pos + n_fft);

            // Sample-accurate frame clock: each column is exactly one hop further
            // along the audio timeline, derived from a running SAMPLE count — not
            // from wall time — so there is zero timer drift or jitter.
            total_samples_processed_ += static_cast<uint64_t>(hop);
            const double timestamp_seconds = static_cast<double>(total_samples_processed_) / sr;

            out_.push_column(dsp_.compute_log_mel_column(frame),  // parity-preserving core
                             timestamp_seconds);
            pos += static_cast<size_t>(hop);
        }
        if (pos > 0) window_.erase(window_.begin(), window_.begin() + pos);
    }
}

}  // namespace rt
