// -----------------------------------------------------------------------------
// audio_recorder.cpp — see audio_recorder.h. Sole dr_wav implementation TU for
// the audio_realtime binary (parity_check has its own in parity_check.cpp).
// -----------------------------------------------------------------------------
#define DR_WAV_IMPLEMENTATION
// dr_wav.h is third-party (SYSTEM include), but C4701 "potentially uninitialized
// local" is emitted late in codegen and slips past /external:W0, so /WX would
// flag it. Quarantine only this third-party #include (skill compiler-hygiene).
#pragma warning(push)
#pragma warning(disable: 4701)  // potentially uninitialized local variable
#include "dr_wav.h"
#pragma warning(pop)

#include "audio_recorder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <utility>

namespace rt {
namespace {

constexpr size_t kBlockSize = 160;          // 10 ms VAD block @ 16 kHz
constexpr size_t kPrerollMs = 500;          // pre-roll length

int16_t to_pcm16(float x) {
    const float c = std::clamp(x, -1.0f, 1.0f) * 32767.0f;
    return static_cast<int16_t>(std::lround(c));
}

std::string timestamp_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_s(&tmv, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
    return buf;
}

}  // namespace

AudioRecorder::AudioRecorder(int sample_rate, std::string out_dir)
    : sr_(sample_rate),
      out_dir_(std::move(out_dir)),
      block_size_(kBlockSize),
      block_(kBlockSize, 0.0f),
      preroll_(static_cast<size_t>(sample_rate) * kPrerollMs / 1000, 0.0f) {}

void AudioRecorder::process(const float* pcm, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        block_[block_fill_++] = pcm[i];
        if (block_fill_ == block_size_) {
            evaluate_block();
            block_fill_ = 0;
        }
    }
}

float AudioRecorder::block_rms_db() const {
    double sum = 0.0;
    for (size_t i = 0; i < block_size_; ++i) sum += static_cast<double>(block_[i]) * block_[i];
    const double rms = std::sqrt(sum / static_cast<double>(block_size_));
    return static_cast<float>(20.0 * std::log10(rms + 1e-6));
}

void AudioRecorder::push_preroll_block() {
    for (size_t i = 0; i < block_size_; ++i) {
        preroll_[preroll_pos_] = block_[i];
        if (++preroll_pos_ >= preroll_.size()) {
            preroll_pos_ = 0;
            preroll_full_ = true;
        }
    }
}

void AudioRecorder::start_recording() {
    record_.clear();
    // Flush the pre-roll in chronological order so the onset is not clipped.
    if (preroll_full_) {
        record_.insert(record_.end(), preroll_.begin() + preroll_pos_, preroll_.end());
        record_.insert(record_.end(), preroll_.begin(), preroll_.begin() + preroll_pos_);
    } else {
        record_.insert(record_.end(), preroll_.begin(), preroll_.begin() + preroll_pos_);
    }
    // Pre-roll consumed; reset so a later clip does not re-emit stale audio.
    preroll_pos_ = 0;
    preroll_full_ = false;
}

void AudioRecorder::evaluate_block() {
    const float db = block_rms_db();
    const bool en = enabled.load(std::memory_order_relaxed);
    const float thr = threshold_db.load(std::memory_order_relaxed);
    const size_t hangover =
        static_cast<size_t>(std::max(0.0f, hangover_sec.load(std::memory_order_relaxed)) * sr_);

    if (!en) {
        if (state_ != State::Idle) {  // toggled off mid-clip: save what we have
            record_.insert(record_.end(), block_.begin(), block_.end());
            finalize_and_save();
            state_ = State::Idle;
        } else {
            push_preroll_block();
        }
    } else if (state_ == State::Idle) {
        if (db > thr) {
            start_recording();
            record_.insert(record_.end(), block_.begin(), block_.end());
            state_ = State::Recording;
            silence_samples_ = 0;
        } else {
            push_preroll_block();
        }
    } else {  // Recording or Hangover: keep accumulating the tail
        record_.insert(record_.end(), block_.begin(), block_.end());
        if (db >= thr) {
            state_ = State::Recording;
            silence_samples_ = 0;
        } else {
            silence_samples_ += block_size_;
            if (state_ == State::Recording) state_ = State::Hangover;
            if (silence_samples_ >= hangover) {
                finalize_and_save();
                state_ = State::Idle;
            }
        }
    }

    std::lock_guard<std::mutex> lk(status_mutex_);
    status_.state = state_;
    status_.level_db = db;
    status_.record_seconds =
        (state_ == State::Idle) ? 0.0 : static_cast<double>(record_.size()) / sr_;
}

void AudioRecorder::finalize_and_save() {
    if (record_.empty()) return;

    std::error_code ec;
    std::filesystem::create_directories(out_dir_, ec);
    const std::string path = out_dir_ + "/rec_" + timestamp_now() + ".wav";

    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_PCM;
    fmt.channels = 1;
    fmt.sampleRate = static_cast<drwav_uint32>(sr_);
    fmt.bitsPerSample = 16;

    drwav wav;
    if (drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        std::vector<int16_t> pcm16(record_.size());
        for (size_t i = 0; i < record_.size(); ++i) pcm16[i] = to_pcm16(record_[i]);
        drwav_write_pcm_frames(&wav, pcm16.size(), pcm16.data());
        drwav_uninit(&wav);
        std::lock_guard<std::mutex> lk(status_mutex_);
        status_.last_saved = path;
    }
    record_.clear();
}

AudioRecorder::Status AudioRecorder::snapshot() const {
    std::lock_guard<std::mutex> lk(status_mutex_);
    return status_;
}

}  // namespace rt
