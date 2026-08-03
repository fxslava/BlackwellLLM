// -----------------------------------------------------------------------------
// TTSDuplexBridge implementation — see tts_duplex_bridge.hpp for the three
// design arguments (AEC tap point, the speaker/reference asymmetry on barge-in,
// and why flushing a ring is a consumer-side operation).
// -----------------------------------------------------------------------------
#include "tts_duplex_bridge.hpp"

#include <algorithm>
#include <cstring>
#include <thread>

namespace blackwell::tts {

TTSDuplexBridge::TTSDuplexBridge(ISynthesizer& synth,
                                 const F5Tokenizer& tokenizer,
                                 audio_rt::SpscRing<float>& speaker,
                                 audio_rt::SpscRing<float>& aec_reference,
                                 ChunkerConfig chunker_cfg,
                                 DuplexConfig cfg)
    : synth_(synth),
      tokenizer_(tokenizer),
      speaker_(speaker),
      aec_ref_(aec_reference),
      cfg_(cfg),
      chunker_(chunker_cfg) {}

void TTSDuplexBridge::SetCancelSignal(const std::atomic<bool>* cancel) noexcept {
    cancel_ = cancel;
}

// Relaxed on both: a one-iteration delay in observing a cancel is
// indistinguishable from the cancel having arrived one iteration later, and
// there is no other data published alongside these flags that needs ordering.
bool TTSDuplexBridge::CancelRequested() const noexcept {
    if (cancelled_.load(std::memory_order_relaxed)) return true;
    return cancel_ != nullptr && cancel_->load(std::memory_order_relaxed);
}

void TTSDuplexBridge::PushToken(std::string_view token) {
    const std::lock_guard<std::mutex> lk(text_mu_);
    // Tokens arriving after a barge-in belong to a reply the user already talked
    // over. Dropping them here means Resume() starts from a clean buffer instead
    // of speaking the tail of an abandoned sentence.
    if (cancelled_.load(std::memory_order_relaxed)) return;
    chunker_.PushToken(token);
}

void TTSDuplexBridge::EndOfStream() {
    const std::lock_guard<std::mutex> lk(text_mu_);
    if (cancelled_.load(std::memory_order_relaxed)) return;
    chunker_.Flush();
}

bool TTSDuplexBridge::HasPendingText() const {
    const std::lock_guard<std::mutex> lk(text_mu_);
    return chunker_.HasPendingChunk();
}

bool TTSDuplexBridge::speaking() const noexcept {
    return speaker_.available() > 0;
}

TtsStatus TTSDuplexBridge::PumpOnce(std::size_t* out_samples) {
    if (out_samples != nullptr) *out_samples = 0;

    // Checked BEFORE taking a chunk: a barge-in that landed while the previous
    // chunk was synthesising must not start another one.
    if (CancelRequested()) {
        Cancel();                       // idempotent; ensures the drain is armed
        return TtsStatus::Interrupted;
    }

    std::string chunk;
    {
        const std::lock_guard<std::mutex> lk(text_mu_);
        if (!chunker_.HasPendingChunk()) return TtsStatus::EmptyResult;
        chunk = chunker_.PopChunk();
    }
    if (chunk.empty()) return TtsStatus::EmptyResult;

    std::size_t unknown = 0;
    std::vector<std::int32_t> ids32 = tokenizer_.Tokenize(chunk, &unknown);
    if (ids32.empty()) return TtsStatus::EmptyResult;
    ids_scratch_.assign(ids32.begin(), ids32.end());

    const TtsStatus st = synth_.Synthesize(ids_scratch_, cancel_, pcm_scratch_);

    if (st == TtsStatus::Interrupted) {
        // Not a fault: the user asked for silence. The chunk is already popped
        // and is deliberately NOT put back -- re-speaking a line someone talked
        // over is exactly what barge-in exists to prevent.
        chunks_cancelled_.fetch_add(1, std::memory_order_relaxed);
        Cancel();
        return TtsStatus::Interrupted;
    }
    if (st != TtsStatus::Success) {
        if (st != TtsStatus::EmptyResult) {
            synthesis_errors_.fetch_add(1, std::memory_order_relaxed);
        }
        return st;
    }
    if (pcm_scratch_.empty()) return TtsStatus::EmptyResult;

    const TtsStatus pub = Publish(pcm_scratch_);
    if (pub != TtsStatus::Success) return pub;

    chunks_spoken_.fetch_add(1, std::memory_order_relaxed);
    if (out_samples != nullptr) *out_samples = pcm_scratch_.size();
    return TtsStatus::Success;
}

TtsStatus TTSDuplexBridge::Publish(const std::vector<float>& pcm) {
    std::size_t pushed = 0;
    std::size_t spins = 0;

    while (pushed < pcm.size()) {
        // Re-checked every iteration: a chunk can be several seconds long, and a
        // barge-in arriving while we are still feeding a full ring must stop the
        // feed rather than finish it.
        if (CancelRequested()) {
            chunks_cancelled_.fetch_add(1, std::memory_order_relaxed);
            samples_discarded_.fetch_add(pcm.size() - pushed, std::memory_order_relaxed);
            Cancel();
            return TtsStatus::Interrupted;
        }

        // write(), not write_or_drop(): this producer runs many times faster
        // than realtime, so a full ring means "the sink is still playing what it
        // already has", not "audio was lost". Counting an overrun on every retry
        // would make that metric meaningless -- spsc_ring.hpp is explicit.
        const std::size_t n = speaker_.write(pcm.data() + pushed, pcm.size() - pushed);
        if (n == 0) {
            if (cfg_.write_spin_limit != 0 && ++spins > cfg_.write_spin_limit) {
                // The consumer has stopped pulling. Report back-pressure rather
                // than spin forever; the caller decides whether that is a stalled
                // device or a paused session.
                samples_discarded_.fetch_add(pcm.size() - pushed, std::memory_order_relaxed);
                return TtsStatus::RuntimeFailure;
            }
            // Yield rather than spin hot: the consumer is a real-time callback,
            // and burning a core here is the one thing that could make it miss
            // the deadline we are waiting on.
            std::this_thread::yield();
            continue;
        }
        spins = 0;
        pushed += n;

        if (cfg_.aec_tap == AecTap::Synthesis) {
            // Deliberately write_or_drop: at this tap point the reference is
            // already mis-aligned (see the header), so a full reference ring is
            // a genuine loss to account for rather than something to wait on.
            (void)aec_ref_.write_or_drop(pcm.data() + (pushed - n), n);
        }
    }

    samples_published_.fetch_add(pcm.size(), std::memory_order_relaxed);
    return TtsStatus::Success;
}

void TTSDuplexBridge::Cancel() {
    // Bump FIRST so a consumer waking between these two lines sees the epoch
    // change and drains, rather than serving one more buffer of stale audio.
    speak_epoch_.fetch_add(1, std::memory_order_release);
    cancelled_.store(true, std::memory_order_release);

    const std::lock_guard<std::mutex> lk(text_mu_);
    chunker_.Reset();
}

void TTSDuplexBridge::Resume() {
    cancelled_.store(false, std::memory_order_release);
}

std::size_t TTSDuplexBridge::PullForPlayback(float* dst, std::size_t count) noexcept {
    if (dst == nullptr || count == 0) return 0;

    // ---- stale-audio drain, consumer side ----------------------------------
    // This is the only place it is legal to discard queued samples: reading is
    // the consumer's own operation, so nothing races the producer. See the
    // header on why SpscRing::reset() cannot be used for barge-in.
    const std::uint64_t epoch = speak_epoch_.load(std::memory_order_acquire);
    if (epoch != drained_epoch_) {
        drained_epoch_ = epoch;
        std::size_t dropped = 0;
        float sink[256];
        for (;;) {
            const std::size_t got = speaker_.read(sink, 256);
            if (got == 0) break;
            dropped += got;
        }
        if (dropped != 0) {
            samples_discarded_.fetch_add(dropped, std::memory_order_relaxed);
        }
        // The AEC reference is NOT drained. Audio already inside the device
        // buffer is still going to be emitted, and the room keeps ringing after
        // that -- which is exactly the window in which the user is talking over
        // us and the canceller needs its reference most. See the header.
    }

    // read_or_silence: a device callback must return a FULL buffer, so a
    // shortfall is padded and counted as an underrun by the ring itself.
    const std::size_t avail = speaker_.available();
    const std::size_t real = std::min(avail, count);
    speaker_.read_or_silence(dst, count);

    if (cfg_.aec_tap == AecTap::Playback) {
        // THE reference tap. `dst` now holds exactly what the device is about to
        // play, silence included -- and the silence matters: the AEC's reference
        // stream must stay continuous and sample-aligned with the mic, so gaps
        // would shift every subsequent sample and destroy the alignment this
        // tap point exists to provide.
        //
        // write_or_drop, not write: this runs on the audio callback and cannot
        // wait for anything. If the AEC is not draining, that is a real loss and
        // the ring counts it.
        (void)aec_ref_.write_or_drop(dst, count);
    }
    return real;
}

}  // namespace blackwell::tts
