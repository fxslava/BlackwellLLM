// -----------------------------------------------------------------------------
// TTSDuplexBridge implementation — see tts_duplex_bridge.hpp for the three
// design arguments (AEC tap point, the speaker/reference asymmetry on barge-in,
// and why flushing a ring is a consumer-side operation).
// -----------------------------------------------------------------------------
#include "tts_duplex_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>    // printf (the worker-side handoff diagnostics)
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

bool TTSDuplexBridge::SetAccentor(const IAccentor* accentor) {
    if (accentor == nullptr) {
        accentor_ = nullptr;
        return true;
    }
    // THE GUARD THAT MATTERS. A '+' the voice does not know does not read as
    // "no stress"; it reads as id 0, which is a space, i.e. a pause in the
    // middle of every marked word. Checking costs one hash lookup at startup
    // and turns a checkpoint mismatch from a stutter into a log line.
    if (tokenizer_.IdForCodepoint('+') == F5Tokenizer::kUnknownId) {
        std::printf("[tts] stress marks DISABLED: '+' is not in this voice's vocabulary, "
                    "so every mark would reach the model as a pause inside a word. "
                    "Use a voice pack whose vocab.txt contains '+'.\n");
        std::fflush(stdout);
        accentor_ = nullptr;
        return false;
    }
    accentor_ = accentor;
    return true;
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

    // ---- the text frontend, stages 1-3 (see the block above DuplexConfig) ---
    // Here and not in PushToken because every one of these needs whole
    // constructs, and a token stream splits "**", "2026" and "золотая" down the
    // middle wherever the LLM's tokenizer felt like it.
    //
    // 1. Markdown the model was asked not to put in the spoken half but did,
    //    plus whatever stress notation IT used, normalised away.
    chunk = NormalizeForSpeech(chunk, cfg_.text);
    if (chunk.empty()) return TtsStatus::EmptyResult;   // decoration only, no words

    // 2. Digits and symbols become words. Before the accentor, because "123"
    //    has no vowels to stress until it is "сто двадцать три".
    if (cfg_.expand_numbers) {
        chunk = ExpandForSpeech(chunk, cfg_.normalizer);
        if (chunk.empty()) return TtsStatus::EmptyResult;
    }

    // 3. Stress marks go in LAST, so nothing downstream of here rewrites them.
    if (accentor_ != nullptr) {
        chunk = accentor_->Accentuate(chunk);
    }

    std::size_t unknown = 0;
    std::vector<std::int32_t> ids32 = tokenizer_.Tokenize(chunk, &unknown);
    if (ids32.empty()) {
        // A SILENT FAILURE THAT USED TO LOOK LIKE NOTHING HAPPENING. Text
        // arrived, was chunked, and then tokenised to nothing -- which is what a
        // vocab that does not cover the script produces, and the only visible
        // symptom is silence. Say so.
        std::printf("[tts-worker] DROPPED %zu chars: vocabulary produced no tokens "
                    "(wrong vocab.txt for this language?)\n",
                    chunk.size());
        std::fflush(stdout);
        return TtsStatus::EmptyResult;
    }
    ids_scratch_.assign(ids32.begin(), ids32.end());

    // The line that proves synthesis actually STARTED, as opposed to text having
    // been pushed at a bridge that never got to it. `unknown` is on it because a
    // chunk that tokenises mostly to the unknown id still synthesises -- it just
    // produces confident gibberish or near-silence, and the count is the only
    // way to tell that from a bad reference clip.
    std::printf("[tts-worker] F5-TTS synthesizing %zu chars on CUDA (%zu tokens, %zu unknown)...\n",
                chunk.size(), ids_scratch_.size(), unknown);
    std::fflush(stdout);

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
        // Counted before this existed, but only visible in the exit summary --
        // which is no help while a user is standing there asking why it went
        // quiet. The status distinguishes an ORT/CUDA fault from a graph that
        // returned nothing.
        std::printf("[tts-worker] synthesis FAILED (%s) for %zu chars\n", to_string(st),
                    chunk.size());
        std::fflush(stdout);
        return st;
    }
    if (pcm_scratch_.empty()) return TtsStatus::EmptyResult;

    const TtsStatus pub = Publish(pcm_scratch_);
    if (pub != TtsStatus::Success) return pub;

    chunks_spoken_.fetch_add(1, std::memory_order_relaxed);
    if (out_samples != nullptr) *out_samples = pcm_scratch_.size();
    return TtsStatus::Success;
}

TtsStatus TTSDuplexBridge::PushPcm(const std::vector<float>& pcm) {
    if (pcm.empty()) return TtsStatus::EmptyResult;
    if (CancelRequested()) return TtsStatus::Interrupted;
    const TtsStatus st = Publish(pcm);
    if (st == TtsStatus::Success) chunks_spoken_.fetch_add(1, std::memory_order_relaxed);
    return st;
}

// PUBLISHING IS A HANDOFF, NOT A RENDEZVOUS. The speaker ring is sized to hold
// a whole chunk (see TtsRuntime), so the steady state is: this writes the chunk
// in one call, returns, and the worker goes straight back to synthesising chunk
// N+1 on the GPU while chunk N plays. The wait below is the exceptional path --
// a chunk longer than the ring, or a sink that has stopped pulling.
//
// IT SLEEPS RATHER THAN YIELDS, and that is a correctness argument rather than
// a tidiness one. yield() returns immediately whenever no other thread is
// runnable on that core, so the "wait" was a hot spin: on a machine whose cores
// are already carrying an LLM decode and a real-time audio callback, it burns a
// core for the entire duration of playback -- competing with the exact callback
// it is waiting on. A 2 ms sleep gives the same wake-up latency (the ring
// drains at device rate; one poll interval is a fraction of one buffer) at no
// cpu cost at all.
TtsStatus TTSDuplexBridge::Publish(const std::vector<float>& pcm) {
    std::size_t pushed = 0;
    // Measured from the last PROGRESS, not from entry: a chunk longer than the
    // ring is handed over across several playback periods, and that is normal
    // operation rather than a stall. What the budget must catch is a consumer
    // that has stopped consuming.
    auto last_progress = std::chrono::steady_clock::now();
    const auto poll = std::chrono::milliseconds(
        cfg_.write_poll_ms == 0 ? 1 : static_cast<long long>(cfg_.write_poll_ms));

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
            if (cfg_.write_wait_ms != 0) {
                const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - last_progress)
                                        .count();
                if (waited > static_cast<long long>(cfg_.write_wait_ms)) {
                    // The consumer has stopped pulling. Report back-pressure
                    // rather than wait forever; the caller decides whether that
                    // is a stalled device or a paused session.
                    samples_discarded_.fetch_add(pcm.size() - pushed,
                                                 std::memory_order_relaxed);
                    return TtsStatus::RuntimeFailure;
                }
            }
            std::this_thread::sleep_for(poll);
            continue;
        }
        last_progress = std::chrono::steady_clock::now();
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
