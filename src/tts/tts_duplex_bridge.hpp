#pragma once
// -----------------------------------------------------------------------------
// TTSDuplexBridge — LLM token stream in, speaker audio + AEC reference out, with
// barge-in that actually stops the sound.
//
// This is the piece that makes the assistant full-duplex without triggering on
// itself. Three concerns meet here and each one has a failure mode that is quiet
// rather than loud:
//
// =============================================================================
// 1. THE AEC REFERENCE MUST BE TIME-ALIGNED WITH THE SPEAKER, NOT WITH SYNTHESIS
// =============================================================================
// An echo canceller subtracts a REFERENCE (what the speaker is playing) from the
// microphone signal. It can only do that if the reference is aligned to what the
// mic is hearing right now, to within its delay-search window -- roughly
// 100-250 ms for WebRTC AEC3, less for SpeexDSP.
//
// Synthesis does NOT run at that rate. F5 produces ~9 seconds of audio in ~3
// seconds of wall clock, so tapping the reference where the PCM is produced
// puts it up to several seconds AHEAD of the microphone. No delay estimator
// recovers from that: the AEC silently fails to cancel, the assistant hears
// itself, the VAD scores it as speech -- correctly, because it IS speech -- and
// barge-in cancels the generation currently being spoken. The system's own
// correctness works against it.
//
// So the reference is tapped where the samples are actually handed to the audio
// device: PullForPlayback(). That call is the audio callback, it runs at device
// rate, and the sample it serves is the sample about to be heard. This is
// AecTap::Playback and it is the default.
//
// AecTap::Synthesis exists because it is what a caller might expect, and it is
// correct ONLY if something downstream re-aligns (a delay-compensating queue, or
// an AEC fed from a hardware loopback). Choosing it without that is the bug
// described above.
//
// THE DEFAULT IS NOT WHAT THE VOICE ASSISTANT USES. It selects AecTap::None and
// feeds its canceller from a WASAPI loopback of the render endpoint instead --
// post-mix and post-volume, which no tap inside this class can see. Playback
// remains the default because it is the right answer for a caller that has no
// loopback; see AecTap::None below.
//
// =============================================================================
// 2. BARGE-IN FLUSHES THE SPEAKER BUT *KEEPS* THE AEC REFERENCE
// =============================================================================
// The asymmetry is deliberate and it is the least obvious thing in this file.
//
// When the user talks over the assistant we stop producing audio and drop what
// is queued -- that is the point of barge-in. But audio already handed to the
// device is still going to come out of the speaker, and the room keeps
// reverberating for another 100-300 ms after that. During exactly that window
// the user IS speaking, so the AEC is doing the most important work it will ever
// do. Flushing the reference there blinds it precisely when the mic contains
// both the user and the assistant's tail, and the assistant re-triggers on its
// own decaying output.
//
// So: speaker ring is drained, AEC reference is left alone.
//
// =============================================================================
// 3. FLUSHING AN SPSC RING IS A CONSUMER-SIDE OPERATION
// =============================================================================
// SpscRing::reset() is documented "NOT SAFE while either side is running", and
// barge-in is by definition a moment when both sides are running. spsc_ring.hpp
// prescribes the alternative directly: an epoch bump at the layer above, which
// lets the consumer skip stale spans without racing the producer.
//
// That is what Cancel() does -- it bumps speak_epoch_ and stops the producer.
// The drain itself happens inside PullForPlayback(), on the consumer thread,
// where reading is legal. The consequence worth knowing: if the audio callback
// is not running, queued audio is not dropped until it next runs. That is
// correct (nothing is playing either) but it means a test must drive the
// consumer to observe the drain.
//
// =============================================================================
// WHY ISynthesizer AND NOT F5TtsEngine DIRECTLY
// =============================================================================
// Two reasons, both structural. blackwell_tts is engine-free and CUDA-free by
// contract (src/tts/CMakeLists.txt), and naming F5TtsEngine here would drag
// ONNXRuntime and cudart into a target whose other consumers are a text
// frontend and a CPU test suite. And the behaviour that most needs testing --
// cancelling mid-synthesis -- would otherwise require a 1.3 GB model and a GPU,
// so it could never run in the fast `validation` suite. The seam makes the
// interesting case testable with a mock that blocks on command.
//
// F5TtsEngine satisfies this interface through a thin adapter in
// blackwell_tts_f5 (F5EngineSynthesizer), which is where the CUDA lives.
//
// THREADING
//   PushToken / EndOfStream : the LLM stream thread.
//   PumpOnce                : the TTS worker thread (owns synthesis).
//   PullForPlayback         : the audio callback. SOLE consumer of the speaker
//                             ring; must not block, allocate, or lock.
//   Cancel / observers      : any thread.
// The chunker is not thread-safe, so a mutex serialises the text side. It is not
// on a real-time path (PullForPlayback never takes it), so the lock is free.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "f5_tokenizer.hpp"
#include "spsc_ring.hpp"
#include "text_chunker.hpp"
#include "tts_status.hpp"

namespace blackwell::tts {

// The synthesis seam. One chunk of text in, PCM out, abortable.
class ISynthesizer {
public:
    virtual ~ISynthesizer() = default;
    ISynthesizer(const ISynthesizer&) = delete;
    ISynthesizer& operator=(const ISynthesizer&) = delete;

    // RUNTIME tier: noexcept, returns a status. `cancel` may be null; when it is
    // not, the implementation MUST poll it and abandon promptly -- for F5 that
    // is once per completed ODE step, so worst-case latency is one step.
    // On TtsStatus::Interrupted, out_pcm must be left empty.
    virtual TtsStatus Synthesize(const std::vector<std::int64_t>& text_ids,
                                 const std::atomic<bool>* cancel,
                                 std::vector<float>& out_pcm) noexcept = 0;

    virtual int sample_rate() const noexcept = 0;

protected:
    ISynthesizer() = default;
};

enum class AecTap : std::int32_t {
    // Reference is tapped as samples go to the device. Correct; see the header.
    Playback = 0,
    // Reference is tapped where PCM is produced. Only correct if something
    // downstream re-aligns it against the playback clock.
    Synthesis = 1,
    // NO reference is published at all. For callers whose canceller takes its
    // far end from somewhere this class cannot see -- specifically a WASAPI
    // LOOPBACK capture of the render endpoint, which observes what the speaker
    // actually emits (post-mix, post-volume, including audio this process never
    // produced) rather than what we handed the device.
    //
    // Choosing it is not merely skipping a copy on the audio thread, though it
    // does that: it also retires the pre-gain/post-gain correction the Playback
    // tap needs (AecCaptureFilter::SetReferenceGain), because a loopback
    // reference is already scaled by whatever the user set.
    None = 2,
};

struct DuplexConfig {
    AecTap aec_tap = AecTap::Playback;

    // How long PumpOnce will spin trying to hand a finished chunk to a full
    // speaker ring before giving up and reporting back-pressure. The producer
    // runs far faster than realtime, so a full ring means "the sink is still
    // playing", not "audio was lost" -- it should wait, not drop. 0 = wait
    // indefinitely (still abandoning if cancel fires).
    std::size_t write_spin_limit = 100000;
};

class TTSDuplexBridge {
public:
    // `speaker` and `aec_reference` are owned by the caller and outlive the
    // bridge. They are separate rings because they have different consumers
    // (the audio device and the AEC) and, on barge-in, different lifetimes.
    TTSDuplexBridge(ISynthesizer& synth,
                    const F5Tokenizer& tokenizer,
                    audio_rt::SpscRing<float>& speaker,
                    audio_rt::SpscRing<float>& aec_reference,
                    ChunkerConfig chunker_cfg = ChunkerConfig{},
                    DuplexConfig cfg = DuplexConfig{});

    TTSDuplexBridge(const TTSDuplexBridge&) = delete;
    TTSDuplexBridge& operator=(const TTSDuplexBridge&) = delete;

    // The barge-in signal, owned by whoever detects speech onset (the VAD /
    // segmenter). The bridge only ever READS it: it is checked before each chunk
    // and threaded into the solver so a long generation aborts mid-flight.
    // May be null (uninterruptible).
    void SetCancelSignal(const std::atomic<bool>* cancel) noexcept;

    // ---- text side (LLM stream thread) --------------------------------------
    void PushToken(std::string_view token);
    void EndOfStream();          // flush the chunker; the reply is complete
    bool HasPendingText() const;

    // ---- synthesis (TTS worker thread) --------------------------------------
    // Synthesises at most ONE pending chunk and publishes its PCM. Returns:
    //   Success        - a chunk was spoken; *out_samples gets its length
    //   EmptyResult    - nothing pending, or the chunk held nothing speakable
    //   Interrupted    - cancel fired; nothing was published (NOT a fault)
    //   RuntimeFailure - the synthesizer faulted; counted, chunk dropped
    // One chunk per call so the worker loop can re-check for barge-in between
    // chunks without this class owning a thread.
    TtsStatus PumpOnce(std::size_t* out_samples = nullptr);

    // ---- barge-in (any thread) ----------------------------------------------
    // Stops speaking NOW: drops pending text, abandons anything unspoken, and
    // marks the queued speaker audio stale so the consumer drops it on its next
    // pull. Does NOT touch the AEC reference -- see the header for why that
    // asymmetry is the point. Idempotent.
    void Cancel();

    // Clears the cancelled state so the bridge will speak again. Call when the
    // user's turn has ended, after the caller has cleared its cancel signal.
    void Resume();

    // Publishes ready-made PCM (24 kHz mono f32) as if it had been synthesised:
    // same ring, same barge-in checks, same AEC reference tap. For UI sounds --
    // the "check sound" tone -- which must travel the REAL output path or they
    // do not test it.
    //
    // Any thread EXCEPT the audio callback. It can block briefly while the ring
    // drains, exactly as a synthesised chunk does, which is why the callback is
    // excluded rather than merely discouraged.
    TtsStatus PushPcm(const std::vector<float>& pcm);

    bool cancelled() const noexcept { return cancelled_.load(std::memory_order_acquire); }
    std::uint64_t speak_epoch() const noexcept {
        return speak_epoch_.load(std::memory_order_acquire);
    }

    // ---- playback (audio callback thread; SOLE consumer of `speaker`) -------
    // Serves up to `count` samples, padding with silence on underrun, and -- in
    // AecTap::Playback mode -- copies exactly what it served into the AEC
    // reference. Drains stale audio first if a barge-in happened since the last
    // call. Lock-free and allocation-free.
    //
    // Returns the number of REAL samples served (the remainder of `count` is
    // silence). Always writes `count` samples to `dst`, because a device
    // callback must hand back a full buffer.
    std::size_t PullForPlayback(float* dst, std::size_t count) noexcept;

    // ---- observers (any thread) ---------------------------------------------
    std::uint64_t chunks_spoken() const noexcept { return chunks_spoken_.load(std::memory_order_relaxed); }
    std::uint64_t chunks_cancelled() const noexcept { return chunks_cancelled_.load(std::memory_order_relaxed); }
    std::uint64_t synthesis_errors() const noexcept { return synthesis_errors_.load(std::memory_order_relaxed); }
    std::uint64_t samples_published() const noexcept { return samples_published_.load(std::memory_order_relaxed); }
    std::uint64_t samples_discarded() const noexcept { return samples_discarded_.load(std::memory_order_relaxed); }

    // Samples of silence the speaker ring had to invent because the synthesiser
    // had not produced them yet. THE number that decides whether "no sound"
    // means "nothing was synthesised" or "synthesis could not keep up with the
    // device" -- the second is what GPU contention with a decoding LLM would
    // look like, and it is invisible in every other counter here.
    //
    // Safe from any thread (SpscRing's observers are), though it is only
    // meaningful next to chunks_spoken: an idle stream pads every buffer and is
    // not starving.
    std::uint64_t speaker_underruns() const noexcept { return speaker_.underruns(); }
    // True between the first sample of a chunk being queued and the speaker ring
    // running dry. OBSERVATION ONLY -- it used to drive a mic-gating interlock,
    // and that interlock is gone: the microphone is never gated on whether we
    // are speaking, because that is precisely when a user might interrupt. What
    // removes our voice from the capture stream is the echo canceller, which
    // works off the reference samples rather than off this flag.
    bool speaking() const noexcept;

private:
    // Publishes one chunk's PCM, honouring cancel while waiting for ring space.
    TtsStatus Publish(const std::vector<float>& pcm);
    bool CancelRequested() const noexcept;

    ISynthesizer&            synth_;
    const F5Tokenizer&       tokenizer_;
    audio_rt::SpscRing<float>& speaker_;
    audio_rt::SpscRing<float>& aec_ref_;
    DuplexConfig             cfg_;

    mutable std::mutex       text_mu_;     // guards chunker_ only
    TextChunker              chunker_;

    // Reused across chunks so the steady state allocates only when a chunk grows
    // past the high-water mark.
    std::vector<std::int64_t> ids_scratch_;
    std::vector<float>        pcm_scratch_;

    const std::atomic<bool>* cancel_ = nullptr;   // external, read-only
    std::atomic<bool>        cancelled_{false};
    std::atomic<std::uint64_t> speak_epoch_{0};
    // Last epoch the CONSUMER acted on. Only PullForPlayback touches it.
    std::uint64_t            drained_epoch_ = 0;

    std::atomic<std::uint64_t> chunks_spoken_{0};
    std::atomic<std::uint64_t> chunks_cancelled_{0};
    std::atomic<std::uint64_t> synthesis_errors_{0};
    std::atomic<std::uint64_t> samples_published_{0};
    std::atomic<std::uint64_t> samples_discarded_{0};
};

}  // namespace blackwell::tts
