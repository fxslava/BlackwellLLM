#pragma once
// -----------------------------------------------------------------------------
// WhisperCascadeMode — Mode C behind ISpeechMode: the CASCADE.
//
//   mic -> VAD -> SpeechSegmenter -> AbsoluteAudioRing -> whisper.cpp (GGML)
//       -> UTF-8 -> IntentCommitQueue -> the dispatcher answers it.
//
// WHAT MAKES IT DIFFERENT FROM MODE A, in one sentence: no audio ever reaches the
// backbone. Mode A splices audio soft-tokens into the KV cache and the model
// writes the transcript itself; here a separate ASR produces TEXT and the
// backbone only ever sees text. That is why cascade mode does not load the
// Ultravox audio head at all (engine_bootstrap's load_audio_head=false), and it
// is where the VRAM for the ~1.6 GB GGML model comes from.
//
// =============================================================================
// THE THREE THREADS, AND WHY THERE IS A THIRD ONE
// =============================================================================
//   DSP WORKER    on_pcm_block(): ring write, VAD block accounting, segmenter.
//                 Touches no engine, no CUDA, no window. Hands a finished
//                 utterance to the ASR worker and returns.
//   ASR WORKER    OWNS the WhisperAsr. Transcribes one utterance at a time, then
//                 MARSHALS the result onto the engine thread. Exists because
//                 Transcribe() blocks for the whole encode+decode and neither of
//                 the other two threads can afford to: the DSP worker would drop
//                 microphone blocks, and the engine thread is the decode loop.
//   ENGINE THREAD pump_engine() -> wait_and_pump(), which is what RUNS the
//                 marshaled publish. Still the only thread that calls into the
//                 engine, so the single-threaded engine doctrine is intact.
//
// WHY THE PUBLISH IS MARSHALED RATHER THAN DONE ON THE ASR WORKER.
// IntentCommitQueue is single-producer, and its producer is defined to be the one
// engine-owning thread. Offering from the ASR worker would be a second producer
// on a wait-free SPSC ring -- which does not fail loudly, it corrupts quietly
// under contention that only shows up when the user talks fast. post_engine_task
// is the seam that already exists for exactly this, and it costs one deque push.
//
// CUDA. With the ggml CUDA backend the ASR worker carries the THIRD CUDA context
// in the process (engine + ONNXRuntime CUDA EP + ggml). Nothing can make two
// contexts not share SMs, so the contention is managed by SCHEDULING instead:
// at most one utterance is ever in flight, the encode happens between turns
// rather than during one, and the queue below is deliberately shallow so a
// backlog is refused instead of silently deepening the latency it was built to
// reduce.
//
// ERROR TIER: INIT for the constructor -- it blocks until the ASR worker has
// either loaded the model or failed, and rethrows the failure on this thread, so
// "cascade mode could not arm" is a startup error with the model path in it and
// never a mode that comes up mute. RUNTIME after that: a failed encode is a lost
// utterance, counted and logged, never an exception crossing a thread boundary.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absolute_audio_ring.hpp"          // blackwell::bridge::AbsoluteAudioRing
#include "bridge/speculative_bridge_api.h"  // SpeechVadScoreFn, SpeechStateCallback
#include "engine_control_bridge.hpp"        // EngineControlBridge, IntentCommitQueue
#include "intent_commit.hpp"                // TerminationReason
#include "speech_segmenter.hpp"             // blackwell::vad::SpeechSegmenter

#include "speech_mode.hpp"
#include "whisper_asr.hpp"

namespace rt {

class WhisperCascadeMode final : public ISpeechMode {
public:
    struct Config {
        int sample_rate = 16000;
        // The VAD block the segmenter is clocked by. 10 ms, matching
        // SpeechPipelineController's -- and matching what Silero was exported
        // for, which is the binding constraint: the scorer is handed exactly one
        // block and a different size changes its verdicts.
        int block_samples = 160;

        // Detection. Same shape as Mode A's: one sensitivity knob, release
        // derived from it inside the segmenter.
        float onset_threshold = 0.5f;
        int   hangover_ms     = 800;

        // Utterance geometry. THESE THREE DECIDE WHETHER PUNCTUATION IS RIGHT,
        // which is not obvious and is worth stating: Whisper infers terminal
        // punctuation -- the '?' of a question above all -- from the prosody of
        // the last few hundred milliseconds, and it needs a clean left edge for
        // the first phoneme. A segment clipped at the last loud sample loses
        // both. See WhisperAsr::Transcribe.
        int preroll_ms        = 250;
        int tail_pad_ms       = 200;
        int min_utterance_ms  = 200;
        int max_utterance_ms  = 15000;

        // The frozen prefix this mode prefills. Cascade mode never runs a
        // transcription prompt (whisper.cpp does that job), so the ONE prefix it
        // lays down is the assistant persona -- unlike Mode A, which has two.
        std::string system_prompt;
    };

    using PrefillSystemPromptFn = std::function<std::uint32_t(const std::string&)>;

    // ENGINE THREAD. TWO edges, and the split is load-bearing -- see publish().
    //
    //   on_text    the words themselves. Fires BEFORE the transcript is offered
    //              to the commit gate, so the user's bubble is on screen before
    //              anything can announce a reply to it.
    //   on_verdict whether the gate took it. Necessarily after the offer, and
    //              carries no text: it only closes the bubble on_text opened.
    //
    // `utterance_id` is the epoch the UI keys its live bubble on -- the same
    // role gen_id plays for Mode A's token stream -- and both edges carry it so
    // the second cannot be attached to the wrong turn.
    //
    // Callbacks rather than a direct AssistantView dependency because this
    // header must stay drivable from a test with no UI, no WebView2 and no COM.
    struct TranscriptCallbacks {
        std::function<void(const std::string& utf8, std::uint32_t utterance_id)> on_text;
        std::function<void(std::uint32_t utterance_id, bool committed)>          on_verdict;
    };

    // `control` and `vad_user` are BORROWED and must outlive this object.
    //
    // The VAD arrives as the C ABI's own SpeechVadScoreFn seam, exactly as
    // ConversationalMode takes it and for the same reason: naming SileroVAD here
    // would drag ONNXRuntime into every consumer of this header. A null vad_fn
    // falls back to the RMS ramp below -- which is a genuinely worse detector,
    // and says so once at construction rather than quietly underperforming.
    //
    // THROWS (INIT tier) if the ASR cannot load its model. Cascade mode refuses
    // to arm without one; it never degrades to the legacy path behind the user's
    // back, because a pipeline that silently changed which model heard you is
    // the one failure nobody would think to look for.
    WhisperCascadeMode(blackwell::bridge::EngineControlBridge* control,
                       PrefillSystemPromptFn prefill,
                       SpeechVadScoreFn vad_fn, void* vad_user,
                       const WhisperAsrConfig& asr_cfg,
                       const Config& cfg,
                       SpeechStateCallback state_cb,
                       TranscriptCallbacks transcript_cb,
                       void* callback_user)
        : control_(control), prefill_(std::move(prefill)),
          vad_fn_(vad_fn), vad_user_(vad_user), cfg_(cfg),
          state_cb_(state_cb), transcript_cb_(std::move(transcript_cb)),
          callback_user_(callback_user),
          // The ring must hold the LONGEST window a Final can name -- the cap
          // plus its pre-roll -- or the read that follows would fail on exactly
          // the utterances that matter most (the long ones). The extra second is
          // slack for the ASR worker's own scheduling latency: audio keeps
          // arriving while it transcribes, and the head must not be evicted out
          // from under a job that is already queued.
          ring_(static_cast<std::size_t>(cfg.sample_rate) *
                static_cast<std::size_t>(cfg.max_utterance_ms + cfg.preroll_ms + 1000) / 1000u) {
        if (control_ == nullptr) throw std::runtime_error("WhisperCascadeMode: null control");

        blackwell::vad::SegmenterConfig scfg;
        scfg.sample_rate   = cfg_.sample_rate;
        scfg.block_samples = cfg_.block_samples;
        scfg.onset_threshold = cfg_.onset_threshold;
        scfg.preroll_ms      = cfg_.preroll_ms;
        scfg.hangover_ms     = cfg_.hangover_ms;
        scfg.tail_pad_ms     = cfg_.tail_pad_ms;
        scfg.min_utterance_ms = cfg_.min_utterance_ms;
        scfg.max_utterance_ms = cfg_.max_utterance_ms;
        // NO PARTIALS. The segmenter's redraft cadence exists for Mode B's
        // rolling re-translation; a cascade has nothing to redraft -- it either
        // has a whole utterance to transcribe or it has nothing. Pushing the
        // period past the cap is how you say "never" without a second flag.
        scfg.partial_period_ms = cfg_.max_utterance_ms + 1;
        segmenter_ = blackwell::vad::SpeechSegmenter(scfg);

        block_.reserve(static_cast<std::size_t>(cfg_.block_samples));
        live_onset_.store(cfg_.onset_threshold, std::memory_order_relaxed);
        live_hangover_ms_.store(cfg_.hangover_ms, std::memory_order_relaxed);

        if (vad_fn_ == nullptr) {
            std::fprintf(stderr,
                         "[cascade] NOTE: no neural VAD -- falling back to an RMS ramp. "
                         "Utterance boundaries will be less reliable, which in a cascade "
                         "means clipped transcripts rather than a slightly late commit.\n");
        }

        start_asr_worker(asr_cfg);   // blocks until the model is up, or throws
    }

    ~WhisperCascadeMode() override { shutdown_asr_worker(); }

    WhisperCascadeMode(const WhisperCascadeMode&) = delete;
    WhisperCascadeMode& operator=(const WhisperCascadeMode&) = delete;

    const char* name() const noexcept override { return "WhisperCascade"; }

    void start_on_engine_thread() override {
        if (prefill_) frozen_prefix_ = prefill_(cfg_.system_prompt);
    }

    // ---- DSP WORKER ---------------------------------------------------------
    // Ring, VAD, segmenter. The one rule is that nothing here may block: this
    // thread is draining a live capture device, and a stall is dropped samples.
    void on_pcm_block(const float* samples, std::size_t count) noexcept override {
        if (samples == nullptr || count == 0) return;

        // Retune point. The segmenter's setters are owner-thread-only by
        // contract, and this is that thread -- so the settings panel's atomics
        // are drained HERE and nowhere else (speech_segmenter.hpp says why).
        apply_live_retune();

        ring_.write(samples, count);

        const bool manual = manual_mode_.load(std::memory_order_acquire);
        const std::size_t block_n = static_cast<std::size_t>(cfg_.block_samples);

        for (std::size_t i = 0; i < count; ++i) {
            const double s = static_cast<double>(samples[i]);
            sumsq_ += s * s;
            block_.push_back(samples[i]);
            if (block_.size() < block_n) continue;

            // PUSH-TO-TALK. A zero probability is fed rather than the block being
            // skipped: the segmenter's clock must keep advancing in lockstep with
            // the ring, or its absolute sample indices stop naming the audio the
            // ring holds and every subsequent read is off by the muted span.
            const float p = manual ? 0.0f : score_block();
            if (const auto seg = segmenter_.on_block(p)) {
                if (seg->kind == blackwell::vad::SegmentKind::Final) on_final(*seg);
            } else if (segmenter_.in_speech() && !was_speaking_) {
                // ONSET -- THE BARGE-IN EDGE. Two things have to happen here, and
                // only one of them is cosmetic.
                //
                // 1. ABORT THE IN-FLIGHT GENERATION. An epoch bump is what makes
                //    the decode loop stop at its next token check. Mode A gets
                //    this from the speech pipeline; Mode C has no pipeline, so
                //    without this line the answer to the PREVIOUS turn keeps
                //    decoding while the user is already talking over it -- the
                //    speaker goes quiet (the TTS barge-in below) and the GPU does
                //    not, which is the worst of both.
                //
                //    IT IS ALSO THE SM-CONTENTION ANSWER. The one moment the ASR
                //    encode and the text decode would genuinely overlap is
                //    exactly this one, and the overlap is with work the user has
                //    just cancelled by speaking. Killing it here is why the third
                //    CUDA context does not have to fight the decode loop for SMs.
                //
                //    Callable from this thread by design: the epoch bump is an
                //    O(1) fire-and-forget store, the same one Mode A's VAD makes.
                //
                // 2. Announce it as the pipeline state Mode A announces, so the
                //    app's existing TTS barge-in wiring works here untouched
                //    (main.cpp's on_state).
                control_->cancel_generation(control_->active_generation() + 1);
                emit_state(SPEECH_STATE_PREFILL_SPEAKING);
            }
            was_speaking_ = segmenter_.in_speech();

            block_.clear();
            sumsq_ = 0.0;
        }
    }

    // ---- ENGINE THREAD ------------------------------------------------------
    // Parks at 0% CPU until work arrives. The work is either a bridge command or
    // a task the ASR worker posted -- pump() drains the tasks first either way.
    void pump_engine() override { (void)control_->wait_and_pump(); }

    void stop() noexcept override {
        control_->cancel_generation(control_->active_generation() + 1);
        control_->stop();
        shutdown_asr_worker();
    }

    ModeTelemetry telemetry() const noexcept override {
        ModeTelemetry t;
        t.name = name();
        t.utterances = published_.load(std::memory_order_relaxed);
        return t;
    }

    // ---- LIVE settings ------------------------------------------------------
    // Any thread. Both are stored and drained by the DSP worker at the top of
    // its next block, because the segmenter's own setters may only be touched by
    // the thread that clocks it.
    void set_vad_threshold(float probability) noexcept override {
        live_onset_.store(probability, std::memory_order_relaxed);
        retune_.store(true, std::memory_order_release);
    }
    void set_silence_hangover_ms(std::uint32_t ms) noexcept override {
        live_hangover_ms_.store(static_cast<int>(ms), std::memory_order_relaxed);
        retune_.store(true, std::memory_order_release);
    }
    // No speculative prefill exists in a cascade: there is nothing to warm with,
    // because no audio reaches the backbone and the text does not exist until the
    // ASR has finished. The interface's default no-op is the honest body, and it
    // is spelled out here so the absence reads as a decision rather than an
    // oversight.
    void set_warm_prefill_interval_ms(std::uint32_t /*ms*/) noexcept override {}

    // Honours the interface contract -- no automatic VAD transition may fire --
    // and NOTHING MORE, which is the part to know before turning it on: Mode C
    // has no explicit press/release event source of its own (Mode A gets one
    // through the speech_pipeline_* C ABI). So in this mode push-to-talk
    // currently means the cascade is PARKED, not "trigger it by hand".
    void set_manual_mode(bool enabled) noexcept override {
        manual_mode_.store(enabled, std::memory_order_release);
    }

    // ---- observers ----------------------------------------------------------
    std::uint32_t frozen_prefix_tokens() const noexcept { return frozen_prefix_; }
    std::uint64_t published() const noexcept {
        return published_.load(std::memory_order_relaxed);
    }
    // Utterances the ASR never saw because the queue was already full. Non-zero
    // means the encode is slower than the speaker, which is the ONE failure this
    // design can produce silently -- so it is counted rather than only logged.
    std::uint64_t dropped_overrun() const noexcept {
        return dropped_overrun_.load(std::memory_order_relaxed);
    }
    // Null until the worker has loaded it; borrowed, and only safe to read
    // telemetry from (the object itself belongs to the ASR worker).
    const WhisperAsr* asr() const noexcept { return asr_.get(); }

private:
    // ---- one queued utterance ----------------------------------------------
    // The PCM is COPIED here, on the DSP thread, rather than the ASR worker
    // reading the ring later. That trade is deliberate: the copy is one memcpy
    // of at most a few hundred KB, once per utterance (not per block), and it
    // removes the entire eviction race -- the worker can never be handed a range
    // whose head the ring dropped while it was busy. The vectors live in the
    // slots and keep their capacity, so the steady state allocates nothing.
    struct Job {
        std::vector<float> pcm;
        std::uint32_t      utterance_id = 0;
    };

    // Shallow ON PURPOSE. One utterance in flight plus one waiting; past that the
    // ASR is slower than the speaker and the honest move is to refuse the NEW
    // work rather than deepen a latency queue -- the same asymmetry
    // IntentCommitQueue chose, for the same reason.
    static constexpr std::size_t kMaxPending = 2;

    // ---- DSP-thread helpers -------------------------------------------------

    void apply_live_retune() noexcept {
        if (!retune_.exchange(false, std::memory_order_acq_rel)) return;
        segmenter_.set_onset_threshold(live_onset_.load(std::memory_order_relaxed));
        segmenter_.set_hangover_ms(live_hangover_ms_.load(std::memory_order_relaxed));
    }

    // One block's speech probability. The neural scorer when there is one; a
    // ramp over RMS dB otherwise.
    //
    // WHY A RAMP AND NOT A THRESHOLD. The segmenter's hysteresis works on the
    // GAP between onset and release, and a detector that only ever returns 0 or 1
    // collapses that gap to nothing -- the release threshold becomes unreachable
    // from below and the hysteresis stops absorbing anything. Mapping dB linearly
    // across a 20 dB window keeps both thresholds meaningful.
    float score_block() noexcept {
        if (vad_fn_ != nullptr) {
            const float p = vad_fn_(vad_user_, block_.data(), block_.size());
            // Negative is the ABI's "no opinion"; hold the previous verdict
            // rather than reading it as silence, which would cut an utterance.
            if (p >= 0.0f) last_prob_ = p;
            return last_prob_;
        }
        const double mean = sumsq_ / static_cast<double>(block_.size());
        const double db = 10.0 * std::log10(mean > 1e-20 ? mean : 1e-20);
        constexpr double kFloorDb = -60.0;
        constexpr double kCeilDb  = -40.0;
        const double t = (db - kFloorDb) / (kCeilDb - kFloorDb);
        last_prob_ = static_cast<float>(t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t));
        return last_prob_;
    }

    // A complete utterance. Copy it out of the ring and hand it over.
    //
    // EVERY EXIT FROM HERE CLOSES THE TURN, including the two that drop the
    // audio. The state emitted on the way in parks the UI on "your words are
    // being written down", and only a publish() clears it -- so a path that
    // returned early without publishing would leave the app looking permanently
    // busy after one dropped utterance. publish("") is the empty-transcript
    // case the ASR worker already produces for silence.
    void on_final(const blackwell::vad::SpeechSegment& seg) noexcept {
        // The ASR is now the thing the turn is waiting on. Reported as the state
        // Mode A uses for its transcription decode, so the UI's phase label means
        // the same thing in both modes: "your words are being written down".
        emit_state(SPEECH_STATE_DECODE_TRANSLATING);

        Job job;
        {
            std::lock_guard<std::mutex> lk(job_mu_);
            if (pending_.size() >= kMaxPending) {
                dropped_overrun_.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr,
                             "[cascade] utterance %u DROPPED: the ASR is behind (%zu queued). "
                             "The encode is slower than the speaker.\n",
                             seg.utterance_id, kMaxPending);
                publish(std::string{}, seg.utterance_id);
                return;
            }
            // Reuse a retired slot's buffer when there is one, so a steady
            // conversation stops allocating after the first utterance.
            if (!spare_.empty()) {
                job = std::move(spare_.back());
                spare_.pop_back();
            }
        }

        // read() FAILS rather than truncating when the head is gone -- which is
        // the correct answer, because a window missing its first word
        // transcribes into a confidently wrong sentence rather than a visibly
        // broken one. The ring is sized so this cannot happen; it is logged
        // loudly because if it ever does, the sizing is the bug.
        //
        // Done OUTSIDE the lock: it is a memcpy of up to a few hundred KB, and
        // the ASR worker takes this same mutex on every job.
        if (!ring_.read(seg.begin_sample, seg.end_sample, &job.pcm)) {
            std::fprintf(stderr,
                         "[cascade] utterance %u DROPPED: [%llu,%llu) is no longer in the "
                         "ring (oldest %llu). The ring is undersized for "
                         "whisper_max_utterance_ms.\n",
                         seg.utterance_id,
                         static_cast<unsigned long long>(seg.begin_sample),
                         static_cast<unsigned long long>(seg.end_sample),
                         static_cast<unsigned long long>(ring_.available_from()));
            {
                std::lock_guard<std::mutex> back(job_mu_);
                if (spare_.size() < kMaxPending) spare_.push_back(std::move(job));
            }
            publish(std::string{}, seg.utterance_id);
            return;
        }
        job.utterance_id = seg.utterance_id;

        {
            std::lock_guard<std::mutex> lk(job_mu_);
            pending_.push_back(std::move(job));
        }
        job_cv_.notify_one();
    }

    void emit_state(SpeechPipelineState next) noexcept {
        if (state_cb_ == nullptr) return;
        const SpeechPipelineState prev = last_state_;
        last_state_ = next;
        state_cb_(callback_user_, prev, next, 0);
    }

    // ---- the ASR worker -----------------------------------------------------

    // Starts the worker and BLOCKS until it has either constructed the WhisperAsr
    // or failed. The construction has to happen over there -- CUDA's current
    // device is per-thread and ggml binds its context to whichever device is
    // current when it first allocates (whisper_asr.hpp) -- but the FAILURE has to
    // surface here, as an exception on the caller's thread, or a bad model path
    // would come up as a mode that simply never transcribes anything.
    void start_asr_worker(const WhisperAsrConfig& asr_cfg) {
        std::mutex ready_mu;
        std::condition_variable ready_cv;
        bool ready = false;
        std::string init_error;

        worker_ = std::thread([&, asr_cfg] {
            try {
                asr_ = std::make_unique<WhisperAsr>(asr_cfg);
            } catch (const std::exception& e) {
                init_error = e.what();
            }
            {
                // NOTIFY UNDER THE LOCK, and this is not cargo cult: ready_mu,
                // ready_cv and init_error are LOCALS of start_asr_worker, and
                // this thread outlives that frame. Notifying after the unlock
                // would let the waiter return and destroy them while this thread
                // is still inside notify_one(). Holding the lock across the
                // notify makes the waiter block on reacquiring it until this
                // scope ends, which is what keeps the frame alive long enough.
                std::lock_guard<std::mutex> lk(ready_mu);
                ready = true;
                ready_cv.notify_one();
            }
            if (asr_ == nullptr) return;   // the ctor below throws; nothing to run
            asr_loop();
        });

        {
            std::unique_lock<std::mutex> lk(ready_mu);
            ready_cv.wait(lk, [&] { return ready; });
        }
        if (asr_ == nullptr) {
            // The thread has already returned or is about to; joining here is what
            // keeps this class's failure path from leaving a thread behind for the
            // destructor that will never run (the object is not constructed).
            if (worker_.joinable()) worker_.join();
            throw std::runtime_error("WhisperCascadeMode: ASR unavailable -- " + init_error);
        }
        std::printf("[cascade] whisper ARMED on the %s: %s\n",
                    asr_->gpu() ? "GPU" : "CPU", asr_->model_path().c_str());
        std::fflush(stdout);
    }

    void asr_loop() noexcept {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(job_mu_);
                job_cv_.wait(lk, [&] {
                    return asr_stopping_.load(std::memory_order_acquire) || !pending_.empty();
                });
                if (asr_stopping_.load(std::memory_order_acquire) && pending_.empty()) return;
                job = std::move(pending_.front());
                pending_.pop_front();
            }

            // THE BLOCKING CALL. Tens of ms on the GPU, hundreds on the CPU, and
            // the entire reason this thread exists.
            std::string text = asr_->Transcribe(job.pcm.data(), job.pcm.size());
            const std::uint32_t id = job.utterance_id;

            // Retire the buffer before publishing, so its capacity is available
            // to the next utterance even if the engine thread is busy.
            {
                std::lock_guard<std::mutex> lk(job_mu_);
                if (spare_.size() < kMaxPending) spare_.push_back(std::move(job));
            }

            // An empty transcript is silence, a non-speech artefact, or a failed
            // encode -- indistinguishable here, and all three mean "publish
            // nothing". It STILL goes through publish(), because the UI is
            // sitting on the "your words are being written down" phase this
            // mode set when the utterance ended, and only the turn boundary
            // clears it. Skipping the empty case would leave the app looking
            // busy until the next time somebody spoke.
            publish(std::move(text), id);
        }
    }

    // Hand the transcript to the engine thread. Everything downstream of this
    // point -- the commit gate, the dispatcher, the UI -- runs there.
    //
    // THE ORDER OF THE TWO CALLBACK EDGES BELOW IS THE WHOLE MESSAGE ORDERING
    // GUARANTEE, and it is not obvious enough to leave implicit.
    //
    // offer() does not merely record the intent: it wakes the dispatcher thread
    // out of wait_pop(), and that thread's FIRST act is on_dispatch_start ->
    // "remote.start" -> the page creates the assistant's bubble. That message
    // and this transcript land in the SAME single UI queue
    // (AssistantWindow::post_event), which is FIFO and preserves whatever order
    // it is handed. So whichever thread enqueues first is the bubble that
    // appears first -- and offering before publishing put the ANSWER ABOVE THE
    // QUESTION every time the dispatcher won the race.
    //
    // Publishing first makes it deterministic rather than lucky: the push into
    // the UI queue completes in program order before offer() is called, and
    // offer()/wait_pop() synchronise, so remote.start cannot be enqueued ahead
    // of the words it is answering. Do not merge these back into one call.
    void publish(std::string text, std::uint32_t utterance_id) noexcept {
        const bool empty = text.empty();
        auto task = [this, text = std::move(text), utterance_id, empty]() mutable {
            if (!empty && transcript_cb_.on_text) transcript_cb_.on_text(text, utterance_id);

            bool committed = false;
            if (!empty) {
                // Eos, and it is not a fiction: the commit rule asks whether the
                // thought FINISHED on its own, and a transcript over a
                // hangover-bounded utterance is exactly that -- the speaker
                // stopped talking. The reasons the gate rejects (barge-in, a
                // token cap, a fault) are all properties of a decode loop that
                // does not exist on this path, so mapping them in would be the
                // dishonest direction.
                if (auto* q = control_->commit_queue(); q != nullptr) {
                    // token_count is 0 and stays 0: nothing on this path is
                    // tokenized. It counts LOCAL DECODE tokens, which is what
                    // makes dropped_token_cap() diagnostic on the legacy path --
                    // filling it with a byte count here would put a plausible
                    // wrong number where a reader expects that meaning.
                    committed = q->offer(blackwell::bridge::TerminationReason::Eos,
                                         control_->active_generation(),
                                         std::string(text),
                                         /*token_count=*/0u);
                    if (!committed) {
                        // The only way Eos fails to commit is a full queue: the
                        // dispatcher is still streaming an earlier answer.
                        std::fprintf(stderr,
                                     "[cascade] transcript NOT dispatched (the commit queue is "
                                     "full -- an earlier answer is still streaming)\n");
                    }
                }
                published_.fetch_add(1, std::memory_order_relaxed);
            }
            // THE TURN BOUNDARY, and deliberately NOT an emit_state(IDLE).
            // Mode A's consumers treat IDLE as "read the control's published
            // verdict and close the bubble with it" -- and on this path the
            // control has no verdict to publish, because no decode loop ran.
            // Emitting IDLE would therefore overwrite the transcript's own
            // outcome with a stale TerminationReason::None and paint every
            // committed utterance as undispatched. The callback carries the
            // verdict directly instead.
            //
            // Text-free, and after the offer by necessity: it closes the bubble
            // on_text already opened. It fires even for an empty transcript,
            // because the UI is sitting on the "writing your words down" phase
            // and only a turn boundary clears it.
            if (transcript_cb_.on_verdict) transcript_cb_.on_verdict(utterance_id, committed);
        };
        if (!control_->post_engine_task(std::move(task))) {
            // The task deque is full, which means the engine thread has not
            // pumped in a long time. Losing the transcript is bad; losing it
            // silently would be worse.
            std::fprintf(stderr,
                         "[cascade] transcript LOST: the engine task queue is full "
                         "(utterance %u)\n", utterance_id);
        }
    }

    void shutdown_asr_worker() noexcept {
        if (!worker_.joinable()) return;
        asr_stopping_.store(true, std::memory_order_release);
        job_cv_.notify_all();
        worker_.join();
        // Destroyed ON the owner thread would be ideal; the worker has exited by
        // now, so this is the only thread left that can see it and the
        // whisper_context has no thread affinity for teardown.
        asr_.reset();
    }

    // ---- borrowed -----------------------------------------------------------
    blackwell::bridge::EngineControlBridge* control_ = nullptr;
    PrefillSystemPromptFn                   prefill_;
    SpeechVadScoreFn                        vad_fn_ = nullptr;
    void*                                   vad_user_ = nullptr;
    Config                                  cfg_;
    SpeechStateCallback                     state_cb_ = nullptr;
    TranscriptCallbacks                     transcript_cb_;
    void*                                   callback_user_ = nullptr;

    // ---- DSP-thread state (no synchronisation: one owner) -------------------
    blackwell::vad::SpeechSegmenter segmenter_{};
    std::vector<float>              block_;
    double                          sumsq_ = 0.0;
    float                           last_prob_ = 0.0f;
    bool                            was_speaking_ = false;
    SpeechPipelineState             last_state_ = SPEECH_STATE_IDLE;

    // Written by the DSP thread, read by the ASR worker. Its own mutex.
    blackwell::bridge::AbsoluteAudioRing ring_;

    // ---- the handoff --------------------------------------------------------
    std::mutex                      job_mu_;
    std::condition_variable         job_cv_;
    std::deque<Job>                 pending_;
    std::vector<Job>                spare_;     // retired buffers, capacity intact
    std::atomic<bool>               asr_stopping_{false};
    std::thread                     worker_;
    std::unique_ptr<WhisperAsr>     asr_;       // owned BY the worker thread

    // ---- live retune (written any thread, drained by the DSP worker) --------
    std::atomic<bool>  retune_{false};
    std::atomic<float> live_onset_{0.5f};
    std::atomic<int>   live_hangover_ms_{800};
    std::atomic<bool>  manual_mode_{false};

    std::atomic<std::uint64_t> published_{0};
    std::atomic<std::uint64_t> dropped_overrun_{0};
    std::uint32_t              frozen_prefix_ = 0;
};

}  // namespace rt
