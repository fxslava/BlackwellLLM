#pragma once
// =============================================================================
// bridge/engine_control_bridge.hpp — the concrete IEngineControl implementor:
// maps the audio/VAD thread's speculative-speech commands onto the ONE engine
// thread without ever letting the audio thread touch CUDA (single-thread engine
// doctrine, CLAUDE.md).
//
// ROLE
//   IEngineControl (bridge_internal.hpp) is the seam the pure-C bridge marshals
//   across. EngineControlBridge closes it on the engine side. It is split by
//   thread, exactly like AudioRingBuffer:
//
//     PRODUCER  (audio / VAD / event thread)     CONSUMER  (engine-owning thread)
//     cancel_generation() ─ epoch bump ───▶  [ SPSC Command Ring ]  ──▶ pump()
//     rewind_kv()         ─ enqueue ──────▶        │ doorbell            │ dispatch
//     warm_prefill()      ─ enqueue ──────▶        ▼                     ▼
//     commit_and_decode() ─ enqueue ──────▶  wait_and_pump()   do_rewind / do_warm_
//                                             (0% CPU idle)     prefill / do_commit_
//                                                               decode  (CUDA here)
//
//   The producer edge NEVER launches a kernel, allocates device memory, or
//   rewinds a KV page. It only bumps an atomic epoch and appends a POD command
//   to a pre-allocated ring. The consumer edge (the single thread that owns
//   forward()/prefill) drains the ring inside its own decode loop and performs
//   every mutation there.
//
// INTEGRATION (no Engine::step(); core stays independent of the bridge)
//   There is no engine-owned generate loop — each BlackwellEngine::forward_status
//   decodes one token, and the decode loop lives in the engine-owning thread. So
//   this bridge is drained by that thread, NOT from inside src/core. The driver
//   pattern is:
//
//       while (running) {
//           bridge.wait_and_pump();          // block at 0% CPU until work arrives
//           // ... its do_commit_decode ran the decode loop, checking
//           //     bridge.cancelled(gen) each token for wait-free barge-in ...
//       }
//
//   Dependency direction stays bridge → core (blackwell_bridge links the engine's
//   white-box surface); core never links the bridge.
//
// ZERO-ALLOCATION HOT PATH (Architectural Rule #2)
//   The command ring is allocated ONCE in the ctor (INIT tier). Enqueue, poll,
//   the epoch bump, and the barge-in check are all wait-free, bounded, noexcept —
//   no lock, no CAS loop on the producer, no allocation. wait_and_pump() blocks
//   on a C++20 atomic doorbell (WaitOnAddress under MSVC): idle silence costs 0%
//   CPU, and a producer enqueue / barge-in wakes it with ~0 ms onset latency.
//
// MULTIMODAL EXECUTE SEAM
//   do_rewind is wired to the engine directly (BlackwellEngine::rewind). The
//   warm-prefill / commit-decode multimodal stages (Whisper frontend → Ultravox
//   projector → inject_audio_embeddings → Llama prefill/decode) touch
//   blackwell_audio, which the leaf bridge lib does not link; the engine assembly
//   overrides do_warm_prefill / do_commit_decode to supply them (binding a
//   projector via set_projector). The barge-in epoch, the ring, and the decode
//   loop's cancel checks live here and are fully exercised without a GPU.
// =============================================================================
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "bridge_internal.hpp"        // IEngineControl, TokenSink, AudioStreamHandle, AudioRingBuffer
#include "intent_commit.hpp"          // TerminationReason, IntentCommitQueue (the commit gate)
#include "blackwell/engine.h"         // BlackwellEngine, blackwell::EngineStatus

// The Ultravox projector lives in blackwell_audio; the leaf bridge lib only holds
// a non-owning pointer to it (the engine assembly binds + drives it), so a forward
// declaration keeps this header — and blackwell_bridge — free of any CUDA/audio dep.
namespace blackwell::audio { class UltravoxProjector; }

namespace blackwell::bridge {

// alignas(64) puts each contended atomic on its own cache line (no false sharing
// between the producer and consumer threads). That padding is the design, so
// silence MSVC C4324 which /W4 /WX would otherwise reject.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)  // structure padded due to alignment specifier (intentional)
#endif

class EngineControlBridge : public IEngineControl {
public:
    struct Config {
        // Pinned SPSC audio-ring capacity a new stream is sized to (samples).
        // Sized once from the engine plan; the ring never resizes (Rule #2).
        size_t audio_ring_capacity_samples = static_cast<size_t>(16000) * 30;  // 30 s @ 16 kHz
        // Command-ring depth; rounded UP to a power of two. Control commands are
        // low-rate (throttled warm prefills, rare barge-ins), so a small ring
        // never fills in practice; a full ring reports StateMismatch, never blocks.
        size_t command_ring_capacity = 256;
        // The sequence this bridge drives. Single-stream / batch=1 doctrine: 0.
        int    seq_id = 0;
    };

    // engine may be null ONLY for tests that override every do_* execute hook; a
    // production bridge binds the real single-thread engine. INIT tier: allocates
    // the command ring (may throw std::bad_alloc).
    explicit EngineControlBridge(BlackwellEngine* engine, const Config& cfg = {});
    ~EngineControlBridge() override;

    EngineControlBridge(const EngineControlBridge&) = delete;
    EngineControlBridge& operator=(const EngineControlBridge&) = delete;
    EngineControlBridge(EngineControlBridge&&) = delete;
    EngineControlBridge& operator=(EngineControlBridge&&) = delete;

    // Bind the Ultravox projector the engine-assembly's do_warm_prefill /
    // do_commit_decode override consumes. Call once during setup (engine thread),
    // before streaming begins. Non-owning.
    void set_projector(blackwell::audio::UltravoxProjector* projector) noexcept {
        projector_ = projector;
    }

    // ---- Commit gate to the Cloud API (Local Router pattern) -----------------
    // Bind the queue that carries EOS-terminated intents to the network
    // dispatcher. Non-owning; the app owns the queue and outlives the bridge.
    // Call once during setup (engine thread), before streaming begins.
    //
    // Leaving this null is a supported configuration: the local pipeline runs
    // exactly as before and nothing is ever dispatched. Cloud routing is opt-in
    // by binding a queue, never by default.
    void set_commit_queue(IntentCommitQueue* queue) noexcept { commit_queue_ = queue; }
    IntentCommitQueue* commit_queue() const noexcept { return commit_queue_; }

    // How the most recent generation ended. Published by publish_intent(), which
    // means it is correct for BOTH decode loops -- run_decode_loop AND an
    // override's own loop (RealEngineControl::decode_assistant_turn) -- because
    // both funnel through the same publish call. That is what lets the UI hold a
    // single EngineControlBridge* and still render the right verdict whichever
    // control is live. Recorded even when no commit queue is bound.
    TerminationReason last_reason() const noexcept {
        return static_cast<TerminationReason>(last_reason_.load(std::memory_order_acquire));
    }

    // ---- System-prompt prefix-cache floor (frozen KV boundary) ---------------
    // The system prompt is prefilled ONCE at startup and frozen; its KV must never
    // be truncated by a barge-in micro-rewind. Publishing the prefix length here
    // makes every do_rewind clamp keep_prompt_tokens UP to this floor, so the
    // rewind can drop speculative/decode state but never the system prefix. Set
    // once during setup (engine thread) after the system-prompt prefill; 0 (the
    // default) means "no floor" and leaves rewind semantics unchanged. Callable
    // from any thread (atomic publish).
    void set_system_prefix_tokens(uint32_t n) noexcept {
        system_prefix_tokens_.store(n, std::memory_order_release);
    }
    uint32_t system_prefix_tokens() const noexcept {
        return system_prefix_tokens_.load(std::memory_order_acquire);
    }

    // ---- VAD PRE-ROLL: the acoustic head a speech-start flush must NOT eat ----
    // A speech-start rewind flushes the audio ring, because everything buffered
    // between two utterances is background noise and encoding it makes the model
    // hallucinate. But the ring is fed CONTINUOUSLY, including while the VAD is
    // idle — so at the instant the detector fires, the samples sitting at the head
    // of the ring are not noise: they are the onset of the word the user just
    // started saying. A neural VAD needs 45–110 ms of audio to decide, and a total
    // flush throws exactly that window away, which is why short and plosive first
    // syllables ("p", "k", "t") were being clipped off the transcript.
    //
    // This is the amount of audio a flush RETAINS at the ring's head, so it is
    // consumed as the start of the new utterance instead of discarded. It is the
    // live-engine counterpart of ContinuousStreamingConfig::pre_roll_ms (which the
    // offline re-translation driver already honoured) and shares its default and
    // its bounds. 0 restores the old hard-flush behaviour.
    //
    // UI thread writes, engine thread reads inside do_rewind — one atomic, read at
    // an utterance boundary, so a change applies to the very next onset.
    void set_pre_roll_ms(int ms) noexcept {
        pre_roll_ms_.store(ms < 0 ? 0 : ms, std::memory_order_release);
    }
    int pre_roll_ms() const noexcept {
        return pre_roll_ms_.load(std::memory_order_acquire);
    }
    // The same figure in samples at `sample_rate`. The conversion lives here so
    // every flush site derives it identically (a rounding disagreement between two
    // call sites would show up as a few clipped samples nobody could account for).
    size_t pre_roll_samples(uint32_t sample_rate) const noexcept {
        return static_cast<size_t>(static_cast<uint64_t>(pre_roll_ms()) * sample_rate / 1000u);
    }

    // ---- IEngineControl: PRODUCER edge (audio / VAD / event thread) ----------
    // Every method here is wait-free and never touches CUDA. The speculative ones
    // are noexcept (they only bump an atomic + append a POD); a superseded op
    // (gen < the live epoch) is dropped immediately and reported as Success.
    size_t audio_ring_capacity_samples() const override;
    bool   generation_in_flight() const override;
    blackwell::EngineStatus submit_multimodal(AudioRingBuffer& ring,
                                              const char* prompt_utf8,
                                              TokenSink sink) override;
    void cancel_generation(uint64_t gen) noexcept override;
    blackwell::EngineStatus rewind_kv(AudioStreamHandle stream,
                                      uint32_t keep_prompt_tokens,
                                      uint64_t gen) noexcept override;
    blackwell::EngineStatus warm_prefill(AudioStreamHandle stream,
                                         uint64_t gen) noexcept override;
    blackwell::EngineStatus commit_and_decode(AudioStreamHandle stream,
                                              TokenSink sink,
                                              uint64_t gen) noexcept override;

    // ---- TYPED user turn: the text edge onto the SAME command ring ------------
    // A typed message must reach the commit gate by exactly the path a spoken one
    // does -- marshaled onto the engine thread, decoded by the same loop, offered
    // to the same IntentCommitQueue with the same EOS rule. Anything else would be
    // a second, untested route to a billed cloud call.
    //
    // NOT the audio producer edge, and deliberately not held to its contract: this
    // is driven by a human pressing Enter (single-digit events per minute), so it
    // takes an OWNED copy of the text under a short mutex rather than borrowing the
    // caller's buffer the way submit_multimodal does. The audio hot path never
    // touches this lock. Callable from any thread (the UI thread, in practice).
    //
    // Returns StateMismatch if the command ring is full; the text is then dropped
    // rather than queued behind a backlog the user can no longer see.
    blackwell::EngineStatus submit_text(std::string text, TokenSink sink) noexcept;

    // ---- ENGINE TASK: run arbitrary engine work on the owning thread ----------
    // The general marshaling seam, and the ONLY sanctioned way for a non-engine
    // thread to reach the engine (CLAUDE.md's single-threaded doctrine; this is
    // the port of poc_overlay's LiveTranslationTracker::PostEngineTask). `fn` is
    // executed by pump() on the engine thread and may do anything an engine-thread
    // caller may do -- prefill, rewind, reconfigure.
    //
    // WHY THIS IS NOT A COMMAND ON THE RING. The ring is a pre-allocated POD ring
    // on the AUDIO hot path (Rule #2: no allocation, wait-free). A task owns a
    // heap-allocated closure and is raised by a human clicking a button, so it
    // gets the same treatment submit_text's payload does: an owned copy behind a
    // short mutex, off the ring entirely. The audio path never touches this lock.
    //
    // ORDERING, which is the load-bearing part. Tasks run at a COMMAND-BATCH
    // BOUNDARY -- pump() drains every pending task BEFORE dispatching that
    // batch's commands, and a decode already in flight runs to completion first
    // (the engine thread is inside do_commit_decode, not inside pump). So a task
    // never observes a half-decoded turn, and it can safely invalidate the KV
    // cache out from under work that has not started yet.
    //
    // Callable from any thread. noexcept: a throwing `fn` is caught by the pump
    // like any other engine work. Returns false only if the closure could not be
    // queued (allocation failure), in which case it will never run.
    bool post_engine_task(std::function<void()> fn) noexcept;

    // ---- CONSUMER edge (the single engine-owning thread) ---------------------

    // Drain and execute every queued command that is not superseded. Non-blocking;
    // returns the number of commands actually executed. Call at the top of the
    // engine thread's loop iteration. noexcept: a faulted command reports through
    // its EngineStatus / sink, never by throw across the pump.
    size_t pump() noexcept;

    // Block at 0% CPU until at least one command is queued (or stop() is called),
    // then drain like pump(). Returns the number executed (0 only on stop with an
    // empty ring). Lost-wakeup-safe: the doorbell is re-checked after the empty
    // test and before the wait.
    size_t wait_and_pump() noexcept;

    // Wake a thread blocked in wait_and_pump() for shutdown; subsequent
    // wait_and_pump() calls return promptly. Idempotent; callable from any thread.
    void stop() noexcept;

    // Wait-free barge-in check for the decode loop: true once a newer epoch has
    // superseded `gen` (an on_speech_start bumped the generation). The decode loop
    // reads this every token and aborts immediately when it flips (Rule #1: the
    // interrupt is a monotone epoch, not a lock).
    bool cancelled(uint64_t gen) const noexcept {
        return gen < active_gen_.load(std::memory_order_acquire);
    }

    // The live epoch. Monotone; only ever advanced by cancel_generation().
    uint64_t active_generation() const noexcept {
        return active_gen_.load(std::memory_order_acquire);
    }

protected:
    enum class CommandType : uint32_t { Rewind, WarmPrefill, CommitDecode };

    // POD command marshaled producer → consumer. Trivially copyable (no heap on
    // the ring path). Which fields are live depends on `type`:
    //   Rewind        : stream, keep_prompt_tokens
    //   WarmPrefill   : stream
    //   CommitDecode  : stream (+ ring/prompt for the one-shot submit path), sink
    struct Command {
        CommandType       type = CommandType::Rewind;
        uint64_t          gen = 0;
        AudioStreamHandle stream = nullptr;
        AudioRingBuffer*  ring = nullptr;    // one-shot submit_multimodal path
        const char*       prompt = nullptr;  // one-shot submit_multimodal path (caller-owned)
        uint32_t          keep_prompt_tokens = 0;
        // This CommitDecode carries a TYPED user turn: the text itself is NOT in
        // the POD (the ring stays allocation-free), it sits in the FIFO behind
        // take_text_turn(). The flag is what tells a do_commit_decode override to
        // go and fetch it instead of reading the audio ring.
        bool              text_turn = false;
        TokenSink         sink{};
    };

    // Consumer edge (engine thread): claim the text this CommitDecode was queued
    // with. Returns false if the FIFO is empty (a superseded submission already
    // consumed it). Exactly one successful take per `text_turn` command, so the
    // FIFO stays in lockstep with the ring.
    bool take_text_turn(std::string& out) noexcept;

    // Consumer edge (engine thread): run every task post_engine_task() queued.
    // Returns how many ran. Called by pump() ahead of the command batch; exposed
    // to subclasses so an override that owns its own inner loop can also reach a
    // task boundary. noexcept -- a throwing task is swallowed exactly like a
    // throwing do_* hook.
    size_t drain_engine_tasks() noexcept;

    // Execute hooks, dispatched by pump() on the engine thread (virtual so the
    // engine assembly can supply the multimodal stages and tests can stub them).
    // A command whose gen is already superseded is dropped BEFORE dispatch.
    //   do_rewind       : wired here to BlackwellEngine::rewind (guarded).
    //   do_warm_prefill : base returns InvalidConfig (audio stage lives in the
    //   do_commit_decode  engine assembly; see the header preamble).
    virtual blackwell::EngineStatus do_rewind(const Command& cmd);
    virtual blackwell::EngineStatus do_warm_prefill(const Command& cmd);
    virtual blackwell::EngineStatus do_commit_decode(const Command& cmd);

    // Reusable barge-in decode loop for the engine assembly's do_commit_decode:
    // starting from first_token_id at start_pos, decode up to max_new_tokens
    // tokens, emitting each through `sink` and checking cancelled(gen) BEFORE
    // every step so a barge-in aborts within one token. decode_one() is the
    // per-token engine call, virtual so a CPU test drives the loop without a GPU.
    // Clears the in-flight flag and emits a final callback on exit.
    //
    // *out_reason receives WHY the loop stopped. This is load-bearing, not
    // diagnostic: it is the local arbiter's verdict, and IntentCommitQueue
    // dispatches to the paid Cloud API on TerminationReason::Eos and nothing
    // else. Before this existed, EOS / barge-in / token-cap all exited as
    // `is_final=1, BRIDGE_OK` and were indistinguishable to any consumer.
    //
    // A cancel is a clean supersede, not an error -- the returned EngineStatus
    // is still Success for both BargeIn and TokenCap. Branch on *out_reason,
    // never on the status, when deciding whether to dispatch.
    blackwell::EngineStatus run_decode_loop(uint64_t gen, int first_token_id,
                                            int start_pos, int max_new_tokens,
                                            const TokenSink& sink,
                                            TerminationReason* out_reason = nullptr);

    // One decode step: consume `token_id` at `pos` on seq cfg_.seq_id, sampling
    // greedily; writes the next token id to *out_token. Default forwards to
    // BlackwellEngine::forward_status. Returns the runtime-tier status.
    virtual blackwell::EngineStatus decode_one(int token_id, int pos, int* out_token) noexcept;

    // Is `token_id` the model's end-of-sequence marker (<|eot_id|>,
    // <|end_of_text|>, <|im_end|>, ...)? THE commit predicate: a true here is
    // what makes an utterance eligible for a paid cloud call.
    //
    // The base tier has no tokenizer (same reason do_warm_prefill is a stub),
    // so it returns false and the loop can only ever terminate on cap/cancel/
    // fault -- which fails CLOSED: no EOS means no dispatch, never a spurious
    // one. The engine assembly overrides this with
    // `tokenizer().is_stop(token_id)` (include/blackwell/tokenizer.h), which
    // already resolves the stop set from generation_config.json.
    virtual bool is_eos(int token_id) const noexcept;

    // Hand a finished generation to the commit gate. The ONLY path from the
    // engine thread to the network layer -- publish_intent applies no policy of
    // its own, it just forwards to IntentCommitQueue::offer(), where the commit
    // rule lives. Returns true iff the intent was committed for dispatch.
    //
    // Safe to call with any reason (and with no queue bound at all): a non-Eos
    // reason is counted and dropped inside the queue.
    bool publish_intent(TerminationReason reason, uint64_t gen, std::string&& payload,
                        uint32_t token_count) noexcept;

    // Clear the single-in-flight guard (called by run_decode_loop on exit; also
    // available to an override that finishes decode by another path).
    void clear_in_flight() noexcept { decoding_.store(false, std::memory_order_release); }

    // Apply the system-prompt prefix-cache floor: a rewind may keep MORE than the
    // requested prefix but never LESS than the frozen system prefix. Shared by the
    // base do_rewind and any engine-assembly / simulated override so the frozen-prefix
    // invariant holds no matter which override executes the rewind.
    uint32_t effective_keep_tokens(uint32_t requested) const noexcept {
        const uint32_t floor = system_prefix_tokens_.load(std::memory_order_acquire);
        return requested > floor ? requested : floor;
    }

    BlackwellEngine*                     engine_ = nullptr;    // non-owning; the single-thread engine
    blackwell::audio::UltravoxProjector* projector_ = nullptr; // non-owning; bound by the engine assembly
    IntentCommitQueue*                   commit_queue_ = nullptr; // non-owning; null = cloud routing off
    Config                               cfg_;

private:
    // Drop an op whose epoch is already stale (barge-in superseded it). Wait-free.
    bool superseded(uint64_t gen) const noexcept {
        return gen < active_gen_.load(std::memory_order_acquire);
    }

    // SPSC command-ring primitives (mirror AudioRingBuffer's cursor protocol).
    bool enqueue(const Command& cmd) noexcept;  // producer: append + ring the doorbell
    bool try_pop(Command& out) noexcept;        // consumer: one command or false
    bool ring_empty() const noexcept;
    void ring_notify() noexcept;                // publish the doorbell + wake a waiter

    // ---- pre-allocated command ring (never resized) -------------------------
    std::unique_ptr<Command[]> slots_;
    size_t capacity_ = 0;   // power of two
    size_t mask_ = 0;       // capacity_ - 1

    // SPSC cursors on separate cache lines: producer writes tail_, consumer writes
    // head_; each reads the other with acquire after publishing its own with release.
    alignas(64) std::atomic<uint64_t> head_{0};   // consumer (engine thread)
    alignas(64) std::atomic<uint64_t> tail_{0};   // producer (audio thread)

    // Monotone barge-in epoch: cancel_generation() advances it; the whole system
    // reads it to drop stale work / abort stale decode.
    alignas(64) std::atomic<uint64_t> active_gen_{0};

    // Frozen system-prompt prefix length (tokens). The KV rewind floor: do_rewind
    // never truncates below this. 0 = no floor. Published once at setup; read on
    // the engine thread by every rewind. Its own cache line: read on the hot
    // barge-in path, so keep it off the SPSC cursors' lines (no false sharing).
    alignas(64) std::atomic<uint32_t> system_prefix_tokens_{0};

    // Pre-roll retained by a speech-start flush (see set_pre_roll_ms). Mirrors
    // ContinuousStreamingConfig::pre_roll_ms's 250 ms default; not that type,
    // because this header must not drag the segmenter into every bridge consumer.
    std::atomic<int> pre_roll_ms_{250};

    // Last generation's verdict. Written by publish_intent() on the engine thread,
    // read by the UI thread for the status badge -- hence atomic.
    std::atomic<uint32_t> last_reason_{static_cast<uint32_t>(TerminationReason::None)};

    // Doorbell: a free-running counter the producer bumps + notifies on every
    // enqueue / cancel; wait_and_pump() blocks on it (0% CPU) and wakes on change.
    alignas(64) std::atomic<uint64_t> doorbell_{0};

    // Typed user turns, in submission order. Guarded by a plain mutex on purpose:
    // its only producer is a human typing and its only consumer is the engine
    // thread draining the ring, so the contention is nil and a lock-free FIFO
    // here would be ceremony. NOT reachable from the audio path (see submit_text).
    std::mutex              text_mu_;
    std::deque<std::string> text_turns_;

    // Engine tasks (post_engine_task). Same rationale as text_turns_: a human-rate
    // producer and a single consumer, so a plain mutex is the honest primitive.
    // `pending_tasks_` mirrors the deque's size so wait_and_pump()'s park loop can
    // test for work WITHOUT taking the lock -- the audio thread parks there every
    // idle moment, and it must not contend with the UI thread to do so.
    std::mutex                         task_mu_;
    std::deque<std::function<void()>>  tasks_;
    std::atomic<size_t>                pending_tasks_{0};

    std::atomic<bool> decoding_{false};   // single-in-flight guard (generation_in_flight)
    std::atomic<bool> stopping_{false};   // shutdown latch for wait_and_pump()
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace blackwell::bridge
