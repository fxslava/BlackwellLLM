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
#include <memory>

#include "bridge_internal.hpp"        // IEngineControl, TokenSink, AudioStreamHandle, AudioRingBuffer
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
        TokenSink         sink{};
    };

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
    // Clears the in-flight flag and emits a final callback on exit (cancel, cap,
    // or fault). A cancel is a clean supersede, not an error.
    blackwell::EngineStatus run_decode_loop(uint64_t gen, int first_token_id,
                                            int start_pos, int max_new_tokens,
                                            const TokenSink& sink);

    // One decode step: consume `token_id` at `pos` on seq cfg_.seq_id, sampling
    // greedily; writes the next token id to *out_token. Default forwards to
    // BlackwellEngine::forward_status. Returns the runtime-tier status.
    virtual blackwell::EngineStatus decode_one(int token_id, int pos, int* out_token) noexcept;

    // Clear the single-in-flight guard (called by run_decode_loop on exit; also
    // available to an override that finishes decode by another path).
    void clear_in_flight() noexcept { decoding_.store(false, std::memory_order_release); }

    BlackwellEngine*                     engine_ = nullptr;    // non-owning; the single-thread engine
    blackwell::audio::UltravoxProjector* projector_ = nullptr; // non-owning; bound by the engine assembly
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

    // Doorbell: a free-running counter the producer bumps + notifies on every
    // enqueue / cancel; wait_and_pump() blocks on it (0% CPU) and wakes on change.
    alignas(64) std::atomic<uint64_t> doorbell_{0};

    std::atomic<bool> decoding_{false};   // single-in-flight guard (generation_in_flight)
    std::atomic<bool> stopping_{false};   // shutdown latch for wait_and_pump()
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace blackwell::bridge
