// =============================================================================
// bridge/engine_control_bridge.cpp — EngineControlBridge: the concrete
// IEngineControl implementor. See engine_control_bridge.hpp for the full
// threading contract and the producer/consumer split.
// =============================================================================
#include "engine_control_bridge.hpp"

#include <exception>

#include "bridge/engine_api.h"  // BridgeStatus / BRIDGE_OK for the token sink

using blackwell::EngineStatus;

namespace blackwell::bridge {
namespace {

// Round UP to the next power of two (>= 2). The ring's `pos & mask_` slot trick
// needs a power-of-two capacity.
size_t round_up_pow2(size_t n) noexcept {
    if (n < 2) return 2;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// EngineStatus -> BridgeStatus, 1:1 with the doctrine's vocabulary (mirrors
// engine_api.cpp's map_status; kept local so the sink can report a final status).
BridgeStatus to_bridge_status(EngineStatus s) noexcept {
    switch (s) {
        case EngineStatus::Success:          return BRIDGE_OK;
        case EngineStatus::CudaRuntimeError: return BRIDGE_ERR_CUDA;
        case EngineStatus::OutOfVram:        return BRIDGE_ERR_OUT_OF_MEMORY;
        case EngineStatus::InvalidArgument:  return BRIDGE_ERR_INVALID_ARG;
        case EngineStatus::InvalidConfig:    return BRIDGE_ERR_INVALID_ARG;
        case EngineStatus::StateMismatch:    return BRIDGE_ERR_STATE;
    }
    return BRIDGE_ERR_INTERNAL;
}

}  // namespace

// ---- construction -----------------------------------------------------------
EngineControlBridge::EngineControlBridge(BlackwellEngine* engine, const Config& cfg)
    : engine_(engine), cfg_(cfg) {
    capacity_ = round_up_pow2(cfg_.command_ring_capacity);
    mask_ = capacity_ - 1;
    // INIT tier: the command ring is allocated ONCE here and never resized.
    slots_ = std::make_unique<Command[]>(capacity_);
}

EngineControlBridge::~EngineControlBridge() {
    // Unblock any thread parked in wait_and_pump() before the ring is destroyed.
    stop();
}

// -----------------------------------------------------------------------------
// SPSC command-ring primitives. Single producer (the audio/VAD/event thread — the
// app funnels all boundary events onto it, exactly like AudioRingBuffer's own
// "exactly one producer" contract) and single consumer (the engine thread).
// -----------------------------------------------------------------------------
void EngineControlBridge::ring_notify() noexcept {
    // Free-running doorbell: publish, then wake a parked consumer. WaitOnAddress
    // under MSVC, so an idle wait costs 0% CPU and this wakes it with ~0 ms onset.
    doorbell_.fetch_add(1, std::memory_order_release);
    doorbell_.notify_one();
}

bool EngineControlBridge::enqueue(const Command& cmd) noexcept {
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    const uint64_t h = head_.load(std::memory_order_acquire);
    if (t - h >= capacity_) return false;  // ring full -> backpressure (StateMismatch)
    slots_[static_cast<size_t>(t) & mask_] = cmd;
    tail_.store(t + 1, std::memory_order_release);  // publish AFTER the slot write
    ring_notify();
    return true;
}

bool EngineControlBridge::try_pop(Command& out) noexcept {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    const uint64_t t = tail_.load(std::memory_order_acquire);
    if (h == t) return false;  // empty
    out = slots_[static_cast<size_t>(h) & mask_];
    head_.store(h + 1, std::memory_order_release);  // free the slot for the producer
    return true;
}

bool EngineControlBridge::ring_empty() const noexcept {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
}

// -----------------------------------------------------------------------------
// IEngineControl: PRODUCER edge (audio / VAD / event thread). Wait-free; no CUDA.
// -----------------------------------------------------------------------------
size_t EngineControlBridge::audio_ring_capacity_samples() const {
    return cfg_.audio_ring_capacity_samples;
}

bool EngineControlBridge::generation_in_flight() const {
    return decoding_.load(std::memory_order_acquire);
}

void EngineControlBridge::cancel_generation(uint64_t gen) noexcept {
    // Monotone publish of the live epoch. Any in-flight decode with an older gen
    // aborts at its next cancelled() check; any still-queued op with an older gen
    // is dropped by pump(). Bounded CAS (no lock, no allocation).
    uint64_t cur = active_gen_.load(std::memory_order_relaxed);
    while (gen > cur) {
        if (active_gen_.compare_exchange_weak(cur, gen, std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
            break;
        }
    }
    // Wake the engine thread so the marshaled micro-rewind is applied promptly.
    ring_notify();
}

EngineStatus EngineControlBridge::rewind_kv(AudioStreamHandle stream,
                                            uint32_t keep_prompt_tokens,
                                            uint64_t gen) noexcept {
    if (superseded(gen)) return EngineStatus::Success;  // a newer epoch already owns the state
    Command cmd;
    cmd.type = CommandType::Rewind;
    cmd.gen = gen;
    cmd.stream = stream;
    cmd.keep_prompt_tokens = keep_prompt_tokens;
    return enqueue(cmd) ? EngineStatus::Success : EngineStatus::StateMismatch;
}

EngineStatus EngineControlBridge::warm_prefill(AudioStreamHandle stream, uint64_t gen) noexcept {
    if (superseded(gen)) return EngineStatus::Success;
    Command cmd;
    cmd.type = CommandType::WarmPrefill;
    cmd.gen = gen;
    cmd.stream = stream;
    return enqueue(cmd) ? EngineStatus::Success : EngineStatus::StateMismatch;
}

EngineStatus EngineControlBridge::commit_and_decode(AudioStreamHandle stream, TokenSink sink,
                                                    uint64_t gen) noexcept {
    if (superseded(gen)) return EngineStatus::Success;
    // Latch single-in-flight before publishing so generation_in_flight() is true
    // the instant the command is visible; the decode loop clears it on exit.
    decoding_.store(true, std::memory_order_release);
    Command cmd;
    cmd.type = CommandType::CommitDecode;
    cmd.gen = gen;
    cmd.stream = stream;
    cmd.sink = sink;
    if (!enqueue(cmd)) {
        decoding_.store(false, std::memory_order_release);
        return EngineStatus::StateMismatch;
    }
    return EngineStatus::Success;
}

EngineStatus EngineControlBridge::submit_multimodal(AudioRingBuffer& ring,
                                                    const char* prompt_utf8, TokenSink sink) {
    // One-shot (non-speculative) multimodal generate. Tagged with the CURRENT
    // epoch so a concurrent barge-in still supersedes it. `prompt_utf8` is
    // caller-owned and must outlive the marshaled job (the async contract).
    decoding_.store(true, std::memory_order_release);
    Command cmd;
    cmd.type = CommandType::CommitDecode;
    cmd.gen = active_gen_.load(std::memory_order_acquire);
    cmd.ring = &ring;
    cmd.prompt = prompt_utf8;
    cmd.sink = sink;
    if (!enqueue(cmd)) {
        decoding_.store(false, std::memory_order_release);
        return EngineStatus::StateMismatch;
    }
    return EngineStatus::Success;
}

// -----------------------------------------------------------------------------
// CONSUMER edge (the single engine-owning thread).
// -----------------------------------------------------------------------------
size_t EngineControlBridge::pump() noexcept {
    size_t executed = 0;
    Command cmd;
    while (try_pop(cmd)) {
        // Drop a command a barge-in superseded between enqueue and now.
        if (superseded(cmd.gen)) continue;
        // Runtime tier: no exception escapes the pump. A do_* override that raises
        // is caught and folded to a dropped command (its own sink/status already
        // carries any error).
        try {
            switch (cmd.type) {
                case CommandType::Rewind:       (void)do_rewind(cmd); break;
                case CommandType::WarmPrefill:  (void)do_warm_prefill(cmd); break;
                case CommandType::CommitDecode: (void)do_commit_decode(cmd); break;
            }
        } catch (const std::exception&) {
            // Swallow: the session self-heals on the next reconcile.
        } catch (...) {
        }
        ++executed;
    }
    return executed;
}

size_t EngineControlBridge::wait_and_pump() noexcept {
    for (;;) {
        const size_t executed = pump();
        if (executed > 0) return executed;
        if (stopping_.load(std::memory_order_acquire)) return 0;

        // Lost-wakeup guard: snapshot the doorbell, then re-check the ring and the
        // stop latch BEFORE parking. A producer that enqueued (bumping the
        // doorbell) between pump() and here changes `observed`, so wait() returns
        // immediately instead of sleeping through the event.
        const uint64_t observed = doorbell_.load(std::memory_order_acquire);
        if (!ring_empty() || stopping_.load(std::memory_order_acquire)) continue;
        doorbell_.wait(observed, std::memory_order_acquire);  // 0% CPU until notified
    }
}

void EngineControlBridge::stop() noexcept {
    stopping_.store(true, std::memory_order_release);
    ring_notify();  // wake any parked wait_and_pump()
}

// -----------------------------------------------------------------------------
// Execute hooks (engine thread). do_rewind is wired to the engine; the multimodal
// stages are the engine-assembly seam (see the header preamble).
// -----------------------------------------------------------------------------
EngineStatus EngineControlBridge::do_rewind(const Command& cmd) {
    if (engine_ == nullptr) return EngineStatus::InvalidConfig;
    try {
        // Micro-rewind: drop KV/SSM state past the verified prefix, keeping at
        // least keep_prompt_tokens — but NEVER below the frozen system-prompt
        // prefix (effective_keep_tokens clamps up to that floor), so the cached
        // system prefix survives every barge-in. BlackwellEngine::rewind is
        // capability-gated (throws under Continuous / hybrid-without-branching) —
        // runtime tier folds that to a status so the pump never unwinds.
        engine_->rewind(cfg_.seq_id,
                        static_cast<int>(effective_keep_tokens(cmd.keep_prompt_tokens)));
        return EngineStatus::Success;
    } catch (const std::exception&) {
        return EngineStatus::StateMismatch;
    }
}

EngineStatus EngineControlBridge::do_warm_prefill(const Command& /*cmd*/) {
    // The speculative warm-prefill runs the Whisper frontend -> Ultravox projector
    // -> inject_audio_embeddings -> Llama prefill over the buffered audio. That
    // stage links blackwell_audio, which the leaf bridge lib does not; the engine
    // assembly overrides this (binding a projector via set_projector). Base:
    // capability-gate with a clear status rather than silently no-op.
    return EngineStatus::InvalidConfig;
}

EngineStatus EngineControlBridge::do_commit_decode(const Command& /*cmd*/) {
    // Commit soft-tokens + decode: same multimodal stage as warm-prefill followed
    // by the barge-in decode loop (run_decode_loop). Supplied by the engine
    // assembly override; base clears the in-flight guard so the stream stays usable.
    clear_in_flight();
    return EngineStatus::InvalidConfig;
}

EngineStatus EngineControlBridge::decode_one(int token_id, int pos, int* out_token) noexcept {
    if (engine_ == nullptr) return EngineStatus::InvalidConfig;
    int next = 0;
    // Greedy (argmax) continuation: temperature 0, top_p 1 on the bridge's sequence.
    const EngineStatus st =
        engine_->forward_status(token_id, pos, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                cfg_.seq_id, &next);
    if (out_token != nullptr) *out_token = next;
    return st;
}

EngineStatus EngineControlBridge::run_decode_loop(uint64_t gen, int first_token_id,
                                                  int start_pos, int max_new_tokens,
                                                  const TokenSink& sink) {
    EngineStatus st = EngineStatus::Success;
    bool aborted = false;
    int token = first_token_id;
    int pos = start_pos;
    int emitted = 0;

    for (; emitted < max_new_tokens; ++emitted) {
        // Wait-free barge-in: a newer epoch (on_speech_start) aborts within one
        // token. This is the zero-cost atomic check the doctrine mandates.
        if (cancelled(gen)) {
            aborted = true;
            break;
        }
        int next = 0;
        st = decode_one(token, pos, &next);
        if (st != EngineStatus::Success) break;
        // Base emits ids-only (no detokenizer at this tier); the engine assembly
        // override detokenizes `next` before emitting.
        sink.emit("", emitted, /*is_final=*/0, BRIDGE_OK);
        token = next;
        ++pos;
    }

    clear_in_flight();
    // A barge-in is a clean supersede, not a fault: report OK on the final emit.
    const BridgeStatus final_status = aborted ? BRIDGE_OK : to_bridge_status(st);
    sink.emit("", emitted, /*is_final=*/1, final_status);
    return aborted ? EngineStatus::Success : st;
}

}  // namespace blackwell::bridge
