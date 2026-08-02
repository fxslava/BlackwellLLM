#pragma once
// =============================================================================
// bridge/intent_commit.hpp — THE COMMIT GATE between the local arbiter and the
// paid Cloud API. Header-only, CUDA-free, no engine dependency.
//
// WHY THIS EXISTS
//   The cloud request is billed at dispatch: prefill/input tokens are charged
//   the moment the request is accepted, and nothing downstream can refund them.
//   So the decision "is this thought finished?" MUST be made before the socket
//   is touched, not after. The local LLM (Llama via Ultravox) is the sole
//   arbiter; this queue is the one-way valve its verdict flows through.
//
// THE COMMIT RULE (enforced in exactly ONE place: offer(), below)
//   A local generation is dispatchable IFF it terminated on EOS — the model
//   emitted <|eot_id|> / <|end_of_text|> of its own accord. Every other exit is
//   a fragment and is dropped:
//
//     TerminationReason::Eos      -> COMMIT. The only accepting branch.
//     TerminationReason::BargeIn  -> drop. The user interrupted the thought.
//     TerminationReason::TokenCap -> drop. A budget truncated it.
//     TerminationReason::Fault    -> drop. The engine faulted mid-generation.
//     TerminationReason::None     -> drop. Loop never ran / still running.
//
//   The gate is structural, not advisory: offer() is the only way into the
//   queue and it takes the reason as its FIRST argument, so a caller cannot
//   enqueue without stating why the loop stopped. The failure mode of a caller
//   that forgets to call offer() at all is "nothing is dispatched" — silence,
//   never a spurious paid request. That asymmetry is deliberate.
//
// OBSERVABILITY IS NOT OPTIONAL HERE
//   "Missing EOS = no dispatch" turns every local truncation into a silent
//   black hole: the user speaks, nothing reaches the cloud, and no error is
//   raised anywhere. dropped_token_cap() in particular means max_new_tokens is
//   mis-sized and the feature is quietly dead. Surface these counters in the
//   UI/telemetry — do not let them only exist in a debugger.
//
// CONCURRENCY
//   Single-producer (the ONE engine-owning thread, per CLAUDE.md) /
//   single-consumer (the network dispatcher thread). Wait-free on the producer
//   edge; the consumer may block on a C++20 atomic doorbell at 0% CPU, the same
//   pattern EngineControlBridge::wait_and_pump() uses.
//
//   Producer-side allocation: offer() MOVES the payload string in. The decode
//   loop already built that string token by token, so the move is a pointer
//   swap — no allocation on the engine thread at commit time.
// =============================================================================
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace blackwell::bridge {

// Why a local generation loop stopped. Produced by
// EngineControlBridge::run_decode_loop; consumed by IntentCommitQueue::offer.
enum class TerminationReason : uint32_t {
    None = 0,   // loop never ran, or is still running
    Eos,        // model emitted its end-of-sequence token -- THE COMMIT CASE
    BargeIn,    // the user interrupted: a newer epoch superseded this generation
    TokenCap,   // a budget cut the thought short: per-turn cap OR context exhausted
    Fault,      // the decode step returned a non-Success EngineStatus
};

[[nodiscard]] inline const char* to_string(TerminationReason r) noexcept {
    switch (r) {
        case TerminationReason::None:     return "None";
        case TerminationReason::Eos:      return "Eos";
        case TerminationReason::BargeIn:  return "BargeIn";
        case TerminationReason::TokenCap: return "TokenCap";
        case TerminationReason::Fault:    return "Fault";
    }
    return "Unknown";
}

// The single predicate behind the commit rule. Free function so it can be
// asserted against directly in tests without constructing a queue.
[[nodiscard]] inline constexpr bool is_dispatchable(TerminationReason r) noexcept {
    return r == TerminationReason::Eos;
}

// One finalized, immutable intent. Once this lands in the queue the payload is
// frozen -- nothing downstream edits it, which is what makes the cloud request
// reproducible and safely retryable.
struct IntentRecord {
    uint64_t    generation = 0;   // the local epoch that produced it
    uint64_t    sequence = 0;     // monotone commit counter (assigned by the queue)
    uint32_t    token_count = 0;  // local tokens generated, EOS excluded
    std::string payload;          // finalized prompt / structured JSON intent
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)  // padded for cache-line alignment (intentional)
#endif

class IntentCommitQueue {
public:
    // capacity is rounded UP to a power of two (mask instead of modulo).
    // Deliberately small: an intent is one finished utterance, and a deep
    // backlog of stale intents is worse than refusing new ones. INIT tier --
    // the single allocation happens here and never again.
    explicit IntentCommitQueue(size_t capacity = 8)
        : capacity_(round_up_pow2(capacity == 0 ? 1 : capacity)),
          mask_(capacity_ - 1),
          slots_(std::make_unique<IntentRecord[]>(capacity_)) {}

    IntentCommitQueue(const IntentCommitQueue&) = delete;
    IntentCommitQueue& operator=(const IntentCommitQueue&) = delete;

    // ---- PRODUCER edge: the engine thread. THE GATE. ------------------------
    // The ONLY way an intent reaches the network layer. Returns true iff the
    // intent was committed, i.e. iff `reason == Eos` AND there was room.
    //
    // `payload` is moved from on the accepting branch only; on a rejection the
    // caller's string is left untouched (so it can still be logged/inspected).
    // Wait-free and noexcept: no allocation, no lock, no CAS loop.
    [[nodiscard]] bool offer(TerminationReason reason, uint64_t generation,
                             std::string&& payload, uint32_t token_count) noexcept {
        if (!is_dispatchable(reason)) {
            // Fragment. Count it by cause so a systematic failure (a too-small
            // max_new_tokens starving the pathway) is visible rather than mute.
            switch (reason) {
                case TerminationReason::BargeIn: ++dropped_barge_in_; break;
                case TerminationReason::TokenCap:  ++dropped_token_cap_; break;
                case TerminationReason::Fault:     ++dropped_fault_; break;
                default:                           ++dropped_none_; break;
            }
            return false;
        }

        const uint64_t tail = tail_.load(std::memory_order_relaxed);
        const uint64_t head = head_.load(std::memory_order_acquire);
        if (tail - head >= capacity_) {
            // The dispatcher is behind (a cloud request is still streaming).
            // Refuse the NEW intent rather than evicting an older committed one:
            // every record in here already passed the gate and is owed a call.
            ++dropped_queue_full_;
            return false;
        }

        IntentRecord& slot = slots_[static_cast<size_t>(tail & mask_)];
        slot.generation = generation;
        slot.sequence = ++commit_seq_;
        slot.token_count = token_count;
        slot.payload = std::move(payload);  // pointer swap; no allocation here

        tail_.store(tail + 1, std::memory_order_release);
        ++committed_;
        doorbell_.fetch_add(1, std::memory_order_release);
        doorbell_.notify_one();
        return true;
    }

    // ---- CONSUMER edge: the network dispatcher thread ----------------------

    // Non-blocking. Moves one record out; false if the queue is empty.
    [[nodiscard]] bool try_pop(IntentRecord& out) noexcept {
        const uint64_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) return false;

        IntentRecord& slot = slots_[static_cast<size_t>(head & mask_)];
        out.generation = slot.generation;
        out.sequence = slot.sequence;
        out.token_count = slot.token_count;
        out.payload = std::move(slot.payload);
        slot.payload.clear();  // release the moved-from buffer back to the slot

        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Block at 0% CPU until an intent is committed, then pop it. Returns false
    // only on stop() with an empty queue. Lost-wakeup-safe: the doorbell is
    // re-read before the wait, so a commit that lands in the gap is not missed.
    [[nodiscard]] bool wait_pop(IntentRecord& out) noexcept {
        for (;;) {
            if (try_pop(out)) return true;
            if (stopping_.load(std::memory_order_acquire)) return false;
            const uint64_t ticket = doorbell_.load(std::memory_order_acquire);
            // Re-check AFTER sampling the doorbell: a commit between the pop
            // attempt and here bumps the counter, so wait() returns at once.
            if (!empty() || stopping_.load(std::memory_order_acquire)) continue;
            doorbell_.wait(ticket, std::memory_order_acquire);
        }
    }

    // Wake a blocked wait_pop() for shutdown. Idempotent; any thread.
    void stop() noexcept {
        stopping_.store(true, std::memory_order_release);
        doorbell_.fetch_add(1, std::memory_order_release);
        doorbell_.notify_all();
    }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }
    [[nodiscard]] size_t size() const noexcept {
        return static_cast<size_t>(tail_.load(std::memory_order_acquire) -
                                   head_.load(std::memory_order_acquire));
    }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

    // ---- Gate telemetry. Read from any thread; see the header preamble. -----
    [[nodiscard]] uint64_t committed() const noexcept { return committed_; }
    [[nodiscard]] uint64_t dropped_barge_in() const noexcept { return dropped_barge_in_; }
    [[nodiscard]] uint64_t dropped_token_cap() const noexcept { return dropped_token_cap_; }
    [[nodiscard]] uint64_t dropped_fault() const noexcept { return dropped_fault_; }
    [[nodiscard]] uint64_t dropped_none() const noexcept { return dropped_none_; }
    [[nodiscard]] uint64_t dropped_queue_full() const noexcept { return dropped_queue_full_; }
    [[nodiscard]] uint64_t dropped_total() const noexcept {
        return dropped_barge_in_ + dropped_token_cap_ + dropped_fault_ + dropped_none_ +
               dropped_queue_full_;
    }

private:
    static size_t round_up_pow2(size_t v) noexcept {
        size_t p = 1;
        while (p < v) p <<= 1;
        return p;
    }

    const size_t capacity_;
    const size_t mask_;
    std::unique_ptr<IntentRecord[]> slots_;

    // SPSC cursors on separate cache lines (producer owns tail_, consumer head_).
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) std::atomic<uint64_t> doorbell_{0};

    // Producer-thread-private counters -- plain, not atomic, on purpose: only
    // the engine thread writes them and a torn read of a telemetry counter is
    // not worth a lock on the commit path.
    uint64_t commit_seq_ = 0;
    uint64_t committed_ = 0;
    uint64_t dropped_barge_in_ = 0;
    uint64_t dropped_token_cap_ = 0;
    uint64_t dropped_fault_ = 0;
    uint64_t dropped_none_ = 0;
    uint64_t dropped_queue_full_ = 0;

    std::atomic<bool> stopping_{false};
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace blackwell::bridge
