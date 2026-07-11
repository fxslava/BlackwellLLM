#pragma once
#include <cstdint>
#include <deque>
#include <vector>

#include "blackwell/engine.h"
#include "blackwell/engine_status.h"

// ============================================================================
// HybridSnapshotRing — snapshot-driven VIRTUAL rewind for hybrid SSM models.
// ============================================================================
// A dense/paged model rolls a sequence back physically: BlackwellEngine::rewind
// (or the coordinator's truncate_sequence micro-rewind) drops trailing KV pages
// and the position-addressed cache self-heals. A HYBRID linear-attention model
// cannot: its recurrent SSM state advances with every forward() and keeps no
// per-position history to invert, so rewind() throws for it (engine.cpp). What a
// hybrid model CAN do is fork() -- a physical D2D snapshot of all three state
// stores (SSM recurrent/conv, gated full-attention KV, paged CoW pool). This
// class turns that forward-only primitive into a bounded backtrack by keeping a
// RING of checkpoints and, on a rewind request, restoring the nearest one and
// replaying the retained token tail on top of it.
//
// Slot model (all ids share the engine's branch_capacity budget [0, B)):
//   slot 0            -- the ACTIVE decode head, pinned. The caller prefills the
//                        prompt here and forward()s here; it is the engine's
//                        primordial sequence and stays the sole live head across
//                        sessions, so the caller's cold-prefill path is unchanged.
//   slots [1, B)      -- the checkpoint ring: immutable fork()s of slot 0,
//                        captured every `stride` tokens, newest-wins eviction.
// Backtrack horizon ~= (B-1) * stride tokens. B is RuntimeConfig::
// paged_branch_factor; a model with B < 2 (or no branching) is unsupported()
// and the caller keeps the cold-restart path (reset_state + reprefill).
//
// Three invariants make the snapshots correct:
//   I1  Snapshots are FORWARD-captured: on_token() forks slot 0 the instant it
//       reaches a stride boundary, before it decodes past it -- fork() copies the
//       whole per-seq stride, so a late snapshot would capture the wrong-branch
//       tail. Capture-on-arrival is what makes a snapshot "the state at length L".
//   I2  Snapshots are consumed by COPY, never by handoff: virtual_rewind() forks
//       the chosen checkpoint back INTO slot 0 and leaves the checkpoint intact,
//       so the same anchor can be rewound to again (a critic may reject several
//       candidate continuations from one point).
//   I3  The host token mirror is the source of truth: virtual_rewind() replays
//       the retained tail [snapshot_len, target_len) from the caller's token
//       vector so the active head lands EXACTLY at target_len with state that
//       matches that prefix (the nearest snapshot merely bounds how far back the
//       physical restore reaches).
//
// Restoring into the pinned slot 0 means release()ing it and fork()ing the
// snapshot back in (fork rejects an id that still exists). That is safe: the
// restore fork allocates NO device memory -- it CoW-shares the snapshot's paged
// pages by refcount and D2D-copies the recurrent/full-attn state into slot 0's
// PRE-ALLOCATED per-slot slices -- so it can only fault on an already-dead CUDA
// context, which is unrecoverable regardless. On such a fault virtual_rewind()
// reports the status and the caller cold-restarts.
//
// Threading: single engine-owning thread, like everything below the engine's
// control plane (the overlay's LiveTranslationTracker worker). Not thread-safe.
// The destructor releases every ring slot, so it too must run on that thread.
namespace blackwell {

class HybridSnapshotRing {
public:
    // `engine` must outlive the ring and must be a branching-capable hybrid
    // model (assert supported() first). `checkpoint_stride` is K: one snapshot
    // every K decoded tokens. Slot 0 is the fixed active head; the ring owns
    // slots [1, engine.branch_capacity()).
    HybridSnapshotRing(BlackwellEngine& engine, int checkpoint_stride = 8);

    // Releases every live ring slot (engine-owning thread only). Slot 0 (the
    // active head) is the engine's primordial sequence and is NOT released here.
    ~HybridSnapshotRing();

    HybridSnapshotRing(const HybridSnapshotRing&) = delete;
    HybridSnapshotRing& operator=(const HybridSnapshotRing&) = delete;

    // Virtual rewind needs at least one snapshot slot on top of the active head,
    // and the recurrent state has no positional undo (so a dense model would use
    // physical rewind() instead, not this). True iff the model forks AND has an
    // SSM subsystem AND branch_capacity() >= 2.
    static bool supported(const BlackwellEngine& engine) noexcept;

    // Open a fresh decode session: slot 0 already holds `prompt_len` prefilled
    // tokens (the anchor). Clears any prior ring and captures the anchor snapshot
    // at prompt_len, so a later rewind can reach back to the prompt boundary
    // until eviction pushes the horizon forward. Returns the snapshot status
    // (a failed anchor fork leaves the ring empty -- every rewind then reports
    // below-horizon and the caller cold-restarts).
    EngineStatus begin(int prompt_len);

    // Record that the active head (slot 0) now holds exactly `active_len` tokens
    // (call once after each decoded token). Every `stride` tokens past the last
    // snapshot this forks slot 0 into a fresh ring slot, evicting the OLDEST
    // snapshot first when the ring is full (bounded horizon). A snapshot failure
    // is non-fatal -- it returns the fault status but leaves the session usable
    // with a shorter reachable horizon; decode should continue.
    EngineStatus on_token(int active_len);

    // Roll the active head back so exactly `target_len` tokens remain, then leave
    // it ready to decode: on success *out_next is the FIRST token to generate at
    // position target_len (already sampled with the caller's temperature/top_p).
    //   1. pick the newest snapshot with len STRICTLY < target_len; if none
    //      exists (target_len at/below the ring's oldest snapshot, or <=
    //      prompt_len), return -1 -- the caller cold-restarts.
    //   2. restore slot 0 from it by copy (I2) and replay the retained tail
    //      tokens[snapshot_len, target_len) so the head lands exactly at
    //      target_len (I3). The strict "<" guarantees the final retained token
    //      (target_len-1) is re-run, so its logits at target_len are freshly
    //      valid -- the recurrent SSM state has no stored per-position logits and
    //      cannot be re-run idempotently, so the replay itself must produce them.
    //      That last step samples with (temperature, top_p) into *out_next;
    //      earlier steps advance deterministically (their sampled token is a
    //      retained token we already know, so it is discarded).
    //   3. evict every snapshot newer than target_len (the abandoned path).
    // `tokens` is the caller's full sequence mirror (prompt + generated), indexed
    // by absolute position; its size must be >= target_len. On success returns
    // target_len; on an engine fault during restore/replay returns -1 with *st
    // carrying the status; on below-horizon returns -1 with *st = Success (a
    // clean "use cold restart", not an error).
    int virtual_rewind(int target_len, const std::vector<int>& tokens,
                       float temperature, float top_p, int* out_next,
                       EngineStatus* st = nullptr);

    // Oldest reachable length: the len of the ring's oldest snapshot, or -1 when
    // the ring is empty. A virtual_rewind(target) with target < horizon_floor()
    // cannot be served without a cold restart.
    int horizon_floor() const noexcept;

    // Drop every ring slot (releasing their paged pages) without touching the
    // active head. Call when starting an unrelated sequence on slot 0 after a
    // reset_state()/reprefill, so stale snapshots of the old sequence do not
    // linger holding pages or masquerade as valid anchors.
    void clear();

    int stride() const noexcept { return m_stride; }

private:
    struct Snapshot {
        int slot;  // ring slot id in [1, B) holding the fork
        int len;   // token count the active head had when this was captured
    };

    // Capture a snapshot of slot 0 at length `len` into a fresh ring slot,
    // evicting the oldest snapshot first if none is free. Status tier: a fork
    // fault returns its status and leaves the free list / ring consistent.
    EngineStatus take_snapshot(int len);

    // Release the oldest snapshot, returning its slot to the free list. No-op
    // (false) when the ring is already empty.
    bool evict_oldest();

    BlackwellEngine&      m_engine;
    int                   m_stride;
    int                   m_prompt_len   = 0;     // anchor boundary (session start)
    std::vector<int>      m_free_slots;           // available ids in [1, B), stack
    std::deque<Snapshot>  m_ring;                 // oldest at front, newest at back
};

} // namespace blackwell
