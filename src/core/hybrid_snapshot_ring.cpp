#include "hybrid_snapshot_ring.h"

#include <exception>

namespace blackwell {

namespace {
// A fork()/release_sequence() throw is exception-tier (CoW branch admin). For
// the ring's status-tier surface we convert it: a restore/snapshot fork writes
// only pre-allocated per-slot slices and CoW-shares pages by refcount (no device
// allocation), so a throw here is a broken CUDA context, reported conservatively
// as CudaRuntimeError. (A genuine OOM can still arise when a growing snapshot
// fork must CoW a boundary page; it surfaces the same way -- the caller treats
// any non-Success identically: shorter horizon, or cold restart.)
EngineStatus status_of(const std::exception&) noexcept {
    return EngineStatus::CudaRuntimeError;
}
}  // namespace

bool HybridSnapshotRing::supported(const BlackwellEngine& engine) noexcept {
    const ModelCapabilities caps = engine.get_capabilities();
    return caps.supports_cow_branching && caps.requires_ssm_subsystem &&
           engine.branch_capacity() >= 2;
}

HybridSnapshotRing::HybridSnapshotRing(BlackwellEngine& engine, int checkpoint_stride)
    : m_engine(engine), m_stride(checkpoint_stride > 0 ? checkpoint_stride : 1) {
    // Slots [1, B) are the recyclable ring; slot 0 is the pinned active head.
    // Push highest-first so the stack hands out low ids first (cosmetic).
    for (int s = m_engine.branch_capacity() - 1; s >= 1; --s) m_free_slots.push_back(s);
}

HybridSnapshotRing::~HybridSnapshotRing() { clear(); }

void HybridSnapshotRing::clear() {
    for (const Snapshot& s : m_ring) {
        try {
            m_engine.release_sequence(s.slot);
            m_free_slots.push_back(s.slot);  // reclaim only on a clean release
        } catch (...) {
            // Leave the slot out of the free list: its paged sequence may still
            // exist, and re-forking a live id would throw. Losing one slot for
            // the session is strictly better than a corrupt free list.
        }
    }
    m_ring.clear();
}

bool HybridSnapshotRing::evict_oldest() {
    if (m_ring.empty()) return false;
    const Snapshot victim = m_ring.front();
    m_ring.pop_front();
    try {
        m_engine.release_sequence(victim.slot);
        m_free_slots.push_back(victim.slot);
    } catch (...) {
        // See clear(): drop the entry but do not reclaim a possibly-live id.
    }
    return true;
}

EngineStatus HybridSnapshotRing::take_snapshot(int len) {
    if (m_free_slots.empty()) {
        // Ring full: make room by dropping the oldest reachable anchor. With
        // branch_capacity >= 2 (supported()) there is always a ring slot to
        // reclaim, so this succeeds; the bounded horizon is the whole point.
        if (!evict_oldest() || m_free_slots.empty()) return EngineStatus::OutOfVram;
    }
    const int slot = m_free_slots.back();
    try {
        m_engine.fork(/*parent=*/0, /*child=*/slot);
    } catch (const std::exception& e) {
        return status_of(e);  // slot stays free; nothing was pushed to the ring
    }
    m_free_slots.pop_back();
    m_ring.push_back({slot, len});
    return EngineStatus::Success;
}

EngineStatus HybridSnapshotRing::begin(int prompt_len) {
    clear();
    m_prompt_len = prompt_len;
    return take_snapshot(prompt_len);  // the anchor at the prompt boundary
}

EngineStatus HybridSnapshotRing::on_token(int active_len) {
    const int last = m_ring.empty() ? m_prompt_len : m_ring.back().len;
    if (active_len - last >= m_stride) return take_snapshot(active_len);
    return EngineStatus::Success;
}

int HybridSnapshotRing::horizon_floor() const noexcept {
    return m_ring.empty() ? -1 : m_ring.front().len;
}

int HybridSnapshotRing::virtual_rewind(int target_len, const std::vector<int>& tokens,
                                       float temperature, float top_p, int* out_next,
                                       EngineStatus* st) {
    if (st) *st = EngineStatus::Success;
    if (target_len <= m_prompt_len) return -1;                // regenerating all == cold restart
    if (static_cast<int>(tokens.size()) < target_len) {       // mirror too short: caller bug
        if (st) *st = EngineStatus::InvalidArgument;
        return -1;
    }

    // Newest snapshot STRICTLY before target_len (deque is ascending by len), so
    // the replay below always re-runs the final retained token and regenerates
    // valid logits at target_len (see the header: SSM has no stored logits).
    int cp_slot = -1, cp_len = -1;
    for (auto it = m_ring.rbegin(); it != m_ring.rend(); ++it) {
        if (it->len < target_len) { cp_slot = it->slot; cp_len = it->len; break; }
    }
    if (cp_slot < 0) return -1;  // below horizon: *st stays Success -> cold restart

    // Restore the snapshot INTO the pinned active head (slot 0): release slot 0,
    // then fork the snapshot back into id 0 (fork rejects an id that still
    // exists). The snapshot slot itself is left intact (I2). Allocation-free, so
    // a throw here means a dead context -> report and let the caller cold-restart.
    try {
        m_engine.release_sequence(0);
        m_engine.fork(/*parent=*/cp_slot, /*child=*/0);
    } catch (const std::exception& e) {
        if (st) *st = status_of(e);
        return -1;
    }

    // Replay the retained tail [cp_len, target_len) so the head lands EXACTLY at
    // target_len (I3). Earlier steps advance deterministically (their sampled
    // token is a retained token we discard); the FINAL step (position
    // target_len-1) samples with the caller's temperature/top_p, and that sample
    // -- drawn from the freshly-produced logits at target_len -- is the first
    // token to generate on resume.
    int next = -1;
    for (int p = cp_len; p < target_len; ++p) {
        const bool last = (p == target_len - 1);
        const EngineStatus fs = m_engine.forward_status(
            tokens[p], p, last ? temperature : 0.0f, last ? top_p : 1.0f,
            /*seq_id=*/0, &next);
        if (fs != EngineStatus::Success) {
            if (st) *st = fs;
            return -1;  // slot 0 is partially advanced; caller cold-restarts
        }
    }
    if (out_next) *out_next = next;

    // Evict every snapshot strictly newer than target_len -- the abandoned path.
    // The retained snapshots (len <= cp_len <= target_len) stay valid anchors
    // because the replayed tail reproduced their prefix content exactly.
    while (!m_ring.empty() && m_ring.back().len > target_len) {
        const Snapshot dead = m_ring.back();
        m_ring.pop_back();
        try {
            m_engine.release_sequence(dead.slot);
            m_free_slots.push_back(dead.slot);
        } catch (...) {
            // Drop the entry; do not reclaim a possibly-live id (see clear()).
        }
    }
    return target_len;
}

} // namespace blackwell
