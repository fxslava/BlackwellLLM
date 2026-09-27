#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include "safetensors.h"
#include "blackwell/config.h"

struct QuantizedTensorPtrs {
    const void* qweight;
    const void* scales;
    const void* qzeros;
};

// E8W5 needs four tensors rather than AWQ's three, and one of them (the codebook) is
// per-tensor metadata rather than a weight plane, so it gets its own struct instead of
// overloading QuantizedTensorPtrs. See docs/E8W5_FORMAT_SPEC.md section 5.
struct E8W5TensorPtrs {
    const void* plane_lo;   // uint32[out_features][in_features/8]
    const void* plane_hi;   // uint8 [out_features][in_features/8]
    const void* scales;     // half  [out_features][in_features/128]
    const void* codebook;   // half  [64], fitted per tensor
};

// Page-locked (pinned) host memory pool. Every host<->device tensor transfer in
// the offloading path must source/sink cudaHostAlloc'd memory so cudaMemcpyAsync
// can DMA over PCIe without an internal pageable staging copy; pageable malloc
// is forbidden for tensor traffic.
class PinnedHostPool {
public:
    PinnedHostPool() = default;
    explicit PinnedHostPool(size_t bytes) { reserve(bytes); }
    ~PinnedHostPool();

    PinnedHostPool(const PinnedHostPool&) = delete;
    PinnedHostPool& operator=(const PinnedHostPool&) = delete;

    // One-shot arena reservation; throws on a second call or cudaHostAlloc failure.
    void reserve(size_t bytes);

    // Bump allocation, 16-byte aligned; throws std::bad_alloc on exhaustion.
    void* allocate(size_t bytes);

    void*  base()     const { return m_base; }
    size_t capacity() const { return m_capacity; }
    size_t used()     const { return m_used; }

private:
    void*  m_base = nullptr;
    size_t m_capacity = 0;
    size_t m_used = 0;
};

// Static Memory Orchestrator for 12GB VRAM limit.
//
// Layers [0, num_gpu_layers) keep weights and KV cache resident in VRAM exactly
// as before. Layers [num_gpu_layers, num_layers) live in pinned host RAM and are
// streamed through two ping-pong device staging slots (slot = layer % 2) on a
// dedicated non-blocking transfer stream, overlapping PCIe traffic of layer N+1
// with the legacy-stream compute of layer N. Ordering between the two streams is
// expressed exclusively through cudaEvents; the math kernels are untouched.
class VRAMArena {
public:
    // Cumulative weight-load progress (bytes staged so far, total bytes). Called
    // once per tensor on the constructing thread; drive a "(45%)" HUD from it.
    // NOTE: with a batched loader (DirectStorage) the resident-tensor portion
    // reports ENQUEUE progress and the real I/O completes at the flush barrier,
    // so the bar may run ahead of the disk -- honest enough for a loading HUD.
    using LoadProgressFn = std::function<void(size_t bytes_done, size_t bytes_total)>;

    // activation_token_capacity: how many token rows the ping-pong activation
    // buffers must hold at once. 1 (default) is the batch=1 decode contract.
    // Batched prefill / true batch mode raise it (max chunk width) so d_X_accum /
    // d_X_norm can stage [num_tokens, hidden_dim]; the KV cache and weight arena
    // are unaffected (KV is position-addressed, not batch-scaled).
    VRAMArena(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader,
              const ModelConfig& config, size_t max_seq_len = 2048,
              size_t num_gpu_layers = static_cast<size_t>(-1),
              LoadProgressFn load_progress = {},
              size_t activation_token_capacity = 1);
    ~VRAMArena();

    VRAMArena(const VRAMArena&) = delete;
    VRAMArena& operator=(const VRAMArena&) = delete;

    // Retrieve device pointer for static weights
    void* get_weight_ptr(const std::string& name) const;

    // Safely retrieves a pointer if the weight exists, returns nullptr otherwise
    const void* get_weight_ptr_optional(const std::string& name) const;

    QuantizedTensorPtrs get_quantized_pointers(const std::string& base_name) const;

    // All four E8W5 tensors for one projection. Throws on any missing tensor when the
    // checkpoint declares e8w5: a half-populated projection would decode to garbage
    // silently, which is worse than failing the load.
    E8W5TensorPtrs get_e8w5_pointers(const std::string& base_name) const;

    // Ping-Pong activation buffers (reused across all 32 layers)
    float* get_activation_buffer_A() const { return d_activation_A; }
    float* get_activation_buffer_B() const { return d_activation_B; }

    // Global KV-Cache buffers (VRAM-resident layers only)
    float* get_k_cache() const { return d_k_cache; }
    float* get_v_cache() const { return d_v_cache; }

    // 🎯 НОВЫЕ МЕТОДЫ: Получение сохраненных размеров и емкостей
    size_t get_max_seq_len() const { return m_max_seq_len; }
    size_t get_k_cache_size() const { return m_total_cache_bytes; }
    size_t get_v_cache_size() const { return m_total_cache_bytes; }
    size_t get_activation_buffer_size() const { return m_activation_bytes; }
    // Token rows the ping-pong buffers can stage (>= 1); the batched-forward width.
    size_t get_activation_token_capacity() const { return m_activation_token_capacity; }

    // ------------------------------------------------------------------
    // Soft hibernation (inactivity lifecycle, stage 2)
    // ------------------------------------------------------------------
    // hibernate() evacuates the resident weights arena — the multi-GB VRAM
    // block — into a pinned host stash (D2H DMA) and frees the device
    // allocation; the arena object itself, the name->pointer registry, and
    // every dynamic pool stay alive. wakeup() re-allocates device memory,
    // DMAs the stash back over PCIe, and REBASES the registry to the new
    // device base, so callers keep using get_weight_ptr() with no re-load
    // from disk. The pinned stash is allocated once (first hibernate) and
    // kept for the process lifetime: repeat cycles never re-pay the pinning
    // cost, and a wakeup can never fail on host allocation.
    //
    // Contract: the engine must be idle — no kernel launched against the
    // arena may still be in flight (hibernate() synchronizes the device
    // before copying) and no forward() may run until wakeup() returns.
    // get_weight_ptr() throws while hibernated to make violations loud.
    // Both are idempotent. Not thread-safe; call from the single engine-
    // owning thread, like every other engine entry point.
    void hibernate();
    void wakeup();
    bool hibernated() const { return m_hibernated; }

    // ------------------------------------------------------------------
    // Layer offloading control plane (all no-ops for VRAM-resident layers)
    // ------------------------------------------------------------------
    size_t num_gpu_layers() const { return m_num_gpu_layers; }
    bool layer_is_offloaded(int layer) const {
        return static_cast<size_t>(layer) >= m_num_gpu_layers &&
               static_cast<size_t>(layer) <  m_config.num_layers;
    }

    // Enqueue layer weights + KV prefix [0, pos) onto the transfer stream.
    // Never blocks the host or the compute stream: call for layer N+1 while
    // layer N is executing. Out-of-range / resident layers are ignored.
    void prefetch_layer(int layer, int pos);

    // Make the compute (legacy) stream wait until the layer's weights are
    // staged. Cold path stages synchronously-in-order on the transfer stream
    // first; the host never blocks either way. Must precede any kernel that
    // reads the layer's weights.
    void ensure_layer_ready(int layer);

    // Per-layer KV cache routing: resident pool pointer or staging-slot pointer.
    // For offloaded layers the layer must currently own its staging slot.
    float* get_layer_k_cache(int layer) const;
    float* get_layer_v_cache(int layer) const;

    // Guarantee positions [0, pos) of the layer's KV cache are in the staging
    // slot before the attention kernels run (incremental: only missing tokens
    // are transferred). Compute-stream-ordered like ensure_layer_ready.
    void prepare_layer_kv(int layer, int pos);

    // Spill the KV column written at `pos` back to the pinned host mirror once
    // the attention kernels of this step have finished (event-ordered, async).
    void commit_layer_kv(int layer, int pos);

    // Roll every offload high-water mark back to `target_pos` tokens (KV rewind).
    // Clamps each offloaded layer's host-mirror fill count AND every device
    // staging slot's valid_upto down to target_pos, so positions >= target_pos are
    // treated as stale and re-spilled/re-staged by the next decode step. The
    // resident (VRAM) KV slabs are position-addressed and overwritten in place, so
    // they need no truncation. No-op when nothing is offloaded. Idempotent; does
    // not touch the device (no kernels, no sync) -- the single engine-owning thread
    // calls it between decode steps.
    void truncate_kv(int target_pos);

private:
    void allocate_weights_pool(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader);
    void allocate_dynamic_pool(size_t max_seq_len);
    void init_streams();
    // Frees every device pool and nulls the pointers (idempotent). Shared by the
    // destructor and the constructor's failure path.
    void release_pools();

    // Enqueue the full weight block of an offloaded layer into its slot (H2D,
    // transfer stream). No-op if the slot already holds the layer.
    void stage_layer_weights(int layer);
    // Enqueue missing KV prefix tokens [slot.valid_upto, min(pos, host_filled))
    // into the layer's KV slot. Handles slot eviction of the previous occupant.
    void stage_layer_kv(int layer, int pos);

    void* resolve_offloaded(const std::string& name) const;

    // Contiguous memory blocks
    void* d_weights_arena = nullptr;
    size_t total_weights_bytes = 0;

    // Soft-hibernation state: the pinned host stash the weights evacuate to,
    // and the device base they were registered against (wakeup() rebases the
    // registry by the delta between the old and the fresh allocation).
    void* m_h_hibernate_stash = nullptr;
    void* m_hibernated_old_base = nullptr;
    bool  m_hibernated = false;

    // Activation buffers (FP32)
    float* d_activation_A = nullptr;
    float* d_activation_B = nullptr;

    // KV Cache pool
    float* d_k_cache = nullptr;
    float* d_v_cache = nullptr;

    ModelConfig m_config;
    LoadProgressFn m_load_progress;  // optional weight-load progress observer
    size_t m_max_seq_len = 0;
    size_t m_total_cache_bytes = 0;
    size_t m_activation_bytes = 0;
    size_t m_activation_token_capacity = 1;
    size_t m_num_gpu_layers = 0;

    // Offset registry mapping tensor names to their absolute addresses in d_weights_arena
    std::unordered_map<std::string, void*> weight_pointers;

    // ------------------------------------------------------------------
    // Offloading state
    // ------------------------------------------------------------------
    static constexpr int kNumSlots = 2;

    struct OffloadedTensor {
        int    layer;
        size_t offset;   // byte offset inside the layer's packed block
    };

    struct WeightSlot {
        uint8_t*    d_base = nullptr;    // device staging buffer
        int         layer = -1;          // current occupant
        bool        compute_synced = false; // legacy stream already waits on ev_ready
        cudaEvent_t ev_ready  = nullptr; // recorded on transfer stream after H2D
        cudaEvent_t ev_retire = nullptr; // recorded on compute stream before eviction
    };

    struct KVSlot {
        float*      d_k = nullptr;       // [kv_heads, max_seq_len, head_dim]
        float*      d_v = nullptr;
        int         layer = -1;
        int         valid_upto = 0;      // tokens of `layer` present in the slot
        bool        compute_synced = false;
        cudaEvent_t ev_ready  = nullptr;
        cudaEvent_t ev_retire = nullptr;
        cudaEvent_t ev_commit = nullptr; // compute done -> safe to spill column
    };

    PinnedHostPool m_pinned;                       // weights + KV mirror arena
    std::unordered_map<std::string, OffloadedTensor> m_offloaded_tensors;
    std::vector<uint8_t*> m_layer_host_base;       // [num_layers], null if resident
    std::vector<size_t>   m_layer_bytes;           // packed block size per layer
    size_t m_max_layer_bytes = 0;

    std::vector<float*> m_h_k_mirror;              // per offloaded layer
    std::vector<float*> m_h_v_mirror;
    std::vector<int>    m_kv_host_filled;          // tokens spilled so far per offloaded layer
    size_t m_single_layer_kv_bytes = 0;

    WeightSlot m_wslots[kNumSlots];
    KVSlot     m_kvslots[kNumSlots];
    cudaStream_t m_transfer_stream = nullptr;
};
