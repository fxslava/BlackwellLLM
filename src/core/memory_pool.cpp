#include "memory_pool.h"
#include "blackwell/weight_loader.h"
#include "common.h"
#include <algorithm>
// Hybrid error doctrine (Roadmap #2): every CUDA check here is CUDA_CHECK_THROW
// -- no exit() from this library. Construction/allocation failures unwind via
// the VRAMArena ctor's try/catch -> release_pools() and map to E_OUTOFMEMORY at
// the DLL edge; hibernate()/wakeup() throws surface as HRESULTs through the
// EngineCom boundary that wraps them; and the rare offload-path throw (staging
// during decode) propagates up through step_*/run_token into forward_status's
// noexcept catch, which converts it to an EngineStatus. Teardown (release_pools,
// destructors) uses RAW cudaFree/cudaFreeHost -- nothing may throw there.
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <new>

// ============================================================================
// PinnedHostPool
// ============================================================================
PinnedHostPool::~PinnedHostPool() {
    if (m_base) cudaFreeHost(m_base);
}

void PinnedHostPool::reserve(size_t bytes) {
    if (m_base)
        throw std::logic_error("PinnedHostPool: arena already reserved");
    if (bytes == 0) return;

    // INIT tier: throws blackwell::cuda_error (cudaErrorMemoryAllocation ->
    // E_OUTOFMEMORY at the DLL edge). On failure cudaHostAlloc sets m_base to
    // null, so ~PinnedHostPool's free is a no-op.
    CUDA_CHECK_THROW(cudaHostAlloc(&m_base, bytes, cudaHostAllocDefault));
    m_capacity = bytes;
    m_used = 0;
}

void* PinnedHostPool::allocate(size_t bytes) {
    const size_t aligned_base = (m_used + 15) & ~size_t(15);
    if (aligned_base + bytes > m_capacity)
        throw std::bad_alloc();
    void* p = static_cast<uint8_t*>(m_base) + aligned_base;
    m_used = aligned_base + bytes;
    return p;
}

// ============================================================================
// Helpers
// ============================================================================
// "model.layers.N.<...>" -> N; anything else (embeddings, final norm, lm_head) -> -1
static int layer_index_of(const std::string& name) {
    constexpr char kPrefix[] = "model.layers.";
    constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (name.compare(0, kPrefixLen, kPrefix) != 0) return -1;

    size_t i = kPrefixLen;
    long value = 0;
    bool any = false;
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) {
        value = value * 10 + (name[i] - '0');
        ++i;
        any = true;
    }
    if (!any || i >= name.size() || name[i] != '.') return -1;
    return static_cast<int>(value);
}

// Minimal cached file reader for filling the pinned weight mirror at startup.
// (IWeightLoader targets VRAM destinations; host-resident layers read directly.)
namespace {
class FileCache {
public:
    ~FileCache() {
        for (auto& [path, f] : m_files)
            if (f) fclose(f);
    }
    void read(const std::string& path, size_t offset, size_t size, void* dst) {
        FILE* f = get(path);
        if (_fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0)
            throw std::runtime_error("[VRAMArena] seek to " + std::to_string(offset) +
                                     " failed in " + path);
        const size_t got = fread(dst, 1, size, f);
        if (got != size)
            throw std::runtime_error("[VRAMArena] short read (" + std::to_string(got) +
                                     " of " + std::to_string(size) + " bytes) from " + path);
    }
private:
    FILE* get(const std::string& path) {
        auto it = m_files.find(path);
        if (it != m_files.end()) return it->second;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("[VRAMArena] cannot open " + path);
        m_files[path] = f;
        return f;
    }
    std::unordered_map<std::string, FILE*> m_files;
};
} // namespace

// ============================================================================
// VRAMArena
// ============================================================================
VRAMArena::VRAMArena(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader,
                     const ModelConfig& config, size_t max_seq_len, size_t num_gpu_layers,
                     LoadProgressFn load_progress, size_t activation_token_capacity)
    : m_config(config), m_load_progress(std::move(load_progress)), m_max_seq_len(max_seq_len),
      m_activation_token_capacity(std::max<size_t>(activation_token_capacity, 1)),
      m_num_gpu_layers(std::min(num_gpu_layers, config.num_layers))
{
    std::cout << "[VRAM Arena] Initializing static memory pools...\n";
    if (m_num_gpu_layers < m_config.num_layers) {
        std::cout << "[VRAM Arena] Layer offloading active: layers 0.."
                  << (m_num_gpu_layers ? std::to_string(m_num_gpu_layers - 1) : std::string("<none>"))
                  << " in VRAM, layers " << m_num_gpu_layers << ".."
                  << (m_config.num_layers - 1) << " in pinned host RAM.\n";
    }
    // If weight loading throws (missing tensor, I/O failure) after the arena has
    // been cudaMalloc'd, the destructor will not run (the object was never fully
    // constructed), so the multi-GB allocation must be released here.
    try {
        init_streams();
        allocate_weights_pool(safetensors_path, metadata_loader);
        allocate_dynamic_pool(max_seq_len);
    } catch (...) {
        release_pools();
        throw;
    }
}

void VRAMArena::init_streams() {
    if (m_num_gpu_layers >= m_config.num_layers) return; // nothing offloaded

    // Non-blocking is essential: the math kernels run on the LEGACY default
    // stream, which implicitly serializes against blocking streams. A
    // non-blocking transfer stream is exempt, so cudaMemcpyAsync H2D/D2H here
    // genuinely overlaps with kernel execution; ordering is restored only at
    // the explicit cudaEvent handshakes below.
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&m_transfer_stream, cudaStreamNonBlocking));

    for (int s = 0; s < kNumSlots; ++s) {
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&m_wslots[s].ev_ready,  cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&m_wslots[s].ev_retire, cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&m_kvslots[s].ev_ready,  cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&m_kvslots[s].ev_retire, cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&m_kvslots[s].ev_commit, cudaEventDisableTiming));
    }
}

void VRAMArena::release_pools() {
    // Drain in-flight transfers before tearing down anything they touch.
    if (m_transfer_stream) {
        cudaStreamSynchronize(m_transfer_stream);
        cudaStreamDestroy(m_transfer_stream);
        m_transfer_stream = nullptr;
    }
    for (int s = 0; s < kNumSlots; ++s) {
        auto destroy = [](cudaEvent_t& ev) { if (ev) { cudaEventDestroy(ev); ev = nullptr; } };
        destroy(m_wslots[s].ev_ready);
        destroy(m_wslots[s].ev_retire);
        destroy(m_kvslots[s].ev_ready);
        destroy(m_kvslots[s].ev_retire);
        destroy(m_kvslots[s].ev_commit);
        if (m_wslots[s].d_base) { cudaFree(m_wslots[s].d_base); m_wslots[s].d_base = nullptr; }
        if (m_kvslots[s].d_k)   { cudaFree(m_kvslots[s].d_k);   m_kvslots[s].d_k   = nullptr; }
        if (m_kvslots[s].d_v)   { cudaFree(m_kvslots[s].d_v);   m_kvslots[s].d_v   = nullptr; }
    }

    if (d_weights_arena) { cudaFree(d_weights_arena); d_weights_arena = nullptr; }
    if (m_h_hibernate_stash) { cudaFreeHost(m_h_hibernate_stash); m_h_hibernate_stash = nullptr; }
    if (d_activation_A)  { cudaFree(d_activation_A);  d_activation_A  = nullptr; }
    if (d_activation_B)  { cudaFree(d_activation_B);  d_activation_B  = nullptr; }
    if (d_k_cache)       { cudaFree(d_k_cache);       d_k_cache       = nullptr; }
    if (d_v_cache)       { cudaFree(d_v_cache);       d_v_cache       = nullptr; }
    // Pinned host arena is released by m_pinned's own destructor.
}

VRAMArena::~VRAMArena() {
    release_pools();
    std::cout << "[VRAM Arena] All static device pools successfully released.\n";
}

void VRAMArena::allocate_weights_pool(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader) {
    // Unused: with a multi-shard checkpoint the authoritative per-tensor path is
    // entry.file_path (the loader resolves each shard itself); the index path is
    // kept in the signature for symmetry with the ctor.
    (void)safetensors_path;
    auto tensor_names = metadata_loader.list_tensors();

    // 1. Partition tensors: VRAM-resident (non-layer tensors + layers below the
    //    split) vs host-offloaded, with strict 16-byte alignment per tensor.
    size_t current_offset = 0;
    size_t qweight_count = 0;
    std::vector<std::pair<std::string, size_t>> aligned_offsets;
    aligned_offsets.reserve(tensor_names.size());

    m_layer_host_base.assign(m_config.num_layers, nullptr);
    m_layer_bytes.assign(m_config.num_layers, 0);
    std::vector<std::vector<std::string>> offloaded_by_layer(m_config.num_layers);

    for (const auto& name : tensor_names) {
        if (name.size() >= 8 && name.compare(name.size() - 8, 8, ".qweight") == 0)
            ++qweight_count;

        const int layer = layer_index_of(name);
        if (layer >= 0 && layer_is_offloaded(layer)) {
            offloaded_by_layer[layer].push_back(name);
            continue;
        }
        const auto& entry = metadata_loader.get_tensor(name);
        aligned_offsets.push_back({name, current_offset});
        current_offset += (entry.byte_size + 15) & ~size_t(15);
    }
    total_weights_bytes = current_offset;

    // 2. Pack each offloaded layer into one contiguous block so a single
    //    cudaMemcpyAsync stages the whole layer.
    size_t pinned_weight_bytes = 0;
    for (size_t l = m_num_gpu_layers; l < m_config.num_layers; ++l) {
        size_t off = 0;
        for (const auto& name : offloaded_by_layer[l]) {
            const auto& entry = metadata_loader.get_tensor(name);
            m_offloaded_tensors[name] = {static_cast<int>(l), off};
            off += (entry.byte_size + 15) & ~size_t(15);
        }
        m_layer_bytes[l] = off;
        m_max_layer_bytes = std::max(m_max_layer_bytes, off);
        pinned_weight_bytes += off;
    }

    // 3. Reserve the pinned arena once for weights AND the KV host mirror
    //    (allocate_dynamic_pool carves the mirror out of the same arena).
    const size_t offloaded_count = m_config.num_layers - m_num_gpu_layers;
    m_single_layer_kv_bytes = m_config.num_key_value_heads * m_max_seq_len *
                              m_config.head_dim * sizeof(float);
    const size_t pinned_total = pinned_weight_bytes +
                                offloaded_count * 2 * m_single_layer_kv_bytes;
    if (pinned_total > 0) {
        m_pinned.reserve(pinned_total);
        std::cout << "[VRAM Arena] Pinned host pool reserved: "
                  << (pinned_total / (1024 * 1024 * 1024.0)) << " GB ("
                  << (pinned_weight_bytes / (1024 * 1024 * 1024.0)) << " GB weights + KV mirror)\n";
    }

    std::cout << "[VRAM Arena] Contiguous VRAM weights footprint: "
              << (total_weights_bytes / (1024 * 1024 * 1024.0)) << " GB\n";

    // 4. Allocate one massive unified arena for all resident weights
    CUDA_CHECK_THROW(cudaMalloc(&d_weights_arena, total_weights_bytes));

    // Weight-load progress: one cumulative byte counter across the resident
    // upload AND the offloaded pinned-mirror fill, so the observer sees a
    // single monotone 0..total sweep. Emitted per tensor; the observer is
    // O(1) by contract.
    const size_t progress_total = total_weights_bytes + pinned_weight_bytes;
    size_t progress_done = 0;
    const auto report_progress = [&] {
        if (m_load_progress) m_load_progress(progress_done, progress_total);
    };
    report_progress();  // 0% -- the HUD can show the bar immediately

    // 🎯 5. Создаем наш полиморфный загрузчик
    auto io_loader = IWeightLoader::create();

    uint8_t* d_base = reinterpret_cast<uint8_t*>(d_weights_arena);

    for (const auto& [name, vram_offset] : aligned_offsets) {
        const auto& entry = metadata_loader.get_tensor(name);
        void* d_dest = d_base + vram_offset;

        // 🚀 ИДЕАЛЬНЫЙ LSP: Передаем путь к конкретному шарду и смещение
        io_loader->load_to_vram(entry.file_path, entry.file_offset, entry.byte_size, d_dest);

        weight_pointers[name] = d_dest;
        progress_done += entry.byte_size;
        report_progress();
    }

    // Batched loaders (DirectStorage) defer the disk reads; flush() is the
    // completion barrier after which every d_dest above is fully populated.
    io_loader->flush();

    // 6. Fill the pinned mirror for offloaded layers straight from disk.
    if (offloaded_count > 0) {
        FileCache files;
        for (size_t l = m_num_gpu_layers; l < m_config.num_layers; ++l) {
            uint8_t* h_block = static_cast<uint8_t*>(m_pinned.allocate(m_layer_bytes[l]));
            m_layer_host_base[l] = h_block;
            for (const auto& name : offloaded_by_layer[l]) {
                const auto& entry = metadata_loader.get_tensor(name);
                files.read(entry.file_path, entry.file_offset, entry.byte_size,
                           h_block + m_offloaded_tensors[name].offset);
                progress_done += entry.byte_size;
                report_progress();
            }
        }

        // Double-buffered device staging: layer N computes from one slot while
        // layer N+1 streams into the other (slot index = layer % 2).
        for (int s = 0; s < kNumSlots; ++s)
            CUDA_CHECK_THROW(cudaMalloc(&m_wslots[s].d_base, m_max_layer_bytes));
        std::cout << "[VRAM Arena] Weight staging slots: 2 x "
                  << (m_max_layer_bytes / (1024 * 1024.0)) << " MB\n";
    }

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    // Force 100%: the per-tensor counter sums raw byte_size while the totals
    // carry 16-byte alignment padding, so it lands slightly short of total.
    progress_done = progress_total;
    report_progress();
    std::cout << "[VRAM Arena] Weights successfully transferred to device arena.\n";
    std::cout << "[VRAM Arena] Loaded " << qweight_count << " quantized AWQ/GPTQ modules.\n";
}

void VRAMArena::allocate_dynamic_pool(size_t max_seq_len) {
    const size_t intermediate_dim = m_config.intermediate_dim;
    const size_t kv_heads         = m_config.num_key_value_heads;
    const size_t head_dim         = m_config.head_dim;

    // Ping-Pong buffers (FP32 accumulation) sized for the widest vector they ever
    // hold. The engine stores hidden_dim activations in them; take the max with
    // intermediate_dim so the capacity contract holds for any config, not only
    // for models where intermediate_dim > hidden_dim. Multiplied by the activation
    // token capacity so batched prefill / true batch mode can stage
    // [num_tokens, hidden_dim] rows contiguously (capacity 1 == batch=1 decode).
    size_t ping_pong_bytes = std::max(intermediate_dim, m_config.hidden_dim)
                             * m_activation_token_capacity * sizeof(float);
    m_activation_bytes = ping_pong_bytes; // 🎯 Сохраняем размер буферов активации

    CUDA_CHECK_THROW(cudaMalloc(&d_activation_A, ping_pong_bytes));
    CUDA_CHECK_THROW(cudaMalloc(&d_activation_B, ping_pong_bytes));

    // Resident KV-Cache pool covers only VRAM-resident layers:
    // [num_gpu_layers, kv_heads, max_seq_len, head_dim]
    const size_t single_layer_kv_bytes = kv_heads * max_seq_len * head_dim * sizeof(float);
    const size_t total_cache_bytes = m_num_gpu_layers * single_layer_kv_bytes;
    m_total_cache_bytes = total_cache_bytes; // 🎯 Сохраняем размер в байтах одного пула кэша

    if (total_cache_bytes > 0) {
        CUDA_CHECK_THROW(cudaMalloc(&d_k_cache, total_cache_bytes));
        CUDA_CHECK_THROW(cudaMalloc(&d_v_cache, total_cache_bytes));

        // 🚨 ЖЕЛЕЗОБЕТОННАЯ ОЧИСТКА: Зануляем весь кэш
        CUDA_CHECK_THROW(cudaMemset(d_k_cache, 0, total_cache_bytes));
        CUDA_CHECK_THROW(cudaMemset(d_v_cache, 0, total_cache_bytes));
    }
    CUDA_CHECK_THROW(cudaMemset(d_activation_A, 0, ping_pong_bytes));
    CUDA_CHECK_THROW(cudaMemset(d_activation_B, 0, ping_pong_bytes));

    // Offloaded layers: pinned host mirror (authoritative copy) + two device
    // staging slots with the exact stride the attention kernel expects.
    const size_t offloaded_count = m_config.num_layers - m_num_gpu_layers;
    if (offloaded_count > 0) {
        m_h_k_mirror.reserve(offloaded_count);
        m_h_v_mirror.reserve(offloaded_count);
        for (size_t i = 0; i < offloaded_count; ++i) {
            auto* h_k = static_cast<float*>(m_pinned.allocate(single_layer_kv_bytes));
            auto* h_v = static_cast<float*>(m_pinned.allocate(single_layer_kv_bytes));
            std::memset(h_k, 0, single_layer_kv_bytes);
            std::memset(h_v, 0, single_layer_kv_bytes);
            m_h_k_mirror.push_back(h_k);
            m_h_v_mirror.push_back(h_v);
        }
        m_kv_host_filled.assign(offloaded_count, 0);

        for (int s = 0; s < kNumSlots; ++s) {
            CUDA_CHECK_THROW(cudaMalloc(&m_kvslots[s].d_k, single_layer_kv_bytes));
            CUDA_CHECK_THROW(cudaMalloc(&m_kvslots[s].d_v, single_layer_kv_bytes));
            CUDA_CHECK_THROW(cudaMemset(m_kvslots[s].d_k, 0, single_layer_kv_bytes));
            CUDA_CHECK_THROW(cudaMemset(m_kvslots[s].d_v, 0, single_layer_kv_bytes));
        }
    }

    std::cout << "[VRAM Arena] Dynamic pool allocated. Context capacity: " << max_seq_len << " tokens.\n";
    std::cout << "[VRAM Arena] Total dynamic memory consumption: "
              << ((ping_pong_bytes * 2 + total_cache_bytes * 2 +
                   (offloaded_count ? 4 * single_layer_kv_bytes : 0)) / (1024 * 1024.0)) << " MB\n";
}

// ============================================================================
// Soft hibernation (weights VRAM <-> pinned host RAM; see memory_pool.h)
// ============================================================================
void VRAMArena::hibernate() {
    if (m_hibernated || !d_weights_arena || total_weights_bytes == 0) return;

    if (!m_h_hibernate_stash) {
        // Admin/lifecycle tier: throws (mapped to an HRESULT at the EngineCom
        // boundary that wraps hibernate()). cudaHostAlloc nulls the pointer on
        // failure, so a later retry re-attempts cleanly.
        CUDA_CHECK_THROW(
            cudaHostAlloc(&m_h_hibernate_stash, total_weights_bytes, cudaHostAllocDefault));
    }

    // The engine is idle by contract, but "idle" host-side still allows queued
    // async work (transfer-stream prefetches, KV commits). Drain everything so
    // the snapshot is consistent and nothing reads the arena after the free.
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaMemcpy(m_h_hibernate_stash, d_weights_arena, total_weights_bytes,
                          cudaMemcpyDeviceToHost));

    m_hibernated_old_base = d_weights_arena;
    CUDA_CHECK_THROW(cudaFree(d_weights_arena));
    d_weights_arena = nullptr;
    m_hibernated = true;
    std::cout << "[VRAM Arena] Hibernated: "
              << (total_weights_bytes / (1024 * 1024 * 1024.0))
              << " GB of weights moved VRAM -> pinned host RAM.\n";
}

void VRAMArena::wakeup() {
    if (!m_hibernated) return;

    CUDA_CHECK_THROW(cudaMalloc(&d_weights_arena, total_weights_bytes));
    // Pinned source => a pure PCIe DMA burst, no pageable staging copy.
    CUDA_CHECK_THROW(cudaMemcpy(d_weights_arena, m_h_hibernate_stash, total_weights_bytes,
                          cudaMemcpyHostToDevice));

    // The registry values are absolute addresses into the OLD arena; rebase
    // them onto the fresh allocation. (Offloaded tensors resolve through the
    // staging slots and never touch this map.)
    const ptrdiff_t delta = static_cast<uint8_t*>(d_weights_arena) -
                            static_cast<uint8_t*>(m_hibernated_old_base);
    if (delta != 0) {
        for (auto& [name, ptr] : weight_pointers) {
            ptr = static_cast<uint8_t*>(ptr) + delta;
        }
    }
    m_hibernated_old_base = nullptr;
    m_hibernated = false;
    std::cout << "[VRAM Arena] Awake: weights DMA'd back to VRAM.\n";
}

// ============================================================================
// Weight resolution
// ============================================================================
void* VRAMArena::resolve_offloaded(const std::string& name) const {
    auto it = m_offloaded_tensors.find(name);
    if (it == m_offloaded_tensors.end()) return nullptr;

    const OffloadedTensor& info = it->second;
    const WeightSlot& slot = m_wslots[info.layer % kNumSlots];
    if (slot.layer != info.layer)
        throw std::runtime_error("[VRAMArena] Offloaded layer " + std::to_string(info.layer) +
                                 " is not staged in VRAM (ensure_layer_ready() must run before "
                                 "resolving " + name + ")");
    return slot.d_base + info.offset;
}

void* VRAMArena::get_weight_ptr(const std::string& name) const {
    if (m_hibernated)
        throw std::runtime_error("[VRAMArena] weight access while hibernated (" + name +
                                 "): call wakeup() before any forward pass");
    auto it = weight_pointers.find(name);
    if (it != weight_pointers.end()) return it->second;

    if (void* p = resolve_offloaded(name)) return p;
    throw std::runtime_error("Pointer not allocated in arena for: " + name);
}

const void* VRAMArena::get_weight_ptr_optional(const std::string& name) const {
    if (m_hibernated)
        throw std::runtime_error("[VRAMArena] weight access while hibernated (" + name +
                                 "): call wakeup() before any forward pass");
    auto it = weight_pointers.find(name);
    if (it != weight_pointers.end()) return it->second;
    return resolve_offloaded(name); // nullptr when the tensor does not exist
}

QuantizedTensorPtrs VRAMArena::get_quantized_pointers(const std::string& base_name) const {
    const bool is_quant = (m_config.quant_method == "awq" || m_config.quant_method == "gptq");

    auto lookup = [&](const std::string& key) -> const void* {
        return get_weight_ptr_optional(key);
    };

    const void* qweight = lookup(base_name + ".qweight");
    const void* scales  = lookup(base_name + ".scales");
    const void* qzeros  = lookup(base_name + ".qzeros");

    if (is_quant) {
        if (!qweight)
            throw std::runtime_error("[VRAMArena] Missing quantized tensor: " + base_name + ".qweight");
        if (!scales)
            throw std::runtime_error("[VRAMArena] Missing quantized tensor: " + base_name + ".scales");
    }

    return {qweight, scales, qzeros};
}

// ============================================================================
// Asynchronous layer staging (weights)
// ============================================================================
void VRAMArena::stage_layer_weights(int layer) {
    WeightSlot& slot = m_wslots[layer % kNumSlots];
    if (slot.layer == layer) return;

    // Eviction handshake: every kernel reading the previous occupant has
    // already been enqueued on the legacy stream, so an event recorded there
    // NOW conservatively brackets all of them. The transfer stream must not
    // overwrite the slot before that point.
    CUDA_CHECK_THROW(cudaEventRecord(slot.ev_retire, 0));
    CUDA_CHECK_THROW(cudaStreamWaitEvent(m_transfer_stream, slot.ev_retire, 0));

    CUDA_CHECK_THROW(cudaMemcpyAsync(slot.d_base, m_layer_host_base[layer], m_layer_bytes[layer],
                               cudaMemcpyHostToDevice, m_transfer_stream));
    CUDA_CHECK_THROW(cudaEventRecord(slot.ev_ready, m_transfer_stream));

    slot.layer = layer;
    slot.compute_synced = false;
}

void VRAMArena::prefetch_layer(int layer, int pos) {
    if (layer < 0 || static_cast<size_t>(layer) >= m_config.num_layers) return;
    if (!layer_is_offloaded(layer)) return;
    stage_layer_weights(layer);
    stage_layer_kv(layer, pos);
}

void VRAMArena::ensure_layer_ready(int layer) {
    if (!layer_is_offloaded(layer)) return;
    stage_layer_weights(layer); // cold path; no-op when prefetched

    WeightSlot& slot = m_wslots[layer % kNumSlots];
    if (!slot.compute_synced) {
        // Block the COMPUTE STREAM (not the host) until the H2D copy lands.
        CUDA_CHECK_THROW(cudaStreamWaitEvent(0, slot.ev_ready, 0));
        slot.compute_synced = true;
    }
}

// ============================================================================
// KV-cache routing & spill/prefetch
// ============================================================================
float* VRAMArena::get_layer_k_cache(int layer) const {
    if (!layer_is_offloaded(layer))
        return d_k_cache + static_cast<size_t>(layer) *
               (m_config.num_key_value_heads * m_max_seq_len * m_config.head_dim);

    const KVSlot& slot = m_kvslots[layer % kNumSlots];
    if (slot.layer != layer)
        throw std::runtime_error("[VRAMArena] KV cache of layer " + std::to_string(layer) +
                                 " is not staged (prepare_layer_kv() must run first)");
    return slot.d_k;
}

float* VRAMArena::get_layer_v_cache(int layer) const {
    if (!layer_is_offloaded(layer))
        return d_v_cache + static_cast<size_t>(layer) *
               (m_config.num_key_value_heads * m_max_seq_len * m_config.head_dim);

    const KVSlot& slot = m_kvslots[layer % kNumSlots];
    if (slot.layer != layer)
        throw std::runtime_error("[VRAMArena] KV cache of layer " + std::to_string(layer) +
                                 " is not staged (prepare_layer_kv() must run first)");
    return slot.d_v;
}

void VRAMArena::stage_layer_kv(int layer, int pos) {
    KVSlot& slot = m_kvslots[layer % kNumSlots];

    if (slot.layer != layer) {
        // Evict previous occupant: wait for its enqueued attention work, then
        // the slot may be rewritten. Its dirty column was already spilled by
        // commit_layer_kv on the same transfer stream (FIFO order protects
        // spill-before-refetch of the host mirror).
        CUDA_CHECK_THROW(cudaEventRecord(slot.ev_retire, 0));
        CUDA_CHECK_THROW(cudaStreamWaitEvent(m_transfer_stream, slot.ev_retire, 0));
        slot.layer = layer;
        slot.valid_upto = 0;
        slot.compute_synced = false;
    }

    // Incremental block prefetch: only tokens the slot is missing (and the host
    // mirror actually has) cross PCIe. When the slot still holds this layer
    // from the previous decode step, this is a single-token copy.
    const int oi = layer - static_cast<int>(m_num_gpu_layers);
    const int need = std::min(pos, m_kv_host_filled[oi]);
    if (need > slot.valid_upto) {
        const size_t pitch = m_max_seq_len * m_config.head_dim * sizeof(float);
        const size_t row_offset = static_cast<size_t>(slot.valid_upto) * m_config.head_dim;
        const size_t width = static_cast<size_t>(need - slot.valid_upto) *
                             m_config.head_dim * sizeof(float);

        // Layout [kv_heads, max_seq_len, head_dim]: a token range is contiguous
        // within each head row, so one strided 2D copy moves all heads.
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(slot.d_k + row_offset, pitch,
                                     m_h_k_mirror[oi] + row_offset, pitch,
                                     width, m_config.num_key_value_heads,
                                     cudaMemcpyHostToDevice, m_transfer_stream));
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(slot.d_v + row_offset, pitch,
                                     m_h_v_mirror[oi] + row_offset, pitch,
                                     width, m_config.num_key_value_heads,
                                     cudaMemcpyHostToDevice, m_transfer_stream));
        CUDA_CHECK_THROW(cudaEventRecord(slot.ev_ready, m_transfer_stream));

        slot.valid_upto = need;
        slot.compute_synced = false;
    }
}

void VRAMArena::prepare_layer_kv(int layer, int pos) {
    if (!layer_is_offloaded(layer)) return;
    stage_layer_kv(layer, pos);

    KVSlot& slot = m_kvslots[layer % kNumSlots];
    if (!slot.compute_synced) {
        CUDA_CHECK_THROW(cudaStreamWaitEvent(0, slot.ev_ready, 0));
        slot.compute_synced = true;
    }
}

void VRAMArena::commit_layer_kv(int layer, int pos) {
    if (!layer_is_offloaded(layer)) return;

    KVSlot& slot = m_kvslots[layer % kNumSlots];
    if (slot.layer != layer)
        throw std::runtime_error("[VRAMArena] commit_layer_kv: layer " + std::to_string(layer) +
                                 " does not own its staging slot");

    // The RoPE/attention kernels that wrote column `pos` are enqueued on the
    // legacy stream; spill must wait for them but NOT for future compute.
    CUDA_CHECK_THROW(cudaEventRecord(slot.ev_commit, 0));
    CUDA_CHECK_THROW(cudaStreamWaitEvent(m_transfer_stream, slot.ev_commit, 0));

    const size_t pitch = m_max_seq_len * m_config.head_dim * sizeof(float);
    const size_t col_offset = static_cast<size_t>(pos) * m_config.head_dim;
    const size_t width = m_config.head_dim * sizeof(float);

    const int oi = layer - static_cast<int>(m_num_gpu_layers);
    CUDA_CHECK_THROW(cudaMemcpy2DAsync(m_h_k_mirror[oi] + col_offset, pitch,
                                 slot.d_k + col_offset, pitch,
                                 width, m_config.num_key_value_heads,
                                 cudaMemcpyDeviceToHost, m_transfer_stream));
    CUDA_CHECK_THROW(cudaMemcpy2DAsync(m_h_v_mirror[oi] + col_offset, pitch,
                                 slot.d_v + col_offset, pitch,
                                 width, m_config.num_key_value_heads,
                                 cudaMemcpyDeviceToHost, m_transfer_stream));

    // The kernel itself extended in-slot validity through `pos`; the mirror
    // catches up asynchronously, ordered ahead of any future mirror read by
    // transfer-stream FIFO.
    slot.valid_upto = std::max(slot.valid_upto, pos + 1);
    m_kv_host_filled[oi] = std::max(m_kv_host_filled[oi], pos + 1);
}

void VRAMArena::truncate_kv(int target_pos) {
    if (target_pos < 0) target_pos = 0;

    // Pull every offloaded layer's host-mirror high-water mark back to target_pos.
    // A later commit_layer_kv re-extends it with std::max, so any speculative
    // column above target_pos is overwritten rather than read as valid.
    for (int& filled : m_kv_host_filled)
        filled = std::min(filled, target_pos);

    // Invalidate any staging slot content above target_pos. Clearing
    // compute_synced forces prepare_layer_kv to re-wait on a fresh stage before the
    // next attention read, so a rewound-then-refilled prefix is never served stale.
    for (int s = 0; s < kNumSlots; ++s) {
        KVSlot& slot = m_kvslots[s];
        if (slot.valid_upto > target_pos) {
            slot.valid_upto = target_pos;
            slot.compute_synced = false;
        }
    }
}
