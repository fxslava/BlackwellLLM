#include "memory_pool.h"
#include "blackwell/weight_loader.h"
#include "common.h"
#include <iostream>

VRAMArena::VRAMArena(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader, const ModelConfig& config, size_t max_seq_len)
    : m_config(config), m_max_seq_len(max_seq_len)
{
    std::cout << "[VRAM Arena] Initializing static memory pools...\n";
    allocate_weights_pool(safetensors_path, metadata_loader);
    allocate_dynamic_pool(max_seq_len);
}

VRAMArena::~VRAMArena() {
    if (d_weights_arena) cudaFree(d_weights_arena);
    if (d_activation_A)  cudaFree(d_activation_A);
    if (d_activation_B)  cudaFree(d_activation_B);
    if (d_k_cache)       cudaFree(d_k_cache);
    if (d_v_cache)       cudaFree(d_v_cache);
    std::cout << "[VRAM Arena] All static device pools successfully released.\n";
}

void VRAMArena::allocate_weights_pool(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader) {
    auto tensor_names = metadata_loader.list_tensors();
    
    // 1. Calculate absolute byte footprint with strict 16-byte alignment per tensor
    size_t current_offset = 0;
    std::vector<std::pair<std::string, size_t>> aligned_offsets;
    aligned_offsets.reserve(tensor_names.size());

    for (const auto& name : tensor_names) {
        const auto& entry = metadata_loader.get_tensor(name);
        aligned_offsets.push_back({name, current_offset});
        
        size_t size = entry.byte_size;
        current_offset += (size + 15) & ~15; 
    }
    total_weights_bytes = current_offset;

    std::cout << "[VRAM Arena] Contiguous weights footprint: " 
              << (total_weights_bytes / (1024 * 1024 * 1024.0)) << " GB\n";

    // 2. Allocate one massive unified arena for all weights
    CUDA_CHECK(cudaMalloc(&d_weights_arena, total_weights_bytes));

    // 🎯 3. Создаем наш полиморфный загрузчик
    auto io_loader = IWeightLoader::create();

    uint8_t* d_base = reinterpret_cast<uint8_t*>(d_weights_arena);
    
    for (const auto& [name, vram_offset] : aligned_offsets) {
        const auto& entry = metadata_loader.get_tensor(name);
        void* d_dest = d_base + vram_offset;
        
        // 🚀 ИДЕАЛЬНЫЙ LSP: Передаем путь к конкретному шарду и смещение
        io_loader->load_to_vram(entry.file_path, entry.file_offset, entry.byte_size, d_dest);
        
        weight_pointers[name] = d_dest;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[VRAM Arena] Weights successfully transferred to device arena.\n";
}

void VRAMArena::allocate_dynamic_pool(size_t max_seq_len) {
    const size_t intermediate_dim = m_config.intermediate_dim;
    const size_t kv_heads         = m_config.num_key_value_heads;
    const size_t head_dim         = m_config.head_dim;
    const size_t num_layers       = m_config.num_layers;

    // Ping-Pong buffers (FP32 accumulation) allocated for the maximum possible intermediate dimension
    size_t ping_pong_bytes = intermediate_dim * sizeof(float);
    m_activation_bytes = ping_pong_bytes; // 🎯 Сохраняем размер буферов активации
    
    CUDA_CHECK(cudaMalloc(&d_activation_A, ping_pong_bytes));
    CUDA_CHECK(cudaMalloc(&d_activation_B, ping_pong_bytes));

    // Global KV-Cache calculation: [num_layers, kv_heads, max_seq_len, head_dim]
    size_t single_layer_kv_bytes = kv_heads * max_seq_len * head_dim * sizeof(float);
    size_t total_cache_bytes = num_layers * single_layer_kv_bytes;
    m_total_cache_bytes = total_cache_bytes; // 🎯 Сохраняем размер в байтах одного пула кэша

    // 🎯 ИСПРАВЛЕНО: Удалено дублирование вызовов cudaMalloc
    CUDA_CHECK(cudaMalloc(&d_k_cache, total_cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache, total_cache_bytes));

    // 🚨 ЖЕЛЕЗОБЕТОННАЯ ОЧИСТКА: Зануляем весь кэш и буферы активаций
    CUDA_CHECK(cudaMemset(d_k_cache, 0, total_cache_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, total_cache_bytes));
    CUDA_CHECK(cudaMemset(d_activation_A, 0, ping_pong_bytes));
    CUDA_CHECK(cudaMemset(d_activation_B, 0, ping_pong_bytes));

    std::cout << "[VRAM Arena] Dynamic pool allocated. Context capacity: " << max_seq_len << " tokens.\n";
    std::cout << "[VRAM Arena] Total dynamic memory consumption: " 
              << ((ping_pong_bytes * 2 + total_cache_bytes * 2) / (1024 * 1024.0)) << " MB\n";
}

void* VRAMArena::get_weight_ptr(const std::string& name) const {
    auto it = weight_pointers.find(name);
    if (it == weight_pointers.end()) {
        throw std::runtime_error("Pointer not allocated in arena for: " + name);
    }
    return it->second;
}

const void* VRAMArena::get_weight_ptr_optional(const std::string& name) const {
    auto it = weight_pointers.find(name);
    if (it != weight_pointers.end()) {
        return it->second;
    }
    return nullptr;
}