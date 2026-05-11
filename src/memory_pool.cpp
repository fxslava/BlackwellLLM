#include "memory_pool.h"
#include "common.h"
#include <iostream>

VRAMArena::VRAMArena(const SafetensorsLoader& loader, size_t max_seq_len) {
    std::cout << "[VRAM Arena] Initializing static memory pools...\n";
    allocate_weights_pool(loader);
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

void VRAMArena::allocate_weights_pool(const SafetensorsLoader& loader) {
    auto tensor_names = loader.list_tensors();
    
    // 1. Calculate absolute byte footprint with strict 16-byte alignment per tensor
    size_t current_offset = 0;
    std::vector<std::pair<std::string, size_t>> aligned_offsets;
    aligned_offsets.reserve(tensor_names.size());

    for (const auto& name : tensor_names) {
        const auto& entry = loader.get_tensor(name);
        aligned_offsets.push_back({name, current_offset});
        
        // Align offsets to 16 bytes (128-bit boundary required by CUDA memory accesses)
        size_t size = entry.byte_size;
        current_offset += (size + 15) & ~15; 
    }
    total_weights_bytes = current_offset;

    std::cout << "[VRAM Arena] Contiguous weights footprint: " 
              << (total_weights_bytes / (1024 * 1024 * 1024.0)) << " GB\n";

    // 2. Allocate one massive unified arena for all weights
    CUDA_CHECK(cudaMalloc(&d_weights_arena, total_weights_bytes));

    // 3. Populate device memory asynchronously via direct copies from OS file cache
    uint8_t* d_base = reinterpret_cast<uint8_t*>(d_weights_arena);
    
    for (const auto& [name, offset] : aligned_offsets) {
        const auto& entry = loader.get_tensor(name);
        void* d_dest = d_base + offset;
        
        CUDA_CHECK(cudaMemcpyAsync(
            d_dest, 
            entry.host_data_ptr, 
            entry.byte_size, 
            cudaMemcpyHostToDevice
        ));
        
        weight_pointers[name] = d_dest;
    }

    // Synchronize to ensure all 8.46 GB are safely residing in VRAM
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[VRAM Arena] Weights successfully transferred to device arena.\n";
}

void VRAMArena::allocate_dynamic_pool(size_t max_seq_len) {
    // Llama 3 8B dimensions
    const size_t hidden_dim = 4096;
    const size_t intermediate_dim = 14336; // Largest buffer needed for FFN SwiGLU
    const size_t kv_heads = 8;             // GQA configuration
    const size_t head_dim = 128;
    const size_t num_layers = 32;

    // Ping-Pong buffers (FP32 accumulation) allocated for the maximum possible intermediate dimension
    size_t ping_pong_bytes = intermediate_dim * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_activation_A, ping_pong_bytes));
    CUDA_CHECK(cudaMalloc(&d_activation_B, ping_pong_bytes));

    // Global KV-Cache calculation: [num_layers, kv_heads, max_seq_len, head_dim]
    size_t single_layer_kv_bytes = kv_heads * max_seq_len * head_dim * sizeof(float);
    size_t total_cache_bytes = num_layers * single_layer_kv_bytes;

    CUDA_CHECK(cudaMalloc(&d_k_cache, total_cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache, total_cache_bytes));

    // Выделение KV-кэша
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