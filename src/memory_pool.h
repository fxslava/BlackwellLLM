#pragma once
#include <cuda_runtime.h>
#include <cstddef>
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

// Static Memory Orchestrator for 12GB VRAM limit
class VRAMArena {
public:
    VRAMArena(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader, const ModelConfig& config, size_t max_seq_len = 2048);
    ~VRAMArena();

    VRAMArena(const VRAMArena&) = delete;
    VRAMArena& operator=(const VRAMArena&) = delete;

    // Retrieve device pointer for static weights
    void* get_weight_ptr(const std::string& name) const;

    // Safely retrieves a pointer if the weight exists, returns nullptr otherwise
    const void* get_weight_ptr_optional(const std::string& name) const;

    QuantizedTensorPtrs get_quantized_pointers(const std::string& base_name) const;

    // Ping-Pong activation buffers (reused across all 32 layers)
    float* get_activation_buffer_A() const { return d_activation_A; }
    float* get_activation_buffer_B() const { return d_activation_B; }

    // Global KV-Cache buffers
    float* get_k_cache() const { return d_k_cache; }
    float* get_v_cache() const { return d_v_cache; }

    // 🎯 НОВЫЕ МЕТОДЫ: Получение сохраненных размеров и емкостей
    size_t get_max_seq_len() const { return m_max_seq_len; }
    size_t get_k_cache_size() const { return m_total_cache_bytes; }
    size_t get_v_cache_size() const { return m_total_cache_bytes; }
    size_t get_activation_buffer_size() const { return m_activation_bytes; }

private:
    void allocate_weights_pool(const std::string& safetensors_path, const SafetensorsLoader& metadata_loader);
    void allocate_dynamic_pool(size_t max_seq_len);
    // Frees every device pool and nulls the pointers (idempotent). Shared by the
    // destructor and the constructor's failure path.
    void release_pools();

    // Contiguous memory blocks
    void* d_weights_arena = nullptr;
    size_t total_weights_bytes = 0;

    // Activation buffers (FP32)
    float* d_activation_A = nullptr;
    float* d_activation_B = nullptr;

    // KV Cache pool
    float* d_k_cache = nullptr;
    float* d_v_cache = nullptr;

    ModelConfig m_config;
    size_t m_max_seq_len = 0;
    size_t m_total_cache_bytes = 0;
    size_t m_activation_bytes = 0;

    // Offset registry mapping tensor names to their absolute addresses in d_weights_arena
    std::unordered_map<std::string, void*> weight_pointers;
};