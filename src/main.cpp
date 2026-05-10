#include <iostream>
#include <cuda_runtime.h>
#include "common.h"
#include "safetensors.h"

int main() {
    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM Engine: Multi-File Weights Test\n";
    std::cout << "==================================================\n\n";

    try {
        // Pointing directly to the index file mapped in your folder structure
        std::string index_path = "llama3-8b-fp8/model.safetensors.index.json"; 
        
        std::cout << "Initializing multi-file Zero-Copy memory loader...\n";
        SafetensorsLoader loader(index_path);

        // Let's test picking tensors from different physical files
        std::vector<std::string> test_tensors = {
            "model.embed_tokens.weight",           // Usually in chunk 1
            "model.layers.0.self_attn.q_proj.weight", // Usually in chunk 1
            "model.layers.31.mlp.up_proj.weight",  // Guaranteed in chunk 2
            "lm_head.weight"                       // Usually at the very end
        };

        std::cout << "\nValidating distributed tensor access:\n";
        std::cout << "--------------------------------------------------\n";

        size_t total_model_bytes = 0;

        for (const auto& name : test_tensors) {
            if (loader.has_tensor(name)) {
                const auto& entry = loader.get_tensor(name);
                std::cout << " [FOUND] " << entry.name << "\n";
                std::cout << "         Dtype: " << entry.dtype << " | Bytes: " << entry.byte_size << "\n";
            } else {
                std::cout << " [ERROR] Missing critical tensor: " << name << "\n";
            }
        }

        // Calculate aggregate size across the entire registry
        for (const auto& name : loader.list_tensors()) {
            total_model_bytes += loader.get_tensor(name).byte_size;
        }

        std::cout << "--------------------------------------------------\n";
        std::cout << "[System VRAM Check] Total pure FP8 tensor volume: " 
                  << (total_model_bytes / (1024 * 1024 * 1024.0)) << " GB\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR]: " << e.what() << "\n";
        std::cerr << "Ensure the folder name matches 'llama3-8b-fp8' and contains all chunks.\n";
    }

    return 0;
}