#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#include "safetensors.h"
#include "memory_pool.h"
#include "cpu_models.h"
#include "nvcomp_baseline.h"

// 🎯 ADJUSTABLE CONFIGURATION
const size_t CHUNK_SIZE_BYTES = 65536; // 64 KB chunks
const size_t HUFFMAN_TABLE_OVERHEAD_BYTES = 128; // 1024 bits for canonical code lengths

struct TensorStats {
    std::string name;
    size_t original_bytes;
    std::vector<uint64_t> global_tensor_hist;
    size_t per_tensor_compressed_bytes;
    size_t per_chunk_compressed_bytes;
};

struct HuffmanNode {
    uint8_t byte_value;
    uint64_t freq;
    HuffmanNode* left;
    HuffmanNode* right;
    HuffmanNode(uint8_t val, uint64_t f) : byte_value(val), freq(f), left(nullptr), right(nullptr) {}
};

struct CompareNodes {
    bool operator()(HuffmanNode* const& n1, HuffmanNode* const& n2) {
        return n1->freq > n2->freq;
    }
};

void free_huffman_tree(HuffmanNode* root) {
    if (!root) return;
    free_huffman_tree(root->left);
    free_huffman_tree(root->right);
    delete root;
}

void calculate_bit_lengths(HuffmanNode* root, int current_depth, std::vector<int>& bit_lengths) {
    if (!root) return;
    if (!root->left && !root->right) {
        bit_lengths[root->byte_value] = (current_depth == 0) ? 1 : current_depth;
        return;
    }
    calculate_bit_lengths(root->left, current_depth + 1, bit_lengths);
    calculate_bit_lengths(root->right, current_depth + 1, bit_lengths);
}

// Internal helper to get bit lengths from any 256-bin histogram
std::vector<int> get_huffman_lengths(const std::vector<uint64_t>& histogram) {
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNodes> min_heap;
    for (int i = 0; i < 256; ++i) {
        if (histogram[i] > 0) {
            min_heap.push(new HuffmanNode(static_cast<uint8_t>(i), histogram[i]));
        }
    }
    std::vector<int> bit_lengths(256, 0);
    if (min_heap.empty()) return bit_lengths;

    while (min_heap.size() > 1) {
        HuffmanNode* left = min_heap.top(); min_heap.pop();
        HuffmanNode* right = min_heap.top(); min_heap.pop();
        HuffmanNode* parent = new HuffmanNode(0, left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        min_heap.push(parent);
    }
    HuffmanNode* root = min_heap.top();
    calculate_bit_lengths(root, 0, bit_lengths);
    free_huffman_tree(root);
    return bit_lengths;
}

int main(int argc, char** argv) {
    std::cout << "🔬 Launching Multi-Mode Blackwell Compression Framework...\n";
    std::cout << "ℹ️ Current Chunk Size configuration: " << (CHUNK_SIZE_BYTES / 1024) << " KB\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        
        auto all_tensors = metadata_loader.list_tensors();
        std::vector<TensorStats> model_stats;
        std::vector<uint64_t> model_global_histogram(256, 0);

        std::cout << "📥 Processing tensors and analyzing local chunk distributions...\n";

        for (const auto& tensor_name : all_tensors) {
            const void* d_weight_ptr = arena.get_weight_ptr_optional(tensor_name);
            if (!d_weight_ptr) continue;
            
            size_t tensor_size = metadata_loader.get_tensor(tensor_name).byte_size;
            if (tensor_size == 0) continue;

            // Single download pass to Host RAM
            std::vector<uint8_t> h_weights(tensor_size);
            cudaMemcpy(h_weights.data(), d_weight_ptr, tensor_size, cudaMemcpyDeviceToHost);

            // 1. Compute Global Tensor Histogram
            std::vector<uint64_t> tensor_hist(256, 0);
            for (size_t i = 0; i < tensor_size; ++i) {
                tensor_hist[h_weights[i]]++;
                model_global_histogram[h_weights[i]]++; // Accumulate to total model histogram
            }

            // 2. Evaluate Per-Tensor Mode (Bits + 128 bytes table overhead)
            std::vector<int> tensor_lengths = get_huffman_lengths(tensor_hist);
            size_t per_tensor_bits = 0;
            for (int i = 0; i < 256; ++i) per_tensor_bits += tensor_hist[i] * tensor_lengths[i];
            size_t per_tensor_bytes = ((per_tensor_bits + 7) / 8) + HUFFMAN_TABLE_OVERHEAD_BYTES;

            // 3. Evaluate Per-Chunk Mode (Iterate weights in chunks)
            size_t per_chunk_total_bytes = 0;
            for (size_t offset = 0; offset < tensor_size; offset += CHUNK_SIZE_BYTES) {
                size_t current_chunk_size = std::min(CHUNK_SIZE_BYTES, tensor_size - offset);
                
                std::vector<uint64_t> chunk_hist(256, 0);
                for (size_t i = 0; i < current_chunk_size; ++i) {
                    chunk_hist[h_weights[offset + i]]++;
                }

                std::vector<int> chunk_lengths = get_huffman_lengths(chunk_hist);
                size_t chunk_bits = 0;
                for (int i = 0; i < 256; ++i) chunk_bits += chunk_hist[i] * chunk_lengths[i];
                
                // Each chunk pays 128 bytes overhead for its unique table
                per_chunk_total_bytes += ((chunk_bits + 7) / 8) + HUFFMAN_TABLE_OVERHEAD_BYTES;
            }

            model_stats.push_back({tensor_name, tensor_size, tensor_hist, per_tensor_bytes, per_chunk_total_bytes});
        }

        // 4. Generate the Per-Model Global Unified Table
        std::vector<int> model_global_lengths = get_huffman_lengths(model_global_histogram);

        // 5. Render Final Comparative Report
        std::cout << "\n=================================== MULTI-STRATEGY VLC COMPRESSION REPORT ===================================\n";
        std::cout << std::left 
                  << std::setw(42) << "Tensor Name" 
                  << std::setw(14) << "Orig (MB)" 
                  << std::setw(16) << "Per-Model" 
                  << std::setw(16) << "Per-Tensor" 
                  << std::setw(16) << "Per-Chunk" << "\n";
        std::cout << "--------------------------------------------------------------------------------------------------------------\n";

        size_t total_orig = 0;
        size_t total_model_bytes = 0;
        size_t total_tensor_bytes = 0;
        size_t total_chunk_bytes = 0;

        for (const auto& t : model_stats) {
            // Mathematically derive Per-Model size from cached tensor histograms
            size_t per_model_bits = 0;
            for (int i = 0; i < 256; ++i) per_model_bits += t.global_tensor_hist[i] * model_global_lengths[i];
            size_t per_model_bytes = (per_model_bits + 7) / 8; // Global table pays overhead once, negligible here

            float r_model  = static_cast<float>(t.original_bytes) / per_model_bytes;
            float r_tensor = static_cast<float>(t.original_bytes) / t.per_tensor_compressed_bytes;
            float r_chunk  = static_cast<float>(t.original_bytes) / t.per_chunk_compressed_bytes;

            std::cout << std::left 
                      << std::setw(42) << (t.name.length() > 39 ? t.name.substr(0, 36) + "..." : t.name)
                      << std::setw(14) << std::fixed << std::setprecision(2) << (t.original_bytes / 1024.0 / 1024.0)
                      << std::setw(16) << (std::to_string(r_model).substr(0,4) + "x")
                      << std::setw(16) << (std::to_string(r_tensor).substr(0,4) + "x")
                      << std::setw(16) << (std::to_string(r_chunk).substr(0,4) + "x") << "\n";

            total_orig += t.original_bytes;
            total_model_bytes += per_model_bytes;
            total_tensor_bytes += t.per_tensor_compressed_bytes;
            total_chunk_bytes += t.per_chunk_compressed_bytes;
        }

        float final_r_model  = static_cast<float>(total_orig) / total_model_bytes;
        float final_r_tensor = static_cast<float>(total_orig) / total_tensor_bytes;
        float final_r_chunk  = static_cast<float>(total_orig) / total_chunk_bytes;

        std::cout << "--------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left 
                  << std::setw(42) << "TOTAL MODEL SUMMARY"
                  << std::setw(14) << std::fixed << std::setprecision(2) << (total_orig / 1024.0 / 1024.0)
                  << std::setw(16) << (std::to_string(final_r_model).substr(0,4) + "x")
                  << std::setw(16) << (std::to_string(final_r_tensor).substr(0,4) + "x")
                  << std::setw(16) << (std::to_string(final_r_chunk).substr(0,4) + "x") << "\n";
        std::cout << "==============================================================================================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}