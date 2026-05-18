#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <queue>
#include <map>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#include "safetensors.h"
#include "memory_pool.h"
#include "cpu_models.h"
#include "nvcomp_baseline.h"

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

void calculate_bit_lengths(HuffmanNode* root, int current_depth, std::vector<int>& bit_lengths) {
    if (!root) return;

    if (!root->left && !root->right) {
        bit_lengths[root->byte_value] = (current_depth == 0) ? 1 : current_depth;
        return;
    }

    calculate_bit_lengths(root->left, current_depth + 1, bit_lengths);
    calculate_bit_lengths(root->right, current_depth + 1, bit_lengths);
}

void evaluate_global_vlc(const std::vector<uint64_t>& raw_histogram, size_t tensor_size) {
    std::cout << "\n🌳 Building optimal VLC (Huffman) tree for tensor...\n";

    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNodes> min_heap;
    
    for (int i = 0; i < 256; ++i) {
        if (raw_histogram[i] > 0) {
            min_heap.push(new HuffmanNode(static_cast<uint8_t>(i), raw_histogram[i]));
        }
    }

    while (min_heap.size() > 1) {
        HuffmanNode* left = min_heap.top(); min_heap.pop();
        HuffmanNode* right = min_heap.top(); min_heap.pop();

        HuffmanNode* parent = new HuffmanNode(0, left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        
        min_heap.push(parent);
    }

    HuffmanNode* root = min_heap.top();

    std::vector<int> bit_lengths(256, 0);
    calculate_bit_lengths(root, 0, bit_lengths);

    size_t total_compressed_bits = 0;
    for (int i = 0; i < 256; ++i) {
        total_compressed_bits += raw_histogram[i] * bit_lengths[i];
    }

    size_t original_bits = tensor_size * 8;
    float compression_ratio = static_cast<float>(original_bits) / total_compressed_bits;

    struct DiagnosticEntry {
        uint8_t byte_val;
        float float_val;
        uint64_t freq;
        int bits;
    };
    std::vector<DiagnosticEntry> diag;
    for (int i = 0; i < 256; ++i) {
        if (raw_histogram[i] > 0) {
            diag.push_back({static_cast<uint8_t>(i), cpu_unpack_fp8_e4m3(static_cast<uint8_t>(i)), raw_histogram[i], bit_lengths[i]});
        }
    }
    std::sort(diag.begin(), diag.end(), [](const auto& a, const auto& b) { return a.freq > b.freq; });

    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(12) << "Float Val" << std::setw(15) << "Frequency" << std::setw(12) << "VLC Length" << "\n";
    std::cout << "--------------------------------------------------------\n";
    for (size_t i = 0; i < std::min(diag.size(), (size_t)10); ++i) {
        std::cout << std::left << std::setw(12) << diag[i].float_val 
                  << std::setw(15) << diag[i].freq 
                  << std::setw(12) << (std::to_string(diag[i].bits) + " bits") << "\n";
    }
    std::cout << "--------------------------------------------------------\n";

    std::cout << "📊 ORIGINAL TENSOR SIZE : " << (original_bits / 8 / 1024 / 1024) << " MB\n";
    std::cout << "📦 THEORETICAL VLC SIZE: " << (total_compressed_bits / 8.0 / 1024.0 / 1024.0) << " MB\n";
    std::cout << "🚀 EXPECTED RATIO      : " << std::fixed << std::setprecision(2) << compression_ratio << "x\n\n";
}

std::vector<uint64_t> export_weight_histogram(const void* d_weight_ptr, size_t tensor_size, const std::string& csv_filename) {
    std::cout << "📊 Direct collection of ordered weight histogram (MAX -> 0 -> -MIN)...\n";
    
    std::vector<std::pair<float, uint8_t>> sorted_pairs(256);
    for (int i = 0; i < 256; ++i) {
        sorted_pairs[i] = { cpu_unpack_fp8_e4m3(static_cast<uint8_t>(i)), static_cast<uint8_t>(i) };
    }
    std::sort(sorted_pairs.begin(), sorted_pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> byte_to_bucket(256);
    for (int bucket_idx = 0; bucket_idx < 256; ++bucket_idx) {
        byte_to_bucket[sorted_pairs[bucket_idx].second] = bucket_idx;
    }

    std::vector<uint8_t> h_weights(tensor_size);
    cudaMemcpy(h_weights.data(), d_weight_ptr, tensor_size, cudaMemcpyDeviceToHost);

    std::vector<uint64_t> bucket_histogram(256, 0);
    std::vector<uint64_t> raw_histogram(256, 0);

    for (size_t i = 0; i < tensor_size; ++i) {
        uint8_t raw_byte = h_weights[i];
        bucket_histogram[byte_to_bucket[raw_byte]]++;
        raw_histogram[raw_byte]++;
    }

    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Failed to create file: " + csv_filename);
    }

    csv_file << "Bucket_Index,Byte_Value,Unpacked_Float,Frequency,Percentage\n";
    for (int bucket_idx = 0; bucket_idx < 256; ++bucket_idx) {
        uint8_t raw_byte = sorted_pairs[bucket_idx].second;
        float fp_val = sorted_pairs[bucket_idx].first;
        uint64_t freq = bucket_histogram[bucket_idx];
        float pct = (static_cast<float>(freq) / tensor_size) * 100.0f;
        csv_file << bucket_idx << "," << static_cast<int>(raw_byte) << "," << fp_val << "," << freq << "," << std::fixed << std::setprecision(4) << pct << "\n";
    }
    csv_file.close();
    
    std::cout << "✅ Histogram successfully saved to: " << csv_filename << "\n";
    return raw_histogram;
}

int main(int argc, char** argv) {
    std::cout << "🔬 Launching Blackwell Compression experimental stand...\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        
        std::string target_tensor = "model.layers.0.mlp.down_proj.weight";
        if (!metadata_loader.has_tensor(target_tensor)) {
            for (const auto& t : metadata_loader.list_tensors()) {
                if (t.find("down_proj.weight") != std::string::npos) { target_tensor = t; break; }
            }
        }
        
        const void* d_weight_ptr = arena.get_weight_ptr_optional(target_tensor);
        if (!d_weight_ptr) throw std::runtime_error("Tensor not found in VRAMArena!");
        size_t tensor_size = metadata_loader.get_tensor(target_tensor).byte_size;
        
        std::cout << "✅ Target tensor verified: " << target_tensor << " (" << tensor_size / 1024 / 1024 << " MB)\n";
        
        // 1. Run standard nvCOMP benchmarks
        run_nvcomp_baseline(d_weight_ptr, tensor_size);
        
        // 2. Export distribution data & run global VLC check
        auto raw_hist = export_weight_histogram(d_weight_ptr, tensor_size, "weight_distribution.csv");
        evaluate_global_vlc(raw_hist, tensor_size);
        
    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}