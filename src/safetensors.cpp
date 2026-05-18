#include "safetensors.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

SafetensorsLoader::SafetensorsLoader(const std::string& index_or_file_path) {
    // Check if the input is an index JSON file or a raw safetensors file
    if (index_or_file_path.find(".index.json") != std::string::npos) {
        load_index_file(index_or_file_path);
    } else {
        load_single_file(index_or_file_path);
    }
}

SafetensorsLoader::~SafetensorsLoader() {
#ifdef _WIN32
    for (auto& mf : mapped_files) {
        if (mf.mapped_data) UnmapViewOfFile(mf.mapped_data);
        if (mf.h_map && mf.h_map != INVALID_HANDLE_VALUE) CloseHandle(mf.h_map);
        if (mf.h_file && mf.h_file != INVALID_HANDLE_VALUE) CloseHandle(mf.h_file);
    }
#endif
    std::cout << "[Safetensors] Successfully unmapped " << mapped_files.size() << " files from virtual memory.\n";
}

void SafetensorsLoader::load_single_file(const std::string& file_path) {
    std::ifstream f(file_path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open: " + file_path);

    // 1. Читаем размер шапки (8 байт)
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);

    // 2. Выкачиваем JSON-строку
    std::string json_str(header_size, '\0');
    f.read(&json_str[0], header_size);
    json header = json::parse(json_str);

    // Точка на диске, где начинается бинарный блок текущего файла
    size_t binary_start_pos = 8 + header_size;

    for (auto& [key, value] : header.items()) {
        if (key == "__metadata__") continue;

        TensorEntry entry;
        entry.name = key;
        entry.dtype = value["dtype"].get<std::string>();
        entry.file_path = file_path; // 🎯 Запоминаем конкретный шард
        
        for (auto& dim : value["shape"]) entry.shape.push_back(dim.get<size_t>());

        auto offsets = value["data_offsets"];
        size_t start_offset = offsets[0].get<size_t>();
        size_t end_offset   = offsets[1].get<size_t>();

        entry.byte_size = end_offset - start_offset;
        // 🎯 Считаем абсолютную позицию байт в файле
        entry.file_offset = binary_start_pos + start_offset; 

        registry[key] = entry;
    }
    std::cout << "[Safetensors] Registered metadata for slice: " << fs::path(file_path).filename().string() << "\n";
}

void SafetensorsLoader::load_index_file(const std::string& index_path) {
    std::cout << "[Safetensors] Parsing multi-file index: " << index_path << "\n";
    std::ifstream f(index_path);
    if (!f.is_open()) throw std::runtime_error("Failed to open index JSON: " + index_path);

    json index_json = json::parse(f);
    auto weight_map = index_json["weight_map"];

    // Deduplicate file names to map each chunk exactly once
    std::vector<std::string> unique_files;
    for (auto& [tensor_name, file_name_val] : weight_map.items()) {
        std::string file_name = file_name_val.get<std::string>();
        if (std::find(unique_files.begin(), unique_files.end(), file_name) == unique_files.end()) {
            unique_files.push_back(file_name);
        }
    }

    // Resolve base directory from the index path
    fs::path base_dir = fs::path(index_path).parent_path();

    // Map all discovered files into virtual memory
    for (const auto& file_name : unique_files) {
        fs::path full_chunk_path = base_dir / file_name;
        load_single_file(full_chunk_path.string());
    }

    std::cout << "[Safetensors] Multi-file registry complete. Total tensors loaded: " << registry.size() << "\n";
}

const TensorEntry& SafetensorsLoader::get_tensor(const std::string& name) const {
    auto it = registry.find(name);
    if (it == registry.end()) throw std::runtime_error("Tensor not found in registry: " + name);
    return it->second;
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
    return registry.find(name) != registry.end();
}

std::vector<std::string> SafetensorsLoader::list_tensors() const {
    std::vector<std::string> keys;
    keys.reserve(registry.size());
    for (const auto& [key, _] : registry) keys.push_back(key);
    return keys;
}