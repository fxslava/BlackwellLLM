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
#ifdef _WIN32
    MappedFile mf;
    mf.h_file = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (mf.h_file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(mf.h_file, &size)) {
        throw std::runtime_error("Failed to get file size for: " + file_path);
    }
    mf.file_size = static_cast<size_t>(size.QuadPart);

    mf.h_map = CreateFileMappingA(mf.h_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mf.h_map || mf.h_map == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to create File Mapping for: " + file_path);
    }

    mf.mapped_data = reinterpret_cast<const uint8_t*>(MapViewOfFile(mf.h_map, FILE_MAP_READ, 0, 0, 0));
    if (!mf.mapped_data) {
        throw std::runtime_error("Failed to execute MapViewOfFile for: " + file_path);
    }

    // Parse the 8-byte header size
    uint64_t header_size = *reinterpret_cast<const uint64_t*>(mf.mapped_data);
    size_t header_bytes = static_cast<size_t>(header_size);

    std::string json_str(reinterpret_cast<const char*>(mf.mapped_data + 8), header_bytes);
    json header = json::parse(json_str);
    const uint8_t* base_data_ptr = mf.mapped_data + 8 + header_bytes;

    for (auto& [key, value] : header.items()) {
        if (key == "__metadata__") continue;

        TensorEntry entry;
        entry.name = key;
        entry.dtype = value["dtype"].get<std::string>();
        for (auto& dim : value["shape"]) entry.shape.push_back(dim.get<size_t>());

        auto offsets = value["data_offsets"];
        size_t start_offset = offsets[0].get<size_t>();
        size_t end_offset   = offsets[1].get<size_t>();

        entry.byte_size = end_offset - start_offset;
        entry.host_data_ptr = base_data_ptr + start_offset;

        registry[key] = entry;
    }

    mapped_files.push_back(mf);
    std::cout << "[Safetensors] Mapped slice: " << fs::path(file_path).filename().string() 
              << " (" << (mf.file_size / (1024 * 1024)) << " MB)\n";
#else
    throw std::runtime_error("Only Win32 API is currently supported.");
#endif
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