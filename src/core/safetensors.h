#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

struct TensorEntry {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    size_t byte_size;
    std::string file_path; // 🎯 Путь к файлу-шарду
    size_t file_offset;    // 🎯 Абсолютное смещение на диске
};

class SafetensorsLoader {
public:
    // Конструктор теперь принимает путь к .safetensors ИЛИ к .index.json
    explicit SafetensorsLoader(const std::string& index_or_file_path);

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;

    const TensorEntry& get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;
    std::vector<std::string> list_tensors() const;

private:
    void load_single_file(const std::string& file_path);
    void load_index_file(const std::string& index_path);

    std::unordered_map<std::string, TensorEntry> registry;
};