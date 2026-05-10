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
    const uint8_t* host_data_ptr; // Прямой Zero-Copy указатель
};

class SafetensorsLoader {
public:
    // Конструктор теперь принимает путь к .safetensors ИЛИ к .index.json
    SafetensorsLoader(const std::string& index_or_file_path);
    ~SafetensorsLoader();

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;

    const TensorEntry& get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;
    std::vector<std::string> list_tensors() const;

private:
    void load_single_file(const std::string& file_path);
    void load_index_file(const std::string& index_path);

    // Структура для удержания системных ресурсов каждого смаппленного файла
    struct MappedFile {
        void* h_file = nullptr;
        void* h_map = nullptr;
        const uint8_t* mapped_data = nullptr;
        size_t file_size = 0;
    };

    std::vector<MappedFile> mapped_files;
    std::unordered_map<std::string, TensorEntry> registry;
};