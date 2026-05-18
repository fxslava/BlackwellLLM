#include "weight_loader.h"
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <cuda_runtime.h>
#include <vector>

class StandardLoader : public IWeightLoader {
private:
    std::unordered_map<std::string, FILE*> m_file_cache;

    FILE* get_or_open_file(const std::string& path) {
        if (m_file_cache.find(path) != m_file_cache.end()) {
            return m_file_cache[path];
        }
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("StandardLoader: Cannot open " + path);
        m_file_cache[path] = f;
        return f;
    }

public:
    ~StandardLoader() override {
        for (auto& [path, f] : m_file_cache) {
            if (f) fclose(f);
        }
    }

    void load_to_vram(const std::string& filepath, size_t offset, size_t size, void* d_ptr) override {
        FILE* f = get_or_open_file(filepath);
        _fseeki64(f, offset, SEEK_SET);
        
        std::vector<char> temp_buffer(size);
        fread(temp_buffer.data(), 1, size, f);
        cudaMemcpy(d_ptr, temp_buffer.data(), size, cudaMemcpyHostToDevice);
    }
};

// 🎯 Если DirectStorage выключен на уровне CMake, фабрика форсит стандартный лоадер
#ifndef USE_DIRECT_STORAGE
std::unique_ptr<IWeightLoader> IWeightLoader::create() {
    return std::make_unique<StandardLoader>();
}
#endif