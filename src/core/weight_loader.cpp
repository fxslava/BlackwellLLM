#include "blackwell/weight_loader.h"
#include <cstdio>
#include <stdexcept>
#include <string>
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
        if (_fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0)
            throw std::runtime_error("StandardLoader: seek to offset " + std::to_string(offset) +
                                     " failed in " + filepath);

        std::vector<char> temp_buffer(size);
        const size_t bytes_read = fread(temp_buffer.data(), 1, size, f);
        if (bytes_read != size)
            throw std::runtime_error("StandardLoader: short read (" + std::to_string(bytes_read) +
                                     " of " + std::to_string(size) + " bytes) from " + filepath);

        const cudaError_t err = cudaMemcpy(d_ptr, temp_buffer.data(), size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("StandardLoader: cudaMemcpy to VRAM failed: " +
                                     std::string(cudaGetErrorString(err)));
    }
};

std::unique_ptr<IWeightLoader> create_standard_loader() {
    return std::make_unique<StandardLoader>();
}

// 🎯 Если DirectStorage выключен на уровне CMake, фабрика форсит стандартный лоадер
#ifndef USE_DIRECT_STORAGE
std::unique_ptr<IWeightLoader> IWeightLoader::create() {
    return create_standard_loader();
}
#endif