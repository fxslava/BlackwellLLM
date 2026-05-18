#ifdef USE_DIRECT_STORAGE

#include "weight_loader.h"
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <windows.h>
#include <d3d12.h>
#include <dstorage.h>
#include <cuda_runtime.h>
#include <comdef.h> 

class DirectStorageLoader : public IWeightLoader {
private:
    ID3D12Device* d3dDevice = nullptr;
    IDStorageFactory* factory = nullptr;
    IDStorageQueue* queue = nullptr;
    
    // 🎯 1. ЕДИНЫЙ ОБЪЕКТ СИНХРОНИЗАЦИИ
    ID3D12Fence* m_fence = nullptr;
    uint64_t m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;
    
    std::unordered_map<std::string, IDStorageFile*> m_file_cache;
    std::unordered_map<std::string, size_t> m_file_sizes; 
    
    // 🎯 2. ВОЗВРАЩАЕМ PINNED MEMORY ДЛЯ ПРЯМОГО DMA В ВИДЕОКАРТУ
    void* m_staging_buffer = nullptr;
    const size_t m_staging_size = 32 * 1024 * 1024; // 32 МБ

    std::wstring utf8_to_utf16(const std::string& str) {
        if (str.empty()) return L"";
        int size = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
        std::wstring wstr(size, 0);
        MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size);
        return wstr;
    }

    IDStorageFile* get_or_open_file(const std::string& path) {
        if (m_file_cache.find(path) != m_file_cache.end()) return m_file_cache[path];
        
        std::wstring wpath = utf8_to_utf16(path);
        IDStorageFile* file = nullptr;
        HRESULT hr = factory->OpenFile(wpath.c_str(), IID_PPV_ARGS(&file));
        
        if (FAILED(hr)) {
            _com_error err(hr);
            throw std::runtime_error("DirectStorage: Failed to open file " + path + " - " + err.ErrorMessage());
        }
        m_file_cache[path] = file;
        m_file_sizes[path] = std::filesystem::file_size(path);
        return file;
    }

public:
    DirectStorageLoader() {
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3dDevice)))) {
            throw std::runtime_error("DirectStorage: D3D12 Device creation failed");
        }
        if (FAILED(DStorageGetFactory(IID_PPV_ARGS(&factory)))) {
            throw std::runtime_error("DirectStorage: Factory initialization failed");
        }

        DSTORAGE_QUEUE_DESC queueDesc = {};
        queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        queueDesc.Capacity = 128;
        queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
        factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue));
        
        // 🎯 Инициализируем Fence только один раз при старте!
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (FAILED(d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
            throw std::runtime_error("DirectStorage: Failed to create Fence");
        }

        // 🎯 Выделяем закрепленную ОЗУ. Она в сотни раз быстрее VirtualAlloc при работе с CUDA!
        if (cudaHostAlloc(&m_staging_buffer, m_staging_size, cudaHostAllocDefault) != cudaSuccess) {
            throw std::runtime_error("DirectStorage: Failed to allocate Pinned RAM");
        }
    }

    ~DirectStorageLoader() override {
        if (m_staging_buffer) cudaFreeHost(m_staging_buffer);
        for (auto& [path, handle] : m_file_cache) {
            if (handle) handle->Release();
        }
        if (queue) queue->Release();
        if (factory) factory->Release();
        if (m_fence) m_fence->Release();
        if (m_fenceEvent) CloseHandle(m_fenceEvent);
        if (d3dDevice) d3dDevice->Release();
    }

    void load_to_vram(const std::string& filepath, size_t offset, size_t size, void* d_ptr) override {
        IDStorageFile* file = get_or_open_file(filepath);
        size_t total_file_size = m_file_sizes[filepath];

        const size_t SECTOR_SIZE = 4096;
        size_t bytes_remaining = size;
        size_t current_offset = offset;
        uint8_t* current_vram_ptr = static_cast<uint8_t*>(d_ptr);

        while (bytes_remaining > 0) {
            size_t aligned_offset = (current_offset / SECTOR_SIZE) * SECTOR_SIZE;
            size_t shift = current_offset - aligned_offset;

            size_t max_useful_data = m_staging_size - SECTOR_SIZE - shift;
            size_t chunk_data_size = min(bytes_remaining, max_useful_data);

            size_t raw_read_size = shift + chunk_data_size;
            size_t aligned_hardware_size = ((raw_read_size + SECTOR_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;

            if (aligned_offset + aligned_hardware_size > total_file_size) {
                FILE* f = fopen(filepath.c_str(), "rb");
                if (!f) throw std::runtime_error("DirectStorage Hybrid Fallback: Failed to open " + filepath);
                
                _fseeki64(f, current_offset, SEEK_SET);
                std::vector<uint8_t> tail_buffer(chunk_data_size);
                fread(tail_buffer.data(), 1, chunk_data_size, f);
                
                cudaMemcpy(current_vram_ptr, tail_buffer.data(), chunk_data_size, cudaMemcpyHostToDevice);
                fclose(f);
                
                bytes_remaining -= chunk_data_size;
                current_offset += chunk_data_size;
                current_vram_ptr += chunk_data_size;
                continue; 
            }

            DSTORAGE_REQUEST request = {};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY; 
            
            request.Source.File.Source = file;
            request.Source.File.Offset = aligned_offset;       
            request.Source.File.Size = aligned_hardware_size;  
            
            request.Destination.Memory.Buffer = m_staging_buffer; 
            request.Destination.Memory.Size = aligned_hardware_size;

            queue->EnqueueRequest(&request);

            // 🎯 УСКОРЕНИЕ: Просто увеличиваем счетчик и ждем, никакого CreateFence!
            m_fenceValue++;
            queue->EnqueueSignal(m_fence, m_fenceValue);
            queue->Submit();

            m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
            WaitForSingleObject(m_fenceEvent, INFINITE);

            DSTORAGE_ERROR_RECORD errorRecord;
            queue->RetrieveErrorRecord(&errorRecord);
            if (FAILED(errorRecord.FirstFailure.HResult)) {
                char hex_err[64];
                sprintf(hex_err, "0x%08X", errorRecord.FirstFailure.HResult);
                _com_error err(errorRecord.FirstFailure.HResult);
                
                std::string errMsg = "\n[DS ERROR] Read Failed!\n";
                errMsg += "File: " + filepath + "\n";
                errMsg += "HRESULT: " + std::string(hex_err) + "\n";
                errMsg += "Message: " + std::string(err.ErrorMessage());
                throw std::runtime_error(errMsg);
            }

            uint8_t* hardware_buffer_ptr = static_cast<uint8_t*>(m_staging_buffer) + shift;
            
            // 🎯 СВЕРХБЫСТРЫЙ DMA-ПЕРЕНОС (до 25 ГБ/с на PCIe 4.0)
            cudaMemcpy(current_vram_ptr, hardware_buffer_ptr, chunk_data_size, cudaMemcpyHostToDevice);

            bytes_remaining -= chunk_data_size;
            current_offset += chunk_data_size;
            current_vram_ptr += chunk_data_size;
        }
    }
};

std::unique_ptr<IWeightLoader> IWeightLoader::create() {
    return std::make_unique<DirectStorageLoader>();
}
#endif