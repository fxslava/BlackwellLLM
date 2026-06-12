#pragma once
#include <string>
#include <memory>

class IWeightLoader {
public:
    virtual ~IWeightLoader() = default;

    // 🎯 Теперь передаем путь к конкретному файлу-шарду прямо сюда.
    // Лоадер сам внутри решит, открывал он его ранее или нужно открыть сейчас.
    //
    // Batched loaders may DEFER the actual transfer: after this call the bytes
    // are only guaranteed to be in VRAM once flush() returns. d_ptr must stay
    // a valid device pointer until then.
    virtual void load_to_vram(const std::string& filepath, size_t offset, size_t size, void* d_ptr) = 0;

    // Completion barrier for every load_to_vram() issued so far. Synchronous
    // loaders keep the default no-op.
    virtual void flush() {}

    static std::unique_ptr<IWeightLoader> create();
};

// Always-available synchronous fread/cudaMemcpy loader; used directly when
// USE_DIRECT_STORAGE is off and as the runtime fallback when DirectStorage
// initialization fails.
std::unique_ptr<IWeightLoader> create_standard_loader();