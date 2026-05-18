#pragma once
#include <string>
#include <memory>

class IWeightLoader {
public:
    virtual ~IWeightLoader() = default;
    
    // 🎯 Теперь передаем путь к конкретному файлу-шарду прямо сюда.
    // Лоадер сам внутри решит, открывал он его ранее или нужно открыть сейчас.
    virtual void load_to_vram(const std::string& filepath, size_t offset, size_t size, void* d_ptr) = 0;

    static std::unique_ptr<IWeightLoader> create();
};