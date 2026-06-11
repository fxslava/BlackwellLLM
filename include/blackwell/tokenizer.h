#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class LlamaTokenizer {
public:
    // Загружает конфигурацию напрямую из tokenizer.json
    LlamaTokenizer(const std::string& json_path);
    ~LlamaTokenizer();

    // Быстрый поиск токенов
    std::vector<int> encode(const std::string& text, bool add_bos = true);
    std::string decode(int token_id);
    std::string decode(const std::vector<int>& token_ids);

private:
    std::unordered_map<std::string, int> encoder_;
    std::unordered_map<int, std::string> decoder_;
};