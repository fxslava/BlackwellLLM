#include "blackwell/tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

// 🎯 Секретная математика OpenAI/Tiktoken: перевод сырого байта в безопасный Unicode
int byte_to_unicode(unsigned char b) {
    if (b <= 32) return b + 256;
    if (b >= 127 && b <= 160) return b + 162;
    if (b == 173) return 323;
    return b;
}

// 🎯 Обратное преобразование: достаем оригинальный байт из Unicode
int unicode_to_byte(int u) {
    if (u >= 256 && u <= 288) return u - 256;
    if (u >= 289 && u <= 322) return u - 162;
    if (u == 323) return 173;
    return u;
}

LlamaTokenizer::LlamaTokenizer(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) throw std::runtime_error("Failed to open " + json_path);

    json data;
    file >> data;

    if (!data.contains("model") || !data["model"].contains("vocab")) {
        throw std::runtime_error("Invalid tokenizer.json format");
    }

    auto vocab = data["model"]["vocab"];
    for (auto& el : vocab.items()) {
        encoder_[el.key()] = el.value();
        decoder_[el.value()] = el.key();
    }
    std::cout << "[Tokenizer] Loaded " << encoder_.size() << " BPE tokens.\n";
}

LlamaTokenizer::~LlamaTokenizer() = default;

std::vector<int> LlamaTokenizer::encode(const std::string& text, bool add_bos) {
    std::vector<int> tokens;
    if (add_bos) tokens.push_back(128000); // BOS маркер Llama 3

    // 1. Стартовое состояние: каждый байт конвертируем в отдельный токен-строку
    std::vector<std::string> symbols;
    for (unsigned char b : text) {
        int u = byte_to_unicode(b);
        std::string s;
        if (u <= 0x7F) {
            s.push_back(static_cast<char>(u));
        } else {
            s.push_back(static_cast<char>(0xC0 | ((u >> 6) & 0x1F)));
            s.push_back(static_cast<char>(0x80 | (u & 0x3F)));
        }
        symbols.push_back(s);
    }

    // 2. Настоящий цикл BPE (Byte-Pair Encoding)
    while (symbols.size() > 1) {
        int best_idx = -1;
        int min_rank = 2147483647; // В Llama 3 чем меньше ID токена, тем раньше его нужно склеить
        std::string best_merged;

        // Ищем соседнюю пару с самым высоким приоритетом (минимальным ID в словаре)
        for (size_t i = 0; i < symbols.size() - 1; i++) {
            std::string pair = symbols[i] + symbols[i+1];
            auto it = encoder_.find(pair);
            if (it != encoder_.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = static_cast<int>(i);
                    best_merged = pair;
                }
            }
        }

        // Если ни одна из соседних пар не найдена в словаре — склеивать больше нечего
        if (best_idx == -1) {
            break;
        }

        // Склеиваем лучшую пару
        symbols[best_idx] = best_merged;
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    // 3. Собираем финальные ID токенов
    for (const auto& s : symbols) {
        auto it = encoder_.find(s);
        if (it != encoder_.end()) {
            tokens.push_back(it->second);
        } else {
            // Если каким-то чудом остался мусор, кладем фолбэк (обычно это никогда не срабатывает)
            tokens.push_back(0); 
        }
    }

    return tokens;
}

std::string LlamaTokenizer::decode(int token_id) {
    if (decoder_.find(token_id) == decoder_.end()) return "";

    std::string token_str = decoder_[token_id];

    // Глушим системные маркеры
    if (token_str.front() == '<' && token_str.back() == '>') {
        if (token_str == "<|end_of_text|>" || token_str == "<|eot_id|>") return "\n[EOS]\n";
        return ""; 
    }

    // 🎯 Декодируем маскированные байты обратно в чистый UTF-8 (Кириллица, пробелы, спецсимволы)
    std::string result;
    for (size_t i = 0; i < token_str.length(); ) {
        unsigned char c = token_str[i];
        if ((c & 0x80) == 0) {
            result.push_back(static_cast<char>(unicode_to_byte(c)));
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < token_str.length()) {
                unsigned char c2 = token_str[i + 1];
                int u = ((c & 0x1F) << 6) | (c2 & 0x3F);
                result.push_back(static_cast<char>(unicode_to_byte(u)));
            }
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            i += 3;
        } else {
            i += 1;
        }
    }
    return result;
}