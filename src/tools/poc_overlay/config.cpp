#include "config.h"

#include <nlohmann/json.hpp>

#include <fstream>

using nlohmann::json;

namespace {
void ReadShortcut(const json& parent, const char* key, Shortcut& out) {
    if (!parent.contains(key) || !parent[key].is_object()) {
        return;
    }
    const json& j = parent[key];
    out.modifiers = j.value("modifiers", out.modifiers);
    out.vk = j.value("vk", out.vk);
}

json WriteShortcut(const Shortcut& s) {
    return json{{"modifiers", s.modifiers}, {"vk", s.vk}};
}
}  // namespace

std::wstring ConfigStore::DefaultPath() {
    wchar_t exePath[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring path(exePath);
    const size_t slash = path.find_last_of(L"\\/");
    if (slash != std::wstring::npos) {
        path.resize(slash + 1);
    }
    path += L"config.json";
    return path;
}

Config ConfigStore::Load() {
    Config config;
    std::ifstream in(DefaultPath());  // MSVC ifstream accepts a wide path
    if (!in) {
        return config;  // no file yet -> defaults
    }
    try {
        json j;
        in >> j;
        ReadShortcut(j, "activationShortcut", config.activationShortcut);
        ReadShortcut(j, "commitShortcut", config.commitShortcut);
        config.modelPath = FromUtf8(j.value("modelPath", std::string()));
        config.contextSize = j.value("contextSize", config.contextSize);
        config.temperature = j.value("temperature", config.temperature);
        config.topP = j.value("topP", config.topP);
        config.maxTokens = j.value("maxTokens", config.maxTokens);
    } catch (const std::exception&) {
        // Malformed file -> fall back to whatever defaults survived.
    }
    return config;
}

bool ConfigStore::Save(const Config& config) {
    json j;
    j["activationShortcut"] = WriteShortcut(config.activationShortcut);
    j["commitShortcut"] = WriteShortcut(config.commitShortcut);
    j["modelPath"] = ToUtf8(config.modelPath);
    j["contextSize"] = config.contextSize;
    j["temperature"] = config.temperature;
    j["topP"] = config.topP;
    j["maxTokens"] = config.maxTokens;

    std::ofstream out(DefaultPath());
    if (!out) {
        return false;
    }
    out << j.dump(2);
    return true;
}
