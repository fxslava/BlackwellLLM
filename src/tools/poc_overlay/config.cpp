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

std::wstring ConfigStore::DefaultSpillPath() {
    wchar_t local[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) == 0) {
        return L"spill.bkv";  // last resort: next to the current directory
    }
    return std::wstring(local) + L"\\Blackwell\\spill.bkv";
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
        ReadShortcut(j, "cycleLanguageShortcut", config.cycleLanguageShortcut);
        if (j.contains("languagePairs") && j["languagePairs"].is_array()) {
            std::vector<LanguagePair> pairs;
            for (const auto& e : j["languagePairs"]) {
                LanguagePair p;
                p.label = FromUtf8(e.value("label", std::string()));
                p.target = FromUtf8(e.value("target", std::string()));
                if (!p.target.empty()) pairs.push_back(std::move(p));
            }
            if (!pairs.empty()) config.languagePairs = std::move(pairs);  // else keep defaults
        }
        config.activeLanguage = j.value("activeLanguage", config.activeLanguage);
        if (config.activeLanguage < 0 ||
            config.activeLanguage >= static_cast<int>(config.languagePairs.size())) {
            config.activeLanguage = 0;
        }
        config.modelPath = FromUtf8(j.value("modelPath", std::string()));
        config.contextSize = j.value("contextSize", config.contextSize);
        config.temperature = j.value("temperature", config.temperature);
        config.topP = j.value("topP", config.topP);
        config.maxTokens = j.value("maxTokens", config.maxTokens);
        config.captureGranularity = CaptureGranularityFromString(
            j.value("captureGranularity", std::string()), config.captureGranularity);
        config.idleTimerMs = j.value("idleTimerMs", config.idleTimerMs);
        if (config.idleTimerMs < 100) config.idleTimerMs = 100;  // sane floor
        config.activateOnStartup = j.value("activateOnStartup", config.activateOnStartup);
        // Lifecycle timeouts are stored in SECONDS; configs written before the
        // unit selector carried minutes under the *Min keys -- migrate those.
        if (j.contains("kvSpillTimeoutSec")) {
            config.kvSpillTimeoutSec = j.value("kvSpillTimeoutSec", config.kvSpillTimeoutSec);
        } else if (j.contains("kvSpillTimeoutMin")) {
            config.kvSpillTimeoutSec = j.value("kvSpillTimeoutMin", 10) * 60;
        }
        if (j.contains("hibernateTimeoutSec")) {
            config.hibernateTimeoutSec =
                j.value("hibernateTimeoutSec", config.hibernateTimeoutSec);
        } else if (j.contains("hibernateTimeoutMin")) {
            config.hibernateTimeoutSec = j.value("hibernateTimeoutMin", 30) * 60;
        }
        if (config.kvSpillTimeoutSec < 5) config.kvSpillTimeoutSec = 5;  // sane floor
        // Stage 2 never fires before stage 1: hibernation without the KV spill
        // would strand the radix tree in VRAM while the weights leave it.
        if (config.hibernateTimeoutSec < config.kvSpillTimeoutSec) {
            config.hibernateTimeoutSec = config.kvSpillTimeoutSec;
        }
        config.vramCacheBlocks = j.value("vramCacheBlocks", config.vramCacheBlocks);
        config.ramTierBlocks = j.value("ramTierBlocks", config.ramTierBlocks);
        config.diskSpillEnabled = j.value("diskSpillEnabled", config.diskSpillEnabled);
        config.diskSpillBlocks = j.value("diskSpillBlocks", config.diskSpillBlocks);
        config.spillFilePath = FromUtf8(j.value("spillFilePath", std::string()));
        if (config.vramCacheBlocks < 0) config.vramCacheBlocks = 0;
        if (config.ramTierBlocks < 0) config.ramTierBlocks = 0;
        if (config.diskSpillBlocks < 0) config.diskSpillBlocks = 0;
    } catch (const std::exception&) {
        // Malformed file -> fall back to whatever defaults survived.
    }
    return config;
}

bool ConfigStore::Save(const Config& config) {
    json j;
    j["activationShortcut"] = WriteShortcut(config.activationShortcut);
    j["commitShortcut"] = WriteShortcut(config.commitShortcut);
    j["cycleLanguageShortcut"] = WriteShortcut(config.cycleLanguageShortcut);
    {
        json arr = json::array();
        for (const LanguagePair& p : config.languagePairs) {
            arr.push_back({{"label", ToUtf8(p.label)}, {"target", ToUtf8(p.target)}});
        }
        j["languagePairs"] = std::move(arr);
    }
    j["activeLanguage"] = config.activeLanguage;
    j["modelPath"] = ToUtf8(config.modelPath);
    j["contextSize"] = config.contextSize;
    j["temperature"] = config.temperature;
    j["topP"] = config.topP;
    j["maxTokens"] = config.maxTokens;
    j["captureGranularity"] = ToString(config.captureGranularity);
    j["idleTimerMs"] = config.idleTimerMs;
    j["activateOnStartup"] = config.activateOnStartup;
    j["kvSpillTimeoutSec"] = config.kvSpillTimeoutSec;
    j["hibernateTimeoutSec"] = config.hibernateTimeoutSec;
    j["vramCacheBlocks"] = config.vramCacheBlocks;
    j["ramTierBlocks"] = config.ramTierBlocks;
    j["diskSpillEnabled"] = config.diskSpillEnabled;
    j["diskSpillBlocks"] = config.diskSpillBlocks;
    j["spillFilePath"] = ToUtf8(config.spillFilePath);

    std::ofstream out(DefaultPath());
    if (!out) {
        return false;
    }
    out << j.dump(2);
    return true;
}
