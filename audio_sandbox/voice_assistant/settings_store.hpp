#pragma once
// -----------------------------------------------------------------------------
// settings_store.hpp — everything the voice_assistant UI is allowed to configure,
// and the only place it is persisted.
//
// WHY THIS EXISTS. The main window is a messenger: chat, a text box, a mic
// button, nothing else. Every knob that used to sit in the ImGui debug panel
// (checkpoint paths, VAD sensitivity, context policy, streaming mode) now lives
// behind the Settings modal, which means it needs somewhere durable to live
// between launches -- a CLI flag the user typed once is not a setting.
//
// PRECEDENCE. persisted settings < CLI flags. A flag the user TYPED on this
// launch always wins, and is not written back: a one-off `--simulated` must not
// silently become the permanent configuration. Everything else falls back to the
// persisted file, then to the built-in defaults below.
//
// THE RESTART SPLIT is the load-bearing part of this file. `requires_restart()`
// answers exactly one question -- "can this change be applied to the running
// engine, or does it need a fresh process?" -- and the UI is driven off that
// answer rather than a hand-maintained list in the JS:
//
//   LIVE     vad_threshold / context_mode / history_budget / live_streaming
//            -> plain atomics on the control + one speech_pipeline_* setter,
//               read at the next turn boundary. No engine work at all.
//   RESTART  model_dir / audio_head / projector_path / data_dir / device_id /
//            simulated / neural_vad / system_prompt / max_context / capture_mode
//            -> these choose what gets ALLOCATED at bring-up (5.3 GB of weights,
//               the frozen system prefix, the mel geometry). Mutating them under
//               a live engine is not a settings change, it is a different engine.
//
// nlohmann/json arrives through angle brackets so the root's /external:W0
// quarantine keeps it outside the /W4 /WX budget.
// -----------------------------------------------------------------------------
#include <direct.h>  // _mkdir (the settings dir; this target is Windows-only)

#include <cstdlib>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace rt {

// The persisted, user-editable configuration. Defaults here are the ones a fresh
// install boots with -- deliberately the simulated backend, so the app starts and
// is usable before any 5.3 GB checkpoint has been pointed at.
struct AssistantSettings {
    // ---- restart tier: what gets allocated at bring-up ----------------------
    std::string model_dir;             // backbone checkpoint dir ("" = none -> simulated)
    std::string audio_head;            // Ultravox audio head (encoder + projector)
    std::string projector_path;        // projector config override (optional)
    std::string data_dir = "data";     // mel_filters.bin etc.
    std::string system_prompt =
        "You are a concise voice assistant. Answer in one or two short sentences.";
    int  device_id  = 0;               // CUDA device
    int  max_context = 4096;           // KV ceiling (~256 KB/token at 8B geometry)
    bool simulated  = true;            // run the GPU-free stand-in backend
    bool neural_vad = true;            // Silero; false = the built-in RMS detector
    bool loopback_capture = false;     // capture system audio instead of the mic

    // ---- live tier: applied to the running engine at the next turn boundary --
    float vad_threshold = 0.5f;        // speech probability in [0.1, 0.9]
    std::string context_mode = "bounded";   // "stateless" | "bounded"
    int  history_budget_tokens = 256;  // bounded-mode text budget
    bool live_streaming = true;        // center-slice streaming (if armed at launch)

    // Does moving from `*this` to `other` need a fresh process? Compares ONLY the
    // restart-tier fields -- see the header preamble.
    [[nodiscard]] bool requires_restart(const AssistantSettings& other) const {
        return model_dir != other.model_dir || audio_head != other.audio_head ||
               projector_path != other.projector_path || data_dir != other.data_dir ||
               system_prompt != other.system_prompt || device_id != other.device_id ||
               max_context != other.max_context || simulated != other.simulated ||
               neural_vad != other.neural_vad || loopback_capture != other.loopback_capture;
    }
};

// Where the file lives: %LOCALAPPDATA%\BlackwellVoiceAssistant\settings.json.
// Falls back to a CWD-relative file when the variable is missing (a service
// account, a stripped environment) rather than failing to persist at all.
inline std::string settings_path() {
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && *local != '\0') {
        return std::string(local) + "\\BlackwellVoiceAssistant\\settings.json";
    }
    return "voice_assistant_settings.json";
}

// Coerce out-of-range values rather than trusting a hand-edited file. Applied on
// both load and save, so a bad value can neither reach the engine nor persist.
inline void clamp_settings(AssistantSettings& s) {
    if (s.vad_threshold < 0.1f) s.vad_threshold = 0.1f;
    if (s.vad_threshold > 0.9f) s.vad_threshold = 0.9f;
    if (s.max_context < 512) s.max_context = 512;
    if (s.history_budget_tokens < 0) s.history_budget_tokens = 0;
    if (s.device_id < 0) s.device_id = 0;
    if (s.context_mode != "stateless" && s.context_mode != "bounded") {
        s.context_mode = "bounded";
    }
    if (s.data_dir.empty()) s.data_dir = "data";
    // No checkpoint means there is nothing to load -- the simulated backend is
    // the only runnable configuration, so make the stored state say so instead
    // of failing at bring-up with a flag that contradicts the paths.
    if (s.model_dir.empty()) s.simulated = true;
}

// Best-effort load. A missing or corrupt file is NOT an error: the user gets the
// defaults and the next save rewrites the file. Losing settings must never stop
// the app from starting.
inline AssistantSettings load_settings() {
    AssistantSettings s;
    std::ifstream f(settings_path(), std::ios::binary);
    if (!f) return s;
    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::exception&) {
        return s;
    }
    s.model_dir      = j.value("model_dir", s.model_dir);
    s.audio_head     = j.value("audio_head", s.audio_head);
    s.projector_path = j.value("projector_path", s.projector_path);
    s.data_dir       = j.value("data_dir", s.data_dir);
    s.system_prompt  = j.value("system_prompt", s.system_prompt);
    s.device_id      = j.value("device_id", s.device_id);
    s.max_context    = j.value("max_context", s.max_context);
    s.simulated      = j.value("simulated", s.simulated);
    s.neural_vad     = j.value("neural_vad", s.neural_vad);
    s.loopback_capture = j.value("loopback_capture", s.loopback_capture);
    s.vad_threshold  = j.value("vad_threshold", s.vad_threshold);
    s.context_mode   = j.value("context_mode", s.context_mode);
    s.history_budget_tokens = j.value("history_budget_tokens", s.history_budget_tokens);
    s.live_streaming = j.value("live_streaming", s.live_streaming);
    clamp_settings(s);
    return s;
}

// Best-effort save. Returns false if the file could not be written (the caller
// surfaces it in the modal); the in-memory settings still apply for this session.
inline bool save_settings(const AssistantSettings& in) {
    AssistantSettings s = in;
    clamp_settings(s);

    const std::string path = settings_path();
    if (const std::size_t slash = path.find_last_of("\\/"); slash != std::string::npos) {
        // One level under %LOCALAPPDATA%, which always exists -- _mkdir is enough
        // and keeps <filesystem>/<shlobj.h> out of a header everything includes.
        // EEXIST is the expected outcome from the second launch on.
        (void)_mkdir(path.substr(0, slash).c_str());
    }

    nlohmann::json j;
    j["model_dir"]      = s.model_dir;
    j["audio_head"]     = s.audio_head;
    j["projector_path"] = s.projector_path;
    j["data_dir"]       = s.data_dir;
    j["system_prompt"]  = s.system_prompt;
    j["device_id"]      = s.device_id;
    j["max_context"]    = s.max_context;
    j["simulated"]      = s.simulated;
    j["neural_vad"]     = s.neural_vad;
    j["loopback_capture"] = s.loopback_capture;
    j["vad_threshold"]  = s.vad_threshold;
    j["context_mode"]   = s.context_mode;
    j["history_budget_tokens"] = s.history_budget_tokens;
    j["live_streaming"] = s.live_streaming;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f << j.dump(2);
    return static_cast<bool>(f);
}

}  // namespace rt
