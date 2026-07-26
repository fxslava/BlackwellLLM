#pragma once
// -----------------------------------------------------------------------------
// translator/cli_config.hpp — command-line parsing, dynamic model config.json
// parsing, and STRICT projector<->backbone dimension validation for
// audio_translator.
//
// The point of the validation is to fail BEFORE any CUDA allocation or engine
// construction: the Ultravox projector emits audio soft-tokens in the text
// model's embedding space, so its output width MUST equal the backbone's
// hidden_size or the splice at the <|audio|> placeholder (token 128256) is
// nonsense. A mismatch here (e.g. a 2048-wide 1B projector against a 4096-wide
// 8B backbone) would otherwise surface as a garbage decode or an out-of-bounds
// device access deep inside the kernels; we abort up front with a clear message
// instead (see validate_dimensions).
//
// nlohmann/json is included via angle brackets so the root's /external:W0
// quarantine keeps it out of the /W4 /WX budget.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "language_table.hpp"  // rt::kLanguages / language_index (--src-lang/--tgt-lang)

namespace rt {

// Resolved launch configuration (from argv, with an app-config fallback).
struct TranslatorArgs {
    std::string model_dir;             // --model-dir : HF checkpoint dir (backbone)
    std::string projector_path;        // --projector-path : Ultravox projector override (optional)
    std::string audio_head;            // --audio-head / --audio-tower-path : Ultravox
                                       //   checkpoint dir carrying audio_tower.* (Whisper
                                       //   encoder) + multi_modal_projector.* weights
    std::string data_dir = "data";     // positional : mel_filters.bin etc.
    std::string wav_path;              // --wav : headless one-shot inference on a WAV
    std::string features_path;         // --features : headless inference on a precomputed
                                       //   log-mel .bin [n_mels,3000] (bypasses WhisperDSP)
    int         max_new_tokens = 64;   // --max-new-tokens : decode cap (headless/live)
    bool        streaming = false;      // --streaming : sliding-window + dynamic overlap
                                        //   reconciliation prefill (headless --wav path)
    int         stream_window_ms = 2240;  // --stream-window-ms : sliding acoustic window
    int         stream_hop_ms    = 320;   // --stream-hop-ms    : new audio committed per hop
    int         stream_rewind_cap = 8;    // --stream-rewind-cap : max overlap tokens rewritten
                                          //   per hop (0 = append-only, no reconciliation)
    std::string src_lang = "Auto";        // --src-lang : forced audio language ("Auto" = LID)
    std::string tgt_lang = "Russian";     // --tgt-lang : forced translation target (the
                                          //   pre-dropdown shipping behaviour was Russian)
    std::string context_mode = "stateless";  // --context-mode : stateless | bounded
    int         history_budget_tokens = 256; // --history-budget : bounded-mode text budget
    bool        have_model_dir = false;   // whether model_dir was resolved at all
    bool        have_audio_head = false;  // whether audio_head was resolved at all
};

// Default checkpoint locations, used when neither a CLI flag nor a local
// config.json provides them (so a bare `audio_translator.exe` just runs).
inline constexpr const char* kDefaultModelDir  = "F:/AI/llama-3.1-8B-Instruct-AWQ-INT4";
inline constexpr const char* kDefaultAudioHead = "F:/AI/ultravox-v0_5-llama-3_1-8b";

// The backbone dimensions the validation and the projector configuration need,
// parsed straight from the checkpoint's config.json (NOT from the engine, which
// has not been constructed yet — that is the whole point).
struct BackboneConfig {
    int         hidden_size = 0;
    int         vocab_size = 0;
    std::string quant_method = "none";   // quantization_config.quant_method ("awq", ...)
    std::string rope_type = "default";   // rope_scaling.rope_type ("llama3", ...) or "default"
    bool        has_rope_scaling = false;
};

// The Ultravox projector geometry the frontend is configured with. output_dim
// (== text_hidden) is what must match the backbone hidden_size.
struct ProjectorParams {
    int output_dim = 0;        // text_hidden : projector Linear_2 output width
    int num_mel_bins = 128;    // Whisper large-v3-turbo log-mel bins
    int stack_factor = 8;      // frame stacking / compression
};

// The shipping projector is the 8B Ultravox variant (text_hidden 4096); used as
// the projector output width when no --projector-path config overrides it. See
// src/audio/README.md.
inline constexpr int kShippingProjectorTextHidden = 4096;

namespace detail {

inline bool read_json_file(const std::string& path, nlohmann::json& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    try {
        f >> out;
    } catch (const nlohmann::json::exception&) {
        return false;
    }
    return true;
}

}  // namespace detail

// Parse argv. Recognizes `--model-dir <path>`, `--projector-path <path>`, and a
// single positional data dir. When --model-dir is absent, falls back to a local
// app `config.json` ({"model_dir": "...", "projector_path": "..."}) in the CWD.
inline TranslatorArgs parse_cli(int argc, char** argv) {
    TranslatorArgs a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--model-dir") {
            a.model_dir = next("--model-dir");
        } else if (arg == "--projector-path") {
            a.projector_path = next("--projector-path");
        } else if (arg == "--audio-head" || arg == "--audio-tower-path") {
            a.audio_head = next(arg.c_str());
        } else if (arg == "--wav") {
            a.wav_path = next("--wav");
        } else if (arg == "--features") {
            a.features_path = next("--features");
        } else if (arg == "--max-new-tokens") {
            a.max_new_tokens = std::stoi(next("--max-new-tokens"));
        } else if (arg == "--streaming") {
            a.streaming = true;
        } else if (arg == "--stream-window-ms") {
            a.stream_window_ms = std::stoi(next("--stream-window-ms"));
        } else if (arg == "--stream-hop-ms") {
            a.stream_hop_ms = std::stoi(next("--stream-hop-ms"));
        } else if (arg == "--stream-rewind-cap") {
            a.stream_rewind_cap = std::stoi(next("--stream-rewind-cap"));
        } else if (arg == "--src-lang" || arg == "--tgt-lang") {
            // Strict: a typo must abort, not silently degrade to Auto.
            const std::string v = next(arg.c_str());
            if (language_index(v) < 0) {
                std::string known;
                for (int k = 0; k < kLanguageCount; ++k)
                    known += std::string(k ? ", " : "") + kLanguages[k];
                throw std::runtime_error("unknown language for " + arg + ": '" + v +
                                         "' (known: " + known + ")");
            }
            (arg == "--src-lang" ? a.src_lang : a.tgt_lang) = v;
        } else if (arg == "--context-mode") {
            a.context_mode = next("--context-mode");
            if (a.context_mode != "stateless" && a.context_mode != "bounded")
                throw std::runtime_error("--context-mode must be 'stateless' or 'bounded', got '" +
                                         a.context_mode + "'");
        } else if (arg == "--history-budget") {
            a.history_budget_tokens = std::stoi(next("--history-budget"));
            if (a.history_budget_tokens < 0)
                throw std::runtime_error("--history-budget must be >= 0");
        } else if (!arg.empty() && arg[0] != '-') {
            a.data_dir = arg;  // positional: data dir (mel_filters.bin)
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    // Fallback: consult a local app config.json for any value not given on argv.
    if (a.model_dir.empty() || a.projector_path.empty() || a.audio_head.empty()) {
        nlohmann::json j;
        if (detail::read_json_file("config.json", j)) {
            if (a.model_dir.empty())      a.model_dir      = j.value("model_dir", std::string{});
            if (a.projector_path.empty()) a.projector_path = j.value("projector_path", std::string{});
            if (a.audio_head.empty())     a.audio_head     = j.value("audio_head", std::string{});
        }
    }

    // Last resort: the shipping default checkpoint locations, so a bare launch
    // (no flags, no config.json) still resolves both the backbone and audio head.
    if (a.model_dir.empty())  a.model_dir  = kDefaultModelDir;
    if (a.audio_head.empty()) a.audio_head = kDefaultAudioHead;

    a.have_model_dir  = !a.model_dir.empty();
    a.have_audio_head = !a.audio_head.empty();
    return a;
}

// Parse <model_dir>/config.json for the backbone geometry. Throws with a clear
// message if the file is missing or hidden_size/vocab_size cannot be read.
inline BackboneConfig parse_backbone_config(const std::string& model_dir) {
    const std::string cfg_path = model_dir + "/config.json";
    nlohmann::json j;
    if (!detail::read_json_file(cfg_path, j))
        throw std::runtime_error("cannot read backbone config.json at " + cfg_path);

    BackboneConfig c;
    c.hidden_size = j.value("hidden_size", 0);
    c.vocab_size = j.value("vocab_size", 0);
    if (c.hidden_size <= 0)
        throw std::runtime_error("config.json is missing a positive hidden_size (" + cfg_path + ")");

    if (j.contains("quantization_config") && j["quantization_config"].is_object())
        c.quant_method = j["quantization_config"].value("quant_method", std::string("none"));

    if (j.contains("rope_scaling") && j["rope_scaling"].is_object()) {
        c.has_rope_scaling = true;
        c.rope_type = j["rope_scaling"].value("rope_type",
                       j["rope_scaling"].value("type", std::string("default")));
    }
    return c;
}

// Determine the projector's output width. When --projector-path names a dir (or
// file) carrying an Ultravox config.json, read text_hidden from it; otherwise
// use the shipping default. This is deliberately INDEPENDENT of the backbone so
// validate_dimensions can catch a genuine 1B-projector-vs-8B-backbone mismatch.
inline ProjectorParams resolve_projector_params(const std::string& projector_path) {
    ProjectorParams p;
    p.output_dim = kShippingProjectorTextHidden;

    if (!projector_path.empty()) {
        const bool looks_json = projector_path.size() >= 5 &&
            projector_path.compare(projector_path.size() - 5, 5, ".json") == 0;
        const std::string cfg_path = looks_json ? projector_path : projector_path + "/config.json";
        nlohmann::json j;
        if (detail::read_json_file(cfg_path, j)) {
            if (j.contains("text_hidden"))
                p.output_dim = j.value("text_hidden", p.output_dim);
            else if (j.contains("text_config") && j["text_config"].is_object())
                p.output_dim = j["text_config"].value("hidden_size", p.output_dim);
            else if (j.contains("hidden_size"))
                p.output_dim = j.value("hidden_size", p.output_dim);
        }
    }
    return p;
}

// STRICT dimension gate. Throws std::runtime_error with a descriptive message
// when the projector output width does not equal the backbone hidden_size. Call
// this BEFORE constructing the projector workspace or the BlackwellEngine.
inline void validate_dimensions(const ProjectorParams& proj, const BackboneConfig& backbone,
                                const std::string& model_dir) {
    if (proj.output_dim == backbone.hidden_size) return;
    throw std::runtime_error(
        "projector/backbone dimension mismatch — Ultravox projector output_dim=" +
        std::to_string(proj.output_dim) + " but backbone hidden_size=" +
        std::to_string(backbone.hidden_size) + " (from " + model_dir +
        "/config.json). The audio soft-tokens cannot be spliced into the text "
        "embedding stream: point --projector-path at a projector whose text_hidden "
        "matches the backbone, or pass a matching --model-dir.");
}

}  // namespace rt
