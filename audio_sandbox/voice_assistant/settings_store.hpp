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
// ONE FIELD LIST, FIVE CONSUMERS. visit_fields() below is the single declaration
// of what a setting is. Loading, saving, the JSON the page is seeded with, the
// JSON it saves back, and the restart question are all LOOPS OVER IT -- none of
// them carries its own list. That is not tidiness: the previous shape had the
// same field spelled out in four places plus a `data-restart` attribute in the
// markup, and adding a knob meant editing five files in agreement. The failure
// mode was silent (a setting that persists but never loads, or one the restart
// banner does not know about), which is the kind that ships.
//
// THE TIER SPLIT is the load-bearing part. Tier answers exactly one question --
// "can this change be applied to the running engine, or does it need a fresh
// process?" -- and the UI is driven off that answer, which is now PUSHED to the
// page (restart_fields()) rather than re-declared in the HTML:
//
//   LIVE     sampling / reply cap / VAD / cadence / context policy / hotkeys
//            -> plain atomics on the control + the speech_pipeline_* setters,
//               read at the next VAD block or turn boundary. No engine work.
//   LIVE*    system_prompt -> the ONE live setting that costs engine work: its KV
//            is the frozen prefix every turn is built on, so changing it is a
//            cache REBUILD marshaled onto the engine thread, not an atomic store.
//            See rebuild_system_prompt(). Still live -- the user does not have to
//            restart the app to change how the assistant behaves.
//   RESTART  model_dir / audio_head / projector_path / data_dir / device_id /
//            simulated / neural_vad / max_context / capture_mode
//            -> these choose what gets ALLOCATED at bring-up (5.3 GB of weights,
//               the mel geometry). Mutating them under a live engine is not a
//               settings change, it is a different engine.
//
// nlohmann/json arrives through angle brackets so the root's /external:W0
// quarantine keeps it outside the /W4 /WX budget.
// -----------------------------------------------------------------------------
#include <direct.h>  // _mkdir (the settings dir; this target is Windows-only)

#include <cstdlib>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

namespace rt {

// Can this change be applied to the running engine? See the header preamble.
enum class Tier { Live, Restart };

// The persisted, user-editable configuration. Defaults here are the ones a fresh
// install boots with -- deliberately the simulated backend, so the app starts and
// is usable before any 5.3 GB checkpoint has been pointed at.
//
// ADDING A FIELD: declare it here AND add one line to visit_fields(). Nothing
// else in this file needs to change, and nothing outside it may keep its own list.
struct AssistantSettings {
    // ---- Tab 1: inference & paths -------------------------------------------
    std::string model_dir;             // backbone checkpoint dir ("" = none -> simulated)
    std::string audio_head;            // Ultravox audio head (encoder + projector)
    std::string projector_path;        // projector config override (optional)
    std::string data_dir = "data";     // mel_filters.bin etc.
    int  device_id  = 0;               // CUDA device
    // KV ceiling. THE ONE MEMORY KNOB WITH REAL HEADROOM LEFT on a 12 GB card,
    // so it is budgeted rather than set to the largest round number that fits.
    //
    // This app runs the paged (bf16) cache -- isolated_sessions forces it -- so
    // the pool is 128 KB/token per sequence, over paged_branch_factor = 2
    // sequences: 256 KB/token effective. 4096 tokens is ~1.05 GB; 2048 is
    // ~0.52 GB. The rest of the stack measures ~5.3 GB backbone + ~2.2 GB TTS +
    // the audio head + ~1 GB desktop, which puts 4096 over the ~10.5 GB line
    // where WDDM starts paging to system RAM -- and WDDM paging does not fail,
    // it just makes everything 10x slower with no diagnostic. 2048 is the
    // largest value that leaves headroom for all four consumers at once.
    //
    // Raise it if you drop the audio head or the TTS model; watch the [vram]
    // lines at startup, which report what was ACTUALLY committed at each stage.
    int  max_context = 2048;
    bool simulated  = true;            // run the GPU-free stand-in backend

    // Greedy by default, and that is the shipping behaviour on purpose: the local
    // model transcribes and extracts intent, where sampling buys nothing and
    // costs fidelity. The knobs exist for the text path used as a chat assistant.
    float temperature = 0.0f;          // 0 = greedy
    float top_p       = 1.0f;          // 1 = no nucleus truncation
    // Per-turn generated-token ceiling. NOT cosmetic: a turn that hits it
    // terminates as TokenCap and is therefore NEVER dispatched to the cloud, so
    // this value decides whether long answers get through at all.
    int   max_new_tokens = 256;

    // ---- speech output (F5-TTS) ---------------------------------------------
    // Empty ckpt dir = speech output off. These defaults point at a real layout
    // on the development machine so the feature is discoverable; on any other
    // machine they simply fail to load and the app runs text-only, printing why.
    //
    // WHAT ckpt_dir MUST CONTAIN: the two EXPORTED ONNX graphs, f5_tts_dit.onnx
    // and f5_tts_vocoder.onnx -- not the .safetensors checkpoint they were
    // exported FROM. Point it at a folder of PyTorch weights and startup fails
    // with "missing DiT graph"; run scripts/export_f5_tts_onnx.py first.
    std::string tts_ckpt_dir   = "D:\\TTF";
    std::string tts_vocab_path = "D:\\TTF\\vocab.txt";
    // Reference clip. MUST be 24 kHz mono; the loader refuses other rates rather
    // than resampling, because a wrong-rate reference does not fail -- it clones
    // a voice pitched by the rate ratio, which sounds like a different person.
    std::string tts_ref_audio  = "D:\\TTF\\ref_audio.wav";
    // NOT optional when TTS is on, and NOT a label: F5 conditions on an
    // (audio, TEXT) pair and treats generation as infilling, so this must be the
    // literal transcript of tts_ref_audio. The default below is a PLACEHOLDER --
    // leaving it produces confident nonsense, and main.cpp warns at startup if it
    // is still set.
    std::string tts_ref_text   = "Текст вашего референсного аудипоклипа";
    int  tts_nfe_step = 16;            // solver steps: the latency/quality dial

    // ---- chunking (time-to-first-audio vs. prosody continuity) --------------
    // F5 is not streaming: TTFB equals full synthesis time for whatever text it
    // is given, so the only latency lever is giving it less at once. Each split
    // re-conditions the voice on the reference rather than on what was just
    // spoken, so splitting harder is faster AND flatter.
    bool tts_split_on_commas = true;   // clause-level splits; off = sentences only
    int  tts_min_chunk_chars = 20;     // [5, 50]   stops "Да," becoming an utterance
    int  tts_max_chunk_chars = 150;    // [50, 300] forces a split at the next space

    // Gate the mic while the speaker is live. ON by default: there is no echo
    // canceller in this build, so on open speakers the assistant otherwise hears
    // itself, scores it as speech, and barges in on its own answer. The cost is
    // real and is stated in the UI -- barge-in-while-speaking stops working, and
    // the cancel button becomes the only way to interrupt.
    bool tts_mic_gate = true;

    // ---- Tab 2: audio & speculative decoding --------------------------------
    bool neural_vad = true;            // Silero; false = the built-in RMS detector
    bool loopback_capture = false;     // capture system audio instead of the mic
    float vad_threshold = 0.5f;        // speech probability in [0.1, 0.9]
    // How much stable silence closes an utterance. The single biggest lever on
    // perceived responsiveness, and the one most worth a user's time: too short
    // cuts people off mid-sentence, too long makes the assistant feel deaf.
    int  silence_hangover_ms = 800;
    // How much audio BEFORE the detector fired is kept. The hangover above says
    // where a turn ends; this says where it begins. A VAD needs 45-110 ms to
    // decide a block is speech, and the flush at speech onset used to discard
    // exactly that window -- which is what clipped short first syllables. 0 =
    // flush everything (the old behaviour).
    int  pre_roll_ms = 250;
    // Speculative warm-prefill cadence: how often audio buffered so far is
    // prefilled while the user is still speaking. Trades GPU budget against how
    // much prefill is left at the commit boundary. 0 disables warming.
    int  warm_prefill_interval_ms = 320;
    std::string context_mode = "bounded";   // "stateless" | "bounded"
    int  history_budget_tokens = 256;  // bounded-mode text budget (the KV a turn may retain)
    bool live_streaming = true;        // center-slice streaming (if armed at launch)

    // ---- Tab 3: hotkeys & triggers ------------------------------------------
    // "Ctrl+Alt+Space" style chords; "" disables one. Parsed by parse_hotkey()
    // in assistant_window.cpp and registered with RegisterHotKey on the UI thread.
    std::string hotkey_talk   = "Ctrl+Alt+Space";  // listen on/off, or hold-to-talk
    std::string hotkey_cancel = "Ctrl+Alt+X";      // interrupt the turn in flight
    std::string hotkey_show   = "Ctrl+Alt+A";      // bring the window to the front
    // Hold-to-talk rather than toggle. WM_HOTKEY has no key-up event, so hold
    // mode is driven by polling the chord while it is held -- see the window.
    bool hotkey_push_to_talk = false;

    // Answer locally instead of calling out. Swaps the transport under the
    // dispatcher (see local_transport.hpp); everything above it is unchanged, so
    // the same commit rule gates a local answer and a billed one.
    bool local_inference = false;

    // ---- Tab 4: the TWO prompts ---------------------------------------------
    // They are different things and were previously conflated, which is why the
    // model translated when the user wanted a transcript -- the only editable
    // prompt was the persona, and the persona does not decide the audio task.
    //
    // PERSONA: who the assistant IS. Prefilled ONCE and reused by every turn, so
    // editing it is a KV cache rebuild (live, but it costs engine work).
    std::string system_prompt =
        "You are a concise voice assistant. Answer in one or two short sentences.";
    // AUDIO TASK: what to DO with the next chunk of speech. Lives in the per-turn
    // user block, so editing it is free and applies on the very next utterance.
    // Empty = the engine's generated phrase, which follows the task toggle below.
    std::string audio_task_prompt = "Transcribe the following speech exactly as spoken: ";
    // The language being SPOKEN, free text ("Russian", "English"); "" = let the
    // model identify it. Not cosmetic and not a preference: the audio tower
    // mistakes acoustically adjacent languages on short utterances (Russian heard
    // as Polish, transcribed in Latin letters), and nothing downstream can undo
    // that -- the wrong transcript is what gets committed and answered. Naming
    // the language removes the guess. It is folded into the transcription
    // session's frozen prefix AND the per-turn instruction, so editing it is a
    // (cheap) prefix rebuild -- see RealEngineControl::set_speech_language.
    std::string speech_language;
    // Which output tags a turn must produce: "transcribe" | "translate" | "both".
    // DEFAULT IS TRANSCRIBE-ONLY. It used to be both, which is why every reply
    // carried a [Translation] nobody asked for -- and asking an 8B backbone for
    // two outputs instead of one roughly doubles the tokens a turn must generate.
    std::string speech_task = "transcribe";
};

// THE field list. Every other function in this file is a loop over it.
// `S` is AssistantSettings or const AssistantSettings, which is what lets the
// same list serve both the readers (save / push) and the writers (load / parse).
template <class S, class Fn>
void visit_fields(S& s, Fn&& f) {
    // ---- Tab 1 -------------------------------------------------------------
    f("model_dir",                s.model_dir,                Tier::Restart);
    f("audio_head",               s.audio_head,               Tier::Restart);
    f("projector_path",           s.projector_path,           Tier::Restart);
    f("data_dir",                 s.data_dir,                 Tier::Restart);
    f("device_id",                s.device_id,                Tier::Restart);
    f("max_context",              s.max_context,              Tier::Restart);
    f("simulated",                s.simulated,                Tier::Restart);
    f("temperature",              s.temperature,              Tier::Live);
    f("top_p",                    s.top_p,                    Tier::Live);
    f("max_new_tokens",           s.max_new_tokens,           Tier::Live);
    // ---- Tab 2 -------------------------------------------------------------
    f("neural_vad",               s.neural_vad,               Tier::Restart);
    f("loopback_capture",         s.loopback_capture,         Tier::Restart);
    f("vad_threshold",            s.vad_threshold,            Tier::Live);
    f("silence_hangover_ms",      s.silence_hangover_ms,      Tier::Live);
    f("pre_roll_ms",              s.pre_roll_ms,              Tier::Live);
    f("warm_prefill_interval_ms", s.warm_prefill_interval_ms, Tier::Live);
    f("context_mode",             s.context_mode,             Tier::Live);
    f("history_budget_tokens",    s.history_budget_tokens,    Tier::Live);
    f("live_streaming",           s.live_streaming,           Tier::Live);
    // ---- speech output ------------------------------------------------------
    // All Restart tier: the engine loads a ~1.3 GB graph onto the GPU and the
    // reference mel is uploaded once at startup, so none of these can be swapped
    // under a running session without tearing the whole stack down. Changing the
    // reference voice live is a real feature, but it is a rebuild, not a store.
    f("tts_ckpt_dir",             s.tts_ckpt_dir,             Tier::Restart);
    f("tts_vocab_path",           s.tts_vocab_path,           Tier::Restart);
    f("tts_ref_audio",            s.tts_ref_audio,            Tier::Restart);
    f("tts_ref_text",             s.tts_ref_text,             Tier::Restart);
    f("tts_nfe_step",             s.tts_nfe_step,             Tier::Restart);
    f("tts_split_on_commas",      s.tts_split_on_commas,      Tier::Restart);
    f("tts_min_chunk_chars",      s.tts_min_chunk_chars,      Tier::Restart);
    f("tts_max_chunk_chars",      s.tts_max_chunk_chars,      Tier::Restart);
    f("tts_mic_gate",             s.tts_mic_gate,             Tier::Restart);
    // ---- Tab 3 -------------------------------------------------------------
    f("hotkey_talk",              s.hotkey_talk,              Tier::Live);
    f("hotkey_cancel",            s.hotkey_cancel,            Tier::Live);
    f("hotkey_show",              s.hotkey_show,              Tier::Live);
    f("hotkey_push_to_talk",      s.hotkey_push_to_talk,      Tier::Live);
    f("local_inference",          s.local_inference,          Tier::Live);
    // ---- Tab 4 -------------------------------------------------------------
    f("system_prompt",            s.system_prompt,            Tier::Live);
    f("audio_task_prompt",        s.audio_task_prompt,        Tier::Live);
    f("speech_language",          s.speech_language,          Tier::Live);
    f("speech_task",              s.speech_task,              Tier::Live);
}

// Coerce out-of-range values rather than trusting a hand-edited file or a page
// that got creative. Applied on load, on save, and on every value arriving from
// the UI, so a bad value can neither reach the engine nor persist.
//
// The ranges are the UI's ranges: a value the modal can produce is by
// construction a value this function leaves alone.
inline void clamp_settings(AssistantSettings& s) {
    auto clamp_int = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

    if (s.vad_threshold < 0.1f) s.vad_threshold = 0.1f;
    if (s.vad_threshold > 0.9f) s.vad_threshold = 0.9f;
    if (s.temperature < 0.0f) s.temperature = 0.0f;
    if (s.temperature > 2.0f) s.temperature = 2.0f;
    // top_p == 0 would leave the nucleus empty and the sampler with nothing to
    // pick from; 1.0 is "no truncation", which is the correct reading of "off".
    if (s.top_p <= 0.0f || s.top_p > 1.0f) s.top_p = 1.0f;
    s.max_new_tokens = clamp_int(s.max_new_tokens, 16, 4096);
    s.max_context = clamp_int(s.max_context, 512, 131072);
    // 0 hangover disables auto-commit entirely (push-to-talk only), which is a
    // legitimate configuration -- so the floor is 0, not a minimum delay.
    s.silence_hangover_ms = clamp_int(s.silence_hangover_ms, 0, 5000);
    // The bounds are ContinuousStreamingConfig::kMin/kMaxPreRollMs, restated here
    // rather than included: this header is the UI's contract and must not pull the
    // segmenter in. 0 is legal and means "flush the whole ring", the pre-fix
    // behaviour; past ~1 s the retained head stops being an onset and starts being
    // the previous sentence.
    s.pre_roll_ms = clamp_int(s.pre_roll_ms, 0, 1000);
    // 0 disables speculative warming; above that, anything under a VAD block is
    // indistinguishable from every-block warming.
    s.warm_prefill_interval_ms = clamp_int(s.warm_prefill_interval_ms, 0, 5000);
    if (s.history_budget_tokens < 0) s.history_budget_tokens = 0;
    if (s.device_id < 0) s.device_id = 0;
    if (s.context_mode != "stateless" && s.context_mode != "bounded") {
        s.context_mode = "bounded";
    }
    if (s.speech_task != "transcribe" && s.speech_task != "translate" &&
        s.speech_task != "both") {
        s.speech_task = "transcribe";
    }
    // "" means Auto, so whitespace has to mean Auto too: a stray space would
    // otherwise be a non-empty language that reads "The spoken language is  ."
    // to the model, and would keep re-triggering the prefix rebuild diff.
    if (const std::size_t first = s.speech_language.find_first_not_of(" \t\r\n");
        first == std::string::npos) {
        s.speech_language.clear();
    } else {
        s.speech_language = s.speech_language.substr(
            first, s.speech_language.find_last_not_of(" \t\r\n") - first + 1);
    }
    if (s.data_dir.empty()) s.data_dir = "data";
    // No checkpoint means there is nothing to load -- the simulated backend is
    // the only runnable configuration, so make the stored state say so instead
    // of failing at bring-up with a flag that contradicts the paths.
    if (s.model_dir.empty()) s.simulated = true;
}

// Serialize every field (both tiers). Used for the persisted file AND for the
// payload the page is seeded with -- deliberately the same function, so what the
// UI edits is exactly what is on disk.
inline nlohmann::json to_json(const AssistantSettings& s) {
    nlohmann::json j;
    visit_fields(s, [&](const char* key, const auto& value, Tier) { j[key] = value; });
    return j;
}

// Read every field present in `j`, leaving the rest at whatever `s` already
// holds. That fallback is what makes both an OLD settings file (missing the
// fields added since) and a PARTIAL payload from the page safe to apply.
inline void from_json(const nlohmann::json& j, AssistantSettings& s) {
    visit_fields(s, [&](const char* key, auto& value, Tier) {
        using T = std::decay_t<decltype(value)>;
        if (!j.contains(key)) return;
        try {
            value = j.at(key).get<T>();
        } catch (const nlohmann::json::exception&) {
            // A field of the wrong type keeps its current value rather than
            // taking the whole load down: one bad key must not cost the user
            // every other setting they have.
        }
    });
    clamp_settings(s);
}

// The restart-tier field names, pushed to the page so the "needs restart" banner
// is computed from THIS list rather than a duplicate set of markup attributes.
inline std::vector<std::string> restart_fields() {
    AssistantSettings probe;
    std::vector<std::string> keys;
    visit_fields(probe, [&](const char* key, const auto&, Tier tier) {
        if (tier == Tier::Restart) keys.emplace_back(key);
    });
    return keys;
}

// A stable string of just the restart-tier fields. Two settings needing the same
// engine produce the same signature, so THE restart question is one comparison.
// (nlohmann orders object keys, so the dump is deterministic.)
inline std::string restart_signature(const AssistantSettings& s) {
    nlohmann::json j;
    visit_fields(s, [&](const char* key, const auto& value, Tier tier) {
        if (tier == Tier::Restart) j[key] = value;
    });
    return j.dump();
}

// Does moving from `a` to `b` need a fresh process?
inline bool requires_restart(const AssistantSettings& a, const AssistantSettings& b) {
    return restart_signature(a) != restart_signature(b);
}

// Where the file lives: %LOCALAPPDATA%\BlackwellVoiceAssistant\settings.json.
// Falls back to a CWD-relative file when the variable is missing (a service
// account, a stripped environment) rather than failing to persist at all.
inline std::string settings_path() {
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && *local != '\0') {
        return std::string(local) + "\\BlackwellVoiceAssistant\\settings.json";
    }
    return "voice_assistant_settings.json";
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
    from_json(j, s);   // clamps
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

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f << to_json(s).dump(2);
    return static_cast<bool>(f);
}

}  // namespace rt
