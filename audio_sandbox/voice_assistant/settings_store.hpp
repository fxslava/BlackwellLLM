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
// AudioHotReload is a THIRD tier and not a flavour of Restart, because the thing
// it protects is expensive and unrelated: restarting the app to change an audio
// endpoint would tear down 5.3 GB of AWQ weights, the KV pool and a 1.3 GB ONNX
// session to swap a headphone jack. An ma_device is a handle over a WASAPI
// endpoint -- closing and reopening one costs milliseconds and touches no VRAM.
//
// It is not Live either: Live means "store an atomic and the next block reads
// it", and these need a device to be closed and reopened. So they get their own
// tier, and restart_fields() (which drives the page's banner) deliberately does
// NOT include them -- changing an endpoint must not tell the user to restart.
enum class Tier { Live, Restart, AudioHotReload };

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

    // ---- WHICH SPEECH-TO-TEXT PIPELINE RUNS ---------------------------------
    // "ultravox_legacy"  the end-to-end multimodal path: mic -> log-mel ->
    //                    Whisper encoder -> Ultravox projector -> audio
    //                    soft-tokens spliced into the backbone's KV, decoded on
    //                    the ephemeral transcription sequence. The proven path.
    // "whisper_cascade"  mic -> VAD-bounded utterance -> whisper.cpp (GGML) ->
    //                    UTF-8 -> published as an intent. No audio ever reaches
    //                    the backbone, and the AUDIO HEAD IS NOT LOADED.
    //
    // RESTART TIER, and structurally so: this decides which ISpeechMode object is
    // constructed AND whether ~2 GB of Ultravox audio tower is allocated at
    // bring-up. It is not a mode you can flip under a live engine, it is a
    // different engine configuration.
    //
    // THE TWO ARE MUTUALLY EXCLUSIVE ON PURPOSE. They are two answers to one
    // question ("what did the user say?"), they want the VRAM the other one is
    // holding, and running both would double the transcription cost of every
    // utterance to produce a second answer nothing consumes.
    //
    // Defaults to the LEGACY path. Cascade is opt-in until it has been validated
    // in production, which is the whole point of the flag -- and an unknown value
    // clamps back to legacy rather than to the newer path (see clamp_settings).
    // "simultaneous" (ISpeechMode's Mode B) parses but is REFUSED at bring-up in
    // this binary -- see clamp_settings and bring_up_speech_mode.
    std::string pipeline_mode = "ultravox_legacy";   // | "whisper_cascade" | "simultaneous"

    // The GGML Whisper model the cascade transcribes with (e.g. a turbo-class
    // Whisper-Turbo-Platinum-F16.bin). Empty = fall back to the path CMake
    // configured at build time, then to a copy sitting next to the exe -- the
    // same three-step resolution the Silero model uses. Cascade mode refuses to
    // arm without a model and says so; it never silently degrades to legacy
    // behaviour on a path typo.
    std::string whisper_model_path;
    // The language being spoken, as an ISO-639-1 code ("ru", "en"); "" = let
    // Whisper identify it. NOT cosmetic, and the reasoning is the same one
    // speech_language carries below: on a SHORT utterance an acoustically
    // adjacent language is a live failure mode (Russian heard as Bulgarian and
    // transcribed accordingly), and nothing downstream can undo a wrong
    // transcript -- it is what gets committed and answered.
    std::string whisper_language;
    // CPU worker threads for the ggml graph. Inert on a CUDA-backend build
    // except for the ops ggml keeps on the host, so it is worth exactly what it
    // costs: a small number, not a core count. Ignored entirely when the model
    // fails to load.
    int  whisper_threads = 4;
    // The longest utterance the cascade will transcribe in one piece. It sizes
    // the audio ring, and the ring FAILS a read whose head it has already
    // evicted rather than handing back a truncated window -- so this is the
    // point past which a monologue is dropped with a log line instead of being
    // transcribed missing its first words. Whisper's own encoder tops out at
    // 30 s regardless.
    int  whisper_max_utterance_ms = 15000;

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
    //
    // Ships with the repo (models/f5_tts/, alongside the ONNX graphs): a 4.43 s
    // female Russian clip from google/fleurs ru_ru (CC-BY-4.0), resampled to
    // 24 kHz mono 16-bit, silence-trimmed, peak-normalised. Measured SNR 52 dB,
    // no clipping. scripts/make_f5_reference.py rebuilds it from scratch.
    // Absolute, like tts_ckpt_dir above and for the same reason: the app's CWD
    // is its own exe directory, so a repo-relative path resolves to nothing.
    // The clip lives NEXT TO the ONNX graphs, so this tracks tts_ckpt_dir.
    std::string tts_ref_audio  =
        "D:\\Projects\\BlackwellLLM\\models\\f5_tts\\ref_female_24k.wav";
    // NOT optional when TTS is on, and NOT a label: F5 conditions on an
    // (audio, TEXT) pair and treats generation as infilling, so this must be the
    // literal transcript of tts_ref_audio.
    //
    // THIS IS WHAT "PROMPT BLEEDING" ACTUALLY IS. If the text does not match the
    // audio, the model still has to reconcile the two -- and it does so by
    // inventing content, so the reference leaks into every utterance. The
    // previous value here was a placeholder, and a hand-set one in the field had
    // become "мяу мяу мяу...", which is why synthesis sounded possessed rather
    // than merely wrong. Change the clip and you MUST change this line with it.
    //
    // Ground truth from the dataset, not a transcription: every character is
    // verified present in the checkpoint's vocab.txt.
    std::string tts_ref_text   =
        "Это не казалось мне имеющим смысл; конечно это не было справедливым.";
    int  tts_nfe_step = 16;            // solver steps: the latency/quality dial

    // Playback loudness, [0, 1], applied in the audio callback -- so it takes
    // effect on the next buffer with no restart and no re-synthesis. It is a
    // SOFTWARE gain on our own stream, deliberately not the Windows endpoint
    // volume: changing that would turn every other app on the machine down too.
    //
    // The echo canceller is told the same number (see audio_playback.h): the
    // reference has to describe what the speaker actually emits, or the
    // canceller re-converges every time this moves.
    float tts_volume = 1.0f;

    // ---- chunking (time-to-first-audio vs. prosody continuity) --------------
    // F5 is not streaming: TTFB equals full synthesis time for whatever text it
    // is given, so the only latency lever is giving it less at once. Each split
    // re-conditions the voice on the reference rather than on what was just
    // spoken, so splitting harder is faster AND flatter.
    bool tts_split_on_commas = true;   // clause-level splits; off = sentences only
    int  tts_min_chunk_chars = 20;     // [5, 50]   stops "Да," becoming an utterance
    int  tts_max_chunk_chars = 150;    // [50, 300] forces a split at the next space

    // ---- acoustic echo cancellation -----------------------------------------
    // ON by default, and the microphone is never muted: the assistant's own
    // voice is SUBTRACTED from the capture stream (src/audio_rt/echo_canceller.hpp)
    // rather than the stream being switched off, which is what makes full-duplex
    // barge-in possible on open speakers. The predecessor of this setting was a
    // mic gate; it is gone, and with it the reason the user could not interrupt.
    //
    // Turning this OFF is a diagnostic, not a mode: capture then carries the
    // loudspeaker unmodified and the VAD will trigger on the assistant itself
    // unless you are wearing headphones.
    bool aec_enabled = true;
    // How much room the adaptive filter can model, in milliseconds. Must cover
    // the WHOLE speaker-to-microphone delay -- playback device buffer, flight
    // time, capture buffer -- plus the reverberation tail; echo arriving later
    // than this is uncancellable by construction. 256 ms is generous for a
    // desktop. Raise it for a large or reverberant room; the CPU cost is linear
    // and the convergence gets slower, so raising it "just in case" is not free.
    int  aec_tail_ms = 256;

    // ---- audio endpoints ----------------------------------------------------
    // Endpoint NAMES, not ids: an id is a backend-specific blob that cannot be
    // written into a settings file by hand, and the name is the only form a user
    // can recognise. Empty = follow the system default, which is the right
    // behaviour for a machine whose default changes when a headset is plugged in.
    //
    // Matching is case-insensitive and accepts a unique substring, so
    // "Realtek" resolves if only one endpoint contains it (see
    // audio_sandbox/src/ma_device_select.h). A name that does not resolve does
    // NOT stop the app: it warns and falls back to the default. Launch with
    // --list-audio-devices to print the exact strings.
    //
    // With loopback_capture ON, input_device_name names a PLAYBACK endpoint --
    // loopback taps a speaker, not a microphone.
    std::string output_device_name;    // "" = system default
    std::string input_device_name;     // "" = system default

    // The same two endpoints, addressed by the zero-based [N] the startup listing
    // prints. -1 = not selected by index, which is the default and defers to the
    // name above it.
    //
    // WHY BOTH FORMS EXIST rather than one replacing the other. A name survives a
    // device being unplugged and re-enumerated; an index does not, because it is
    // a position in a list the OS reorders. But a name has to be transcribed
    // exactly out of strings like "Speakers (2- High Definition Audio Device)",
    // and getting it wrong produces SILENCE -- the one failure mode that gives
    // the user nothing to go on. An index is two keystrokes off a printed list.
    // So: index for picking, name for stability, and the app prints the list at
    // every startup so both stay checkable.
    //
    // AN INDEX >= 0 OVERRIDES A NAME. See rt::detail::resolve_device_selection;
    // the short reason is that a stale name left in this file must not defeat the
    // index the user just set to get away from it.
    int output_device_index = -1;
    int input_device_index  = -1;

    // Capture gain, [0, 1], applied to every captured sample before anything
    // reads them. UPSTREAM OF EVERYTHING ON PURPOSE: the VAD, the AEC and the
    // ASR must all see the same signal the user is adjusting, or the meter in
    // the UI stops predicting whether speech will actually trigger.
    //
    // A software gain on OUR stream, not the Windows endpoint level -- same
    // reasoning as tts_volume, and the reason it can be Live.
    float mic_gain = 1.0f;

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
    // The Quake-style log console. Ctrl+~ is the convention this borrows from and
    // the reason the chord is not Ctrl+Alt+something like its neighbours: the
    // gesture is meant to be reachable with one hand without leaving the
    // keyboard, and every game that has this console binds exactly this key.
    //
    // `~` and the backtick are one physical key (VK_OEM_3), and hotkey_spec.hpp
    // accepts either spelling -- so a user who types the chord as they see it
    // printed on the keycap gets what they meant either way.
    std::string hotkey_console = "Ctrl+`";
    // Hold-to-talk rather than toggle. WM_HOTKEY has no key-up event, so hold
    // mode is driven by polling the chord while it is held -- see the window.
    bool hotkey_push_to_talk = false;

    // ---- the log console (Direct2D overlay) ---------------------------------
    // Fraction of the screen HEIGHT the console covers when fully down, [0.2, 1].
    // The Quake proportion is about a third; more than that and the thing it is
    // overlaying stops being visible, which defeats an overlay.
    float console_height_pct = 0.45f;
    // Background alpha, [0.2, 1]. 0.85 is the task's figure and is about where
    // text stays readable over a bright window underneath.
    float console_opacity = 0.85f;
    // MONOSPACED OR THE LAYOUT IS WRONG: the renderer advances by one measured
    // character width per column, so a proportional face would produce ragged
    // columns. Falls back to Consolas (present on every Windows install) when the
    // named family is missing -- Cascadia Code ships with Terminal and VS but is
    // not guaranteed.
    std::string console_font = "Cascadia Code";
    float       console_font_size = 14.0f;
    // How many lines the scrollback holds. Bounded because it is a live ring in
    // memory and an unbounded log is a leak with a friendly name.
    int console_scrollback_lines = 4000;

    // ---- window & tray behaviour --------------------------------------------
    // Intercept [X] and hide to the notification area instead of quitting. OFF
    // means the button does what its shape promises and terminates the app.
    //
    // ON BY DEFAULT because this is a background utility that a hotkey summons:
    // an assistant that has to be relaunched every time the window is dismissed
    // is not a background utility. The tray menu's Exit is the way out, and the
    // first hide says so.
    bool minimize_to_tray = true;
    // Keep the summoned window above other windows (HWND_TOPMOST). OFF by
    // default: a chat window that cannot be put behind anything is a nuisance,
    // and the hotkey already makes it one keystroke away.
    bool always_on_top = false;
    // Show the splash while the heavy initialization runs. Off is for developers
    // who restart the app constantly and do not want the fade.
    bool show_splash = true;
    // Start hidden (tray only) rather than showing the window at launch. For a
    // shortcut in the Startup folder, where a window appearing on login is the
    // wrong behaviour.
    bool start_minimized = false;

    // Answer locally instead of calling out. Swaps the transport under the
    // dispatcher (see local_transport.hpp); everything above it is unchanged, so
    // the same commit rule gates a local answer and a billed one.
    bool local_inference = false;

    // ---- the remote leg: any OpenAI-compatible /chat/completions endpoint ----
    // Used when local_inference is OFF. All three are RESTART tier, and that is
    // structural rather than a shortcut: the HTTP client builds its auth header
    // list and pins its endpoint string ONCE at construction (the slist must
    // outlive every transfer), and the model is bound to the transport object
    // that renders each body. Rebuilding them under a dispatcher thread that may
    // be parked inside send() is a lifetime problem, not a settings change.
    //
    // BASE url, exactly as a provider documents it -- "/chat/completions" is
    // appended by the client (openai_request.hpp::join_url), so a trailing slash
    // or none both work. Empty disables the remote leg: the app falls back to
    // ANTHROPIC_API_KEY if that is set, and to the offline stand-in otherwise.
    std::string remote_api_url = "https://router.cheap/v1";
    // Sent as `Authorization: Bearer <key>`. PERSISTED IN CLEARTEXT under
    // %LOCALAPPDATA% -- this is a settings file, not a credential store, and the
    // UI says so. Empty = the remote leg is not armed.
    //
    // EMPTY BY DEFAULT, AND IT MUST STAY THAT WAY. A key written here is not a
    // convenience: this is a tracked header, so it would enter git history
    // permanently (rewriting history is the only way back out) and it would be
    // compiled into every binary built from this tree. The key belongs in the
    // per-machine settings.json under %LOCALAPPDATA%, or in the environment --
    // both of which are already how a developer's own key reaches the app.
    std::string remote_api_key = "";
    // The model id the endpoint expects. Free text, not a list: which ids a
    // gateway serves is the gateway's business and changes without us.
    std::string remote_model = "gpt-4o-mini";

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
    // ---- the speech-to-text pipeline selector --------------------------------
    // All Restart tier: pipeline_mode picks which ISpeechMode is constructed and
    // whether the Ultravox audio head is allocated; the whisper_* values are read
    // once when the ASR context is created (a ~1.6 GB model load onto the GPU).
    // None of them can be a store the next block reads.
    f("pipeline_mode",            s.pipeline_mode,            Tier::Restart);
    f("whisper_model_path",       s.whisper_model_path,       Tier::Restart);
    f("whisper_language",         s.whisper_language,         Tier::Restart);
    f("whisper_threads",          s.whisper_threads,          Tier::Restart);
    f("whisper_max_utterance_ms", s.whisper_max_utterance_ms, Tier::Restart);
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
    // Live: the gain is read by the audio callback on every buffer, so a change
    // is audible on the next one. Nothing is rebuilt and nothing re-synthesised.
    f("tts_volume",               s.tts_volume,               Tier::Live);
    // AudioHotReload: the device IS closed and reopened, but only the device --
    // see the tier's declaration. The AEC's learned response is reset by the
    // swap (it describes the old speaker), which costs a few hundred ms of
    // re-convergence and is not a reason to restart a process holding 8 GB.
    f("output_device_name",       s.output_device_name,       Tier::AudioHotReload);
    f("input_device_name",        s.input_device_name,        Tier::AudioHotReload);
    f("output_device_index",      s.output_device_index,      Tier::AudioHotReload);
    f("input_device_index",       s.input_device_index,       Tier::AudioHotReload);
    // Live: a plain gain on the capture samples, read per block like tts_volume.
    f("mic_gain",                 s.mic_gain,                 Tier::Live);
    f("tts_split_on_commas",      s.tts_split_on_commas,      Tier::Restart);
    f("tts_min_chunk_chars",      s.tts_min_chunk_chars,      Tier::Restart);
    f("tts_max_chunk_chars",      s.tts_max_chunk_chars,      Tier::Restart);
    // Live: the filter is bypassable in place (and resets its learned response
    // on the way back in), so toggling it costs nothing but a re-convergence.
    // The tail sizes fixed buffers in the constructor, so that one is Restart.
    f("aec_enabled",              s.aec_enabled,              Tier::Live);
    f("aec_tail_ms",              s.aec_tail_ms,              Tier::Restart);
    // ---- Tab 3 -------------------------------------------------------------
    f("hotkey_talk",              s.hotkey_talk,              Tier::Live);
    f("hotkey_cancel",            s.hotkey_cancel,            Tier::Live);
    f("hotkey_show",              s.hotkey_show,              Tier::Live);
    f("hotkey_console",           s.hotkey_console,           Tier::Live);
    f("hotkey_push_to_talk",      s.hotkey_push_to_talk,      Tier::Live);
    // ---- console & window behaviour -----------------------------------------
    // All Live: the console re-reads its geometry and colours on the next slide,
    // the font is re-created in place, and the window flags are one SetWindowPos.
    // Nothing here allocates VRAM or touches a device, which is what keeps the
    // whole group out of the restart tier.
    f("console_height_pct",       s.console_height_pct,       Tier::Live);
    f("console_opacity",          s.console_opacity,          Tier::Live);
    f("console_font",             s.console_font,             Tier::Live);
    f("console_font_size",        s.console_font_size,        Tier::Live);
    // Restart: it sizes the ring buffer at construction, and re-sizing a live
    // ring would either drop scrollback the user is reading or copy it under the
    // writer's feet.
    f("console_scrollback_lines", s.console_scrollback_lines, Tier::Restart);
    f("minimize_to_tray",         s.minimize_to_tray,         Tier::Live);
    f("always_on_top",            s.always_on_top,            Tier::Live);
    f("start_minimized",          s.start_minimized,          Tier::Live);
    // Restart, and only in the sense that it cannot apply retroactively: the
    // splash is over before the settings modal can be opened.
    f("show_splash",              s.show_splash,              Tier::Restart);
    f("local_inference",          s.local_inference,          Tier::Live);
    // Restart tier -- see the declarations for why the HTTP client cannot be
    // reconfigured under a live dispatcher.
    f("remote_api_url",           s.remote_api_url,           Tier::Restart);
    f("remote_api_key",           s.remote_api_key,           Tier::Restart);
    f("remote_model",             s.remote_model,             Tier::Restart);
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
    // Strip surrounding whitespace in place. Pasted credentials and URLs arrive
    // with a trailing newline or a leading space often enough that not doing
    // this is a real bug: a key with a stray "\n" is a well-formed Authorization
    // header that comes back as an opaque 401, with nothing on screen to
    // distinguish it from a wrong key.
    auto trim = [](std::string& v) {
        const std::size_t first = v.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            v.clear();
        } else {
            v = v.substr(first, v.find_last_not_of(" \t\r\n") - first + 1);
        }
    };

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
    s.aec_tail_ms = clamp_int(s.aec_tail_ms, 64, 1000);
    // NaN-safe: `!(v >= lo)` is true for NaN where `v < lo` is false, and a NaN
    // gain would silence the speaker AND poison the canceller's reference.
    if (!(s.tts_volume >= 0.0f)) s.tts_volume = 0.0f;
    if (s.tts_volume > 1.0f) s.tts_volume = 1.0f;
    if (!(s.mic_gain >= 0.0f)) s.mic_gain = 0.0f;   // NaN-safe, as above
    if (s.mic_gain > 1.0f) s.mic_gain = 1.0f;
    // Endpoint names are matched literally, so a pasted trailing space is a name
    // that will not resolve and will fall back to the default with no clue why.
    trim(s.output_device_name);
    trim(s.input_device_name);
    // Only the negative side is normalised, to exactly -1. There is deliberately
    // NO upper clamp: this header cannot see the device list, and clamping an
    // index to a count it would have to guess is how "device 9" quietly becomes
    // "device 2". Out-of-range is caught where the list is actually in hand
    // (resolve_device_by_index), which warns and takes the system default.
    if (s.output_device_index < 0) s.output_device_index = -1;
    if (s.input_device_index < 0)  s.input_device_index  = -1;
    if (s.context_mode != "stateless" && s.context_mode != "bounded") {
        s.context_mode = "bounded";
    }
    if (s.speech_task != "transcribe" && s.speech_task != "translate" &&
        s.speech_task != "both") {
        s.speech_task = "transcribe";
    }
    // An unknown pipeline clamps to LEGACY, not to the newer path. A hand-edited
    // file, an older build's value, or a typo must never be the thing that moves
    // a user onto an unvalidated pipeline -- the fallback direction is toward the
    // one that has been in production.
    //
    // "simultaneous" IS DELIBERATELY IN THIS SET even though voice_assistant
    // cannot run it. It names ISpeechMode's Mode B (continuous rolling
    // re-translation, docs/CONTINUOUS_STREAMING.md), which is a REAL mode this
    // repo builds -- into audio_translator, not into this binary. Silently
    // clamping it to legacy would make "I asked for Mode B" and "I typed
    // ultravx_lgacy" produce the identical, unexplained outcome. Recognised here
    // and REFUSED, loudly, where modes are actually constructed
    // (AppLifecycleManager::bring_up_speech_mode) -- so the user is told which of
    // the two things went wrong.
    if (s.pipeline_mode != "ultravox_legacy" && s.pipeline_mode != "whisper_cascade" &&
        s.pipeline_mode != "simultaneous") {
        s.pipeline_mode = "ultravox_legacy";
    }
    trim(s.whisper_model_path);
    // Whisper wants a bare ISO-639-1 code; "" and whitespace both mean auto-LID,
    // for the same reason speech_language treats them alike below. Lowercased
    // because whisper_lang_id() is case-sensitive and "RU" resolves to nothing --
    // which whisper reports by falling back to auto, i.e. silently ignoring the
    // setting the user just typed.
    trim(s.whisper_language);
    for (char& c : s.whisper_language) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    s.whisper_threads = clamp_int(s.whisper_threads, 1, 32);
    // Floor: below ~1 s Whisper's encoder degrades badly (it zero-pads to 30 s
    // regardless, and a very short window is mostly padding). Ceiling: the
    // encoder's positional embeddings stop at 30 s, so nothing above that can be
    // transcribed in one piece no matter how large the ring is made.
    s.whisper_max_utterance_ms = clamp_int(s.whisper_max_utterance_ms, 1000, 30000);
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
    trim(s.remote_api_url);
    trim(s.remote_api_key);
    trim(s.remote_model);
    // A base URL without a scheme is the other common paste error. Defaulting to
    // https rather than rejecting it keeps "router.cheap/v1" working, and https
    // rather than http because a bearer token must never go out in the clear.
    if (!s.remote_api_url.empty() && s.remote_api_url.find("://") == std::string::npos) {
        s.remote_api_url.insert(0, "https://");
    }
    // ---- console geometry ---------------------------------------------------
    // NaN-safe on the same pattern the gains use: `!(v >= lo)` catches NaN where
    // `v < lo` does not, and a NaN height would make the slide animation compute
    // a target rectangle no window can occupy.
    if (!(s.console_height_pct >= 0.2f)) s.console_height_pct = 0.2f;
    if (s.console_height_pct > 1.0f) s.console_height_pct = 1.0f;
    // The floor is not cosmetic: an alpha near zero makes the console invisible
    // while still swallowing the hotkey, so the user has a toggle that appears to
    // do nothing at all.
    if (!(s.console_opacity >= 0.2f)) s.console_opacity = 0.2f;
    if (s.console_opacity > 1.0f) s.console_opacity = 1.0f;
    if (!(s.console_font_size >= 8.0f)) s.console_font_size = 8.0f;
    if (s.console_font_size > 48.0f) s.console_font_size = 48.0f;
    trim(s.console_font);
    if (s.console_font.empty()) s.console_font = "Consolas";
    s.console_scrollback_lines = clamp_int(s.console_scrollback_lines, 200, 100000);

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

// The same trick for the hot-reload tier: one comparison answers "does a device
// have to be reopened?". Separate from requires_restart so a settings save can
// do the cheap thing when only an endpoint moved -- which is the common case and
// the one that must never cost a process restart.
inline std::string audio_signature(const AssistantSettings& s) {
    nlohmann::json j;
    visit_fields(s, [&](const char* key, const auto& value, Tier tier) {
        if (tier == Tier::AudioHotReload) j[key] = value;
    });
    return j.dump();
}

inline bool requires_audio_reload(const AssistantSettings& a, const AssistantSettings& b) {
    return audio_signature(a) != audio_signature(b);
}

// THE app-data layout, in one function. Everything the app persists lives in
// %LOCALAPPDATA%\BlackwellVoiceAssistant\, and falls back to a CWD-relative file
// when the variable is missing (a service account, a stripped environment)
// rather than failing to persist at all.
//
// `leaf` is the name inside that directory; `fallback` is the flat name used
// when there is no directory to put it in -- deliberately a separate string,
// because a bare "sessions.json" dropped in whatever the CWD happens to be is a
// worse neighbour than a prefixed one.
inline std::string app_data_file(const char* leaf, const char* fallback) {
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && *local != '\0') {
        return std::string(local) + "\\BlackwellVoiceAssistant\\" + leaf;
    }
    return fallback;
}

inline std::string settings_path() {
    return app_data_file("settings.json", "voice_assistant_settings.json");
}

// The persisted conversation log (src/cloud/session_store.hpp). Deliberately a
// SEPARATE file from settings.json and not a key inside it: the two have
// different writers (a person clicking Save vs. the dispatcher thread, once per
// answered turn), different sizes, and different consequences when corrupt --
// and a per-turn rewrite of the settings file would eventually lose settings to
// a crash mid-write.
inline std::string sessions_path() {
    return app_data_file("sessions.json", "voice_assistant_sessions.json");
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
