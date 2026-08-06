#pragma once
// -----------------------------------------------------------------------------
// cli_options.hpp — voice_assistant's OWN command line, layered over the shared
// rt::parse_cli, plus the two self-tests that exist only to serve a flag.
//
// WHY THERE ARE TWO PARSERS. rt::parse_cli (translator/cli_config.hpp) THROWS on
// an unknown argument -- a good property, because a typo'd flag must not be
// silently ignored. So the flags that exist only in this app are removed from
// argv before it runs, and what is left is passed through.
//
// WHY TypedFlags EXISTS, and it is load-bearing for the precedence rule
// (persisted settings < CLI flags): parse_cli RESOLVES a value for several fields
// whether or not anything was typed -- a config.json in the CWD, its built-in
// default paths -- so "args.model_dir is non-empty" says nothing about user
// intent. Only a flag that actually appeared in argv may override a persisted
// setting, and this struct is the record of which ones did.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <string>
#include <vector>

#include "settings_store.hpp"   // rt::AssistantSettings

namespace rt {

// Which flags the user actually TYPED on this launch. See the header preamble.
struct TypedFlags {
    bool model_dir = false;
    bool audio_head = false;
    bool projector_path = false;
    bool data_dir = false;
    bool device = false;
    bool simulated = false;
    bool real = false;
    bool vad_threshold = false;
    bool no_neural_vad = false;
    bool context_mode = false;
    bool history_budget = false;
    bool output_device = false;
    bool input_device = false;
    bool output_device_index = false;
    bool input_device_index = false;
    bool tts_volume = false;
    bool pipeline_mode = false;
    bool whisper_model = false;
    bool whisper_language = false;
};

struct VoiceArgs {
    int device_id = 0;   // --device <id>
    // Audio ENDPOINTS -- unrelated to device_id above, which is a CUDA device.
    // The names are deliberately unambiguous (--audio-output / --audio-input)
    // because "--device" already means something else here.
    std::string output_device;
    std::string input_device;
    // The index form of the same two endpoints, -1 = unset. Kept separate from the
    // names rather than overloading one flag with "a number means an index": an
    // endpoint genuinely called "2" would then be unaddressable by name.
    int output_device_index = -1;
    int input_device_index = -1;
    float tts_volume = 1.0f;
    bool list_audio_devices = false;   // print the endpoints and exit
    bool check_volume = false;         // measure the software gain and exit
    std::string say;                   // --say "<text>": speak it and exit
    // --test-llm-tts ["<prompt>"]: the FULL stack, one injected turn, then exit.
    // Empty string = not requested; the flag supplies a default prompt.
    std::string test_llm_tts;
    // --pipeline <ultravox_legacy|whisper_cascade> and the cascade's two paths.
    // Restart-tier settings all three, so a flag is the ONLY way to try the
    // cascade without opening Settings, saving, and being restarted -- which
    // matters most on exactly the launch where it is being brought up for the
    // first time. Validated by clamp_settings, not here: one validator.
    std::string pipeline_mode;
    std::string whisper_model;
    std::string whisper_language;
    TypedFlags typed;
};

// Strip this app's own flags out of argv, recording which were seen; everything
// else is appended to `passthrough` for rt::parse_cli.
VoiceArgs parse_voice_args(int argc, char** argv, std::vector<char*>& passthrough);

#if defined(VOICE_ASSISTANT_HAS_TTS)
// --check-volume: does the volume setting actually change the loudness?
//
// A self-test rather than a claim. Software volume is one multiply in an audio
// callback, which is exactly the kind of code that is "obviously correct" and
// silently does nothing -- the gain never reaching the callback, or reaching it
// once and never updating, both look identical from the outside and neither shows
// up in a build log.
//
// So this MEASURES it, through the production path and nothing else: the real
// AudioPlayback on the configured endpoint, with WASAPI loopback recording that
// same endpoint's output. It plays a tone at full gain, changes the volume WHILE
// PLAYING (which is the part that must work without a restart), and reports the
// measured ratio of the two. Returns a process exit code.
//
// It makes an audible sound, which is why it is opt-in and brief.
int run_volume_check(const AssistantSettings& settings);

// --say "<text>": drive the SPEECH HALF of a turn with no LLM and no microphone.
//
// WHAT IT PROVES, AND WHAT IT DOES NOT. It pushes text through exactly the calls
// the dispatcher's answer stream makes -- Resume(), PushToken(), EndOfTurn() -- so
// everything downstream of "the answer exists as text" is under test: chunker, F5
// tokenizer, CUDA synthesis, the speaker ring, the device.
//
// It does NOT test the LLM -> dispatcher -> here wiring, because it stands in for
// the LLM. That half has no headless entry point: a real turn needs a microphone
// or the typed box in the UI. So a PASS here plus a "[answer->tts] pushed ..."
// line in a live run is what covers the whole path; this alone covers the half
// where the failure usually is, and is the only half that can be checked without
// a person talking. Returns a process exit code.
int run_say_check(const AssistantSettings& settings, const std::string& text);
#endif

}  // namespace rt
