// -----------------------------------------------------------------------------
// settings_validator.cpp — the checks, and the fan-out. See the header for why
// this sits ON TOP of clamp_settings rather than replacing it.
// -----------------------------------------------------------------------------
#include "settings_validator.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <utility>

#include "hotkey_spec.hpp"
#include "vram_budget.hpp"

namespace rt {
namespace {

// The chords the Windows shell owns. NOT EXHAUSTIVE AND CANNOT BE -- third-party
// launchers, the GPU vendor's overlay and the user's own utilities all take
// global hotkeys, and none of them are enumerable. That is exactly why a match
// here is a WARNING: this list can only ever say "we know this one is taken",
// never "this one is free".
//
// Ctrl+Alt+Del is the interesting entry: it is the Secure Attention Sequence and
// is intercepted below the window manager, so RegisterHotKey does not even fail
// -- it succeeds and never fires. That failure mode (a hotkey that registered
// and does nothing) is the one a user cannot possibly diagnose, which is why the
// list exists at all.
struct ReservedChord {
    UINT        mods;
    UINT        vk;
    const char* owner;
};

const ReservedChord kReserved[] = {
    {MOD_CONTROL | MOD_ALT, VK_DELETE, "Windows (the secure attention sequence)"},
    {MOD_ALT,               VK_TAB,    "Windows (task switcher)"},
    {MOD_ALT,               VK_ESCAPE, "Windows (window cycling)"},
    {MOD_ALT,               VK_F4,     "Windows (close window)"},
    {MOD_CONTROL,           VK_ESCAPE, "Windows (Start menu)"},
    {MOD_CONTROL | MOD_SHIFT, VK_ESCAPE, "Windows (Task Manager)"},
    {MOD_WIN,               'L',       "Windows (lock workstation)"},
    {MOD_WIN,               'D',       "Windows (show desktop)"},
    {MOD_WIN,               'E',       "Windows (File Explorer)"},
    {MOD_WIN,               'R',       "Windows (Run dialog)"},
    {MOD_WIN,               'X',       "Windows (power-user menu)"},
    {MOD_WIN,               'P',       "Windows (display projection)"},
    {MOD_WIN,               'S',       "Windows (search)"},
    {MOD_WIN,               'I',       "Windows (Settings)"},
    {MOD_WIN,               VK_TAB,    "Windows (Task View)"},
    {MOD_WIN,               VK_SPACE,  "Windows (input-language switch)"},
};

const char* reserved_owner(const Hotkey& hk) {
    for (const auto& r : kReserved) {
        if (r.mods == hk.mods && r.vk == hk.vk) return r.owner;
    }
    return nullptr;
}

bool contains_ci(const std::string& hay, const char* needle) {
    const std::string n(needle);
    if (n.size() > hay.size()) return false;
    for (std::size_t i = 0; i + n.size() <= hay.size(); ++i) {
        bool hit = true;
        for (std::size_t j = 0; j < n.size(); ++j) {
            char a = hay[i + j];
            char b = n[j];
            if (a >= 'A' && a <= 'Z') a = static_cast<char>(a + 32);
            if (b >= 'A' && b <= 'Z') b = static_cast<char>(b + 32);
            if (a != b) { hit = false; break; }
        }
        if (hit) return true;
    }
    return false;
}

// ---- the cloud leg ---------------------------------------------------------
void check_remote(const AssistantSettings& s, ValidationReport& out) {
    // An empty URL is a legitimate configuration (the remote leg falls back to
    // ANTHROPIC_API_KEY, then to the offline stand-in), so it is not a finding.
    // Reporting it would put a warning on the default install.
    if (!s.remote_api_url.empty()) {
        // clamp_settings already prepended https:// when there was no scheme, so
        // by here anything not starting https:// was written http:// on purpose.
        if (s.remote_api_url.rfind("http://", 0) == 0) {
            out.add("remote_api_url", Severity::Warning,
                    "this endpoint is plain HTTP, so the API key is sent in clear text over "
                    "the network. Use https:// unless this is a loopback address.");
        }
        // The client appends /chat/completions (openai_request.hpp::join_url), so
        // a base that already carries it produces a doubled path and a 404 that
        // reads exactly like a wrong model id.
        if (contains_ci(s.remote_api_url, "/chat/completions")) {
            out.add("remote_api_url", Severity::Error,
                    "this must be the BASE url (e.g. https://router.cheap/v1) -- "
                    "/chat/completions is appended by the client, and leaving it here "
                    "produces a doubled path.");
        }
        if (s.remote_api_url.find(' ') != std::string::npos) {
            out.add("remote_api_url", Severity::Error,
                    "the endpoint contains a space, which no URL can.");
        }
    }

    if (!s.remote_api_key.empty()) {
        // clamp_settings trims the ends; what it cannot fix is whitespace in the
        // MIDDLE, which is a wrapped paste and produces a malformed Authorization
        // header -- an opaque 401 with nothing on screen to distinguish it from a
        // wrong key. This is the single highest-value check in the file.
        if (s.remote_api_key.find_first_of(" \t\r\n") != std::string::npos) {
            out.add("remote_api_key", Severity::Error,
                    "the key contains whitespace, which usually means it was pasted across "
                    "a line break. It will be rejected as a malformed Authorization header, "
                    "which looks identical to a wrong key.");
        }
        // Placeholders that reach the wire cost a round trip and a 401 to
        // diagnose; they cost nothing to catch here.
        if (contains_ci(s.remote_api_key, "your-api-key") ||
            contains_ci(s.remote_api_key, "xxxx") ||
            contains_ci(s.remote_api_key, "<paste") ||
            s.remote_api_key == "sk-") {
            out.add("remote_api_key", Severity::Error,
                    "this looks like a placeholder rather than a key.");
        }
    }

    // The one cross-field check: asking for the cloud leg with nothing to reach
    // it with is the configuration that produces "why is nothing answering".
    if (!s.local_inference && s.remote_api_url.empty() && s.remote_api_key.empty()) {
        out.add("local_inference", Severity::Warning,
                "local inference is off and no remote endpoint or key is configured, so "
                "replies will come from the offline stand-in. Set a remote endpoint, or "
                "turn local inference back on.");
    }
    if (!s.local_inference && !s.remote_api_url.empty() && s.remote_api_key.empty()) {
        out.add("remote_api_key", Severity::Warning,
                "a remote endpoint is set but no key is. Most gateways answer that with a "
                "401 rather than serving anonymously.");
    }
}

// ---- hotkeys ---------------------------------------------------------------
void check_hotkeys(const AssistantSettings& s, ValidationReport& out) {
    const std::vector<HotkeyBinding> bindings = app_hotkeys(s);

    struct Parsed {
        const HotkeyBinding* b;
        Hotkey               hk;
    };
    std::vector<Parsed> parsed;
    parsed.reserve(bindings.size());

    for (const auto& b : bindings) {
        const Hotkey hk = parse_hotkey(b.spec);
        // An EMPTY spec is "this hotkey is off", which is a supported choice and
        // not a finding. A NON-empty spec that parses to nothing is a typo, and
        // it is the failure the user cannot see: the app comes up, the chord does
        // nothing, and there is no message anywhere.
        if (!b.spec.empty() && !hk.armed()) {
            out.add(b.field, Severity::Error,
                    "'" + b.spec + "' is not a chord this app can register, so the " +
                        std::string(b.label) + " hotkey is off. Use a form like "
                        "Ctrl+Alt+Space.");
            continue;
        }
        if (!hk.armed()) continue;

        // A chord with no modifier steals a bare key from every other
        // application on the machine, system-wide, for as long as this app runs.
        if (hk.mods == 0) {
            out.add(b.field, Severity::Warning,
                    "'" + b.spec + "' has no modifier, so it takes that key away from every "
                    "other application while the assistant is running.");
        }
        if (const char* owner = reserved_owner(hk); owner != nullptr) {
            out.add(b.field, Severity::Warning,
                    "'" + b.spec + "' is owned by " + owner +
                        ". It will either fail to register or never fire.");
        }
        parsed.push_back({&b, hk});
    }

    // SELF-CONFLICTS. Exact, and an Error: RegisterHotKey fails the second
    // binding of the same chord, so one of the two silently does nothing. O(n^2)
    // over four items.
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        for (std::size_t j = i + 1; j < parsed.size(); ++j) {
            if (!(parsed[i].hk == parsed[j].hk)) continue;
            out.add(parsed[j].b->field, Severity::Error,
                    std::string("'") + parsed[j].b->spec + "' is already bound to the " +
                        parsed[i].b->label + " hotkey. Only the first registers; this one "
                        "will not fire.");
        }
    }
}

// ---- the VRAM-driven clamp -------------------------------------------------
void check_vram(AssistantSettings& s, const ValidationContext& ctx, ValidationReport& out) {
    // No card, nothing to budget. The simulated backend allocates no KV at all,
    // and clamping a number that costs nothing would refuse a valid setup.
    if (!ctx.have_gpu || s.simulated || s.model_dir.empty()) return;

    VramBudgetInputs in;
    in.model_dir          = s.model_dir;
    in.audio_head_dir     = s.audio_head;
    in.load_audio_head    = !ctx.cascade;
    in.whisper_model_path = ctx.cascade ? ctx.whisper_model : std::string{};
    in.tts_enabled        = ctx.tts_enabled;
    in.requested_context  = s.max_context;
    in.branch_factor      = ctx.branch_factor;

    const VramBudget b = plan_vram_budget(in);

    // ADVISORY MEANS DO NOT TOUCH THE VALUE. A term that could not be measured
    // makes the whole sum a guess, and clamping a working configuration on a
    // fabricated number is strictly worse than leaving it alone -- the same
    // decision plan_vram_budget itself makes.
    if (b.advisory_only) {
        out.add("max_context", Severity::Info,
                "the VRAM budget could not be computed (" + b.advisory_reason +
                    "), so this value was not checked against the card.");
        return;
    }
    if (!b.fits) {
        // NOT clamped, and deliberately: there is no value that fits, so there is
        // nothing to clamp to. Reported as an Error the user must resolve by
        // dropping a consumer (the audio head, the TTS stack) rather than by
        // tuning this field.
        out.add("max_context", Severity::Error,
                "this configuration does not fit on the card even at the minimum context: " +
                    b.failure);
        return;
    }
    if (b.granted_context < s.max_context) {
        const int asked = s.max_context;
        s.max_context = b.granted_context;
        out.add("max_context", Severity::Warning,
                "reduced from " + std::to_string(asked) + " to " +
                    std::to_string(b.granted_context) +
                    " -- that is the largest KV context this card can hold alongside the "
                    "weights and the rest of the stack.");
    }
}

// Anything clamp_settings silently corrected. Detected by diffing, which is what
// lets one function report on another's corrections without either knowing about
// the other's rules.
void report_coercions(const AssistantSettings& before, const AssistantSettings& after,
                      ValidationReport& out) {
    if (before.pipeline_mode != after.pipeline_mode) {
        out.add("pipeline_mode", Severity::Warning,
                "'" + before.pipeline_mode + "' is not a pipeline -- using " +
                    after.pipeline_mode + ".");
    }
    if (before.context_mode != after.context_mode) {
        out.add("context_mode", Severity::Info,
                "'" + before.context_mode + "' is not a context mode -- using " +
                    after.context_mode + ".");
    }
    if (before.speech_task != after.speech_task) {
        out.add("speech_task", Severity::Info,
                "'" + before.speech_task + "' is not a task -- using " + after.speech_task +
                    ".");
    }
    auto note_int = [&](const char* field, int b, int a) {
        if (b == a) return;
        out.add(field, Severity::Info,
                "adjusted from " + std::to_string(b) + " to " + std::to_string(a) +
                    " (outside the supported range).");
    };
    note_int("max_new_tokens", before.max_new_tokens, after.max_new_tokens);
    note_int("max_context", before.max_context, after.max_context);
    note_int("silence_hangover_ms", before.silence_hangover_ms, after.silence_hangover_ms);
    note_int("pre_roll_ms", before.pre_roll_ms, after.pre_roll_ms);
    note_int("aec_tail_ms", before.aec_tail_ms, after.aec_tail_ms);
    note_int("whisper_threads", before.whisper_threads, after.whisper_threads);
    note_int("whisper_max_utterance_ms", before.whisper_max_utterance_ms,
             after.whisper_max_utterance_ms);
}

}  // namespace

const char* to_string(Severity s) noexcept {
    switch (s) {
        case Severity::Info:    return "info";
        case Severity::Warning: return "warn";
        case Severity::Error:   return "error";
    }
    return "info";
}

void ValidationReport::add(std::string field, Severity sev, std::string message) {
    items.push_back({std::move(field), sev, std::move(message)});
}

bool ValidationReport::has_errors() const noexcept {
    return std::any_of(items.begin(), items.end(),
                       [](const Diagnostic& d) { return d.severity == Severity::Error; });
}

bool ValidationReport::has_warnings() const noexcept {
    return std::any_of(items.begin(), items.end(),
                       [](const Diagnostic& d) { return d.severity == Severity::Warning; });
}

void ValidationReport::print(const char* who) const {
    for (const Diagnostic& d : items) {
        std::fprintf(d.severity == Severity::Error ? stderr : stdout, "[%s] %s: %s -- %s\n",
                     who, to_string(d.severity), d.field.c_str(), d.message.c_str());
    }
}

std::vector<HotkeyBinding> app_hotkeys(const AssistantSettings& s) {
    return {
        {"hotkey_talk",    "talk",    s.hotkey_talk},
        {"hotkey_cancel",  "cancel",  s.hotkey_cancel},
        {"hotkey_show",    "show window", s.hotkey_show},
        {"hotkey_console", "console", s.hotkey_console},
    };
}

ValidationReport validate_settings(AssistantSettings& s, const ValidationContext& ctx) {
    ValidationReport out;

    // The value as the user gave it, kept so the coercion pass can diff against
    // it. clamp_settings runs FIRST because every check below is written against
    // a value already in range -- checking a URL before the scheme is prepended
    // would report a problem clamp_settings was about to fix.
    const AssistantSettings before = s;
    clamp_settings(s);
    report_coercions(before, s, out);

    check_remote(s, out);
    check_hotkeys(s, out);
    // LAST, because it MUTATES max_context and its message quotes the value the
    // user asked for. Running it before the coercion diff would report the same
    // field twice with two different stories.
    check_vram(s, ctx, out);
    return out;
}

// -----------------------------------------------------------------------------
// SettingsBus
// -----------------------------------------------------------------------------
void SettingsBus::subscribe(std::string name, Subscriber fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lk(mu_);
    subs_.push_back({std::move(name), std::move(fn)});
}

void SettingsBus::publish(const AssistantSettings& s) {
    // COPIED UNDER THE LOCK, RUN WITHOUT IT. Holding the mutex across a
    // subscriber would make the bus a lock ordering problem: subscribers take
    // their own locks (the audio binder's device mutex, the view's), and one that
    // blocked would stall every other publisher including the engine thread.
    std::vector<Entry> snapshot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        snapshot = subs_;
    }
    for (const Entry& e : snapshot) {
        try {
            e.fn(s);
        } catch (const std::exception& ex) {
            // One subsystem failing to apply a setting must not cost every other
            // subsystem the same setting -- which is exactly what a hand-written
            // fan-out of six statements does when the third one throws.
            std::fprintf(stderr, "[settings] subscriber '%s' threw: %s\n", e.name.c_str(),
                         ex.what());
        } catch (...) {
            std::fprintf(stderr, "[settings] subscriber '%s' threw\n", e.name.c_str());
        }
    }
}

std::size_t SettingsBus::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return subs_.size();
}

}  // namespace rt
