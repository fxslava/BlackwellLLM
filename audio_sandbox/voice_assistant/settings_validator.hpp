#pragma once
// -----------------------------------------------------------------------------
// settings_validator.hpp — the schema pass over AssistantSettings, and the bus
// that fans an accepted change out to everything running.
//
// WHY THIS IS NOT clamp_settings. clamp_settings (settings_store.hpp) is a
// COERCION: it silently drags every value into range, and that is exactly right
// for its two callers -- a hand-edited file and a page payload, where the only
// sane outcome is a usable configuration. What it cannot do is TELL ANYONE. A
// value that was corrected, a hotkey that will not register, an API key that is
// obviously not a key, a context length this card cannot afford: all of them
// come out of clamp_settings looking identical to a value nobody touched.
//
// So the two are layered rather than merged, and the order is load-bearing:
//
//   clamp_settings()    makes it SAFE      (always runs, silent, cannot fail)
//   validate_settings() makes it EXPLAINED (runs on top, reports, may clamp more)
//
// validate_settings therefore runs clamp_settings FIRST and then reports on what
// it finds -- including a diagnostic for anything clamp_settings had to change,
// which it detects by comparing against the value it was handed.
//
// THE THREE CHECKS THE TASK NAMES, and what each can honestly conclude:
//
//   CLOUD ENDPOINT / TOKEN. Shape only, and deliberately: reachability and
//   authorization are network questions, and answering them here would mean
//   issuing a request from inside a settings save -- on the UI thread, blocking
//   the modal, spending money on a prewarm the user did not ask for. What IS
//   decidable offline is the whole class of paste errors that currently surface
//   as an opaque 401: a URL with no scheme or a non-https one, a path that
//   already ends in /chat/completions (the client appends it), a key with
//   internal whitespace, a key that is obviously a placeholder.
//
//   HOTKEY CONFLICTS. Two kinds, and both are real. SELF-conflicts (two of the
//   app's four chords bound to the same keys) are decidable exactly and are an
//   Error, because RegisterHotKey will fail the second one and the user gets a
//   silently dead hotkey. SYSTEM-reserved chords are a fixed list of things the
//   shell owns (Ctrl+Alt+Del, Win+L, Alt+Tab, ...) -- a Warning rather than an
//   Error, because the list cannot be complete and refusing a chord that would
//   actually have worked is worse than warning about one that works anyway.
//
//   VRAM-DRIVEN CLAMPS. max_context is the one setting with no relationship to
//   the card it lands on, and vram_budget already knows how to answer it. The
//   validator runs the same planner the engine bring-up will run, so the number
//   the modal accepts is the number that will actually be granted -- rather than
//   the user setting 32768, seeing it saved, and finding 2048 in the log.
//
// ERROR TIER: none. Nothing here throws and nothing here refuses to save. The
// report is ADVICE plus in-place correction; `has_errors()` is the UI's cue to
// show the problems, not the app's cue to reject the form. A settings dialog
// that will not close is a worse failure than a setting that was corrected.
// -----------------------------------------------------------------------------
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "settings_store.hpp"

namespace rt {

enum class Severity {
    Info,      // something was adjusted; nothing is wrong
    Warning,   // it will run, but probably not the way the user expects
    Error,     // this field cannot do what it says (a dead hotkey, an unusable key)
};

const char* to_string(Severity s) noexcept;

struct Diagnostic {
    // The AssistantSettings field name, exactly as visit_fields spells it, so the
    // page can attach the message to the input that produced it rather than
    // dumping everything into one banner.
    std::string field;
    Severity    severity = Severity::Info;
    std::string message;
};

struct ValidationReport {
    std::vector<Diagnostic> items;

    [[nodiscard]] bool has_errors() const noexcept;
    [[nodiscard]] bool has_warnings() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return items.empty(); }
    void add(std::string field, Severity sev, std::string message);
    // For the log and the CLI: one line per diagnostic, already prefixed.
    void print(const char* who) const;
};

// What the validator needs to know about the LAUNCH that it cannot read off the
// settings. Everything here is a fact about the running process, so a validation
// performed before bring-up and one performed from the settings modal reach the
// same conclusions.
struct ValidationContext {
    // False disables every VRAM check: there is no card to budget against, and
    // clamping max_context on the simulated backend would refuse a configuration
    // that costs nothing.
    bool have_gpu = false;
    // Charged into the budget exactly as AppLifecycleManager charges them, so
    // the validator's answer matches the one bring-up will reach.
    bool cascade = false;
    bool tts_enabled = false;
    std::string whisper_model;
    // The two-sequence topology doubles the KV pool, so it doubles the cost of
    // every context token. 2 when isolated sessions are on.
    int branch_factor = 2;
};

// Validate (and correct) `s` in place. Runs clamp_settings first -- see the
// header block for why the two are layered rather than merged.
ValidationReport validate_settings(AssistantSettings& s, const ValidationContext& ctx);

// The chords the app itself binds, checked against each other. Exposed because
// the console overlay adds a fifth and the window registers four -- one list.
struct HotkeyBinding {
    const char* field;   // the AssistantSettings field name
    const char* label;   // what to call it in a message
    std::string spec;
};
std::vector<HotkeyBinding> app_hotkeys(const AssistantSettings& s);

// -----------------------------------------------------------------------------
// SettingsBus — the Observer/fan-out the task asks for.
//
// WHAT PROBLEM IT SOLVES. Live settings currently reach the running system
// through one function, WebUIBridge::apply_live_settings, which reads the whole
// struct and pokes six subsystems by hand. That works and it is not wrong, but
// it puts the knowledge of "who cares about mic_gain" in the UI layer, and every
// new consumer means editing a function in a file that has nothing to do with
// the consumer. It also means anything that is not the UI -- a CLI flag, a
// hotkey, the residency machine -- has no way to publish a settings change at
// all.
//
// So subscribers register themselves and the bus delivers. A subscriber is named
// purely so a throw can be attributed; the name is not an identity and
// re-subscribing does not replace.
//
// THREADING. publish() may be called from ANY thread and runs its subscribers
// SYNCHRONOUSLY on the caller's thread, holding no lock while it does (the
// subscriber list is copied under the mutex first, then released). That is the
// contract every subscriber is written against:
//
//   * a subscriber must be safe on whatever thread published -- in practice the
//     UI thread, but the residency machine publishes from the engine thread;
//   * a subscriber must NOT call into the engine directly (single-thread
//     doctrine) -- marshal with post_engine_task, exactly as the fan-out it
//     replaces already does;
//   * a subscriber must not publish (re-entrant publish would deliver a stale
//     snapshot to everyone after it); the bus does not detect this, it is a
//     rule.
//
// A THROWING SUBSCRIBER IS CAUGHT AND LOGGED, and the rest still run. One
// subsystem failing to apply a setting must not cost every other subsystem the
// same setting -- which is precisely what a hand-written fan-out of six
// statements does when the third one throws.
// -----------------------------------------------------------------------------
class SettingsBus {
public:
    using Subscriber = std::function<void(const AssistantSettings&)>;

    // `name` appears in the log when the subscriber throws. Subscriptions live
    // for the bus's lifetime: every subscriber in this app outlives it (they are
    // main()'s locals, and the bus is constructed after all of them), so an
    // unsubscribe token would be a lifetime mechanism with no user.
    void subscribe(std::string name, Subscriber fn);

    // Deliver `s` to every subscriber, in subscription order. See the threading
    // contract above.
    void publish(const AssistantSettings& s);

    [[nodiscard]] std::size_t size() const;

private:
    struct Entry {
        std::string name;
        Subscriber  fn;
    };
    mutable std::mutex mu_;
    std::vector<Entry> subs_;
};

}  // namespace rt
