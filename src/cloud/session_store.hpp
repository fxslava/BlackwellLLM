#pragma once
// =============================================================================
// cloud/session_store.hpp — the DURABLE half of the remote leg's memory: a
// collection of named conversations, persisted next to settings.json.
//
// THE SPLIT WITH chat_history.hpp, which is the thing to understand first:
//
//     ChatHistory   the LIVE window. Bounded to a handful of pairs, feeds the
//                   request body, forgets the front. Owned by the turn loop.
//     SessionStore  the LOG. Keeps far more than any request will ever replay,
//                   survives the process, and is the only thing here that
//                   touches a disk.
//
//   They are deliberately not one class. The window is a COST decision (what a
//   request may carry) and the log is a RETENTION decision (what the app
//   remembers between launches); fusing them would make one number answer both
//   questions, and the honest answers differ by an order of magnitude. So the
//   store holds the full transcript, the window takes its TAIL at startup, and
//   what goes on the wire is bounded a third time by the renderer
//   (OpenAiRequestOptions::max_history_pairs). See docs/LOCAL_ROUTER.md.
//
// MULTI-SESSION AT THE DATA LAYER, SINGLE-SESSION IN THE UI. Nothing here knows
// about `default_session` beyond offering the constant: every operation is keyed
// by id, the file format is a LIST of sessions, and eviction is per-store. That
// is not speculative generality -- it is the difference between adding a session
// picker later and reformatting everyone's history file later. The app currently
// names exactly one id, and that costs nothing.
//
// WHY THE LOCAL LEG IS ABSENT FROM THIS FILE. Local inference remembers by
// construction: its memory IS the KV cache above the system-prefix floor, and
// that cache is VRAM the process frees on the way out. A local conversation
// therefore has no durable form and is not given one -- see the ownership note
// at the load/append call sites in voice_assistant/main.cpp. The store is a
// record of what the CLOUD leg was told and what it answered, and nothing else.
//
// WRITES ARE ATOMIC (temp file + rename) because the writer is a per-turn flush
// on the dispatcher thread and the reader is the next process launch. A plain
// truncate-and-write loses the whole log to a crash or a power cut in the
// millisecond it is empty; rename either lands or does not.
//
// THREADING. One mutex over everything, for the same reason ChatHistory has
// one: append_turn/save run on the DISPATCHER thread, load and the startup seed
// run on the UI thread before it exists, and a future session picker would read
// from the UI thread. The cost is one uncontended lock on a path that just
// finished a network round trip.
//
// nlohmann/json arrives through angle brackets so the root's /external:W0
// quarantine keeps it outside the /W4 /WX budget -- same as settings_store.hpp,
// which is also the file this one shares a directory with on disk.
// =============================================================================
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat_history.hpp"  // ChatTurn + detail::append_clamped / trim_utf8_tail

namespace blackwell::cloud {

// One named conversation. `turns` is oldest-first, exactly like the window's
// snapshot, so seeding one from the other is a copy and not a reversal.
struct Session {
    std::string           id;
    std::vector<ChatTurn> turns;
    // Epoch seconds of the last append. Drives eviction (see Config::max_sessions)
    // and is the one field a future session picker cannot derive from the turns
    // themselves. 0 for a session that has never been appended to.
    std::int64_t updated_at = 0;
};

// What a session picker needs to draw one row, without copying the transcript.
//
// A SUMMARY AND NOT THE SESSION, because the list is drawn far more often than a
// conversation is opened: the sidebar refreshes on every open, every new chat and
// every answered turn, and handing it 32 × 200 turns to render 32 lines of text
// would be the one gratuitous copy on a UI path.
struct SessionSummary {
    std::string  id;
    std::int64_t updated_at = 0;
    std::size_t  turn_count = 0;
    // The first thing the user said, truncated. A conversation is recognised by
    // how it OPENED -- "what's the weather" -- not by how it drifted, so this is
    // deliberately the first user message and not the last.
    std::string preview;
};

// A fresh session id: epoch seconds plus a per-process counter.
//
// NOT a formatted timestamp, and not a UUID. Epoch seconds sort correctly as
// text, match the `updated_at` field sitting next to them in the file (which is
// what makes hand-inspecting sessions.json bearable), and need no calendar
// call -- std::localtime is both deprecated on MSVC and not thread-safe. The
// counter closes the only real collision window: "New chat" twice within one
// second, which is a double-click away.
[[nodiscard]] inline std::string make_session_id() {
    static std::atomic<std::uint32_t> seq{0};
    return "chat-" + std::to_string(static_cast<std::int64_t>(std::time(nullptr))) + "-" +
           std::to_string(seq.fetch_add(1, std::memory_order_relaxed));
}

class SessionStore {
public:
    // The id the app uses for the conversation it boots into. Every OTHER session
    // is named by make_session_id(); this one is a constant because it has to be
    // the same string on the next launch -- a startup seed cannot look up an id it
    // would have to have remembered somewhere else first.
    static constexpr std::string_view kDefaultSessionId = "default_session";

    struct Config {
        // Sessions kept. Eviction drops the LEAST RECENTLY APPENDED, which is
        // the only ordering a picker-less app can defend: with no UI there is no
        // "pinned" and no "archived", so recency is the whole signal.
        std::size_t max_sessions = 32;
        // Turns kept per session -- an order of magnitude above any window that
        // will replay them, because this is the retention question and not the
        // cost one (see the preamble). Truncation is from the front.
        std::size_t max_turns_per_session = 200;
        // The same per-message ceiling ChatHistory applies, re-applied on the way
        // IN from disk: a hand-edited file is untrusted input, and a 50 KB
        // "assistant" line pasted into it would otherwise be replayed verbatim
        // on the first turn after a restart.
        std::size_t max_chars_per_message = 4000;
    };

    // `path` is the file this store owns outright -- it is rewritten whole on
    // every save, so it must not be shared with anything else.
    explicit SessionStore(std::string path, Config cfg = {})
        : path_(std::move(path)), cfg_(cfg) {}

    [[nodiscard]] const std::string& path() const { return path_; }

    // ---- persistence --------------------------------------------------------

    // Best-effort read. A missing, unreadable, corrupt, or wrong-shaped file is
    // NOT an error: the store stays empty, the app starts with a blank slate,
    // and the next save rewrites the file. Losing history must never be the
    // reason an assistant will not launch -- the same doctrine load_settings()
    // follows one directory over.
    //
    // Returns true only when at least one session was read, so a caller can tell
    // "nothing to restore" from "restored".
    bool load() {
        std::vector<Session> parsed;
        {
            std::ifstream f(path_, std::ios::binary);
            if (!f) return false;
            nlohmann::json j;
            try {
                f >> j;
                parse_into(j, parsed);
            } catch (const nlohmann::json::exception&) {
                // Includes the invalid-UTF-8 case: nlohmann rejects it at parse
                // time, which is precisely the byte sequence we must never put
                // back on the wire (it comes back as an opaque HTTP 400).
                return false;
            } catch (const std::exception&) {
                return false;
            }
        }
        const std::lock_guard<std::mutex> lk(mu_);
        sessions_ = std::move(parsed);
        enforce_session_cap();
        return !sessions_.empty();
    }

    // Best-effort atomic write: temp file, flush, rename over the target. Returns
    // false if the log could not be persisted; the caller keeps running on the
    // in-memory copy, because a full disk is not a reason to stop answering.
    bool save() const {
        std::string text;
        {
            const std::lock_guard<std::mutex> lk(mu_);
            nlohmann::json j;
            j["version"] = kFormatVersion;
            nlohmann::json arr = nlohmann::json::array();
            for (const Session& s : sessions_) {
                nlohmann::json js;
                js["id"] = s.id;
                js["updated_at"] = s.updated_at;
                nlohmann::json turns = nlohmann::json::array();
                for (const ChatTurn& t : s.turns) {
                    turns.push_back({{"user", t.user}, {"assistant", t.assistant}});
                }
                js["turns"] = std::move(turns);
                arr.push_back(std::move(js));
            }
            j["sessions"] = std::move(arr);
            try {
                // error_handler_t::replace rather than the default THROW: a
                // malformed byte that reached us from a decoder must degrade to
                // U+FFFD in the log, not cost the user every session they have.
                text = j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
            } catch (const nlohmann::json::exception&) {
                return false;
            }
        }

        std::error_code ec;
        const std::filesystem::path target(path_);
        if (target.has_parent_path()) {
            // create_directories reports "already exists" through `ec`-free
            // success, so only a real failure lands below.
            std::filesystem::create_directories(target.parent_path(), ec);
            if (ec) return false;
        }
        // Sibling temp file: rename is only guaranteed atomic within one volume,
        // and %TEMP% is routinely on another one.
        std::filesystem::path tmp = target;
        tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            f << text;
            f.flush();
            if (!f) return false;
        }
        std::filesystem::rename(tmp, target, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);  // do not leave litter behind
            return false;
        }
        return true;
    }

    // ---- the session collection --------------------------------------------

    // A copy, oldest-first, of everything kept for `id`. Empty for an unknown id,
    // which is the correct answer for a first launch and needs no special case at
    // the call site.
    [[nodiscard]] std::vector<ChatTurn> turns(std::string_view id) const {
        const std::lock_guard<std::mutex> lk(mu_);
        const Session* s = find(id);
        return s != nullptr ? s->turns : std::vector<ChatTurn>{};
    }

    // Records one FINISHED exchange, creating the session if needed. A turn with
    // an empty half is refused (and returns false) for the same protocol reason
    // ChatHistory::commit_turn refuses one: a lone `user` replayed later is two
    // consecutive user messages on the wire, which the strict gateways answer
    // with a 400.
    //
    // Does NOT save -- the caller decides when the flush is worth it, because
    // the useful batching (one write per turn vs. one per burst) is only visible
    // from up there.
    bool append_turn(std::string_view id, ChatTurn t) {
        clamp_turn(t);
        if (t.user.empty() || t.assistant.empty()) return false;
        const std::lock_guard<std::mutex> lk(mu_);
        Session& s = touch(id);
        s.turns.push_back(std::move(t));
        while (s.turns.size() > cfg_.max_turns_per_session) {
            s.turns.erase(s.turns.begin());  // oldest first
        }
        return true;
    }

    // Replaces a session's whole transcript. For an importer or a future "edit
    // this conversation", not for the turn loop -- which appends.
    void set_turns(std::string_view id, std::vector<ChatTurn> turns) {
        for (ChatTurn& t : turns) clamp_turn(t);
        const std::lock_guard<std::mutex> lk(mu_);
        Session& s = touch(id);
        s.turns = std::move(turns);
        while (s.turns.size() > cfg_.max_turns_per_session) {
            s.turns.erase(s.turns.begin());
        }
    }

    // Empties a session but keeps it (and its recency). The app calls this when
    // the persona changes: the transcript was produced by a different assistant,
    // so replaying it after a restart would reproduce exactly the incoherence the
    // live window's clear() exists to prevent -- just delayed by one launch.
    void clear_session(std::string_view id) {
        const std::lock_guard<std::mutex> lk(mu_);
        if (Session* s = find(id); s != nullptr) s->turns.clear();
    }

    // Forgets a session entirely. Returns false for an unknown id.
    bool erase(std::string_view id) {
        const std::lock_guard<std::mutex> lk(mu_);
        for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
            if (it->id == id) {
                sessions_.erase(it);
                return true;
            }
        }
        return false;
    }

    void clear_all() {
        const std::lock_guard<std::mutex> lk(mu_);
        sessions_.clear();
    }

    // Ids in the store's own order: least recently appended first, which is
    // eviction order and therefore the order a picker would want reversed.
    [[nodiscard]] std::vector<std::string> session_ids() const {
        const std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::string> ids;
        ids.reserve(sessions_.size());
        for (const Session& s : sessions_) ids.push_back(s.id);
        return ids;
    }

    // One row per session, NEWEST FIRST -- the reverse of the internal order.
    //
    // Reversed HERE rather than in the page, because the eviction order and the
    // display order are opposite readings of the same fact and only one of them
    // is the caller's business. A UI that had to remember to flip the list is a
    // UI that will eventually forget.
    //
    // `preview_chars` is a byte budget, clamped to a codepoint boundary: this app
    // transcribes Russian, so a naive substr would put a split character into the
    // DOM (and, since the same string travels as JSON, into a serializer that
    // rejects it).
    [[nodiscard]] std::vector<SessionSummary> list_summaries(
        std::size_t preview_chars = 80) const {
        const std::lock_guard<std::mutex> lk(mu_);
        std::vector<SessionSummary> out;
        out.reserve(sessions_.size());
        for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
            SessionSummary sum;
            sum.id = it->id;
            sum.updated_at = it->updated_at;
            sum.turn_count = it->turns.size();
            if (!it->turns.empty()) {
                sum.preview = it->turns.front().user;
                if (sum.preview.size() > preview_chars) {
                    sum.preview.resize(preview_chars);
                    detail::trim_utf8_tail(sum.preview);
                    sum.preview += "\xE2\x80\xA6";  // U+2026, one glyph rather than "..."
                }
            }
            out.push_back(std::move(sum));
        }
        return out;
    }

    [[nodiscard]] std::size_t session_count() const {
        const std::lock_guard<std::mutex> lk(mu_);
        return sessions_.size();
    }

    [[nodiscard]] bool contains(std::string_view id) const {
        const std::lock_guard<std::mutex> lk(mu_);
        return find(id) != nullptr;
    }

private:
    // Bumped only for a change no reader can absorb by ignoring unknown keys.
    // The parser below already tolerates missing fields, so additive changes do
    // not need it.
    static constexpr int kFormatVersion = 1;

    void clamp_turn(ChatTurn& t) const {
        clamp_message(t.user);
        clamp_message(t.assistant);
    }

    void clamp_message(std::string& s) const {
        if (s.size() > cfg_.max_chars_per_message) {
            s.resize(cfg_.max_chars_per_message);
            // The resize above cuts at a BYTE offset and this app transcribes
            // Russian, so the cut lands mid-codepoint roughly half the time.
            detail::trim_utf8_tail(s);
        }
    }

    // Callers hold mu_ for all four of these.
    [[nodiscard]] const Session* find(std::string_view id) const {
        for (const Session& s : sessions_) {
            if (s.id == id) return &s;
        }
        return nullptr;
    }
    [[nodiscard]] Session* find(std::string_view id) {
        for (Session& s : sessions_) {
            if (s.id == id) return &s;
        }
        return nullptr;
    }

    // Finds or creates `id`, stamps it, and moves it to the BACK -- so the vector
    // is always least-recently-appended first and eviction is a pop from the
    // front. O(n) on a container of tens; a map plus a recency list would be
    // three data structures to answer a question one linear scan already answers.
    Session& touch(std::string_view id) {
        const std::int64_t now = static_cast<std::int64_t>(std::time(nullptr));
        for (std::size_t i = 0; i < sessions_.size(); ++i) {
            if (sessions_[i].id != id) continue;
            Session moved = std::move(sessions_[i]);
            sessions_.erase(sessions_.begin() + static_cast<std::ptrdiff_t>(i));
            moved.updated_at = now;
            sessions_.push_back(std::move(moved));
            return sessions_.back();
        }
        Session s;
        s.id = std::string(id);
        s.updated_at = now;
        sessions_.push_back(std::move(s));
        enforce_session_cap();
        return sessions_.back();
    }

    void enforce_session_cap() {
        while (sessions_.size() > cfg_.max_sessions) {
            sessions_.erase(sessions_.begin());  // least recently appended
        }
    }

    // Tolerant by design: an unknown version, a missing key, or one malformed
    // session costs that session and nothing else. A settings file the user
    // hand-edited half-correctly should lose the half they broke.
    void parse_into(const nlohmann::json& j, std::vector<Session>& out) const {
        if (!j.is_object() || !j.contains("sessions")) return;
        const nlohmann::json& arr = j.at("sessions");
        if (!arr.is_array()) return;
        for (const nlohmann::json& js : arr) {
            if (!js.is_object() || !js.contains("id")) continue;
            Session s;
            try {
                s.id = js.at("id").get<std::string>();
            } catch (const nlohmann::json::exception&) {
                continue;
            }
            if (s.id.empty()) continue;
            if (js.contains("updated_at")) {
                try {
                    s.updated_at = js.at("updated_at").get<std::int64_t>();
                } catch (const nlohmann::json::exception&) {
                    s.updated_at = 0;
                }
            }
            if (js.contains("turns") && js.at("turns").is_array()) {
                for (const nlohmann::json& jt : js.at("turns")) {
                    if (!jt.is_object()) continue;
                    ChatTurn t;
                    try {
                        if (jt.contains("user")) t.user = jt.at("user").get<std::string>();
                        if (jt.contains("assistant")) {
                            t.assistant = jt.at("assistant").get<std::string>();
                        }
                    } catch (const nlohmann::json::exception&) {
                        continue;
                    }
                    clamp_turn(t);
                    // Half turns are dropped HERE rather than at replay time, so
                    // nothing downstream has to know the file can contain them.
                    if (t.user.empty() || t.assistant.empty()) continue;
                    s.turns.push_back(std::move(t));
                }
            }
            while (s.turns.size() > cfg_.max_turns_per_session) {
                s.turns.erase(s.turns.begin());
            }
            out.push_back(std::move(s));
        }
    }

    mutable std::mutex   mu_;
    std::string          path_;
    Config               cfg_;
    std::vector<Session> sessions_;  // least recently appended at the front
};

}  // namespace blackwell::cloud
