// =============================================================================
// session_store_test.cpp — the DURABLE half of the remote leg's memory
// (cloud/session_store.hpp), plus the restart behaviour it exists to produce.
//
// WHAT IS ACTUALLY AT RISK HERE, in the order it would hurt:
//
//   1. A local conversation reaching the disk. The local leg's whole promise is
//      that it is ephemeral; the store cannot enforce that on its own, so the
//      rule is tested where it is decided -- at the leg latch (RestartBehaviour
//      below), not just in the store.
//   2. A restart that does not restore. The feature is invisible until then, and
//      nothing else in the suite would notice its absence.
//   3. An unbounded log. The file is rewritten whole on every answered turn, so
//      "keeps everything forever" is a write that gets slower for as long as the
//      user keeps the app installed.
//   4. A corrupt or hand-edited file taking the app down, or worse, surviving as
//      far as the wire -- an invalid-UTF-8 message replayed into a request comes
//      back as an opaque HTTP 400.
//
// It lives in bridge_tests rather than cloud_tests for the same reason
// chat_history_test.cpp does: session_store.hpp needs no curl and no simdjson,
// so it must stay testable in a BUILD_CLOUD_CLIENT=OFF build -- which is also a
// build in which the store still runs.
// =============================================================================
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat_history.hpp"
#include "session_store.hpp"

namespace {

using blackwell::cloud::ChatHistory;
using blackwell::cloud::ChatTurn;
using blackwell::cloud::SessionStore;

const std::string kDefault{SessionStore::kDefaultSessionId};

// A store path unique to one test, removed on the way out. Uses the temp
// directory rather than the build tree: these tests write real files, and a
// leftover one must not be mistaken for a fixture.
class TempStore {
public:
    explicit TempStore(const char* tag) {
        path_ = (std::filesystem::temp_directory_path() /
                 ("blackwell_sessions_test_" + std::string(tag) + ".json"))
                    .string();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    ~TempStore() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        std::filesystem::remove(path_ + ".tmp", ec);
    }
    TempStore(const TempStore&) = delete;
    TempStore& operator=(const TempStore&) = delete;

    [[nodiscard]] const std::string& path() const { return path_; }

    void write_raw(const std::string& text) const {
        std::ofstream f(path_, std::ios::binary | std::ios::trunc);
        f << text;
    }
    [[nodiscard]] std::string read_raw() const {
        std::ifstream f(path_, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    [[nodiscard]] bool exists() const { return std::filesystem::exists(path_); }

private:
    std::string path_;
};

ChatTurn turn(std::string u, std::string a) {
    ChatTurn t;
    t.user = std::move(u);
    t.assistant = std::move(a);
    return t;
}

// =============================================================================
// The store on its own.
// =============================================================================

// THE FEATURE, at its smallest: what one process wrote, the next one reads.
TEST(SessionStore, PersistsAcrossAFreshInstance) {
    TempStore tmp("roundtrip");
    {
        SessionStore w(tmp.path());
        ASSERT_TRUE(w.append_turn(kDefault, turn("my name is Ann", "Hello, Ann.")));
        ASSERT_TRUE(w.append_turn(kDefault, turn("what is my name", "Ann.")));
        ASSERT_TRUE(w.save());
    }
    SessionStore r(tmp.path());
    ASSERT_TRUE(r.load());

    const std::vector<ChatTurn> v = r.turns(kDefault);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0].user, "my name is Ann");
    EXPECT_EQ(v[0].assistant, "Hello, Ann.");
    EXPECT_EQ(v[1].user, "what is my name");
}

// MULTI-SESSION AT THE DATA LAYER. The UI names one id today; the file format and
// every operation are keyed, so adding a picker is not a migration.
TEST(SessionStore, KeepsSessionsIndependent) {
    TempStore tmp("multi");
    {
        SessionStore w(tmp.path());
        w.append_turn("work", turn("standup notes", "noted"));
        w.append_turn("home", turn("grocery list", "listed"));
        w.append_turn("work", turn("and the deadline", "friday"));
        ASSERT_TRUE(w.save());
    }
    SessionStore r(tmp.path());
    ASSERT_TRUE(r.load());

    EXPECT_EQ(r.session_count(), 2u);
    EXPECT_TRUE(r.contains("work"));
    EXPECT_TRUE(r.contains("home"));
    EXPECT_EQ(r.turns("work").size(), 2u);
    ASSERT_EQ(r.turns("home").size(), 1u);
    EXPECT_EQ(r.turns("home")[0].user, "grocery list");
    // An id nobody wrote is empty, not an error -- the first-launch shape.
    EXPECT_TRUE(r.turns("nonexistent").empty());
}

TEST(SessionStore, EraseAndClearDifferInWhatTheyLeaveBehind) {
    TempStore tmp("erase");
    SessionStore s(tmp.path());
    s.append_turn("a", turn("u", "a"));
    s.append_turn("b", turn("u", "a"));

    s.clear_session("a");           // kept, emptied
    EXPECT_TRUE(s.contains("a"));
    EXPECT_TRUE(s.turns("a").empty());

    EXPECT_TRUE(s.erase("b"));      // gone
    EXPECT_FALSE(s.contains("b"));
    EXPECT_FALSE(s.erase("b"));     // and idempotent about it
}

// THE RETENTION BOUND. The file is rewritten whole on every answered turn, so an
// unbounded log is a write that gets slower for as long as the app is installed.
TEST(SessionStore, DropsTheOldestTurnPastThePerSessionCap) {
    TempStore tmp("turncap");
    SessionStore::Config cfg;
    cfg.max_turns_per_session = 3;
    SessionStore s(tmp.path(), cfg);
    for (int i = 1; i <= 10; ++i) {
        s.append_turn(kDefault, turn("u" + std::to_string(i), "a" + std::to_string(i)));
    }
    const std::vector<ChatTurn> v = s.turns(kDefault);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0].user, "u8");     // truncation is from the FRONT
    EXPECT_EQ(v[2].user, "u10");
}

// The same bound one level up. Eviction is least-recently-APPENDED, so a session
// that is still being used cannot be evicted by newer, idle ones.
TEST(SessionStore, EvictsTheLeastRecentlyAppendedSession) {
    TempStore tmp("sessioncap");
    SessionStore::Config cfg;
    cfg.max_sessions = 2;
    SessionStore s(tmp.path(), cfg);
    s.append_turn("first", turn("u", "a"));
    s.append_turn("second", turn("u", "a"));
    s.append_turn("first", turn("u2", "a2"));   // first is now the RECENT one
    s.append_turn("third", turn("u", "a"));     // evicts `second`, not `first`

    EXPECT_EQ(s.session_count(), 2u);
    EXPECT_TRUE(s.contains("first"));
    EXPECT_TRUE(s.contains("third"));
    EXPECT_FALSE(s.contains("second"));
}

// A half turn on the wire is two consecutive `user` messages, which the strict
// gateways reject. It must not enter the log either -- the log is what seeds the
// window, so a half turn stored today is a 400 after the next restart.
TEST(SessionStore, RefusesAHalfTurn) {
    TempStore tmp("halfturn");
    SessionStore s(tmp.path());
    EXPECT_FALSE(s.append_turn(kDefault, turn("asked, never answered", "")));
    EXPECT_FALSE(s.append_turn(kDefault, turn("", "answered nothing")));
    EXPECT_TRUE(s.turns(kDefault).empty());
}

// =============================================================================
// What the sidebar draws. These are cheap to get wrong in a way no other test
// notices: the list is the only handle the user has on a conversation, so a
// mis-ordered or mislabelled row is a conversation they cannot find again.
// =============================================================================

// NEWEST FIRST -- the reverse of the internal (eviction) order. Flipped in the
// store rather than in the page, so the two orderings have exactly one owner.
TEST(SessionSummaries, AreOrderedNewestFirst) {
    TempStore tmp("summaries_order");
    SessionStore s(tmp.path());
    s.append_turn("oldest", turn("u", "a"));
    s.append_turn("middle", turn("u", "a"));
    s.append_turn("newest", turn("u", "a"));

    const std::vector<blackwell::cloud::SessionSummary> v = s.list_summaries();
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0].id, "newest");
    EXPECT_EQ(v[2].id, "oldest");

    // ...and "newest" tracks the last APPEND, not creation: re-using an old
    // conversation must float it back to the top.
    s.append_turn("oldest", turn("u2", "a2"));
    EXPECT_EQ(s.list_summaries()[0].id, "oldest");
}

// The row's label is how the conversation OPENED. Not the last message: a user
// scanning for "the one where I asked about the invoice" is remembering the
// question they asked, not wherever the exchange drifted to.
TEST(SessionSummaries, PreviewIsTheFirstUserMessage) {
    TempStore tmp("summaries_preview");
    SessionStore s(tmp.path());
    s.append_turn(kDefault, turn("what is my name", "Ann."));
    s.append_turn(kDefault, turn("and my dog's", "Rex."));

    const std::vector<blackwell::cloud::SessionSummary> v = s.list_summaries();
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].preview, "what is my name");
    EXPECT_EQ(v[0].turn_count, 2u);
    EXPECT_GT(v[0].updated_at, 0);
}

// A long opening line is truncated for the row, and the cut lands on a codepoint
// boundary. This one travels through nlohmann on its way to the page, which
// REJECTS invalid UTF-8 outright -- so a split character here does not render
// badly, it drops the whole session.list event and the sidebar comes up empty.
TEST(SessionSummaries, TruncatesThePreviewOnACodepointBoundary) {
    TempStore tmp("summaries_utf8");
    SessionStore s(tmp.path());
    // "привет" -- 2 bytes per character against an odd 5-byte budget.
    s.append_turn(kDefault, turn("\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", "ok"));

    const std::vector<blackwell::cloud::SessionSummary> v = s.list_summaries(5);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].preview, "\xD0\xBF\xD1\x80\xE2\x80\xA6");   // "пр" + U+2026
    EXPECT_NO_THROW({
        // The actual failure mode, reproduced: this is the call the view makes.
        nlohmann::json j;
        j["preview"] = v[0].preview;
        (void)j.dump();
    });
}

// An empty conversation still gets a row (the sidebar has to be able to show
// where you are), just without a label to put on it.
TEST(SessionSummaries, EmptySessionHasNoPreviewButStillListsCleanly) {
    TempStore tmp("summaries_empty");
    SessionStore s(tmp.path());
    s.append_turn(kDefault, turn("u", "a"));
    s.clear_session(kDefault);

    const std::vector<blackwell::cloud::SessionSummary> v = s.list_summaries();
    ASSERT_EQ(v.size(), 1u);
    EXPECT_TRUE(v[0].preview.empty());
    EXPECT_EQ(v[0].turn_count, 0u);
}

// "New chat" twice inside one second is a double-click away, and two sessions
// sharing an id are one session the user cannot separate again.
TEST(SessionIds, AreUniqueWithinTheSameSecond) {
    std::vector<std::string> ids;
    for (int i = 0; i < 200; ++i) ids.push_back(blackwell::cloud::make_session_id());
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
    EXPECT_EQ(ids.front().rfind("chat-", 0), 0u);
}

// A generated id must survive the round trip like any other, since it is what
// every conversation past the first is keyed by.
TEST(SessionIds, WorkAsStoreKeys) {
    TempStore tmp("generated_id");
    const std::string id = blackwell::cloud::make_session_id();
    {
        SessionStore w(tmp.path());
        ASSERT_TRUE(w.append_turn(id, turn("u", "a")));
        ASSERT_TRUE(w.save());
    }
    SessionStore r(tmp.path());
    ASSERT_TRUE(r.load());
    EXPECT_TRUE(r.contains(id));
    EXPECT_EQ(r.turns(id).size(), 1u);
}

// =============================================================================
// The file, treated as untrusted input. It sits next to settings.json in a
// directory the user can open, so every one of these is reachable by hand.
// =============================================================================

// A missing file is the FIRST LAUNCH, not an error.
TEST(SessionStore, MissingFileIsNotAnError) {
    TempStore tmp("missing");
    SessionStore s(tmp.path());
    EXPECT_FALSE(s.load());          // "nothing restored"
    EXPECT_EQ(s.session_count(), 0u);
    // ...and the store is immediately usable, which is what makes the caller's
    // startup path branch-free.
    EXPECT_TRUE(s.append_turn(kDefault, turn("u", "a")));
    EXPECT_TRUE(s.save());
    EXPECT_TRUE(tmp.exists());
}

TEST(SessionStore, CorruptFileDegradesToEmptyRatherThanThrowing) {
    TempStore tmp("corrupt");
    tmp.write_raw("{ this is not json at all ][");
    SessionStore s(tmp.path());
    EXPECT_NO_THROW({ EXPECT_FALSE(s.load()); });
    EXPECT_EQ(s.session_count(), 0u);
    // And the next save repairs the file rather than refusing to touch it.
    s.append_turn(kDefault, turn("u", "a"));
    EXPECT_TRUE(s.save());
    SessionStore again(tmp.path());
    EXPECT_TRUE(again.load());
}

// One malformed session costs that session, not the file. A user who
// hand-edited half of it correctly should keep the half that parses.
TEST(SessionStore, SkipsMalformedEntriesAndKeepsTheRest) {
    TempStore tmp("partial");
    tmp.write_raw(R"({
      "version": 1,
      "sessions": [
        { "id": 42, "turns": [] },
        { "turns": [{"user":"u","assistant":"a"}] },
        { "id": "", "turns": [] },
        { "id": "good", "turns": [
            {"user":"kept","assistant":"kept too"},
            {"user":"dangling"},
            "not an object",
            {"assistant":"orphan"}
        ]}
      ]
    })");
    SessionStore s(tmp.path());
    ASSERT_TRUE(s.load());
    EXPECT_EQ(s.session_count(), 1u);

    const std::vector<ChatTurn> v = s.turns("good");
    ASSERT_EQ(v.size(), 1u);         // only the complete pair survives
    EXPECT_EQ(v[0].user, "kept");
}

// A message longer than the cap is clamped ON THE WAY IN. Without this, a large
// block pasted into the file by hand is replayed verbatim on the first request
// after a restart -- billed, and possibly over the endpoint's own limit.
TEST(SessionStore, ClampsAnOversizedMessageFromDisk) {
    TempStore tmp("clamp");
    SessionStore::Config cfg;
    cfg.max_chars_per_message = 16;
    {
        SessionStore w(tmp.path());   // default cap: writes it in full
        w.append_turn(kDefault, turn(std::string(500, 'u'), std::string(500, 'a')));
        ASSERT_TRUE(w.save());
    }
    SessionStore r(tmp.path(), cfg);
    ASSERT_TRUE(r.load());
    const std::vector<ChatTurn> v = r.turns(kDefault);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].user.size(), 16u);
    EXPECT_EQ(v[0].assistant.size(), 16u);
}

// The clamp cuts at a BYTE offset and this app transcribes Russian, so it lands
// mid-codepoint about half the time. Invalid UTF-8 on the wire is an HTTP 400
// that reaches the user as "the assistant did not answer".
TEST(SessionStore, NeverClampsInsideAUtf8Codepoint) {
    TempStore tmp("utf8");
    SessionStore::Config cfg;
    cfg.max_chars_per_message = 5;    // odd cap vs. 2-byte Cyrillic
    {
        SessionStore w(tmp.path());
        w.append_turn(kDefault,
                      turn("\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82",  // "привет"
                           "\xD0\xB4\xD0\xB0"));                                 // "да"
        ASSERT_TRUE(w.save());
    }
    SessionStore r(tmp.path(), cfg);
    ASSERT_TRUE(r.load());
    const std::vector<ChatTurn> v = r.turns(kDefault);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].user, "\xD0\xBF\xD1\x80");   // "пр" -- 2 whole codepoints, not 2.5
    EXPECT_NE(static_cast<unsigned char>(v[0].user.back()) & 0xC0, 0xC0);  // no dangling lead
}

// Cyrillic must survive the round trip byte-for-byte: nlohmann escapes
// non-ASCII only if asked to, and a store that mangled it would be a store
// nobody noticed was broken until the model answered in the wrong language.
TEST(SessionStore, RoundTripsNonAsciiExactly) {
    TempStore tmp("cyrillic");
    const std::string user = "\xD0\xBA\xD0\xB0\xD0\xBA \xD1\x82\xD0\xB5\xD0\xB1\xD1\x8F "
                             "\xD0\xB7\xD0\xBE\xD0\xB2\xD1\x83\xD1\x82";  // "как тебя зовут"
    {
        SessionStore w(tmp.path());
        w.append_turn(kDefault, turn(user, "\xD0\x90\xD0\xBD\xD0\xBD\xD0\xB0"));  // "Анна"
        ASSERT_TRUE(w.save());
    }
    SessionStore r(tmp.path());
    ASSERT_TRUE(r.load());
    ASSERT_EQ(r.turns(kDefault).size(), 1u);
    EXPECT_EQ(r.turns(kDefault)[0].user, user);
}

// The write is temp-file-plus-rename, so a reader never sees a half-written
// file and a crash mid-save cannot empty an existing log. Checked by its
// observable consequence: no .tmp is left behind, and the previous contents are
// intact until the new ones are complete.
TEST(SessionStore, SaveLeavesNoTemporaryFileBehind) {
    TempStore tmp("atomic");
    SessionStore s(tmp.path());
    s.append_turn(kDefault, turn("u", "a"));
    ASSERT_TRUE(s.save());
    EXPECT_TRUE(std::filesystem::exists(tmp.path()));
    EXPECT_FALSE(std::filesystem::exists(tmp.path() + ".tmp"));

    s.append_turn(kDefault, turn("u2", "a2"));
    ASSERT_TRUE(s.save());           // overwrites an EXISTING target
    EXPECT_FALSE(std::filesystem::exists(tmp.path() + ".tmp"));
    SessionStore r(tmp.path());
    ASSERT_TRUE(r.load());
    EXPECT_EQ(r.turns(kDefault).size(), 2u);
}

// =============================================================================
// The window seam: ChatHistory::restore, which is how the log becomes context.
// =============================================================================

// THE TAIL, not the log. The store keeps ~200 turns and the window keeps a
// handful; restoring everything would defeat both caps below it and put the
// whole transcript back on the wire.
TEST(ChatHistoryRestore, TakesTheTailOfALongerLog) {
    ChatHistory::Config cfg;
    cfg.max_turns = 3;
    ChatHistory h(cfg);

    std::vector<ChatTurn> log;
    for (int i = 1; i <= 20; ++i) {
        log.push_back(turn("u" + std::to_string(i), "a" + std::to_string(i)));
    }
    h.restore(log);

    std::vector<ChatTurn> v;
    h.snapshot(v);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0].user, "u18");
    EXPECT_EQ(v[2].user, "u20");
}

// restore() REPLACES. Called twice (or after a live turn), it must not accumulate.
TEST(ChatHistoryRestore, ReplacesRatherThanAppends) {
    ChatHistory h;
    h.begin_turn("live");
    h.append_reply("answer");
    ASSERT_TRUE(h.commit_turn());

    h.restore({turn("from disk", "restored")});
    std::vector<ChatTurn> v;
    h.snapshot(v);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].user, "from disk");
}

// A pending turn cannot survive a restore: the window it belonged to is gone.
TEST(ChatHistoryRestore, DropsAPendingTurn) {
    ChatHistory h;
    h.begin_turn("interrupted by a restore");
    h.restore({turn("u", "a")});
    h.append_reply("stray");
    EXPECT_FALSE(h.commit_turn());   // nothing was open
    EXPECT_EQ(h.size(), 1u);
}

// The file is untrusted input, so restore() re-applies the pair rule rather than
// assuming the store already did.
TEST(ChatHistoryRestore, SkipsHalfTurnsFromDisk) {
    ChatHistory h;
    h.restore({turn("complete", "pair"), turn("dangling", ""), turn("", "orphan")});
    EXPECT_EQ(h.size(), 1u);
}

// commit_turn hands back what it SEALED -- clamped and UTF-8-repaired -- because
// that is what the log must store. Re-deriving it from the raw payload would let
// the log and the window disagree, visibly only after a restart.
TEST(ChatHistoryRestore, CommitHandsBackTheSealedTurn) {
    ChatHistory::Config cfg;
    cfg.max_chars_per_message = 4;
    ChatHistory h(cfg);
    h.begin_turn("\xD0\xBF\xD1\x80\xD0\xB8");   // "при" -- 6 bytes, cap 4
    h.append_reply("okay");

    ChatTurn sealed;
    ASSERT_TRUE(h.commit_turn(&sealed));
    EXPECT_EQ(sealed.user, "\xD0\xBF\xD1\x80");  // "пр", not a split codepoint
    EXPECT_EQ(sealed.assistant, "okay");

    // ...and it matches what the window kept, which is the actual invariant.
    std::vector<ChatTurn> v;
    h.snapshot(v);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].user, sealed.user);
}

TEST(ChatHistoryRestore, DoesNotFillTheOutParamOnARefusedCommit) {
    ChatHistory h;
    h.begin_turn("asked, never answered");
    ChatTurn sealed;
    sealed.user = "untouched";
    EXPECT_FALSE(h.commit_turn(&sealed));
    EXPECT_EQ(sealed.user, "untouched");
}

// =============================================================================
// RESTART BEHAVIOUR — the four acceptance criteria, as the app sequences them.
//
// This is the part that would still be broken with every test above passing: the
// store is correct in isolation and the rule that decides WHETHER to call it
// lives in main.cpp. Reproduced here rather than trusted, because "the local
// engine writes a conversation to disk" is a promise, not a preference.
// =============================================================================

// A stand-in for the one bit main.cpp reads: which leg actually ran the intent.
// The real one is RoutedTransport::last_send_was_local(), latched at the top of
// send() so a toggle flipped mid-answer cannot change the answer afterwards.
struct AppRestartSim {
    SessionStore store;
    ChatHistory  history;
    bool         local_inference;

    AppRestartSim(const std::string& path, bool local)
        : store(path), local_inference(local) {
        // main.cpp's startup seed: gated on the leg the app BOOTS on.
        if (!local_inference && store.load()) {
            history.restore(store.turns(kDefault));
        }
    }

    // One answered turn. `on_local` is the latched leg, which is what the
    // persistence edge keys off -- NOT the toggle's current position.
    void answer(const std::string& user, const std::string& reply, bool on_local) {
        history.begin_turn(user);
        history.append_reply(reply);
        ChatTurn sealed;
        if (!history.commit_turn(&sealed)) return;
        if (on_local) return;                      // ephemeral: never touches disk
        store.append_turn(kDefault, sealed);
        (void)store.save();
    }

    [[nodiscard]] std::size_t window_size() { return history.size(); }
};

// AC: "Application restart with Cloud Engine selected retains the previous
// conversation context."
TEST(RestartBehaviour, CloudEngineRetainsContextAcrossARestart) {
    TempStore tmp("restart_cloud");
    {
        AppRestartSim first(tmp.path(), /*local=*/false);
        first.answer("my name is Ann", "Nice to meet you, Ann.", /*on_local=*/false);
    }
    AppRestartSim second(tmp.path(), /*local=*/false);
    EXPECT_EQ(second.window_size(), 1u);

    std::vector<ChatTurn> v;
    second.history.snapshot(v);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0].user, "my name is Ann");
}

// AC: "Application restart with Local Engine selected starts with a blank slate."
// Note the file is NOT empty here -- a previous cloud session wrote it. The blank
// slate is a decision, not an absence of data.
TEST(RestartBehaviour, LocalEngineStartsBlankEvenWithAPopulatedFile) {
    TempStore tmp("restart_local");
    {
        AppRestartSim cloud(tmp.path(), /*local=*/false);
        cloud.answer("remember this", "remembered", /*on_local=*/false);
    }
    ASSERT_TRUE(std::filesystem::exists(tmp.path()));

    AppRestartSim local(tmp.path(), /*local=*/true);
    EXPECT_EQ(local.window_size(), 0u);
}

// THE PROMISE. A locally-answered turn leaves no durable trace, so a later cloud
// launch cannot replay it -- and, on a machine that has never used the cloud leg,
// no file is created at all.
TEST(RestartBehaviour, LocalTurnsNeverReachTheDisk) {
    TempStore tmp("local_ephemeral");
    {
        AppRestartSim local(tmp.path(), /*local=*/true);
        local.answer("said out loud", "answered on the GPU", /*on_local=*/true);
        EXPECT_EQ(local.window_size(), 1u);   // in memory for THIS session, though
    }
    EXPECT_FALSE(std::filesystem::exists(tmp.path()));

    AppRestartSim cloud(tmp.path(), /*local=*/false);
    EXPECT_EQ(cloud.window_size(), 0u);
}

// The mixed session, which is the case the leg latch exists for: the user starts
// on the cloud, flips to local, flips back. Only the cloud-answered turns are in
// the log -- the disk is a record of what the CLOUD was told, not of the screen.
TEST(RestartBehaviour, OnlyCloudAnsweredTurnsSurviveAMixedSession) {
    TempStore tmp("mixed");
    {
        AppRestartSim app(tmp.path(), /*local=*/false);
        app.answer("cloud one", "billed", /*on_local=*/false);
        app.answer("local one", "free", /*on_local=*/true);
        app.answer("cloud two", "billed again", /*on_local=*/false);
        EXPECT_EQ(app.window_size(), 3u);     // all three are on screen and in context
    }
    AppRestartSim after(tmp.path(), /*local=*/false);
    std::vector<ChatTurn> v;
    after.history.snapshot(v);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0].user, "cloud one");
    EXPECT_EQ(v[1].user, "cloud two");
}

// AC: "The remote payload limits the sent history despite the disk storing the
// full logs." The two bounds are independent and the disk one is much larger, so
// a long-running install must not turn into a long request.
TEST(RestartBehaviour, TheDiskKeepsMoreThanTheWindowReplays) {
    TempStore tmp("bounds");
    {
        SessionStore w(tmp.path());
        for (int i = 1; i <= 40; ++i) {
            w.append_turn(kDefault, turn("u" + std::to_string(i), "a" + std::to_string(i)));
        }
        ASSERT_TRUE(w.save());
    }
    AppRestartSim app(tmp.path(), /*local=*/false);
    EXPECT_EQ(app.store.turns(kDefault).size(), 40u);   // the log kept everything
    EXPECT_EQ(app.window_size(), 4u);                   // ChatHistory's default cap

    std::vector<ChatTurn> v;
    app.history.snapshot(v);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v.back().user, "u40");                    // and it is the TAIL
}

// A persona change wipes both halves. Restoring a transcript produced by a
// different assistant is the incoherence the live clear() exists to prevent --
// keeping the file would only delay it by one launch.
TEST(RestartBehaviour, APersonaChangeClearsTheLogToo) {
    TempStore tmp("persona");
    {
        AppRestartSim app(tmp.path(), /*local=*/false);
        app.answer("under the old persona", "old answer", /*on_local=*/false);
        // main.cpp's on_system_prompt_apply, in the order it runs.
        app.history.clear();
        app.store.clear_session(kDefault);
        ASSERT_TRUE(app.store.save());
    }
    AppRestartSim after(tmp.path(), /*local=*/false);
    EXPECT_EQ(after.window_size(), 0u);
}

}  // namespace
