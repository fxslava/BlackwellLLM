#pragma once
// -----------------------------------------------------------------------------
// conversation_router.hpp — the Local Router itself: the commit gate, the two
// legs, the dispatcher that drains one into the other, and the memory on both
// sides of it.
//
// THE PIPELINE THIS OWNS
//
//   intent (a transcript, or a typed message)
//        |
//        v
//   IntentCommitQueue  ... the gate. Dispatch IFF the local generation reached
//        |                 EOS; barge-in / token-cap / fault are dropped and
//        |                 counted. This is the arbiter in front of a BILLED API.
//        v
//   IntentDispatcher   ... its own thread. Blocks on the gate, then on the
//        |                 transport. Never touches the engine.
//        v
//   RoutedTransport    ... picks the leg. LocalEngineTransport marshals a decode
//        |                 onto the engine thread; the remote transport makes a
//        |                 network call. One atomic decides, per turn.
//        v
//   ReplySplitter      ... ONE answer, TWO audiences: a short <voice> summary to
//                          the speaker, the full <ui> answer to the screen.
//
// WHY IT IS ONE CLASS. Every piece above holds a reference into the piece below
// it, and the dispatcher's THREAD is live between them: its callbacks fire on a
// thread that reads the history window, the session latch and the reply splitter.
// Splitting them across owners means publishing that lifetime contract as a set
// of comments and hoping. Here it is member declaration order, which the compiler
// enforces -- construction forwards, destruction backwards, and stop() joins the
// thread before any of it unwinds.
//
// TEXT AND VOICE ARE THE SAME PATH. A typed message is an intent: offered to the
// SAME gate a transcript is, drained by the SAME dispatcher, answered by whichever
// leg RoutedTransport picks. See submit_typed_turn() for what that replaced and
// why the old arrangement charged for a local decode nobody displayed.
//
// THREADING, per method. Anything marked UI is called from the window message
// loop and may not touch the engine directly -- it marshals via post_engine_task.
// The dispatcher callbacks run on the DISPATCHER thread (and, for the local leg's
// token stream, on the ENGINE thread); every piece of shared state they touch
// names its guard at its declaration.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "chat_history.hpp"       // blackwell::cloud::ChatHistory
#include "intent_commit.hpp"      // blackwell::bridge::IntentCommitQueue
#include "intent_dispatcher.hpp"  // blackwell::cloud::IntentDispatcher
#include "offline_transport.hpp"  // blackwell::cloud::OfflineTransport
#include "session_store.hpp"      // blackwell::cloud::SessionStore

#include "app_context.hpp"
#include "local_transport.hpp"    // rt::LocalEngineTransport, rt::RoutedTransport
#include "reply_split.hpp"        // rt::ReplySplitter, compose_system_prompt
#include "settings_store.hpp"     // rt::sessions_path

namespace rt {

class ConversationRouter {
public:
    // How this launch reaches a remote model, and what the local one is told.
    struct Config {
        // The PERSONA as the user wrote it -- NOT the composed prompt. Composition
        // (persona + the two-audience output contract) happens inside, in one
        // place, so both legs and both speech modes are asked for the same reply
        // shape. See compose_system_prompt.
        std::string persona;
        // PRECEDENCE: a configured OpenAI-compatible endpoint wins, then
        // ANTHROPIC_API_KEY from the environment, then the offline stand-in. The
        // settings file is checked first because it is the surface the user can
        // actually SEE -- an environment variable silently overriding what the
        // Settings modal shows would be the worst of both.
        std::string remote_api_key;
        std::string remote_api_url;
        std::string remote_model;
        // Mirrors the LOCAL per-turn cap so one number bounds a turn wherever it
        // is answered -- a remote leg with no ceiling is an unbounded bill on a
        // runaway generation.
        int  max_new_tokens = 512;
        bool local_inference = false;
    };

    // Re-freeze the local model's system prefix. Lives on the CONCRETE control
    // (how a frozen prefix is laid down genuinely differs per backend), so it
    // arrives as a callable rather than a virtual -- the same seam
    // prefill_system_prompt uses. Runs on the ENGINE thread.
    using RebuildSystemPromptFn = std::function<std::uint32_t(const std::string&)>;
    // Report a rebuild's outcome back to the page. A precompute that takes seconds
    // must not look like a button that did nothing.
    using PromptAppliedFn = std::function<void(bool ok, unsigned tokens, const std::string&)>;

    // `ctx`, `view` and `control` are BORROWED and must outlive this object.
    // INIT tier: a remote leg that will not arm degrades to offline with a log
    // line; nothing here throws for that reason.
    ConversationRouter(const Config& cfg, AppContext& ctx, AssistantView& view,
                       blackwell::bridge::EngineControlBridge* control,
                       LocalEngineTransport::GenerateFn generate_locally,
                       RebuildSystemPromptFn rebuild_system_prompt);
    ~ConversationRouter();

    ConversationRouter(const ConversationRouter&) = delete;
    ConversationRouter& operator=(const ConversationRouter&) = delete;

    // Where a rebuild's verdict is delivered. Set once, after the window exists.
    void set_prompt_applied_sink(PromptAppliedFn fn) { prompt_applied_ = std::move(fn); }

    // Load sessions.json and seed the remote leg's window from it. Gated on the
    // leg the app BOOTS on: with Local selected nothing is loaded and nothing will
    // ever be saved (the local leg's memory is VRAM this process frees on the way
    // out). Call once, before start().
    void load_persisted_history(bool local_inference);

    // Start the dispatcher thread. Nothing is drained from the gate until this
    // runs.
    void start();
    // Join the dispatcher thread. Called AFTER the engine thread, so no commit can
    // arrive post-join. Idempotent.
    void stop() noexcept;

    // ---- the entry points the UI drives (UI thread) --------------------------

    // A typed message, routed exactly like a spoken one. Offered on the ENGINE
    // thread via post_engine_task, because IntentCommitQueue is single-producer
    // and its producer is DEFINED to be the engine thread (intent_commit.hpp).
    void submit_typed_turn(const std::string& text);

    // Stop whatever is generating -- decode, NETWORK and sound, unconditionally
    // and in that order. Three unrelated gestures mean it (the cancel hotkey, the
    // Stop button, opening another conversation), and they must not drift apart.
    void cancel_in_flight();

    // ---- the session sidebar (UI thread) -------------------------------------
    void push_session_list();
    void select_session(const std::string& id);
    void new_session();
    void delete_session(const std::string& id);
    // Repaint the transcript from whatever the remote window currently holds.
    // Called once the page exists, to turn restored context into visible bubbles.
    void seed_startup_view();

    // The persona changed: recompose, wipe both legs' memory (and the active
    // session's log), and marshal a KV rebuild onto the engine thread. Only called
    // when the text actually differs from what is running.
    void apply_system_prompt(const std::string& persona);

    // ---- live settings -------------------------------------------------------
    // Where the NEXT intent gets answered. One atomic; an answer already streaming
    // finishes on the leg it started on (local_transport.hpp).
    void set_use_local(bool local);
    const char* transport_name() const noexcept { return router_.name(); }
    bool transport_is_live() const noexcept { return router_.is_live(); }

    // The gate counters, for the diagnostics poller and the shutdown summary.
    blackwell::bridge::IntentCommitQueue& commit_queue() noexcept { return commit_queue_; }
    void print_shutdown_summary() const;

private:
    void wire_dispatcher_callbacks();
    // Everything a conversation change has to touch, in the order it has to
    // happen. See the .cpp -- step 4 (rewinding the LOCAL KV) is the one with no
    // visible symptom until it bites.
    void switch_to_session(const std::string& id, bool restore_from_disk);
    std::string current_session_id() const;

    // ---- DECLARATION ORDER IS DESTRUCTION ORDER (reversed). Do not reorder. ----

    AppContext&                             ctx_;
    AssistantView&                          view_;
    blackwell::bridge::EngineControlBridge* control_;
    RebuildSystemPromptFn                   rebuild_system_prompt_;
    PromptAppliedFn                         prompt_applied_;

    // Capacity 4: an intent is one finished utterance. A deep backlog of stale
    // intents is worse than refusing new ones -- see intent_commit.hpp. IDENTICAL
    // on both backends: the gate does not know or care which decode loop produced
    // the verdict it is given.
    blackwell::bridge::IntentCommitQueue commit_queue_{4};

    // The stand-in, always constructed: it is what `remote_` points at when no key
    // is configured, and it costs nothing.
    blackwell::cloud::OfflineTransport offline_transport_;

    // THE LIVE REMOTE LEG, type-erased on purpose. The concrete clients
    // (OpenAiStreamClient / ClaudeStreamClient) and their transports live only in
    // the .cpp, which is what keeps libcurl and the cloud headers out of every TU
    // that includes this one.
    struct RemoteLeg {
        // FIRST, because a transport holds a REFERENCE to its client and must be
        // destroyed before it. std::shared_ptr<void> is the type erasure: it
        // carries the concrete deleter without this header naming the type.
        std::shared_ptr<void>                               client;
        std::unique_ptr<blackwell::cloud::IIntentTransport> transport;
    };
    // Resolved in the member-init list (it must be, because router_ below takes
    // the pointer): a leg that will not arm leaves `transport` null and the app
    // falls back to offline_transport_.
    static RemoteLeg arm_remote_leg(const Config& cfg);

    RemoteLeg remote_leg_;
    // Points at remote_leg_.transport when a leg armed, at offline_transport_
    // otherwise. Never null.
    blackwell::cloud::IIntentTransport* remote_ = nullptr;

    // The same committed intent, answered by the local backbone instead of sent
    // anywhere. LocalEngineTransport marshals the decode onto the engine thread;
    // see local_transport.hpp for why this is a transport and not a special case.
    LocalEngineTransport local_transport_;
    // The dispatcher binds ONE transport for its lifetime, so the router is what
    // makes "answer locally" a live toggle rather than a restart.
    RoutedTransport router_;

    // THE PERSONA PLUS THE OUTPUT CONTRACT, composed once and never again. A mutex
    // rather than an atomic: the value is a std::string (no lock-free store exists
    // for one), the reader runs once per cloud call on the dispatcher thread, and
    // the writer is a person clicking Save.
    mutable std::mutex prompt_mu_;
    std::string        system_prompt_;

    // ---- the remote leg's memory --------------------------------------------
    // /chat/completions is stateless, so without this the cloud model forgets the
    // user's name the moment the turn ends. The local leg needs nothing
    // equivalent: its memory IS the KV cache above the system-prefix floor.
    //
    // `history_scratch_` needs no lock: the context provider is the only thing
    // that touches it, and it runs on the dispatcher thread alone. It must outlive
    // the dispatcher because RequestContext::history is a SPAN over it.
    blackwell::cloud::ChatHistory              chat_history_;
    std::vector<blackwell::cloud::ChatTurn>    history_scratch_;

    // ---- the DURABLE half: sessions.json, next to settings.json --------------
    // THE ONE RULE: the disk holds what the CLOUD leg was told and answered.
    // Nothing else. The local leg is ephemeral BY CONSTRUCTION -- nothing is
    // loaded at startup when Local is selected, and no locally-answered turn is
    // ever appended (see the completion edge, which keys off which leg ACTUALLY
    // ran).
    blackwell::cloud::SessionStore session_store_;

    // WHICH conversation is being written into. Mutable since the sidebar landed,
    // and therefore guarded: the UI thread writes it (a click in the drawer) while
    // the dispatcher thread reads it at the top of every turn.
    mutable std::mutex session_mu_;
    std::string        active_session_id_;
    // THE SESSION LATCH, and the same argument as RoutedTransport's leg latch: a
    // turn belongs to the conversation it was ASKED IN. The user can open another
    // session while an answer streams, and filing that answer under whichever
    // session happens to be active when it lands would drop a reply into a
    // conversation it has nothing to do with. Written at dispatch start and read at
    // completion -- both on the dispatcher thread, with one intent in flight
    // (intent_dispatcher.hpp), so it needs no lock of its own.
    std::string        dispatch_session_id_;

    // GUARDED, because the text sink fires on the ENGINE thread for the local leg
    // and the DISPATCHER thread for the remote one. The two never actually overlap
    // (the dispatcher is parked inside LocalEngineTransport::send for the whole
    // local generation) -- but that is a property of another header's blocking
    // behaviour, and the same argument that put a mutex on ctx_.answer_text applies
    // here for the same price.
    std::mutex    reply_mu_;
    ReplySplitter reply_split_;

    // LAST, so it is destroyed FIRST: its thread's callbacks read every member
    // above.
    blackwell::cloud::IntentDispatcher dispatcher_;

    bool started_ = false;
};

}  // namespace rt
