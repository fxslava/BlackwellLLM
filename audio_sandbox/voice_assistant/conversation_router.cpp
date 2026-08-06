// conversation_router.cpp — see conversation_router.hpp.
//
// MOVED, NOT CHANGED. The gate capacity, the transport precedence, the dispatcher
// callbacks, the persistence rule, the session-switch order and every log line are
// byte-for-byte what main() ran. The one structural difference is that the three
// dispatcher callbacks are now member functions wired in wire_dispatcher_callbacks()
// instead of lambdas capturing eighteen locals by reference -- which is the whole
// point of the extraction, since every one of those captures was a lifetime
// contract held together by declaration order in a 2000-line function.
#include "conversation_router.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#if defined(BLACKWELL_HAVE_CLOUD_CLIENT)
#include "claude_stream_client.hpp"
#include "claude_transport.hpp"
#include "openai_stream_client.hpp"
#include "openai_transport.hpp"
#endif

namespace rt {

ConversationRouter::RemoteLeg ConversationRouter::arm_remote_leg(
    [[maybe_unused]] const Config& cfg) {
    RemoteLeg leg;
#if defined(BLACKWELL_HAVE_CLOUD_CLIENT)
    // The key may come from the Settings modal (persisted per machine under
    // %LOCALAPPDATA%) or from OPENAI_API_KEY. It is deliberately NOT checked into
    // settings_store.hpp's defaults: a default in a tracked header enters git
    // history permanently and is compiled into every binary built from the tree.
    // The settings file wins when both are set, because it is the one the user can
    // see.
    std::string remote_key = cfg.remote_api_key;
    if (remote_key.empty()) {
        if (const char* env = std::getenv("OPENAI_API_KEY"); env != nullptr) remote_key = env;
    }

    if (!remote_key.empty() && !cfg.remote_api_url.empty()) {
        // INIT tier: a curl handle we cannot create is fatal for this leg only, so
        // it degrades to offline rather than taking the app down. The user still
        // has a working assistant and a log line saying why.
        try {
            blackwell::cloud::OpenAiStreamClient::Config ocfg;
            ocfg.api_key  = remote_key;
            ocfg.base_url = cfg.remote_api_url;
            auto client =
                std::make_shared<blackwell::cloud::OpenAiStreamClient>(std::move(ocfg));

            blackwell::cloud::OpenAiRequestOptions oopt;
            oopt.max_tokens = cfg.max_new_tokens;
            leg.transport = std::make_unique<blackwell::cloud::OpenAiTransport>(
                *client, cfg.remote_model, oopt);
            // The KEY IS NEVER PRINTED. The endpoint is, because a wrong base URL
            // is the likeliest misconfiguration and otherwise only surfaces as a
            // 404 on the first real utterance.
            std::printf("[cloud] remote leg: %s (model %s) -- LIVE\n",
                        client->endpoint().c_str(), cfg.remote_model.c_str());
            leg.client = std::move(client);
            return leg;
        } catch (const std::exception& e) {
            leg.transport.reset();
            leg.client.reset();
            std::fprintf(stderr, "[cloud] remote leg failed to arm (%s) -- falling back\n",
                         e.what());
        }
    }

    if (const char* key = std::getenv("ANTHROPIC_API_KEY"); key != nullptr && *key != '\0') {
        blackwell::cloud::ClaudeStreamClient::Config ccfg;
        ccfg.api_key = key;
        ccfg.beta = "server-side-fallback-2026-07-01";
        auto client = std::make_shared<blackwell::cloud::ClaudeStreamClient>(std::move(ccfg));
        leg.transport = std::make_unique<blackwell::cloud::ClaudeTransport>(*client);
        leg.client = std::move(client);
        std::printf("[cloud] ANTHROPIC_API_KEY present -- LIVE transport armed\n");
        return leg;
    }

    std::printf("[cloud] no remote API key (Settings -> Remote API endpoint, or "
                "OPENAI_API_KEY) -- Mode: Offline (Simulated)\n");
#else
    std::printf("[cloud] built without BUILD_CLOUD_CLIENT -- Mode: Offline (Simulated)\n");
#endif
    return leg;
}

ConversationRouter::ConversationRouter(const Config& cfg, AppContext& ctx, AssistantView& view,
                                       blackwell::bridge::EngineControlBridge* control,
                                       LocalEngineTransport::GenerateFn generate_locally,
                                       RebuildSystemPromptFn rebuild_system_prompt)
    : ctx_(ctx),
      view_(view),
      control_(control),
      rebuild_system_prompt_(std::move(rebuild_system_prompt)),
      remote_leg_(arm_remote_leg(cfg)),
      remote_(remote_leg_.transport ? remote_leg_.transport.get()
                                    : static_cast<blackwell::cloud::IIntentTransport*>(
                                          &offline_transport_)),
      local_transport_(control, std::move(generate_locally)),
      router_(&local_transport_, remote_),
      // THE PERSONA PLUS THE OUTPUT CONTRACT, composed once here and never again.
      // `cfg.persona` is what the user wrote and what the Settings modal shows;
      // what a MODEL is told is always this. Composing at the point the persona
      // enters the pipeline -- rather than at each of the four places it is read --
      // is what keeps the two legs and the two speech modes from drifting into
      // asking for different reply shapes.
      system_prompt_(compose_system_prompt(cfg.persona)),
      // Multi-session at the data layer, one id in the UI: the store is a
      // collection keyed by session id, and this app names exactly one of them
      // until a picker exists (session_store.hpp's preamble on why that split is
      // worth its keep).
      session_store_(sessions_path()),
      active_session_id_(blackwell::cloud::SessionStore::kDefaultSessionId),
      dispatch_session_id_(active_session_id_),
      // ---- ONE answer, TWO audiences --------------------------------------
      // The reply arrives as one stream carrying two tagged blocks -- a short
      // <voice> summary and the full <ui> answer -- and this pulls them apart as
      // they stream. See reply_split.hpp for the contract, the fail-open behaviour
      // when the model ignores it, and why the parser lives next to the
      // instruction that asks for the tags.
      //
      // THE SINKS ARE WHERE THE SPLIT BECOMES REAL: everything that used to happen
      // to the whole reply now happens to the half it belongs to. The screen, the
      // transcript and the durable log take the <ui> text; the speaker takes the
      // <voice> text and nothing else.
      reply_split_(
          // -> the screen and the remote leg's memory
          [this](std::string_view s) {
              view_.on_remote_token(s);
              // UNCONDITIONAL, unlike the voice sink below: that one is compiled
              // out with speech output, whereas the remote leg's memory must not
              // depend on whether this build can speak. Clamped and bounded inside
              // the store.
              chat_history_.append_reply(s);
          },
          // -> the speaker
          [this]([[maybe_unused]] std::string_view s) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
              // PushToken appends under a short mutex and returns, so neither
              // producer thread is blocked and all synthesis stays on the TTS
              // worker.
              //
              // Muted: the text is not handed over at all. Redundant with the
              // cancel latch left set at dispatch-start -- and kept anyway, because
              // this is the edge a mute that lands MID-answer has to stop, and it
              // must not depend on the reader agreeing that the bridge drops
              // post-cancel pushes.
              if (ctx_.tts != nullptr && !ctx_.tts_muted.load(std::memory_order_acquire)) {
                  ctx_.tts->PushToken(s);
              }
              {
                  const std::lock_guard<std::mutex> lk(ctx_.answer_mu);
                  ctx_.answer_text.append(s);
              }
#endif
          }),
      dispatcher_(commit_queue_, router_, [this] {
          blackwell::cloud::RequestContext c;
          // FROZEN BETWEEN EDITS. Nothing volatile may appear here: a timestamp or
          // a session id in this string drives the prompt-cache hit rate to zero,
          // silently and expensively (intent_request.hpp). A user editing the
          // system prompt DOES invalidate that cache -- correctly, since it is a
          // different prompt -- but only once, and only when they asked for it.
          {
              const std::lock_guard<std::mutex> lk(prompt_mu_);
              c.instructions = system_prompt_;
          }
          c.glossary = "This is a spoken-language voice assistant session.";
          c.committed_prefix = "";
          // Only FINISHED turns: the one being dispatched right now is still
          // pending in the store and arrives separately as `c.intent`, so it cannot
          // appear twice. Re-taken per attempt, which is free -- a retry replays the
          // identical window.
          chat_history_.snapshot(history_scratch_);
          c.history = history_scratch_;
          return c;
      }) {
    router_.set_use_local(cfg.local_inference);
    wire_dispatcher_callbacks();
}

ConversationRouter::~ConversationRouter() { stop(); }

std::string ConversationRouter::current_session_id() const {
    const std::lock_guard<std::mutex> lk(session_mu_);
    return active_session_id_;
}

void ConversationRouter::load_persisted_history(bool local_inference) {
    // Loaded UNCONDITIONALLY, local mode included -- the sidebar has to be able to
    // list what is on disk, and READING the file is not the same act as seeding a
    // conversation from it. What local mode suppresses is the seed below.
    const bool store_loaded = session_store_.load();

    // THE STARTUP SEED, gated on the leg the app BOOTS on. `local_inference` is a
    // live toggle, but this decision is not re-taken when it flips: splicing a
    // conversation from three days ago into one already on screen would be a worse
    // answer than starting the cloud leg cold, and the user never asked for it.
    // Flipping to Local mid-session likewise leaves the in-memory window alone --
    // what the user sees on screen stays one conversation; it is the DISK that
    // never learns about the local turns.
    if (!local_inference) {
        if (store_loaded) {
            // The TAIL of the log, not the log: sessions.json keeps ~200 turns, the
            // window keeps 4, and the wire carries 3 (see docs/LOCAL_ROUTER.md's
            // table). Restoring the whole log into the window would defeat both of
            // the caps below it.
            const std::vector<blackwell::cloud::ChatTurn> saved =
                session_store_.turns(active_session_id_);
            chat_history_.restore(saved);
            std::printf("[history] restored %zu turn(s) from %s (session \"%s\")\n",
                        saved.size(), session_store_.path().c_str(),
                        active_session_id_.c_str());
        } else {
            std::printf("[history] no persisted session at %s -- starting cold\n",
                        session_store_.path().c_str());
        }
    } else {
        // Said out loud, because "the assistant forgot everything" is the first
        // thing a user reports and this is the reason.
        std::printf("[history] local inference selected -- ephemeral session, nothing "
                    "loaded and nothing will be saved\n");
    }
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// THE SPEECH TAP LIVES HERE, on the ANSWER stream, and nowhere else.
//
// These three callbacks are the only place the ASSISTANT'S REPLY exists as text.
// Everything the speech pipeline emits through on_token is the TRANSCRIPT of what
// the USER said -- session B decodes the user's words, publish_turn extracts them,
// and the gate hands them here as an intent. Tapping on_token therefore reads the
// user their own sentence back; tapping the dispatcher reads them the answer.
//
// It also means local and cloud replies are spoken by the same code:
// RoutedTransport picks the leg, and neither branch is visible from here
// (local_transport.hpp's whole argument).
// ---------------------------------------------------------------------------
void ConversationRouter::wire_dispatcher_callbacks() {
    dispatcher_.set_on_dispatch_start([this](const blackwell::bridge::IntentRecord& r) {
        view_.on_dispatch_start(r.sequence);
        // A new answer: drop every scrap of the previous one. A held partial tag
        // surviving into this turn would splice half a "</ui>" onto the front of it.
        {
            const std::lock_guard<std::mutex> lk(reply_mu_);
            reply_split_.reset();
        }
        // The session this turn belongs to, frozen for its whole lifetime. See the
        // latch's declaration for why it is taken here and not read again at
        // completion.
        dispatch_session_id_ = current_session_id();
        // Opens the turn HERE and not at the commit gate: an intent the gate
        // accepted but the dispatcher never sent is not part of the conversation,
        // and the gate runs one stage earlier on the other sequence.
        chat_history_.begin_turn(r.payload);
#if defined(VOICE_ASSISTANT_HAS_TTS)
        // A new answer is starting: clear the cancelled latch that the barge-in
        // edge set when the user began speaking, so the tokens about to arrive are
        // actually spoken. This is the ANSWER's start, which is why it is here and
        // not on a pipeline state -- the pipeline's DECODE state belongs to the
        // transcription pass.
        //
        // UNLESS THE SPEAKER IS MUTED, and this is the arm point that makes the
        // mute mean something: leaving the latch SET is what turns the bridge's
        // PushToken/EndOfStream into no-ops for the whole turn (see
        // TTSDuplexBridge::PushToken), so no chunk is ever handed to the solver and
        // nothing can survive in the chunker to be spliced onto the front of a
        // later, unmuted reply.
        if (ctx_.tts != nullptr && !ctx_.tts_muted.load(std::memory_order_acquire)) {
            ctx_.tts->Resume();
        }
        {
            const std::lock_guard<std::mutex> lk(ctx_.answer_mu);
            ctx_.answer_text.clear();
        }
#endif
    });

    // THE ONE PRODUCER OF THE ASSISTANT'S TEXT, and it produces two streams from
    // it. Nothing here knows which leg is answering -- the splitter sits above
    // RoutedTransport's choice, so a local reply and a billed one are divided by
    // the same code.
    dispatcher_.set_on_text([this](std::string_view s) {
        const std::lock_guard<std::mutex> lk(reply_mu_);
        reply_split_.push(s);
    });

    dispatcher_.set_on_complete([this](const blackwell::bridge::IntentRecord&,
                                       const blackwell::cloud::Result& r) {
        // FIRST, before anything seals or flushes. finish() surrenders the bytes
        // still held against a possible tag and, when the model ignored the
        // contract, hands the written answer to the speaker -- so it must run
        // before chat_history_ seals the turn (which is what the UI half feeds) and
        // before the TTS flush below (which is what the voice half feeds).
        {
            const std::lock_guard<std::mutex> lk(reply_mu_);
            reply_split_.finish();
        }
        // THE USER PRESSED STOP. Not a failure, and it must not be painted as one:
        // an answer the user cut off themselves needs no red note explaining what
        // went wrong. The local leg reports it as a barge-in (its decode loop was
        // superseded by the epoch bump), the remote leg as Cancelled -- two
        // mechanisms, one outcome, and the UI is told the outcome.
        const bool cancelled =
            r.status == blackwell::cloud::Status::Cancelled || r.stop_reason == "barge_in";
        view_.on_remote_final(r.status == blackwell::cloud::Status::Ok,
                              r.error_detail.empty()
                                  ? std::string(blackwell::cloud::to_string(r.status))
                                  : r.error_detail,
                              cancelled);
        // Seals the turn, or drops it whole. Only a completed exchange becomes
        // memory: a failed request, a refusal or a barge-in would otherwise enter
        // the window as an assistant message the user never heard -- and a barged-in
        // half-sentence replayed as an accepted answer is worse than no memory at
        // all. commit_turn also refuses a pair with an empty half, so the
        // alternation the API requires holds by construction.
        blackwell::cloud::ChatTurn sealed;
        const bool committed =
            r.status == blackwell::cloud::Status::Ok && chat_history_.commit_turn(&sealed);
        if (r.status != blackwell::cloud::Status::Ok) chat_history_.abandon_turn();

        // ---- THE PERSISTENCE EDGE -------------------------------------------
        // Gated on which leg ACTUALLY RAN this intent, not on the toggle's current
        // position: the user can flip the switch while an answer streams, and
        // asking the router "are you local?" here would occasionally write a local
        // conversation to disk (which the local leg promises never happens) or drop
        // a turn the cloud was paid for. RoutedTransport latches the leg at the top
        // of send() for exactly this reader -- see last_send_was_local().
        //
        // The SEALED turn is what is written, not the raw payload: the window's
        // clamps and its UTF-8 repair have already run, and the log must agree with
        // the window about what was said or the two diverge in a way that only
        // appears after a restart.
        //
        // Flushed per turn rather than at shutdown. This process holds ~8 GB of GPU
        // state and is killed rather than closed often enough that a shutdown-only
        // flush is a history that mostly does not survive; a few KB of JSON on a
        // path that just waited out a network round trip is not a cost worth
        // optimising against that. Filed under the LATCHED session, not the active
        // one: if the user opened another conversation while this answer streamed,
        // the answer still belongs to the one they asked in.
        if (committed && !router_.last_send_was_local()) {
            if (session_store_.append_turn(dispatch_session_id_, sealed) &&
                !session_store_.save()) {
                // Non-fatal and deliberately quiet-ish: the in-memory window is
                // unaffected, so this session keeps its memory and only the NEXT
                // launch is poorer for it.
                std::fprintf(stderr, "[history] could not persist to %s\n",
                             session_store_.path().c_str());
            }
            // The sidebar shows a preview and an ordering, and BOTH just changed --
            // a conversation that was never persisted before has only now become a
            // row at all. Pushed from the dispatcher thread, which is safe by the
            // same route every other producer takes: the view serialises and the
            // window's post_event marshals (assistant_view.hpp's threading note).
            view_.set_sessions(session_store_.list_summaries(), current_session_id(),
                               router_.use_local());
        }
#if defined(VOICE_ASSISTANT_HAS_TTS)
        // EOS / stop / error -- whichever ended the generation, the turn is over.
        // Flush so a tail shorter than min_chunk_chars is still spoken.
        // Unconditional on status: a failed answer may still have streamed a partial
        // sentence, and leaving it buffered would splice it onto the FRONT of the
        // next reply.
        //
        // MUTED TURNS END WITH A CANCEL, NOT A FLUSH, and the difference is the next
        // reply rather than this one: the flush exists to speak a tail, which is
        // precisely what must not happen here, while the buffer it would have
        // flushed still has to be emptied before the next Resume() can reach it.
        // Cancel does both and is idempotent with the mute edge that already fired.
        const bool muted = ctx_.tts_muted.load(std::memory_order_acquire);
        if (ctx_.tts != nullptr) {
            if (muted) {
                ctx_.tts->BargeIn();
            } else {
                ctx_.tts->EndOfTurn();
            }
        }

        // THE HANDOFF LINE. This is the seam people go looking for when the
        // assistant answers on screen but says nothing, so it prints the text that
        // reached the speaker and how much of it there was.
        //
        // It is logged HERE, on the answer stream's completion, and not at the
        // commit gate. The gate sits one stage EARLIER and on the other sequence: it
        // decides whether the user's TRANSCRIPT becomes an intent worth dispatching.
        // Its "EOS -> DISPATCHED (N tokens)" line counts transcript tokens, and the
        // answer that follows deliberately never re-enters it.
        //
        // A 0-char line here means the LLM emitted nothing (look up at [Decode
        // Stop]); a non-zero line with no [tts-worker] line after it means the text
        // died between the chunker and the synthesiser, which is the next place to
        // look.
        std::string spoken;
        {
            const std::lock_guard<std::mutex> lk(ctx_.answer_mu);
            spoken.swap(ctx_.answer_text);
        }
        if (ctx_.tts != nullptr) {
            // The muted wording is not cosmetic: this line is the seam people check
            // when the assistant answers on screen and says nothing, and "pushed N
            // chars" next to a silent speaker is exactly the wrong thing to tell
            // them when the reason is a muted icon rather than a broken synthesiser.
            std::printf(muted
                            ? "[answer->tts] MUTED -- not synthesised (%zu chars): \"%s\"\n"
                            : "[answer->tts] pushed LLM response to TTS (%zu chars): \"%s\"\n",
                        spoken.size(), spoken.c_str());
            std::fflush(stdout);
        }
#endif
    });
}

void ConversationRouter::start() {
    if (started_) return;
    started_ = true;
    dispatcher_.start();
}

void ConversationRouter::stop() noexcept {
    if (!started_) return;
    started_ = false;
    dispatcher_.stop();
}

// ---- THE TYPED TURN, ROUTED LIKE A SPOKEN ONE -------------------------------
// A typed message is an INTENT, exactly like a transcript, and it enters the
// pipeline at exactly the same point one does: offered to the commit gate, drained
// by the dispatcher, answered by whichever leg RoutedTransport picks. Nothing here
// is new machinery: the routing, the leg latch, the session latch and the
// persistence rule all already existed and applied to every spoken turn. This path
// was the one thing bypassing them, so the fix was to stop bypassing.
//
// WHAT IT USED TO DO, AND WHY THAT WAS WRONG. It called submit_text(), which runs a
// full LOCAL decode of an answer and then offers THAT ANSWER to the gate as the
// intent. So with the cloud leg selected, the GPU generated a reply nobody
// displayed and the remote model was asked to respond to it -- the toggle was, in
// effect, ignored for typed input, and paid for twice over.
//
// OFFERED ON THE ENGINE THREAD, via post_engine_task. IntentCommitQueue is
// single-producer and its producer is defined to be the engine thread
// (intent_commit.hpp); the UI thread offering directly would break that contract
// for the convenience of saving a marshal on a path a human drives at single-digit
// events per minute. Mode C's transcript takes the identical route for the
// identical reason.
void ConversationRouter::submit_typed_turn(const std::string& text) {
    if (text.empty()) return;
    // THE BUBBLE FIRST, THEN THE OFFER -- program order, and it is the whole
    // message-ordering guarantee. offer() wakes the dispatcher thread, whose first
    // act is on_dispatch_start -> "remote.start" -> the page creates the
    // assistant's bubble. Both land in the SAME FIFO UI queue, so whichever is
    // enqueued first is the bubble drawn first; offering first put the answer above
    // the question every time the dispatcher won the race. Do not merge these.
    view_.on_user_text(text);
    // `text` is captured BY VALUE: this task outlives the call that queued it, and
    // a reference capture would compile and dangle.
    const bool queued = control_->post_engine_task([this, text] {
        // Eos, and it is not a fiction: the commit rule asks whether the thought
        // finished on its own, and a message the user pressed Enter on is the most
        // finished a thought gets. The reasons the gate rejects (barge-in, a token
        // cap, a fault) are properties of a decode loop that does not run here.
        if (commit_queue_.offer(blackwell::bridge::TerminationReason::Eos,
                                control_->active_generation(), std::string(text),
                                // token_count counts LOCAL DECODE tokens, which is
                                // what makes dropped_token_cap() diagnostic. Nothing
                                // is tokenized here, so a plausible-looking byte
                                // count would put a wrong number where a reader
                                // expects that meaning.
                                /*token_count=*/0u)) {
            return;
        }
        // The only way an Eos fails to commit is a full queue: an earlier answer is
        // still streaming. SAID TO THE USER, not just to stderr -- nothing else will
        // speak for this turn, and a message that vanished with the composer still
        // showing Stop is the worst of the available failures.
        std::fprintf(stderr, "[typed] NOT dispatched (the commit queue is full -- an earlier "
                             "answer is still streaming)\n");
        view_.on_remote_final(false,
                              "The assistant is still answering — try again in a moment.");
    });
    if (!queued) {
        view_.on_remote_final(false, "The assistant is busy — try again in a moment.");
    }
}

// ALL THREE LEGS, UNCONDITIONALLY, and this is the same argument
// RoutedTransport::abort() makes one layer down: this is not routing a turn, it is
// stopping whatever is running. Asking which leg is live would mean a Stop pressed
// just after the user flipped the toggle cancels the idle one. Each is a no-op when
// there is nothing to stop.
void ConversationRouter::cancel_in_flight() {
    // 1. THE LOCAL DECODE. A barge-in, minus the speech: bump the epoch and the
    // in-flight loop aborts at its next token check. Same mechanism the VAD uses, so
    // a cancelled turn lands as BargeIn and is correctly NOT re-dispatched.
    control_->cancel_generation(control_->active_generation() + 1);
    // 2. THE NETWORK. The epoch above cannot reach a transfer -- the dispatcher
    // thread is blocked inside send() and has never had a way to be interrupted by a
    // barge-in (docs/LOCAL_ROUTER.md's "one honest residual"). This is that way, and
    // it is deliberately narrow: it aborts the transfer IN FLIGHT and leaves the
    // transport armed for the next turn, so it is not a shutdown().
    //
    // What it saves is OUTPUT tokens, which bill as they stream. The input and
    // prefill were paid at acceptance and no cancellation can recover them -- which
    // is exactly why the gate upstream is what stops unfinished thoughts from being
    // sent at all, and why this button is not, and must not become, a substitute for
    // it.
    router_.abort();
#if defined(VOICE_ASSISTANT_HAS_TTS)
    // 3. THE SOUND. Without this the engine halts and the network drops while the
    // speaker keeps playing everything already synthesised -- which reads as the
    // button not working.
    if (ctx_.tts != nullptr) ctx_.tts->BargeIn();
#endif
}

// ---- the session sidebar ----------------------------------------------------
// All UI-thread. The engine work each of these implies is marshaled, and the disk
// work is not on any hot path -- a person clicking a row is not a rate.

// Publishes the drawer's contents. `ephemeral` comes from the ROUTER, not from
// `settings.local_inference`: the router's flag is atomic (so this is also callable
// from the dispatcher thread, which the completion edge does) and it is the same bit
// that will actually decide whether the next turn is written. A "not being saved"
// banner derived from a second copy of that fact is a banner that will eventually
// disagree with the behaviour.
void ConversationRouter::push_session_list() {
    view_.set_sessions(session_store_.list_summaries(), current_session_id(),
                       router_.use_local());
}

// THE SWITCH. Everything a conversation change has to touch, in the order it has to
// happen:
//
//   1. stop the turn in flight   -- it belongs to the session being left
//   2. swap the active id        -- so the next dispatch latches the new one
//   3. reload the remote window  -- from disk, tail-first (chat_history)
//   4. rewind the LOCAL KV       -- see below; this is the easy one to forget
//   5. repaint                   -- transcript, then the drawer
//
// STEP 4 IS NOT OPTIONAL and is the one that has no visible symptom until it bites.
// The local leg's memory is its KV above the system-prefix floor, and it survives a
// session switch by default -- so without a rewind the on-device model answers the
// newly-opened conversation using the context of the one just closed.
// rebuild_system_prompt() is exactly that rewind (it re-freezes the prefix and drops
// everything above it), which is why the persona path already calls it.
void ConversationRouter::switch_to_session(const std::string& id, bool restore_from_disk) {
    cancel_in_flight();
    {
        const std::lock_guard<std::mutex> lk(session_mu_);
        active_session_id_ = id;
    }

    std::vector<blackwell::cloud::ChatTurn> turns;
    if (restore_from_disk) turns = session_store_.turns(id);
    // Even when empty: restore() also drops any pending turn, which is what stops a
    // half-recorded exchange from the previous conversation sealing itself into this
    // one.
    chat_history_.restore(turns);

    // Marshaled, like every other engine touch on this thread. A failure to queue is
    // reported and NOT fatal: the transcript still switches, and the consequence is
    // confined to the local leg carrying stale context until the next rebuild.
    std::string prompt;
    {
        const std::lock_guard<std::mutex> lk(prompt_mu_);
        prompt = system_prompt_;
    }
    // Captured BY VALUE, not by reference: this task outlives the call that queued
    // it.
    const bool queued = control_->post_engine_task([this, prompt] {
        try {
            (void)rebuild_system_prompt_(prompt);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[session] local context reset failed: %s\n", e.what());
        }
    });
    if (!queued) {
        std::fprintf(stderr, "[session] could not reach the engine thread -- the local model "
                             "keeps the previous conversation's context\n");
    }

    view_.on_session_restored(id, turns);
    push_session_list();
    std::printf("[session] opened \"%s\" (%zu turn(s) restored)\n", id.c_str(), turns.size());
    std::fflush(stdout);
}

void ConversationRouter::select_session(const std::string& id) {
    // Clicking the conversation already open is a no-op, not a reload: rebuilding
    // the prefix and repainting the transcript would throw away a warm KV and make
    // the sidebar feel like it lost the user's place. The page suppresses this too;
    // both, because the list it clicks from can be a moment stale.
    if (id == current_session_id()) return;
    if (!session_store_.contains(id)) {
        // A row that was deleted in another window, or a stale click. Do not create
        // it: an id the store has never heard of is not a conversation, and silently
        // opening an empty one under that name would be indistinguishable from the
        // click having worked.
        push_session_list();
        return;
    }
    switch_to_session(id, /*restore_from_disk=*/true);
}

void ConversationRouter::new_session() {
    // NOT written to disk here. A new conversation becomes a row when its first turn
    // is answered (the completion edge appends), so opening five and talking in none
    // leaves nothing behind -- and the drawer does not fill up with empty rows the
    // user cannot tell apart.
    switch_to_session(blackwell::cloud::make_session_id(), /*restore_from_disk=*/false);
}

void ConversationRouter::delete_session(const std::string& id) {
    if (!session_store_.erase(id)) {
        push_session_list();   // already gone; just re-sync the drawer
        return;
    }
    (void)session_store_.save();
    // Deleting the conversation you are IN is legal, and lands you in a fresh empty
    // one. Refusing would make the same button mean different things on different
    // rows, and leaving the user in a session that no longer exists would quietly
    // recreate it on the next answered turn.
    if (id == current_session_id()) {
        switch_to_session(blackwell::cloud::make_session_id(), /*restore_from_disk=*/false);
    } else {
        push_session_list();
    }
}

// ---- THE STARTUP REPAINT ----------------------------------------------------
// The window's context was seeded from disk long before this point (the dispatcher
// needed it), but the page did not exist to be told. Emitting it here closes that
// gap: a restored conversation arrives as visible BUBBLES rather than as invisible
// context the assistant merely happens to remember, which is the difference between
// the feature working and the feature looking broken.
//
// Safe to emit before the page has loaded: post_event queues, and the window flushes
// everything on NavigationCompleted.
//
// The turns come from the WINDOW and not from the store, so what is drawn is exactly
// what the model will be told -- if the tail was clamped to four pairs, four pairs is
// what the user sees. Painting the full log next to a model that only knows its tail
// would be a more elaborate lie than painting nothing.
void ConversationRouter::seed_startup_view() {
    std::vector<blackwell::cloud::ChatTurn> seeded;
    chat_history_.snapshot(seeded);
    view_.on_session_restored(current_session_id(), seeded);
    push_session_list();
}

void ConversationRouter::apply_system_prompt(const std::string& persona) {
    // COMPOSED ONCE, HERE, and used by both branches below. The user edited the
    // persona; what every model is told is the persona plus the output contract, and
    // deriving that twice (once for the cloud string, once for the KV rebuild) is how
    // the two legs would end up being asked for different reply shapes after a
    // settings edit.
    const std::string prompt = compose_system_prompt(persona);
    // The cloud side first, because it is a plain string swap and must not be left
    // describing the old prompt if the GPU rebuild fails.
    {
        const std::lock_guard<std::mutex> lk(prompt_mu_);
        system_prompt_ = prompt;
    }
    // The rebuild below rewinds the LOCAL KV to the system-prefix floor, i.e. the
    // local leg forgets the conversation. Dropping the remote window too keeps the
    // two legs answering as ONE assistant; leaving it would make the same question
    // get a context-aware answer from the cloud and a blank-slate one from the local
    // model.
    chat_history_.clear();
    // AND THE PERSISTED LOG WITH IT -- this is a data-destroying line, so it states
    // its case. The transcript on disk was produced by a DIFFERENT assistant;
    // restoring it into a session running the new persona reproduces precisely the
    // incoherence the clear() above exists to prevent, just delayed by one launch.
    // Keeping the file would also mean a restart silently resurrects what the user's
    // edit just discarded, which is the worse of the two failures.
    //
    // Only fires on a REAL change: the window callback gates this on
    // `prompt_changed`, so saving the modal with the prompt untouched costs nothing.
    // The right long-term answer is a new session id per persona rather than a wipe,
    // which is why the store is keyed by id (session_store.hpp) even though the app
    // names only one today.
    //
    // ONLY THE ACTIVE SESSION. The other conversations in the store were also
    // produced by the old persona, but they are not on screen and not in the window
    // -- wiping them would be a settings edit deleting data the user was not looking
    // at, which is a different and much worse thing than resetting the conversation
    // in front of them.
    session_store_.clear_session(current_session_id());
    (void)session_store_.save();
    push_session_list();

    // THE marshal. This runs on the UI thread; prefill is CUDA work on a 5.3 GB
    // weight set and belongs to the engine thread alone (CLAUDE.md). post_engine_task
    // runs it at the next command-batch boundary -- after any turn already decoding,
    // before any turn not yet started.
    const bool queued = control_->post_engine_task([this, prompt] {
        try {
            const std::uint32_t n = rebuild_system_prompt_(prompt);
            std::printf("[system-prefix] rebuilt: %u tokens frozen (KV rewind floor)\n", n);
            std::fflush(stdout);
            if (prompt_applied_) prompt_applied_(true, static_cast<unsigned>(n), {});
        } catch (const std::exception& e) {
            // A failed rebuild leaves a SHORTER but valid prefix (see
            // prefill_system_prompt), so the app keeps working -- the user just has
            // to be told the prompt is not what they typed.
            std::fprintf(stderr, "[system-prefix] rebuild FAILED: %s\n", e.what());
            if (prompt_applied_) prompt_applied_(false, 0, e.what());
        }
    });
    if (!queued && prompt_applied_) {
        prompt_applied_(false, 0,
                        "Could not reach the engine thread — the prompt was saved but not "
                        "applied. Restart to load it.");
    }
}

void ConversationRouter::set_use_local(bool local) {
    router_.set_use_local(local);
    // The badge must follow the routing, or a user who switched to local still sees
    // "billed" on a turn that never leaves the machine.
    view_.set_transport(router_.name(), router_.is_live());
}

void ConversationRouter::print_shutdown_summary() const {
    std::printf("\n=== commit gate summary ===\n");
    std::printf("  committed (dispatched) : %llu\n",
                static_cast<unsigned long long>(commit_queue_.committed()));
    std::printf("  dropped barge-in       : %llu\n",
                static_cast<unsigned long long>(commit_queue_.dropped_barge_in()));
    std::printf("  dropped token-cap      : %llu%s\n",
                static_cast<unsigned long long>(commit_queue_.dropped_token_cap()),
                commit_queue_.dropped_token_cap() > 0
                    ? "   <-- truncated before EOS; raise the token ceiling"
                    : "");
    std::printf("  dropped queue-full     : %llu\n",
                static_cast<unsigned long long>(commit_queue_.dropped_queue_full()));
}

}  // namespace rt
