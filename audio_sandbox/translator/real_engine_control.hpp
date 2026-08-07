#pragma once
// -----------------------------------------------------------------------------
// translator/real_engine_control.hpp — the PRODUCTION engine-assembly seam for
// audio_translator. Replaces SimulatedEngineControl: it drives a REAL BlackwellEngine
// (8B AWQ backbone) through the same EngineControlBridge control plane (SPSC
// command ring + barge-in epoch + wait/pump), so every commit runs a real GPU
// decode and streams real detokenized tokens to the GUI.
//
// THREADING (single-thread engine doctrine, CLAUDE.md)
//   Every method here runs on the SINGLE engine-owning thread: prefill_system_
//   prompt() during setup, and the do_* overrides while draining the ring under
//   wait_and_pump(). The audio/VAD thread only ever calls the wait-free producer
//   edge (cancel_generation / rewind_kv / warm_prefill / commit_and_decode) on
//   the base class — it never reaches into here and never touches CUDA.
//
// TWO ISOLATED SESSIONS (native CoW branching; see the Session block below)
//   Transcription and conversation are separate jobs with separate system
//   prompts, so they get separate engine SEQUENCES rather than sharing one
//   linear context:
//     seq 0  CHAT, persistent  -- the persona prompt + dialogue history.
//     seq 1  AUDIO, ephemeral  -- the audio task prompt + the soft-tokens;
//                                 rolled back to its prefix after every turn.
//   The pipeline is therefore: audio -> seq 1 -> transcript TEXT -> commit gate
//   -> user chat template -> seq 0 -> reply. The only thing crossing between the
//   two is that transcript string; no KV is shared. enable_isolated_sessions()
//   forks seq 1 off the empty root and is capability-gated -- when the engine
//   cannot branch (continuous KV) everything below collapses to the single
//   shared session it always was.
//
// THE AUDIO PATH (fully wired, two live modes)
//   The full multimodal path is  live PCM -> Whisper log-mel (whisper_dsp) ->
//   Whisper ENCODER (audio_tower) -> Ultravox projector -> prefill-from-embeddings
//   -> decode. The engine's public API + decode loop stay untouched (hard rule);
//   all embedding injection goes through the white-box step_* sweep.
//   * WHOLE-UTTERANCE: do_commit_decode drains the ring at the VAD boundary and
//     runs the proven --wav transcribe path (commit_audio_decode).
//   * CENTER-SLICE STREAMING (plan.mode == CenterSlice): do_warm_prefill drains
//     the ring DURING speech, recomputes the utterance log-mel, and appends only
//     the stable center soft-tokens per sliding-window hop (append-only, pos
//     monotone) — so the encoder work is off the TTFT critical path. The VAD
//     pause (commit_center_decode) flushes the tail, commits the withheld
//     right-edge tokens (zero encoder work — persisted by the last hop), closes
//     the turn frame and decodes; a SHORT-pause barge-in resumes the SAME
//     utterance via a pointer-only rollback to the pause checkpoint (see
//     do_rewind + is_continuation).
//
// TWO SAFETY RAILS (both learned from live testing)
//   * BARGE-IN ROUTING IS TIME-BASED. A decode can hold the pipeline in
//     DECODE_TRANSLATING for a minute, so "speech arrived while decoding" says
//     nothing about intent. is_continuation() measures the SILENCE since the
//     user stopped speaking: a breath (<= kContinuationTimeoutS) continues the
//     utterance; a longer gap is a new thought and takes the full rewind to
//     history_base_pos_, flushing the stale audio even mid-generation.
//   * DEGENERATE-LOOP GUARDS. Every live decode runs through ONE loop
//     (decode_assistant_turn) carrying a repetition penalty (kRepetitionPenalty,
//     applied in the sampler over this turn's rolling window) and a hard
//     per-turn ceiling (max_new_tokens()) that force-closes the turn cleanly.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "whisper_dsp.h"                // whisper::WhisperDSP (live mic log-mel)

#include "blackwell/engine.h"           // BlackwellEngine, blackwell::EngineStatus
#include "blackwell/tokenizer.h"        // blackwell::ITokenizer, ChatMessage

#include "common.h"                     // CUDA_CHECK_THROW
#include "engine_impl.h"                // BlackwellEngine::Impl (white-box step_* + d_X_accum)
#include "kernels/sampling.cuh"         // sample_top_p + launch_repetition_penalty_kernel
#include "ping_pong_audio_buffer.h"     // blackwell::audio::PingPongAudioBuffer
#include "audio_embedding_pipeline.h"   // blackwell::audio::AudioEmbeddingPipeline
#include "sliding_audio_window.h"       // blackwell::audio::kMelFramesPerSoftToken
#include "audio_head_loader.hpp"        // rt::load_audio_pipeline

#include "engine_control_bridge.hpp"    // EngineControlBridge, Command
#include "bridge/engine_api.h"          // BRIDGE_OK for the token sink
#include "utf8_stream.hpp"              // Utf8StreamAssembler (sub-character token splits)

#include "language_table.hpp"           // rt::kLanguages (forced-language prompt directives)

namespace rt {

class RealEngineControl : public blackwell::bridge::EngineControlBridge {
public:
    using Config = blackwell::bridge::EngineControlBridge::Config;

    // engine + tok are non-owning; both outlive this object (single-thread
    // doctrine). max_context caps the persistent KV growth across utterances.
    // The forced-language atomics seed from the engine's RESOLVED tier-3 plan
    // (CLI/InferenceConfig defaults); the UI dropdowns override them at runtime.
    RealEngineControl(BlackwellEngine* engine, blackwell::ITokenizer* tok,
                      int max_context, const Config& cfg = {})
        : blackwell::bridge::EngineControlBridge(engine, cfg),
          tok_(tok), max_context_(max_context) {
        if (engine != nullptr) {
            const auto& plan = engine->get_impl()->m_runtime;
            src_lang_.store(rt::language_index_or_auto(plan.source_language),
                            std::memory_order_relaxed);
            tgt_lang_.store(rt::language_index_or_auto(plan.target_language),
                            std::memory_order_relaxed);
            // Live streaming toggle seeds from the resolved plan: ON iff the
            // CenterSlice geometry was armed at launch (the UI can flip it).
            live_center_.store(plan.audio_streaming.enabled &&
                               plan.audio_streaming.mode ==
                                   blackwell::AudioStreamingMode::CenterSlice,
                               std::memory_order_relaxed);
        }
    }

    // Bind the engine and tokenizer to a control that was constructed WITHOUT
    // them — the deferred-load seam (see EngineControlBridge::attach_engine).
    //
    // WHY IT IS NOT JUST attach_engine(). The constructor does more than store
    // the pointer: it seeds the forced-language and live-streaming atomics from
    // the engine's RESOLVED tier-3 plan. A control built with no engine holds
    // the compile-time defaults for all three, so attaching without re-seeding
    // would run the deferred-loaded engine with settings the launch never chose
    // — silently, and only on the lazy path.
    //
    // ENGINE THREAD ONLY, engine idle. Returns false if a bind is refused.
    bool adopt_engine(BlackwellEngine* engine, blackwell::ITokenizer* tok) noexcept {
        if (engine == nullptr || tok == nullptr) return false;
        if (!attach_engine(engine)) return false;
        tok_ = tok;
        const auto& plan = engine->get_impl()->m_runtime;
        src_lang_.store(rt::language_index_or_auto(plan.source_language),
                        std::memory_order_relaxed);
        tgt_lang_.store(rt::language_index_or_auto(plan.target_language),
                        std::memory_order_relaxed);
        live_center_.store(plan.audio_streaming.enabled &&
                           plan.audio_streaming.mode ==
                               blackwell::AudioStreamingMode::CenterSlice,
                           std::memory_order_relaxed);
        return true;
    }

    // ---- ISOLATED SESSIONS (the engine's NATIVE CoW branching) ---------------
    // Transcription and conversation are two different jobs with two different
    // system prompts, and running them on one linear sequence poisons both: the
    // transcriber inherits the assistant persona ("be helpful, be concise") and
    // answers the audio instead of writing it down, while the chat context
    // accumulates "[Speech] ... | [Translation] ..." formatting noise it then
    // imitates. So they get one engine SEQUENCE each:
    //
    //   CHAT  (seq 0, PERSISTENT)  the persona system prompt + the dialogue
    //         history. Survives every utterance; this is where a reply is
    //         generated, locally or after the cloud round-trip.
    //   AUDIO (seq 1, EPHEMERAL)   the transcription task prompt. Receives the
    //         audio soft-tokens, produces the transcript, and is rolled back to
    //         its own frozen prefix the moment the turn closes -- it never
    //         retains history in ANY context mode, so a hallucinated transcript
    //         cannot feed the next one.
    //
    // The two are genuinely independent KV: seq 1 is fork()ed off the EMPTY root
    // before either prefix is laid down, so nothing is shared but the page
    // allocator. That is only available under the paged (branching) cache --
    // ContinuousKVManager::supports_branching() is false -- which is why
    // bring_up_real_engine requests InferenceConfig::require_branching.
    //
    // isolated() == false is a fully supported configuration and is EXACTLY the
    // pre-isolation behaviour: one session, one prefix, everything on seq 0.
    struct Session {
        int         seq_id        = 0;
        int         pos           = 0;   // logical decode position on this sequence
        uint32_t    system_prefix = 0;   // frozen prefix == this sequence's KV floor
        uint32_t    verified      = 0;   // committed-prefix boundary (checkpoint/rollback)
        int         history_base  = 0;   // system prefix + retained turns
        bool        ephemeral     = false;  // discard every turn back to the prefix
        const char* name          = "chat";
    };

    // Fork the ephemeral transcription sequence. ENGINE THREAD, SETUP ONLY: both
    // sequences must still be empty (fork of a zero-length parent copies no
    // pages, so the two prefixes below diverge from nothing rather than sharing a
    // persona). Returns false -- leaving the single-session path intact -- when
    // the loaded model or KV mode cannot branch, per the capability-gating rule:
    // never guess what a checkpoint supports, ask.
    bool enable_isolated_sessions() {
        if (isolated_) return true;
        if (engine_ == nullptr) return false;
        if (pos_ != 0) return false;                       // prefixes already laid
        if (!engine_->get_capabilities().supports_cow_branching) {
            std::fprintf(stderr, "[session] branching unsupported by this engine "
                                 "(continuous KV or non-snapshot-able state)\n");
            return false;
        }
        if (engine_->branch_capacity() < 2) {
            std::fprintf(stderr, "[session] branch capacity %d < 2 (raise "
                                 "RuntimeOverrides::paged_branch_factor)\n",
                         engine_->branch_capacity());
            return false;
        }
        try {
            engine_->fork(chat_.seq_id, audio_.seq_id);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[session] fork(%d -> %d) failed: %s\n",
                         chat_.seq_id, audio_.seq_id, e.what());
            return false;
        }
        isolated_ = true;
        return true;
    }
    bool isolated() const noexcept { return isolated_; }

    // The chat session's decode position -- the invariant readout that proves
    // isolation is real: a transcription turn must leave this untouched. Any
    // thread may read it (engine thread is the only writer).
    int chat_position() const noexcept {
        return active_ == &chat_ ? pos_ : chat_.pos;
    }
    int audio_position() const noexcept {
        return active_ == &audio_ ? pos_ : audio_.pos;
    }

    // ---- Live streaming-mode toggle (UI thread writes, engine thread reads at
    //      UTTERANCE boundaries — an open utterance latches its mode, so a
    //      mid-speech flip applies from the next utterance) --------------------
    // Whether the CenterSlice plan geometry + audio frontend are armed at all
    // (static capability: resolved at launch; without it the toggle is inert).
    bool center_slice_available() const {
        if (engine_ == nullptr || !audio_head_loaded() || dsp_ == nullptr) return false;
        const auto& plan = engine_->get_impl()->m_runtime.audio_streaming;
        return plan.enabled && plan.mode == blackwell::AudioStreamingMode::CenterSlice;
    }
    void set_live_center_slice(bool on) noexcept {
        live_center_.store(on, std::memory_order_release);
    }
    bool live_center_slice() const noexcept {
        return live_center_.load(std::memory_order_acquire);
    }

    // Last utterance's TTFT — VAD trigger (commit command execution) to the first
    // decode token being available — in ms; 0 until the first utterance. Written
    // by the engine thread's commit paths, read by the UI readout. Both live
    // paths report it, so the toggle gives a direct A/B in the panel.
    float last_ttft_ms() const noexcept {
        return last_ttft_ms_.load(std::memory_order_acquire);
    }

    // ---- Text-context policy (Stateless vs Bounded dialogue history) ----------
    // UI-thread -> engine-thread handoff is a set of plain atomics read only at
    // TURN BOUNDARIES (finalize_turn / build_user_instruction) on the engine
    // thread, so a mid-decode change simply applies from the next turn — no
    // marshaling, no locks, doctrine intact.
    enum class ContextMode : int {
        Stateless      = 0,  // live translator: flush the whole turn after decode
        BoundedHistory = 1,  // voice assistant: retain last turns as text, <= budget
    };
    void set_context_mode(ContextMode m) noexcept {
        mode_.store(static_cast<int>(m), std::memory_order_release);
    }
    ContextMode context_mode() const noexcept {
        return static_cast<ContextMode>(mode_.load(std::memory_order_acquire));
    }
    void set_history_budget_tokens(int n) noexcept {
        history_budget_tokens_.store(n < 0 ? 0 : n, std::memory_order_release);
    }
    int history_budget_tokens() const noexcept {
        return history_budget_tokens_.load(std::memory_order_acquire);
    }

    // Forced languages as rt::kLanguages indices (0 = Auto). Out-of-range stores
    // clamp to Auto so a stale UI index can never read past the table.
    void set_languages(int src_idx, int tgt_idx) noexcept {
        auto clamp = [](int i) { return (i < 0 || i >= rt::kLanguageCount) ? 0 : i; };
        src_lang_.store(clamp(src_idx), std::memory_order_release);
        tgt_lang_.store(clamp(tgt_idx), std::memory_order_release);
    }
    int source_language_index() const noexcept {
        return src_lang_.load(std::memory_order_acquire);
    }
    int target_language_index() const noexcept {
        return tgt_lang_.load(std::memory_order_acquire);
    }

    // ---- Live sampling knobs (Settings -> Inference) --------------------------
    // Same UI-thread -> engine-thread handoff as the languages: plain atomics read
    // at the START of each decode loop, so a change applies from the next turn and
    // never mutates a generation already in flight.
    //
    // DEFAULT IS GREEDY (0.0 / 1.0), and that is the shipping behaviour for a
    // reason: the local model's job here is transcription and intent extraction,
    // where sampling buys nothing and costs fidelity. The knob exists because a
    // user running the text path as a chat assistant wants it; it is not a
    // recommendation. (The canned no-audio-head path previously hardcoded 0.7/0.9
    // and now follows this setting like everything else -- one decode loop, one
    // sampling policy, which is the whole point of decode_assistant_turn.)
    void set_sampling(float temperature, float top_p) noexcept {
        temperature_.store(temperature < 0.0f ? 0.0f : temperature, std::memory_order_release);
        top_p_.store(top_p <= 0.0f || top_p > 1.0f ? 1.0f : top_p, std::memory_order_release);
    }
    float temperature() const noexcept { return temperature_.load(std::memory_order_acquire); }
    float top_p() const noexcept { return top_p_.load(std::memory_order_acquire); }

    // Per-turn generated-token ceiling: the hard brake behind the degenerate-loop
    // guards. NOT cosmetic -- a turn that hits it terminates as TokenCap and is
    // therefore NOT dispatched, so this value directly decides whether long
    // answers reach the cloud at all (see IntentCommitQueue's telemetry).
    void set_max_new_tokens(int n) noexcept {
        max_new_tokens_.store(n < 1 ? 1 : n, std::memory_order_release);
    }
    int max_new_tokens() const noexcept {
        return max_new_tokens_.load(std::memory_order_acquire);
    }

    // ---- Task selection (Transcribe / Translate) ------------------------------
    // Narrowing the task is the cheapest way to cut the 8B backbone's cognitive
    // load: asking for one output instead of two roughly halves the tokens a turn
    // must generate and removes the format the model most often drifts out of.
    // Same UI-thread -> engine-thread handoff as the languages: plain atomics read
    // only at TURN BOUNDARIES (build_user_instruction), so a mid-decode change
    // applies from the next utterance. Both false is coerced to transcribe-only —
    // an empty task would leave the model with no instruction at all.
    //
    // Nothing cached is invalidated by a flip: the task directive lives in the
    // per-turn user prefix, never in the frozen system prefix that
    // history_base_pos_ measures. (Even if it did, a ~40-token text re-prefill is
    // sub-millisecond.)
    void set_tasks(bool transcribe, bool translate) noexcept {
        task_transcribe_.store(transcribe, std::memory_order_release);
        task_translate_.store(translate, std::memory_order_release);
    }

    // ---- the AUDIO TASK PROMPT (Settings -> System prompts) -------------------
    // The second of the two prompts, and a genuinely different thing from the
    // persona:
    //
    //   PERSONA  prefill_system_prompt() -- who the assistant IS. The frozen
    //            system prefix of the CHAT session (seq 0).
    //   AUDIO    this -- what to do with speech. Under isolation it is the frozen
    //   TASK     system prefix of the TRANSCRIPTION session (seq 1), so the two
    //            never sit on one sequence; edits go through
    //            rebuild_audio_task_prompt (cheap: that session keeps no history).
    //            Without isolation it falls back to riding in the per-turn user
    //            block, which is the pre-branching behaviour.
    //
    // Conflating the two is what made the model translate when the user wanted a
    // transcript: the task verb was hardcoded in build_user_instruction() and the
    // only prompt anyone could edit was the persona, which is the one prompt that
    // does NOT decide the task. Putting them on separate sequences is the
    // structural version of that same fix.
    //
    // SCOPE, deliberately narrow. This replaces the task VERB only; the language
    // directives and the output-tag contract are still generated, because
    // encode_history_turn() parses those tags back out and the UI splits on them.
    // A prompt that could silently drop them would corrupt the history parser --
    // so the user owns the instruction, and the machinery keeps its invariants.
    // Empty (the default) restores the generated phrase verbatim.
    //
    // A std::string cannot be an atomic, and this is read at turn boundaries on
    // the engine thread while the UI thread writes it -- hence the mutex. It is
    // never touched from the audio hot path.
    void set_audio_task_prompt(std::string prompt) {
        std::lock_guard<std::mutex> lk(audio_task_mu_);
        audio_task_prompt_ = std::move(prompt);
    }
    std::string audio_task_prompt() const {
        std::lock_guard<std::mutex> lk(audio_task_mu_);
        return audio_task_prompt_;
    }

    // ---- the PER-TURN FORMAT REMINDER (prompt-drift guard) --------------------
    // Appended to the user block of every turn generate_local_reply() answers,
    // so it is the LAST thing the model reads before it starts writing. Empty
    // (the default) appends nothing.
    //
    // WHY A REMINDER AND NOT A BIGGER PIN. The persona is already pinned as hard
    // as this codebase can pin anything: prefill_prefix_on_active publishes its
    // length as the KV rewind floor, and effective_keep_tokens(),
    // kv_cache_rollback() and rebuild_history_kv() all clamp UP to it, so no
    // barge-in, end-of-turn flush or history eviction can truncate a formatting
    // rule written into it. Eviction is not the failure. RECENCY is: after a few
    // turns that rule is a thousand tokens behind the conversation and the 8B
    // backbone drifts out of the format. Restating it here is the same move
    // build_user_instruction() makes for the transcription contract, and for the
    // same reason -- the per-turn user block is the position that actually holds.
    //
    // THE TEXT BELONGS TO THE CALLER. What a reply must look like is the app's
    // contract, not the engine control's: voice_assistant owns both halves of it
    // in reply_split.hpp (the instruction AND the parser that reads it back), and
    // a tag hardcoded here would be a third copy free to drift from those two.
    // The translator app sets nothing and pays nothing.
    //
    // Same mutex as the audio task text, for the same reason: a std::string read
    // at turn boundaries on the engine thread, written by the UI thread, never
    // touched from the audio hot path.
    void set_reply_format_reminder(std::string reminder) {
        std::lock_guard<std::mutex> lk(audio_task_mu_);
        reply_format_reminder_ = std::move(reminder);
    }
    std::string reply_format_reminder() const {
        std::lock_guard<std::mutex> lk(audio_task_mu_);
        return reply_format_reminder_;
    }

    // ---- the SPOKEN LANGUAGE (Settings -> System prompts) ---------------------
    // Free text ("Russian", "English", ""), NOT an rt::kLanguages index, and that
    // is the point: the table is a fixed twelve-entry dropdown, while the failure
    // this fixes is acoustic and can name any language the backbone knows.
    //
    // WHY IT EXISTS. The audio tower's language identification confuses
    // neighbouring phonologies on short hops -- Russian heard as Polish is the
    // reproducible case -- and once it has guessed wrong the backbone dutifully
    // writes the wrong language in the wrong script (Latin transliteration, Polish
    // vocabulary). Nothing downstream can recover from that: the transcript is
    // what gets committed and answered. So the fix is upstream, and it is a
    // constraint rather than a hint -- naming the language REMOVES the decision
    // instead of biasing it.
    //
    // IT LANDS IN TWO PLACES, deliberately:
    //   1. The transcription session's frozen SYSTEM prefix (see
    //      resolve_audio_system_prompt) -- the persona itself is language-locked,
    //      paid for once. Editing it is therefore a prefix rebuild, exactly like
    //      editing the audio task text.
    //   2. The per-turn user instruction (build_user_instruction) -- restated
    //      immediately before the audio soft-tokens, which is the position that
    //      actually kills the transliteration mode, and the ONLY position that
    //      exists at all when isolation is off.
    //
    // Same mutex as the audio task text: both are std::strings read at turn
    // boundaries on the engine thread and written by the UI thread, never touched
    // from the audio hot path.
    void set_speech_language(std::string language) {
        std::lock_guard<std::mutex> lk(audio_task_mu_);
        speech_language_ = std::move(language);
    }
    std::string speech_language() const {
        std::lock_guard<std::mutex> lk(audio_task_mu_);
        return speech_language_;
    }

    // The forced source language as a NAME, resolving the two ways one can be
    // forced. Free text wins over the kLanguages index because it is the more
    // specific statement: the index is a launch-time plan default (--src-lang),
    // the text is what the user typed into this session's settings. "" = Auto,
    // i.e. leave language identification to the model. Any thread.
    std::string forced_source_language() const {
        if (std::string named = speech_language(); !named.empty()) return named;
        const int si = src_lang_.load(std::memory_order_acquire);
        return si > 0 ? std::string(rt::kLanguages[si]) : std::string();
    }

    bool task_transcribe() const noexcept {
        return task_transcribe_.load(std::memory_order_acquire);
    }
    bool task_translate() const noexcept {
        return task_translate_.load(std::memory_order_acquire);
    }

    // The retained-history base (system prefix + retained turns): the KV floor a
    // barge-in rewind honours in bounded mode. Engine thread writes it at turn
    // boundaries; any thread may read it (UI/tests invariant readout).
    int history_base_pos() const noexcept {
        return history_base_pos_.load(std::memory_order_acquire);
    }

    // Engine-thread setup: prefill the frozen system prompt through the REAL
    // engine (BOS + system block via the checkpoint's chat template) and freeze
    // its length as the KV rewind floor. Returns the prefix token count.
    // THE FLOOR IS WHAT LANDED, not what was asked for. A forward that faults
    // part-way through leaves only `done` columns in the KV, and publishing
    // ids.size() as the floor would put the rewind clamp ABOVE the real extent of
    // the cache -- kv_cache_rollback's max(cp.pos, floor) would then park pos_ on
    // uninitialized KV columns and the next turn would decode against garbage.
    // So the floor tracks the actual prefill, and the failure is raised only
    // AFTER the published state is self-consistent (INIT tier: this throws, and
    // rebuild_system_prompt's caller reports it to the user).
    //
    // ISOLATION: this is the CHAT session's prefix (the persona). It also lays
    // the AUDIO session's prefix in the same call, because both are startup KV
    // work on the engine thread and ConversationalMode drives exactly one prefill
    // seam -- splitting them into two callbacks would only create an ordering
    // this function can guarantee for free. Returns the CHAT prefix length (what
    // the seam and the UI report).
    uint32_t prefill_system_prompt(const std::string& system_prompt) {
        activate(chat_);
        const uint32_t done = prefill_prefix_on_active(
            tok_->encode_chat_prelude(system_prompt), "system");
        if (isolated_) prefill_audio_task_prefix(audio_task_prompt());
        return done;
    }

    // The AUDIO session's frozen prefix: what THAT context is for. A dedicated
    // transcription persona is the whole reason the audio task stopped being a
    // sentence buried in the user block -- it now sits where a system prompt
    // belongs, on a sequence the assistant persona cannot reach.
    //
    // Empty (the default) installs kDefaultAudioSystemPrompt; either way the
    // forced spoken language is appended by resolve_audio_system_prompt, which is
    // why a language edit is a rebuild through here and not a plain store.
    // Engine thread only;
    // a no-op without isolation, where there is no second sequence to prefix and
    // the task verb keeps riding in the per-turn user block (build_user_instruction).
    //
    // IDEMPOTENT. A prefix is always laid on an EMPTY sequence, so calling this
    // twice replaces the prefix rather than stacking a second copy above the
    // first -- which matters because both setup (prefill_system_prompt) and every
    // later edit come through here. The floor must be dropped BEFORE the
    // rollback: kv_cache_rollback clamps UP to it and would otherwise refuse to
    // discard the very tokens being replaced.
    uint32_t prefill_audio_task_prefix(const std::string& audio_task) {
        if (!isolated_) return 0;
        activate(audio_);
        set_system_prefix_tokens(0);
        if (pos_ != 0)
            kv_cache_rollback({0, 0}, "audio task prefix (re)build on session B");
        // Everything that described the OLD prefix's sequence is now stale.
        if (audio_pipeline_) audio_pipeline_->reset_history();
        pause_pending_ = false;
        reset_utterance();
        const uint32_t done = prefill_prefix_on_active(
            tok_->encode_chat_prelude(resolve_audio_system_prompt(audio_task)),
            "audio-task");
        activate(chat_);
        return done;
    }

    // Re-lay the AUDIO session's prefix after the user edits the audio task.
    // Cheap by construction: that session retains no history, so this is a
    // rollback to 0 plus a ~30-token re-prefill, and the conversation on the chat
    // sequence is untouched. ENGINE THREAD ONLY (marshal through post_engine_task).
    uint32_t rebuild_audio_task_prompt(const std::string& audio_task) {
        set_audio_task_prompt(audio_task);
        return prefill_audio_task_prefix(audio_task);
    }

    // Re-lay the frozen system prefix after the user edits it. ENGINE THREAD ONLY
    // (marshal through EngineControlBridge::post_engine_task).
    //
    // This is the precompute poc_overlay does for a language branch, applied to
    // the one prefix every turn sits on top of. The system prompt is prefilled
    // ONCE and its KV is then reused by every utterance forever -- that is what
    // makes TTFT a decode-latency number instead of a re-prefill of the whole
    // prompt on every turn. So changing the prompt is not a config edit, it is a
    // cache rebuild, and it has to happen in the right order:
    //
    //   1. DROP THE FLOOR FIRST. kv_cache_rollback clamps UP to
    //      system_prefix_tokens(), so rolling back while the old floor is still
    //      published would refuse to discard the very tokens being replaced.
    //   2. Roll the sequence to 0 -- the new prefix is prefilled from position 0
    //      (prefill_system_prompt appends at pos_ and assumes an empty sequence).
    //   3. Drop retained history: it was encoded under the OLD system prompt and
    //      its KV is about to be invalid.
    //   4. Prefill, which republishes the floor and re-anchors history_base_pos_.
    //
    // Returns the new prefix length in tokens (what the UI reports back).
    uint32_t rebuild_system_prompt(const std::string& system_prompt) {
        activate(chat_);                                   // 0. the PERSONA's session
        set_system_prefix_tokens(0);                       // 1. unpin the floor
        kv_cache_rollback({0, 0}, "system prompt changed (prefix cache rebuild)");  // 2.
        turn_history_.clear();                             // 3. old-prompt history
        history_text_tokens_ = 0;
        if (audio_pipeline_) audio_pipeline_->reset_history();
        return prefill_system_prompt(system_prompt);       // 4. re-freeze
    }

    // ---- FULLY LOCAL INFERENCE: answer an intent without leaving the machine --
    // Generate an assistant reply to `user_text` on the local backbone, streaming
    // each detokenized chunk through `emit`. ENGINE THREAD ONLY (marshal through
    // EngineControlBridge::post_engine_task).
    //
    // WHY THIS IS NOT submit_text(). The typed-input path ends in publish_turn(),
    // which offers the finished turn to the commit gate. That is exactly right for
    // a user's words and exactly wrong for the assistant's: this text IS the answer
    // to an intent that already passed the gate, so re-offering it would dispatch
    // the model's own reply back into the pipeline as a fresh intent, and answer
    // that, forever. The gate is bypassed here for that one reason, and it is the
    // ONLY difference from commit_text_decode.
    //
    // The turn is a NORMAL dialog turn on the live sequence -- same KV, same
    // history policy, same repetition guards -- so the reply sees the persona
    // prefix and the conversation so far, and is retained by the bounded-history
    // policy like any other. The barge-in epoch still cancels it mid-stream: a
    // local answer must be interruptible by speech exactly as a remote one is.
    //
    // Returns Success even when the turn was cut short (cancel / cap); *out_reason
    // carries the verdict for the caller to report.
    blackwell::EngineStatus generate_local_reply(
        const std::string& user_text,
        const std::function<void(std::string_view)>& emit,
        blackwell::bridge::TerminationReason* out_reason = nullptr) {
        using blackwell::bridge::TerminationReason;
        if (out_reason != nullptr) *out_reason = TerminationReason::None;
        if (user_text.empty()) return blackwell::EngineStatus::InvalidArgument;

        // SESSION A. `user_text` is the transcript the AUDIO session produced,
        // already through the commit gate; it arrives here as plain text and is
        // re-encoded below into a user turn on the persona's sequence. That text
        // hand-off IS the isolation boundary -- no KV crosses it.
        activate(chat_);
        pending_user_text_ = user_text;   // history records what was said, verbatim

        // decode_assistant_turn streams through a Command's TokenSink, so wrap the
        // caller's std::function in one rather than duplicating the loop. The gen
        // is the LIVE epoch: a barge-in raised while this reply is streaming
        // supersedes it on the next token, which is what makes the local path
        // interruptible on the same mechanism the remote one uses.
        struct EmitCtx { const std::function<void(std::string_view)>* fn; };
        EmitCtx ctx{&emit};
        Command cmd;
        cmd.gen = active_generation();
        cmd.sink.user = &ctx;
        cmd.sink.fn = [](void* user, const char* utf8, std::int32_t, std::int32_t is_final,
                         BridgeStatus) {
            if (is_final != 0 || utf8 == nullptr || *utf8 == '\0') return;
            const auto* c = static_cast<EmitCtx*>(user);
            (*c->fn)(utf8);
        };

        // WHAT THE MODEL READS vs WHAT THE HISTORY REMEMBERS -- and the split is
        // the point. The KV gets the user's words PLUS the format reminder, so
        // the reminder is the last instruction before the generation cue: the
        // one position that still lands after a long prefix. The retained
        // history gets pending_user_text_ (set above), which is the words alone.
        //
        // Two consequences, both wanted. A bounded-history rebuild re-prefills
        // the conversation WITHOUT the reminders, so the history budget is never
        // spent on N copies of one sentence; and the reminder is always attached
        // to the turn actually being answered rather than accumulating behind it.
        //
        // NOT DONE IN commit_text_decode. That path's output is an INTENT offered
        // to the commit gate, not a reply shown to anyone -- asking it for
        // <voice>/<ui> blocks would wrap the intent itself in tags and dispatch
        // the markup as the user's words.
        std::string user_block = user_text;
        if (const std::string reminder = reply_format_reminder(); !reminder.empty())
            user_block += reminder;

        std::vector<int> turn = tok_->encode_chat_message(
            blackwell::ChatMessage{"user", user_block});
        const std::vector<int> gen = tok_->encode_generation_prompt();
        turn.insert(turn.end(), gen.begin(), gen.end());

        int next = -1;
        const auto t0 = std::chrono::steady_clock::now();
        for (const int id : turn) {
            if (pos_ >= max_context_ ||
                engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                    blackwell::EngineStatus::Success) {
                finalize_turn(std::string(), /*completed=*/false);   // drop partial prefill
                return blackwell::EngineStatus::CudaRuntimeError;
            }
            ++pos_;
        }

        const TurnDecode d = decode_assistant_turn(cmd, next, t0, temperature(), top_p());
        finalize_turn(d.reply, d.completed());
        if (out_reason != nullptr) *out_reason = d.reason;
        return blackwell::EngineStatus::Success;
    }

    // The keep count the most recent barge-in rewind resolved to (after the floor
    // clamp), for the UI's prefix-cache invariant readout.
    uint32_t last_rewind_keep() const noexcept {
        return last_rewind_keep_.load(std::memory_order_acquire);
    }

    // ---- Explicit KV checkpoint / rollback (Phase 0, speculative prefill) -----
    // A lightweight snapshot of the sequence pointer + verified-prefix boundary.
    // Captured before speculatively prefilling audio/text soft-tokens, restored to
    // discard those tokens without leaving stale KV above the rewound position.
    struct KVCheckpoint {
        int      pos = 0;               // logical decode position at capture time
        uint32_t verified_tokens = 0;   // prompt tokens confirmed committed
    };

    // Snapshot the current sequence pointer + verified-prefix boundary. Engine
    // thread only (reads pos_ / verified_prompt_tokens_ without a lock).
    KVCheckpoint kv_cache_checkpoint() {
        const KVCheckpoint cp{pos_, verified_prompt_tokens_};
        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[kv-ckpt] checkpoint pos=%d verified_tokens=%u\n",
                        cp.pos, cp.verified_tokens);
            std::fflush(stdout);
        }
        return cp;
    }

    // Roll the sequence back to a checkpoint, discarding every KV column above it.
    // Enforces the frozen system-prefix floor (a rollback may keep MORE than the
    // checkpoint but never truncate the system prompt), restores the sequence
    // pointer + verified boundary, then reconciles the arena's offload high-water
    // marks through the Continuous KV rewind (VRAMArena::truncate_kv). Engine
    // thread only.
    //
    // EVERY KV drop in this app flows through here or do_rewind — the two choke
    // points — so each prints an UNCONDITIONAL, highly visible invalidation line
    // with its reason: any multi-second re-prefill must be traceable to a named
    // cause in the log, never a silent cache flush.
    void kv_cache_rollback(const KVCheckpoint& cp,
                           const char* reason = "explicit checkpoint rollback") {
        const int floor    = static_cast<int>(effective_keep_tokens(0));
        const int safe_pos = std::max(cp.pos, floor);
        const int from_pos = pos_;
        const auto t0      = std::chrono::steady_clock::now();

        pos_ = safe_pos;
        verified_prompt_tokens_ = cp.verified_tokens;
        // Continuous KV rewind() delegates to VRAMArena::truncate_kv; safe below the
        // engine's public API (single-thread doctrine) and a no-op for a fully
        // resident model (nothing offloaded), so regular decode is unaffected.
        engine_->get_impl()->kv_mgr->rewind(active_seq_, safe_pos);

        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("[KV CACHE INVALIDATION] Reason: %s. pos %d -> %d "
                    "(dropping %d KV tokens; floor=%d, verified=%u)  %.1f us\n",
                    reason, from_pos, safe_pos, from_pos - safe_pos, floor,
                    cp.verified_tokens, us);
        std::fflush(stdout);
    }

    // Engine-thread setup: load the Whisper encoder + Ultravox projector weights
    // from the audio-head checkpoint dir (<audio_head>/model.safetensors) into the
    // async double-buffered pipeline, and size the ping-pong to one embeddings
    // frame. The projector output width is taken from the engine's OWN hidden_size,
    // so the audio soft-tokens always match the backbone they splice into. INIT
    // tier: throws (blackwell::cuda_error / runtime_error) on a missing file or
    // tensor. Idempotent per call (replaces any previous audio head).
    void load_audio_head(const std::string& audio_head) {
        const int hidden = static_cast<int>(engine_->get_impl()->m_config.hidden_dim);
        audio_pipeline_ = rt::load_audio_pipeline(audio_head, hidden);
        audio_pp_ = std::make_unique<blackwell::audio::PingPongAudioBuffer>(
            audio_pipeline_->embed_elems());
        audio_frame_ = 0;
        // Bind the resolved tier-3 streaming plan so the sliding-window hop path can
        // reconcile overlap (inert when the plan is disabled / capability-gated off).
        audio_pipeline_->configure_streaming(engine_->get_impl()->m_runtime.audio_streaming);
    }
    bool audio_head_loaded() const noexcept { return audio_pipeline_ != nullptr; }
    int  audio_out_frames() const {
        return audio_pipeline_ ? audio_pipeline_->out_frames() : 0;
    }

    // Bind the CPU log-mel front-end (non-owning; outlives this object) used by the
    // LIVE mic path to turn buffered utterance PCM into the encoder's [128,3000]
    // input. Without it (or without an audio head) do_commit_decode falls back to a
    // text-only decode. Engine-thread setup. NOTE: mel_filters.bin MUST be the
    // freq-major [n_freqs,n_mels] layout WhisperDSP expects (the Ultravox dump's is
    // the transpose — see the --wav diagnosis).
    void set_dsp(whisper::WhisperDSP* dsp) noexcept { dsp_ = dsp; }

    // One live audio frame end-to-end (requires load_audio_head): PRODUCE the
    // embeddings on Stream 1 (encode + project + stage into the ping-pong) then
    // CONSUME on Stream 2 (event-gated step_* prefill into the KV). d_mel is a
    // device pointer to the log-mel [num_mel_bins, conv_frames]. Returns the decode
    // position after the audio block. Engine-owning thread only.
    int prefill_audio(const float* d_mel) {
        audio_pipeline_->process_frame(d_mel, *audio_pp_, audio_frame_);
        const int p =
            prefill_audio_embeddings(*audio_pp_, audio_frame_, audio_pipeline_->out_frames());
        ++audio_frame_;
        return p;
    }

    // Prefill one Ultravox user turn whose content is the audio in d_mel: the
    // audio soft-tokens ARE the placeholder, spliced between the user header and
    // the turn close, followed by the assistant generation cue. The frozen system
    // prompt (prefill_system_prompt) must already be prefilled. Advances pos_ and
    // writes the first assistant token to decode into *first_token. Returns
    // Success, or InvalidConfig if no audio head is loaded. Engine thread only.
    //
    //   [system prefix] <|start_header_id|>user<|end_header_id|>\n\n
    //     {188 audio soft-tokens}  <|eot_id|>
    //     <|start_header_id|>assistant<|end_header_id|>\n\n   -> decode
    blackwell::EngineStatus prefill_ultravox_turn(const float* d_mel, int* first_token) {
        if (!audio_head_loaded()) return blackwell::EngineStatus::InvalidConfig;
        int next = -1;
        auto prefill_ids = [&](const std::vector<int>& ids) -> bool {
            for (const int id : ids) {
                if (pos_ >= max_context_) return false;
                if (engine_->forward_status(id, pos_, /*temp=*/0.0f, /*top_p=*/1.0f,
                                            active_seq_, &next) != blackwell::EngineStatus::Success)
                    return false;
                ++pos_;
            }
            return true;
        };
        // Strict Llama-3 template with an explicit (language-directed) instruction;
        // the 188 audio soft-tokens occupy the <|audio|> placeholder slot (injected
        // as embeddings, not as a literal token). Special-token strings are
        // recognized by encode() (add_special=false: the BOS already lives in the
        // frozen system prefix).
        const TurnFrame frame = make_turn_frame();

        std::printf("[prompt] <sys:%d tok>%s<|audio x%d|><|eot_id|>%s\n", pos_,
                    tok_->decode(frame.user_hdr, /*render_special=*/true).c_str(),
                    audio_pipeline_->out_frames(),
                    tok_->decode(frame.gen_cue, /*render_special=*/true).c_str());
        std::fflush(stdout);

        if (!prefill_ids(frame.user_hdr)) return blackwell::EngineStatus::StateMismatch;
        prefill_audio(d_mel);  // the 188 audio embeddings occupy the placeholder slot

        // DEBUG (env-gated; does a device sync, so keep it off the live hot path):
        // confirm the injected audio embeddings are real (not zeros/noise).
        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            const int H = static_cast<int>(engine_->get_impl()->m_config.hidden_dim);
            std::vector<float> row(static_cast<size_t>(H));
            cudaDeviceSynchronize();
            cudaMemcpy(row.data(), audio_pp_->slot(audio_frame_ - 1),
                       row.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double n2 = 0.0, amax = 0.0;
            for (float v : row) { n2 += static_cast<double>(v) * v; amax = std::max(amax, std::abs((double)v)); }
            std::printf("[audio-embed] frame row0: L2=%.4f absmax=%.4f  [%.4f %.4f %.4f %.4f]\n",
                        std::sqrt(n2), amax, row[0], row[1], row[2], row[3]);
            std::fflush(stdout);
        }

        if (!prefill_ids(frame.user_eot)) return blackwell::EngineStatus::StateMismatch;
        if (!prefill_ids(frame.gen_cue))  return blackwell::EngineStatus::StateMismatch;

        *first_token = next;
        return blackwell::EngineStatus::Success;
    }

    // Headless one-shot: prefill the audio turn (d_mel) and greedily decode up to
    // max_new_tokens of the assistant reply, returning the detokenized text. Used
    // by the --wav path; the live mic path streams the same decode via a sink.
    std::string transcribe(const float* d_mel, int max_new_tokens) {
        activate(audio_);
        int next = -1;
        if (prefill_ultravox_turn(d_mel, &next) != blackwell::EngineStatus::Success)
            return std::string();
        std::string out;
        for (int i = 0; i < max_new_tokens && pos_ < max_context_; ++i) {
            if (tok_->is_stop(next)) break;
            out += tok_->decode(next, /*render_special=*/false);
            if (engine_->forward_status(next, pos_, /*temp=*/0.0f, /*top_p=*/1.0f,
                                        active_seq_, &next) != blackwell::EngineStatus::Success)
                break;
            ++pos_;
        }
        return out;
    }

    // Streaming transcribe (dynamic overlap reconciliation). Frames the user turn,
    // then slides a [history + hop] window across the utterance's log-mel
    // [num_mel_bins, total_frames] (device, mel-major), feeding each window through
    // prefill_audio_hop -- cold-start seeds the whole first window, later hops append
    // only the (reconciled) delta -- then decodes the assistant reply. Requires the
    // resolved streaming plan to be enabled (else returns empty; the caller falls
    // back to transcribe()). Engine thread only.
    std::string transcribe_streaming(const float* d_mel_full, int num_mel_bins,
                                     int total_frames, int max_new_tokens) {
        const auto& plan = engine_->get_impl()->m_runtime.audio_streaming;
        if (!plan.enabled || !audio_head_loaded() || total_frames <= 0) return std::string();
        activate(audio_);
        const int W   = plan.window_tokens * blackwell::audio::kMelFramesPerSoftToken;
        const int hop = plan.hop_tokens    * blackwell::audio::kMelFramesPerSoftToken;

        int next = -1;
        auto prefill_ids = [&](const std::vector<int>& ids) -> bool {
            for (const int id : ids) {
                if (pos_ >= max_context_) return false;
                if (engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                    blackwell::EngineStatus::Success) return false;
                ++pos_;
            }
            return true;
        };
        const TurnFrame frame = make_turn_frame();
        if (!prefill_ids(frame.user_hdr)) return std::string();

        // Slide the window across [0, total_frames) in `hop`-frame steps. Per hop:
        //   Reconcile   — compare the overlap against the injected history and
        //                 append the delta (rollback+re-inject on divergence);
        //   CenterSlice — append only the NEW stable center rows (no cosine work,
        //                 no mid-speech rollbacks; pos only moves forward).
        const bool center =
            plan.mode == blackwell::AudioStreamingMode::CenterSlice;
        const std::size_t stage_elems = static_cast<std::size_t>(num_mel_bins) * W;
        if (d_window_stage_.count() != stage_elems) d_window_stage_.allocate(stage_elems);
        audio_pipeline_->reset_history();
        for (int e = std::min(W, total_frames); ; e += hop) {
            const int end    = std::min(e, total_frames);
            const int start  = std::max(0, end - W);
            const int frames = end - start;
            // Extract the contiguous [num_mel_bins, frames] window (mel-major slice).
            CUDA_CHECK_THROW(cudaMemcpy2D(
                d_window_stage_.get(), static_cast<std::size_t>(frames) * sizeof(float),
                d_mel_full + start,    static_cast<std::size_t>(total_frames) * sizeof(float),
                static_cast<std::size_t>(frames) * sizeof(float),
                static_cast<std::size_t>(num_mel_bins), cudaMemcpyDeviceToDevice));
            // `start` is the window's absolute mel-frame offset -> monotonic positions.
            if (center) prefill_audio_center_hop(d_window_stage_.get(), frames, start);
            else        prefill_audio_hop(d_window_stage_.get(), frames, start);
            if (end >= total_frames) break;
        }
        // End of utterance == the VAD pause: complete the phrase by committing the
        // withheld right-edge tokens before closing the turn frame.
        if (center) commit_audio_right_edge();

        if (!prefill_ids(frame.user_eot)) return std::string();
        if (!prefill_ids(frame.gen_cue))  return std::string();

        std::string out;
        for (int i = 0; i < max_new_tokens && pos_ < max_context_; ++i) {
            if (tok_->is_stop(next)) break;
            out += tok_->decode(next, /*render_special=*/false);
            if (engine_->forward_status(next, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                blackwell::EngineStatus::Success) break;
            ++pos_;
        }
        return out;
    }

    // The output tags the CURRENT task selection expects. The frozen system prompt
    // can only describe ONE format, so the per-turn instruction restates the
    // contract that matches the live toggles — asking for a [Translation] tag the
    // user turned off is exactly how the model learns to emit junk. Also drives
    // the panel's readout and encode_history_turn's parser. Any thread.
    std::string expected_output_format() const {
        const bool tr = task_transcribe_.load(std::memory_order_acquire);
        const bool tl = task_translate_.load(std::memory_order_acquire);
        if (tr && tl) return "[Speech] <transcript> | [Translation] <translation>";
        if (tl)       return "[Translation] <translation>";
        return "[Speech] <transcript>";
    }

    // Per-turn instruction carrying the TASK selection and the FORCED language
    // directives. It lives in the user turn (re-prefilled every utterance), NOT
    // the frozen system prompt, so a dropdown/checkbox change takes effect on the
    // very next turn. The explicit native-script constraint is what kills the
    // Latin-transliteration failure mode on short audio hops. Any thread (three
    // atomics plus the audio-task mutex, which is only ever taken here at turn
    // boundaries -- never from the audio hot path).
    //
    // ORDER IS LOAD-BEARING: language constraints, then the output-tag contract,
    // then the task phrase LAST — it ends in ": " and must abut the audio
    // soft-tokens that occupy the placeholder slot immediately after this text.
    std::string build_user_instruction() const {
        const int ti = tgt_lang_.load(std::memory_order_acquire);
        bool transcribe = task_transcribe_.load(std::memory_order_acquire);
        const bool translate = task_translate_.load(std::memory_order_acquire);
        // Both tasks off would produce an instruction with no verb; fall back to
        // transcription — the one task that needs no target language. Matches
        // expected_output_format(), which resolves the same case to [Speech].
        if (!transcribe && !translate) transcribe = true;

        std::string s;
        // The forced SOURCE language, however it was forced: the free-text
        // setting or the --src-lang table index (see forced_source_language).
        // Restated here even though the isolated session's frozen prefix already
        // carries it, because THIS is the text that abuts the audio soft-tokens
        // -- and because without isolation the prefix does not exist.
        if (const std::string src = forced_source_language(); !src.empty()) {
            s += "The audio language is ";
            s += src;
            s += ". Write the transcript in ";
            s += src;
            s += " using its native script (never transliterate). ";
        }
        s += "Reply using exactly this format: ";
        s += expected_output_format();
        s += ". ";

        // " into <Target>" — empty when the target is Auto, so every phrasing
        // below stays grammatical without a second set of branches.
        std::string into;
        if (ti > 0) {
            into = " into ";
            into += rt::kLanguages[ti];
        }
        // The user's own task verb, when they supplied one (Settings -> System
        // prompts -> Audio task). Everything above this point is machinery the
        // history parser depends on and is generated either way; only the verb is
        // theirs. It must still end abutting the audio soft-tokens, so a prompt
        // that forgot its trailing separator gets one -- the placeholder slot
        // follows IMMEDIATELY, and "...exactly as spoken.<audio>" reads to the
        // model as one word.
        //
        // UNDER ISOLATION this branch is dead, and deliberately so: the audio
        // task prompt is the transcription session's frozen SYSTEM prefix (see
        // prefill_audio_task_prefix), prefilled once instead of re-encoded into
        // every user block. What stays here is only what genuinely varies per
        // turn -- the language directives and the output-tag contract.
        if (!isolated_) {
            if (std::string custom = audio_task_prompt(); !custom.empty()) {
                s += custom;
                if (const char last = custom.back(); last != ' ') s += ' ';
                return s;
            }
        }

        if (transcribe && translate) {
            s += "Translate" + into + ". Transcribe the following audio: ";
        } else if (translate) {
            s += "Translate the following audio" + into + ": ";
        } else {
            s += "Transcribe the following audio: ";
        }
        return s;
    }

    // End-of-turn context policy (engine thread, after the decode loop):
    //   Stateless      -> roll the KV back to the base (== system floor): the turn
    //                     leaves no trace, so long runs cannot accumulate
    //                     transliteration/error feedback across utterances.
    //   BoundedHistory -> close the assistant turn in place with <|eot_id|> (the
    //                     KV keeps the richer audio-soft-token form) and record
    //                     the turn as TEXT; the deque is pruned to the token
    //                     budget every turn, but the KV rebuild (rollback to the
    //                     floor + re-prefill of the retained turns) is AMORTIZED —
    //                     it only runs when the context headroom demands it, since
    //                     a rebuild costs one forward_status per retained token.
    // An interrupted (barge-in) or empty turn is never recorded — the rollback
    // here is idempotent with the barge-in's pending do_rewind.
    void finalize_turn(const std::string& reply, bool completed) {
        // The known-user-text hint belongs to THIS turn: whichever of the exits
        // below runs, the next turn must not inherit it and start recording a
        // stale question against a fresh answer.
        struct ClearPending {
            std::string* s;
            ~ClearPending() { s->clear(); }
        } clear_pending{&pending_user_text_};

        pause_pending_ = false;   // the turn is over: the edge commit is resolved
        // AN EPHEMERAL SESSION IS NEVER BOUNDED. The transcription context is
        // rebuilt from its frozen prefix for every utterance in EVERY context
        // mode -- retaining transcripts there is how a misheard word (or an
        // outright hallucination over noise) becomes the next utterance's
        // context and then its prior. Dialogue memory belongs to the chat
        // session, which keeps it as clean text; see generate_local_reply.
        const bool bounded = !active_->ephemeral &&
                             context_mode() == ContextMode::BoundedHistory;
        if (!bounded) {
            // Also covers a live Bounded -> Stateless switch: drop the retained
            // text history and re-anchor the base to the frozen system prefix.
            // turn_history_ belongs to the CHAT session alone, so an ephemeral
            // turn re-anchors its own base without touching the dialogue -- were
            // it to clear the deque, every transcription would silently wipe the
            // conversation the user is having.
            if (!active_->ephemeral) {
                turn_history_.clear();
                history_text_tokens_ = 0;
            }
            history_base_pos_.store(static_cast<int>(system_prefix_tokens()),
                                    std::memory_order_relaxed);
        }
        if (!bounded || !completed || reply.empty()) {
            kv_cache_rollback(
                {history_base_pos_.load(std::memory_order_relaxed), verified_prompt_tokens_},
                !bounded ? (active_->ephemeral
                                ? "transcription turn discarded (ephemeral session: neither "
                                  "the audio nor the transcript becomes context)"
                                : "end-of-turn stateless flush (by design: audio + generated "
                                  "text leave no trace)")
                         : "incomplete/empty turn discarded");
            // The pipeline's injected-soft-token bookkeeping mirrors the AUDIO
            // session's KV, so only that session's turns may reset it: a chat
            // turn closing must not un-commit center tokens of a live utterance.
            if (audio_pipeline_ && (!isolated_ || active_->ephemeral))
                audio_pipeline_->reset_history();
            return;
        }

        close_assistant_turn();  // <|eot_id|> keeps the in-place KV template-valid

        TurnRecord rec{encode_history_turn(reply)};
        history_text_tokens_ += rec.ids.size();
        turn_history_.push_back(std::move(rec));
        // Prune the RECORDS to the budget every turn (cheap bookkeeping)...
        const auto budget = static_cast<size_t>(history_budget_tokens());
        while (!turn_history_.empty() && history_text_tokens_ > budget) {
            history_text_tokens_ -= turn_history_.front().ids.size();
            turn_history_.pop_front();
        }
        // ...but rebuild the KV only when the next turn would not fit.
        if (pos_ + kTurnHeadroomTokens > max_context_) {
            rebuild_history_kv();
        } else {
            history_base_pos_.store(pos_, std::memory_order_relaxed);
        }
        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[context] bounded turn recorded: history=%zu turn(s)/%zu text tok, "
                        "base=%d pos=%d\n", turn_history_.size(), history_text_tokens_,
                        history_base_pos_.load(std::memory_order_relaxed), pos_);
            std::fflush(stdout);
        }
    }

protected:
    // ---- the session switch (engine thread only) ------------------------------
    // Park the outgoing session's live decode state and make `s` current. It is
    // NON-REENTRANT by construction and that is safe here: the engine thread runs
    // one turn at a time, and the only two entry points that switch -- the ring's
    // command drain (audio) and post_engine_task (chat) -- are drained
    // sequentially by the same pump, never interleaved mid-turn.
    //
    // A no-op when isolation is off, so every call site below can switch
    // unconditionally and the single-session build stays byte-for-byte the
    // pre-isolation behaviour.
    void activate(Session& s) {
        if (!isolated_ || active_ == &s) return;
        active_->pos           = pos_;
        active_->verified      = verified_prompt_tokens_;
        active_->system_prefix = system_prefix_tokens();
        active_->history_base  = history_base_pos_.load(std::memory_order_relaxed);

        active_     = &s;
        pos_        = s.pos;
        active_seq_ = s.seq_id;
        verified_prompt_tokens_ = s.verified;
        // The base's floor is what effective_keep_tokens() clamps every rewind
        // to, so it MUST follow the session -- a barge-in resolved against the
        // other sequence's floor would truncate into a live prefix.
        set_system_prefix_tokens(s.system_prefix);
        history_base_pos_.store(s.history_base, std::memory_order_relaxed);
    }

    // The AUDIO session's persona when the user has not written one. Deliberately
    // narrow, and deliberately NOT the assistant's: this context exists to turn
    // speech into text and must refuse to do anything else. Answering the audio
    // instead of transcribing it is the exact failure a shared persona caused.
    static constexpr const char* kDefaultAudioSystemPrompt =
        "You are a speech transcription engine. You convert the user's audio into "
        "text and output nothing else -- no commentary, no answers, no questions.";

    // What actually gets frozen onto the transcription sequence: the user's task
    // text (or the default persona) plus, when a spoken language is forced, the
    // clause that removes language identification from the model's hands.
    //
    // THE CLAUSE CONSTRAINS THE TRANSCRIPT, NOT THE WHOLE REPLY. A translate turn
    // legitimately emits a second language, so "output only in X" would contradict
    // the very next instruction the model reads (expected_output_format's
    // [Translation] tag) -- and a prompt that argues with itself is how a model
    // learns to ignore both halves. Scoping it to the transcript keeps the two
    // tasks compatible, which is why the language is named THREE ways: what was
    // spoken, what to write it in, and what NOT to do (the two observed failure
    // modes -- Latin transliteration and drifting into an acoustic neighbour).
    //
    // Any thread (reads the mutex-guarded strings); called on the engine thread.
    std::string resolve_audio_system_prompt(const std::string& audio_task) const {
        std::string s = audio_task.empty() ? kDefaultAudioSystemPrompt : audio_task;
        const std::string lang = forced_source_language();
        if (lang.empty()) return s;
        if (!s.empty() && s.back() != ' ') s += ' ';
        s += "The spoken language is always " + lang + ". Write what you hear in " +
             lang + ", in its own native script -- never transliterate it into "
             "Latin letters, and never render it as any other language.";
        return s;
    }

    // Lay a frozen prefix on the ACTIVE session from position pos_ and publish its
    // length as that session's KV rewind floor.
    //
    // THE FLOOR IS WHAT LANDED, not what was asked for. A forward that faults
    // part-way through leaves only `done` columns in the KV, and publishing
    // ids.size() as the floor would put the rewind clamp ABOVE the real extent of
    // the cache -- kv_cache_rollback's max(cp.pos, floor) would then park pos_ on
    // uninitialized KV columns and the next turn would decode against garbage.
    // So the floor tracks the actual prefill, and the failure is raised only
    // AFTER the published state is self-consistent (INIT tier: this throws, and
    // the rebuild_*_prompt caller reports it to the user).
    uint32_t prefill_prefix_on_active(const std::vector<int>& ids, const char* what) {
        int next = -1;
        auto status = blackwell::EngineStatus::Success;
        uint32_t done = 0;
        for (const int id : ids) {
            status = engine_->forward_status(id, pos_, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                             active_seq_, &next);
            if (status != blackwell::EngineStatus::Success) break;
            ++pos_;
            ++done;
        }
        set_system_prefix_tokens(done);   // KV rewind floor (barge-in never truncates it)
        history_base_pos_.store(pos_, std::memory_order_relaxed);  // no history yet
        active_->system_prefix = done;    // survives the next activate() swap
        active_->history_base  = pos_;
        if (status != blackwell::EngineStatus::Success) {
            throw blackwell::engine_error(
                status, std::string(what) + "-prompt prefill failed on session '" +
                            active_->name + "' after " + std::to_string(done) + " of " +
                            std::to_string(ids.size()) + " tokens");
        }
        std::printf("[session] '%s' prefix frozen: %u token(s) on seq %d\n",
                    active_->name, done, active_seq_);
        std::fflush(stdout);
        return done;
    }

    struct TurnFrame {
        std::vector<int> user_hdr;   // user header + forced-language instruction
        std::vector<int> user_eot;   // "<|eot_id|>"
        std::vector<int> gen_cue;    // assistant generation prompt
    };
    static constexpr const char* kUserHeader =
        "<|start_header_id|>user<|end_header_id|>\n\n";

    // One place frames EVERY audio user turn (whole-utterance and streaming paths
    // both), so the language directives can never drift between them.
    TurnFrame make_turn_frame() const {
        return TurnFrame{
            tok_->encode(std::string(kUserHeader) + build_user_instruction(),
                         /*add_special=*/false),
            tok_->encode("<|eot_id|>", false),
            tok_->encode_generation_prompt()};
    }

    // ---- Barge-in routing: TIME decides, not the decode state ------------------
    // The pipeline can sit in DECODE_TRANSLATING for a minute — a long reply, or a
    // degenerate repetition loop the guards below now cut short. So "speech
    // arrived while we were decoding" is NOT evidence that the user is continuing
    // the same sentence. What distinguishes the two cases is how long they were
    // SILENT:
    //   <= kContinuationTimeoutS : they took a breath -> continuation; keep the
    //                              utterance's committed center tokens and resume
    //                              with a pointer-only rollback;
    //   >  kContinuationTimeoutS : a new thought      -> FULL rewind down to
    //                              history_base_pos_, flushing the stale (and
    //                              possibly hallucination-poisoned) audio, even
    //                              though a generation is still in flight.
    static constexpr float kContinuationTimeoutS = 2.0f;

    // The ONE sample rate this control ever sees. Not a knob: the Whisper front
    // end (whisper_dsp's mel geometry, the encoder's 30 s positional window) and
    // Silero are both fixed at 16 kHz, so a stream at any other rate would be
    // wrong long before it reached here. Named so the ms<->samples conversions
    // below stop being a bare `16`.
    static constexpr std::uint32_t kAudioSampleRate = 16000;
    static double samples_to_ms(std::size_t n) noexcept {
        return static_cast<double>(n) * 1000.0 / static_cast<double>(kAudioSampleRate);
    }

    // Engine thread. *out_silence_s (optional) receives the measured gap. A turn
    // with no recorded speech end — the very first utterance of the session — is
    // never a continuation: there is nothing to continue.
    bool is_continuation(float* out_silence_s = nullptr) const {
        if (!have_speech_end_) {
            if (out_silence_s != nullptr) *out_silence_s = 0.0f;
            return false;
        }
        const float s = static_cast<float>(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          last_speech_end_).count());
        if (out_silence_s != nullptr) *out_silence_s = s;
        return s <= kContinuationTimeoutS;
    }

    // The user stopped speaking: the VAD's stable boundary handed us the
    // utterance. Stamped at the START of the commit, BEFORE any encoder / prefill
    // / decode work — deliberately not at the end of the decode, so a long or
    // runaway generation counts as silence instead of masking it.
    void mark_speech_end() noexcept {
        last_speech_end_ = std::chrono::steady_clock::now();
        have_speech_end_ = true;
    }

    // Barge-in micro-rewind. The 8B AWQ backbone runs Continuous KV: the resident
    // slabs are position-addressed and overwritten in place, so re-anchoring pos_ is
    // enough for the decode math. But when layers are offloaded, the host mirror /
    // staging slots keep a high-water mark that would otherwise leak stale KV above
    // the rewound position across the barge-in; ContinuousKVManager::rewind() now
    // reconciles those via VRAMArena::truncate_kv (a no-op when nothing is
    // offloaded, so the resident-only path is unaffected). We honour the frozen-
    // prefix floor (effective_keep_tokens) so the rewind never drops the system
    // prefix, and re-anchor pos_ to the resolved keep so the next utterance decodes
    // on top of the system prefix, not on top of the previous (superseded) answer.
    blackwell::EngineStatus do_rewind(const Command& cmd) override {
        // A barge-in is about SPEECH, so it rewinds the TRANSCRIPTION sequence:
        // the stale utterance, its soft-tokens and the audio ring. It never
        // touches the chat session's KV -- a local reply that the same barge-in
        // superseded was already rolled back by its own finalize_turn, on its own
        // sequence, before this command was drained (both run on this thread and
        // cannot interleave).
        activate(audio_);

        // ROUTING: continuation vs fresh utterance is decided by the SILENCE
        // DURATION, never by "were we decoding?" — see is_continuation(). A
        // degenerate decode can hold the engine for a minute, and the speech that
        // finally arrives is then a new sentence, not a barge-in.
        float silence_s = 0.0f;
        const bool cont = is_continuation(&silence_s);
        if (utterance_open_ && center_slice_available()) {
            std::printf("[Barge-in Route] %s (silence %.2f s vs %.2f s continuation "
                        "window)\n",
                        cont ? "CONTINUATION — keep the utterance's center tokens"
                             : "FRESH UTTERANCE — full rewind, flush the stale audio",
                        silence_s, kContinuationTimeoutS);
            std::fflush(stdout);
        }

        // CenterSlice mid-utterance resume: a short-pause barge-in continues the
        // SAME user turn, so the committed center tokens are real audio context
        // and must survive. Any tentative state above the pause checkpoint (edge
        // tokens, turn-close template, generated text) was already discarded by
        // the interrupted decode's resume_after_pause(); this call is its
        // idempotent twin for the decode-not-yet-started case. A full rewind to
        // the floor here would throw the utterance's audio away mid-sentence.
        if (utterance_open_ && center_slice_available() && cont) {  // latch, not the live toggle
            resume_after_pause();
            last_rewind_keep_.store(static_cast<uint32_t>(pos_), std::memory_order_release);
            if (std::getenv("BLACKWELL_AV_DEBUG")) {
                std::printf("[center] barge-in resume: utterance continues at pos=%d\n", pos_);
                std::fflush(stdout);
            }
            return blackwell::EngineStatus::Success;
        }

        uint32_t keep = effective_keep_tokens(cmd.keep_prompt_tokens);
        // Bounded-history floor: retained dialogue turns sit directly above the
        // system prefix and are committed context — a barge-in must not truncate
        // them (they would silently vanish until the next rebuild). In stateless
        // mode the base equals the system floor, so this clamp is a no-op there.
        const auto hist_floor =
            static_cast<uint32_t>(history_base_pos_.load(std::memory_order_relaxed));
        if (keep < hist_floor) keep = hist_floor;
        last_rewind_keep_.store(keep, std::memory_order_release);
        std::printf("[KV CACHE INVALIDATION] Reason: speech-start full rewind (fresh "
                    "utterance after %.2f s of silence). pos %d -> %u (dropping %d KV "
                    "tokens; system floor=%u, history base=%u)\n",
                    silence_s, pos_, keep, pos_ - static_cast<int>(keep),
                    system_prefix_tokens(), hist_floor);
        std::fflush(stdout);
        pos_ = static_cast<int>(keep);
        // Full host-mirror reconciliation (Phase 0 truncate_kv), safe below the
        // engine's public API on the single engine-owning thread.
        engine_->get_impl()->kv_mgr->rewind(active_seq_, static_cast<int>(keep));
        // The KV is rewound to (at most) the frozen system prefix, so no audio soft-
        // tokens survive: drop the overlap-reconciliation / center-slice history too
        // (reset_history covers both), any outstanding pause-edge commit, and the
        // live utterance accumulators (a fresh utterance starts here).
        if (audio_pipeline_) audio_pipeline_->reset_history();
        pause_pending_ = false;
        reset_utterance();

        // AUDIO FLUSH, MINUS THE PRE-ROLL. A fresh utterance starts from roughly
        // THIS moment's audio: everything buffered further back is pre-speech-start
        // material — in auto mode the background noise accumulated since the last
        // commit (unbounded between utterances), in manual mode anything that
        // leaked in before the press — and encoding it makes the LLM hallucinate
        // over noise.
        //
        // But NOT everything in the ring is noise, which is what the original hard
        // flush got wrong. The ring is fed continuously, including while the VAD is
        // idle, and a neural detector needs 45–110 ms of audio before it can call a
        // block speech. So at the instant the onset fires, the newest samples in
        // the ring ARE the first phoneme of the word — flushing them clipped short
        // and plosive first syllables off every transcript. pre_roll_ms() names how
        // much of that acoustic head to keep; only the older remainder is dropped,
        // and the retained tail is consumed naturally as the start of the new
        // utterance by the next warm-prefill drain or commit.
        //
        // Discarded HERE because this runs on the engine thread, the only consumer
        // allowed to move the ring's read cursor (SPSC contract). The mid-utterance
        // resume branch above deliberately does not flush at all: there the whole
        // ring holds the freshly resumed speech.
        if (cmd.stream != nullptr) {
            blackwell::bridge::AudioRingBuffer& ring = cmd.stream->ring;
            const std::size_t avail = ring.available_samples();
            // Unsigned-safe max(0, avail - preroll): a ring holding less than the
            // pre-roll is entirely acoustic head, so nothing is dropped.
            // `keep_pcm` is SAMPLES, deliberately not named `keep` — that one is
            // the KV token count resolved above, and the two must never be confused.
            const std::size_t keep_pcm = std::min(avail, pre_roll_samples(kAudioSampleRate));
            const std::size_t stale    = avail - keep_pcm;
            if (stale != 0) {
                pcm_stage_.resize(stale);
                const std::size_t got = ring.read_samples(pcm_stage_.data(), stale);
                if (std::getenv("BLACKWELL_AV_DEBUG")) {
                    std::printf("[flush] speech start: discarded %zu stale PCM samples"
                                " (%.0f ms of pre-onset audio), retained %zu (%.0f ms"
                                " of pre-roll)\n",
                                got, samples_to_ms(got), keep_pcm, samples_to_ms(keep_pcm));
                    std::fflush(stdout);
                }
            }
        }

        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[kv-ckpt] barge-in rewind -> keep=%u (floor=%u)\n",
                        keep, system_prefix_tokens());
            std::fflush(stdout);
        }
        return blackwell::EngineStatus::Success;
    }

    // Speculative warm-prefill — the CenterSlice LIVE feed. While the user is
    // still speaking (throttled by warm_prefill_interval_ms), drain the buffered
    // PCM, recompute the utterance log-mel, and advance the sliding window by
    // whole hops, appending only the stable center tokens. This grows the KV
    // DURING speech, which is what keeps the encoder off the TTFT critical path.
    // Falls back to the documented no-op (Success, state machine keeps warming)
    // when the center-slice live path is not armed.
    blackwell::EngineStatus do_warm_prefill(const Command& cmd) override {
        // utterance_open_ = the mode latch: an utterance the center path opened
        // keeps hopping even if the UI toggled mid-speech (flip applies next turn).
        if (cmd.stream == nullptr || (!center_streaming_live() && !utterance_open_))
            return blackwell::EngineStatus::Success;
        activate(audio_);   // warm hops grow the TRANSCRIPTION sequence, never the chat one
        drain_stream_pcm(cmd.stream->ring);
        open_utterance_if_needed();
        run_pending_center_hops(/*flush_tail=*/false);
        return blackwell::EngineStatus::Success;
    }

    // Commit + decode: run a REAL greedy-ish decode on the engine and stream each
    // detokenized piece through the sink, checking cancelled(gen) BEFORE every
    // token so a barge-in aborts within one token. The user turn is a placeholder
    // (the audio-derived transcript is the marked seam — see the header preamble);
    // everything downstream is the genuine GPU decode path.
    blackwell::EngineStatus do_commit_decode(const Command& cmd) override {
        // TYPED TURN. Submitted from the UI thread through submit_text(), which
        // parked the text and flagged the command. It skips every audio stage --
        // no VAD boundary, no encoder, no soft-token splice -- and joins the
        // pipeline at exactly the point the audio paths do: prefill a user turn,
        // run THE decode loop, finalize, offer to the gate. Same loop, same
        // guards, same EOS rule, so a typed intent is billed (or refused) by the
        // identical arithmetic a spoken one is.
        if (cmd.text_turn) {
            std::string typed;
            if (!take_text_turn(typed) || typed.empty()) {
                clear_in_flight();
                return finish(cmd, 0);
            }
            return commit_text_decode(cmd, typed);
        }

        // SESSION B from here down. Every audio commit path -- center-slice,
        // whole-utterance, and the no-audio-head canned fallback's sibling -- runs
        // on the ephemeral transcription sequence, under the audio task prefix,
        // and hands its result onward as TEXT through the commit gate.
        activate(audio_);
        // An audio turn has no known user text -- the transcript the model is
        // about to write IS the record of what was said (encode_history_turn).
        pending_user_text_.clear();

        // The VAD's stable boundary IS the moment the user stopped speaking: start
        // the silence clock here, before the (possibly multi-second) commit work,
        // so a barge-in arriving later measures the true gap (see is_continuation).
        mark_speech_end();

        // CENTER-SLICE LIVE PATH: the audio was already streamed into the KV by
        // the warm hops; this commit only flushes the tail, completes the phrase
        // (right-edge commit), closes the turn frame, and decodes. An OPEN
        // utterance always commits here regardless of the live toggle (the mode
        // latch — only the center path ever opens one).
        if (cmd.stream != nullptr && (center_streaming_live() || utterance_open_))
            return commit_center_decode(cmd);

        // WHOLE-UTTERANCE LIVE PATH: with an audio head + DSP bound, transcribe
        // the buffered utterance (the SAME encode->project->prefill_audio->decode
        // path the --wav mode proves) instead of the text placeholder below.
        // ROUTING TELEMETRY: taking this path means a FULL utterance re-encode +
        // batch prefill (the multi-second <|audio x188|> event) — say WHY the
        // fast path was not taken, so a misrouted commit can never hide.
        if (audio_head_loaded() && dsp_ != nullptr && cmd.stream != nullptr) {
            std::printf("[Prefill Route] whole-utterance FALLBACK: %s\n",
                        center_slice_available()
                            ? "center-slice armed but live toggle = Whole utterance"
                            : "center-slice plan NOT armed (launch --stream-mode center)");
            std::fflush(stdout);
            return commit_audio_decode(cmd);
        }

        // No audio head / no DSP: fall back to a canned text turn so the live GUI
        // still shows varied real model output rather than one repeated answer.
        static constexpr const char* kPrompts[] = {
            "Briefly greet the user in Russian and English.",
            "Say one short encouraging sentence in Russian.",
            "Translate 'good morning' into Russian and use it in a short greeting.",
        };
        const std::size_t idx =
            turn_counter_.fetch_add(1, std::memory_order_relaxed) % 3;
        return commit_text_decode(cmd, kPrompts[idx]);
    }

    // ONE text-turn commit path, shared by the typed-input edge and the no-audio-
    // head fallback: frame a user turn + assistant cue through the checkpoint's
    // chat template, prefill it, then hand off to THE decode loop. Engine thread.
    blackwell::EngineStatus commit_text_decode(const Command& cmd,
                                               const std::string& user_text) {
        activate(chat_);   // typed words are a dialogue turn, not a transcription job
        pending_user_text_ = user_text;   // history records what was asked, verbatim
        std::vector<int> turn = tok_->encode_chat_message(
            blackwell::ChatMessage{"user", user_text});
        const std::vector<int> gen = tok_->encode_generation_prompt();
        turn.insert(turn.end(), gen.begin(), gen.end());

        int next = -1;
        const auto t_turn0 = std::chrono::steady_clock::now();  // text-path TTFT clock
        for (const int id : turn) {
            if (pos_ >= max_context_ ||
                engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                    blackwell::EngineStatus::Success) {
                finalize_turn(std::string(), /*completed=*/false);  // drop partial prefill
                clear_in_flight();
                return finish(cmd, 0);
            }
            ++pos_;
        }

        // Decode the assistant turn, streaming real detokenized tokens. t_turn0
        // seeds the per-token clock, so the first [Decode] line reports the
        // prefill-to-first-token wall time (no VAD trigger on this text path).
        const TurnDecode d = decode_assistant_turn(cmd, next, t_turn0,
                                                   temperature(), top_p());
        finalize_turn(d.reply, d.completed());
        publish_turn(cmd, d);   // commit gate: dispatches iff reason == Eos
        clear_in_flight();
        return finish(cmd, d.emitted);
    }

    // Consume audio embeddings that the audio pipeline (Stream 1: WhisperEncoder
    // -> UltravoxProjector, src/audio) has staged into pp.slot(frame) as
    // [num_audio, hidden] soft-tokens. This is the CONSUMER edge of the
    // double-buffered async handoff:
    //
    //   1. pp.consumer_acquire(frame, 0) makes the engine's own (default) stream
    //      WAIT on the producer's ready event -- pure cudaStreamWaitEvent, no host
    //      sync -- so Stream 1 can already be encoding the NEXT frame.
    //   2. Each embedding row is injected into the residual stream (d_X_accum) and
    //      prefilled into the KV via the white-box step_* sweep, exactly as the
    //      injected-embedding path (tests/integration/test_llama8b_fp8_integration).
    //      The engine's public API and decode loop are UNTOUCHED (hard rule).
    //   3. pp.consumer_release(frame, 0) frees the slot for the producer (frame+2).
    //
    // Returns the decode position after the audio block; the caller then decodes
    // the assistant turn from here with forward_status. NOTE: this is wired and
    // unit-tested at the buffer/handoff level (tests/integration/
    // test_audio_pingpong_pipeline), but not yet driven from the live UI loop --
    // the remaining seam is the DSP log-mel -> pipeline feed in main.cpp.
    int prefill_audio_embeddings(blackwell::audio::PingPongAudioBuffer& pp,
                                 long long frame, int num_audio) {
        auto* core = engine_->get_impl();
        const int hidden = static_cast<int>(core->m_config.hidden_dim);
        const int num_layers = static_cast<int>(core->m_config.num_layers);

        pp.consumer_acquire(frame, /*engine default stream=*/nullptr);

        for (int a = 0; a < num_audio && pos_ < max_context_; ++a, ++pos_) {
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                core->d_X_accum, pp.slot(frame) + static_cast<size_t>(a) * hidden,
                static_cast<size_t>(hidden) * sizeof(float),
                cudaMemcpyDeviceToDevice, /*stream 0*/ nullptr));
            core->kv_mgr->prepare_decode_step(active_seq_, pos_);

            bool ok = true;
            for (int l = 0; l < num_layers && ok; ++l) {
                using S = blackwell::EngineStatus;
                ok = core->step_attention_norm(l) == S::Success
                  && core->step_attention_qkv_projections(l) == S::Success
                  && core->step_attention_math(l, pos_) == S::Success
                  && core->step_attention_out(l) == S::Success
                  && core->step_mlp_norm(l) == S::Success
                  && core->step_mlp_projections(l) == S::Success
                  && core->step_mlp_out(l) == S::Success;
            }
            if (ok) ok = core->step_final_ops() == blackwell::EngineStatus::Success;
            if (!ok) break;
        }

        pp.consumer_release(frame, /*stream 0*/ nullptr);
        return pos_;
    }

    // Inject `count` projector soft-token rows starting at d_embeds[start_row] into
    // the KV via the white-box step_* sweep, advancing pos_. d_embeds is the audio-
    // stream projector output; the caller must have synchronized that stream first.
    // Returns Success, or the first failing step's status.
    blackwell::EngineStatus inject_embedding_rows(const float* d_embeds, int start_row,
                                                  int count) {
        auto* core = engine_->get_impl();
        const int hidden = static_cast<int>(core->m_config.hidden_dim);
        const int num_layers = static_cast<int>(core->m_config.num_layers);
        for (int a = 0; a < count && pos_ < max_context_; ++a, ++pos_) {
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                core->d_X_accum,
                d_embeds + static_cast<size_t>(start_row + a) * hidden,
                static_cast<size_t>(hidden) * sizeof(float),
                cudaMemcpyDeviceToDevice, /*stream 0*/ nullptr));
            core->kv_mgr->prepare_decode_step(active_seq_, pos_);
            using S = blackwell::EngineStatus;
            bool ok = true;
            for (int l = 0; l < num_layers && ok; ++l) {
                ok = core->step_attention_norm(l) == S::Success
                  && core->step_attention_qkv_projections(l) == S::Success
                  && core->step_attention_math(l, pos_) == S::Success
                  && core->step_attention_out(l) == S::Success
                  && core->step_mlp_norm(l) == S::Success
                  && core->step_mlp_projections(l) == S::Success
                  && core->step_mlp_out(l) == S::Success;
            }
            if (ok) ok = core->step_final_ops() == S::Success;
            if (!ok) return S::CudaRuntimeError;
        }
        return blackwell::EngineStatus::Success;
    }

    // One streaming hop with DYNAMIC OVERLAP RECONCILIATION. d_mel is [num_mel_bins,
    // mel_frames] for the CURRENT sliding window. Encode+project the window, compare
    // its overlapping prefix against the injected history, and either:
    //   * divergence -> kv_cache_rollback(rewind_to_pos) then re-inject the corrected
    //     overlap tail + the new hop (from refill_from), or
    //   * stable     -> inject only the tail delta (hop_tokens).
    // Then record what was committed. Returns pos_ after the hop. Engine thread only;
    // no-op reconcile (plan disabled) degenerates to Phase-1 tail-slicing.
    int prefill_audio_hop(const float* d_mel, int mel_frames, int mel_frame_offset = 0) {
        const auto win = audio_pipeline_->encode_project_window(d_mel, mel_frames,
                                                               mel_frame_offset);
        // The projector ran on the audio stream; make its output visible to the
        // host-side reconcile (D2H) and the stream-0 injection below.
        CUDA_CHECK_THROW(cudaStreamSynchronize(audio_pipeline_->audio_stream()));

        const auto& plan = engine_->get_impl()->m_runtime.audio_streaming;
        // Cold start (no injected history yet): seed the ENTIRE window -- there is no
        // overlap to reconcile, and injecting only the tail delta would drop the
        // window's leading context. Subsequent hops append deltas / reconcile.
        const bool cold = audio_pipeline_->history_size() == 0;
        const blackwell::audio::ReconciliationPlan rec =
            cold ? blackwell::audio::ReconciliationPlan{}
                 : audio_pipeline_->reconcile(win.embeds, win.num_tokens);

        int start_row = 0, count = win.num_tokens;
        if (cold) {
            start_row = 0;
            count     = win.num_tokens;
        } else if (rec.diverged) {
            kv_cache_rollback({rec.rewind_to_pos, verified_prompt_tokens_},
                              "reconcile divergence (overlap cosine below threshold)");
            start_row = rec.refill_from;
            count     = win.num_tokens - rec.refill_from;
        } else {
            const int hop = plan.hop_tokens > 0 ? plan.hop_tokens : win.num_tokens;
            start_row = std::max(0, win.num_tokens - hop);
            count     = win.num_tokens - start_row;
        }

        const int hidden = static_cast<int>(engine_->get_impl()->m_config.hidden_dim);
        const int inject_pos = pos_;
        inject_embedding_rows(win.embeds, start_row, count);
        audio_pipeline_->record_injected(
            win.embeds + static_cast<size_t>(start_row) * hidden, inject_pos, count);

        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            if (rec.diverged)
                std::printf("[reconcile] divergence: rollback pos=%d, refill %d row(s) "
                            "(window %d tok) -> pos=%d\n",
                            rec.rewind_to_pos, count, win.num_tokens, pos_);
            else
                std::printf("[reconcile] stable: appended delta %d row(s) (window %d tok) "
                            "-> pos=%d\n", count, win.num_tokens, pos_);
            std::fflush(stdout);
        }
        return pos_;
    }

    // ---- CenterSlice streaming (append-only during speech) -----------------------
    // One hop: encode+project the window, then inject ONLY the new stable center
    // rows (the pipeline drops both CNN-distorted edges and tracks the absolute
    // committed count). No cosine checks, no rollbacks — pos_ only moves forward
    // while speech is active. Engine thread only.
    int prefill_audio_center_hop(const float* d_mel, int mel_frames, int mel_frame_offset) {
        const auto slice = audio_pipeline_->center_hop(d_mel, mel_frames, mel_frame_offset);
        // The hop ran on the audio stream; the injection below runs on stream 0.
        CUDA_CHECK_THROW(cudaStreamSynchronize(audio_pipeline_->audio_stream()));
        if (slice.count > 0) {
            inject_embedding_rows(slice.embeds, slice.start_row, slice.count);
            // Visible proof the incremental path is alive DURING speech: if these
            // lines are absent before a commit, the commit will be a full re-prefill.
            std::printf("[Warm Hop] +%d center token(s) (utterance committed: %d) -> pos=%d\n",
                        slice.count, audio_pipeline_->center_committed(), pos_);
            std::fflush(stdout);
        } else if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[center] hop: window not grown yet (committed=%d, pos=%d)\n",
                        audio_pipeline_->center_committed(), pos_);
            std::fflush(stdout);
        }
        return pos_;
    }

    // VAD pause: capture the KV checkpoint, then commit the withheld right-edge
    // tokens to complete the phrase (they were persisted by the last hop — zero
    // encoder work here). If speech resumes before the turn finalizes,
    // resume_after_pause() undoes exactly this commit. pause_pending_ arms even
    // when the edge slice is empty: everything the caller prefills ABOVE the
    // checkpoint (turn-close template + generated tokens) is tentative until the
    // turn finalizes, and a resume must discard it either way. Engine thread only.
    int commit_audio_right_edge() {
        pause_cp_ = kv_cache_checkpoint();
        pause_pending_ = true;
        const auto slice = audio_pipeline_->commit_pending_edge();
        if (slice.count > 0)
            inject_embedding_rows(slice.embeds, slice.start_row, slice.count);
        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[center] pause: committed %d edge row(s) -> pos=%d\n",
                        slice.count, pos_);
            std::fflush(stdout);
        }
        return pos_;
    }

    // Speech resumed after a pause-commit (barge-in / micro-pause): pointer-only
    // rollback of the tentative edge tokens (pos -= K; truncate_kv is a no-op for
    // the resident model — 0 ms of GPU compute) and un-commit the pipeline's
    // counter, so the next hop overwrites the edge region with fresh,
    // context-complete center tokens. Idempotent. Engine thread only.
    void resume_after_pause() {
        if (!pause_pending_) return;
        pause_pending_ = false;
        kv_cache_rollback(pause_cp_,
                          "barge-in resume (pointer-only: tentative edge/template/"
                          "generated tokens dropped, center tokens KEPT)");
        audio_pipeline_->uncommit_pending_edge();
        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[center] resume: edge un-committed -> pos=%d (committed=%d)\n",
                        pos_, audio_pipeline_->center_committed());
            std::fflush(stdout);
        }
    }

private:
    // Emit the final callback (clean supersede or cap/fault both report OK to the
    // stream) and return Success — a barge-in is not an error.
    blackwell::EngineStatus finish(const Command& cmd, int emitted) {
        cmd.sink.emit("", emitted, /*is_final=*/1, BRIDGE_OK);
        return blackwell::EngineStatus::Success;
    }

    // ---- degenerate-loop guards (the GPU's seatbelt) --------------------------
    // Live testing found the 8B backbone falling into exact phrase loops in a
    // quiet room, cooking the GPU for 2000+ tokens and pinning the pipeline in
    // DECODE_TRANSLATING for over a minute. Two INDEPENDENT brakes, because
    // either one alone can be defeated:
    //   * kRepetitionPenalty shapes the logits (in the sampler, before argmax /
    //     top-p) so an exact repeat becomes strictly less likely — this breaks
    //     the loop mathematically rather than just truncating it;
    //   * max_new_tokens() is the hard ceiling for when it does not. Hitting
    //     it force-closes the turn CLEANLY (counts as completed -> finalize_turn
    //     emits the EOT / stateless flush), so the pipeline returns to IDLE and
    //     the next utterance is a normal fresh start, not a wedged state.
    // The penalty window is per TURN and rolling: a penalty that leaked across
    // utterances would suppress legitimately repeated words forever, and an
    // unbounded window would suppress common words within one long reply.
    static constexpr float kRepetitionPenalty  = 1.15f;
    // Default for the live ceiling below; the effective value is set_max_new_tokens().
    static constexpr int   kDefaultMaxGeneratedTokens = 256;
    static constexpr int   kRepetitionWindow   = 64;

    // What a decode loop produced.
    //
    // TWO DIFFERENT QUESTIONS, TWO DIFFERENT FIELDS. Do not collapse them.
    //
    //   completed()  "is this turn usable as KV//text history?"  A max-token or
    //                context-cap stop is CLEAN by this measure: the text is
    //                well-formed, just short, so it is kept as history. Only a
    //                barge-in or an engine fault leaves the turn unusable.
    //
    //   reason       "may this turn be DISPATCHED to the paid Cloud API?"  Here
    //                a budget stop is NOT clean -- it is a severed thought, and
    //                sending it burns prefill tokens on a fragment.
    //
    // Before `reason` existed, EOT / per-turn cap / context cap all produced
    // completed()==true and the distinction survived only as a `const char*`
    // stop_reason used for logging. Anything gating on completed() would have
    // paid for truncated intents. Gate on `reason`; keep completed() for KV.
    struct TurnDecode {
        std::string reply;
        int  emitted = 0;
        bool interrupted = false;   // barge-in superseded this generation
        bool faulted = false;       // run_token / sampling reported != Success
        blackwell::bridge::TerminationReason reason =
            blackwell::bridge::TerminationReason::None;
        bool completed() const noexcept { return !interrupted && !faulted; }
        // The commit gate's predicate, restated here so a call site cannot get
        // it subtly wrong (e.g. by reaching for completed()).
        bool dispatchable() const noexcept {
            return blackwell::bridge::is_dispatchable(reason);
        }
    };

    // THE decode loop — one implementation behind every live commit path
    // (center-slice, whole-utterance, text placeholder), so the guards above can
    // never drift between them. `first_token` is the token the prefill's last
    // forward_status already sampled; t_start seeds the per-token clock (the VAD
    // trigger on the audio paths), so the first [Decode] line reports true TTFT.
    // Engine thread only; advances pos_.
    TurnDecode decode_assistant_turn(const Command& cmd, int first_token,
                                     std::chrono::steady_clock::time_point t_start,
                                     float temperature, float top_p) {
        TurnDecode out;
        int next = first_token;
        const char* stop_reason = "max length hit (context budget)";
        reset_repetition_window();
        // A vocabulary is built over BYTES: one Cyrillic/CJK character is
        // routinely split across two tokens, so a per-token piece is often
        // invalid UTF-8 on its own. The assembler holds the trailing fragment
        // until the token that completes it arrives, so every chunk that leaves
        // this loop is a whole number of code points. Fresh per turn -- a
        // superseded generation's dangling bytes must never prepend themselves
        // to the next turn's first token.
        utf8_.reset();
        // Snapshot the ceiling ONCE: a mid-turn change from the UI must not move
        // the finish line under a generation that is already running.
        const int cap = max_new_tokens();
        auto t_tok = t_start;
        // Default: falling out of the `while` means the CONTEXT budget ran out.
        // Seeded here (not left None) so every exit path below carries a verdict
        // and no path can leave `reason` at a value the gate treats as unset.
        out.reason = blackwell::bridge::TerminationReason::TokenCap;
        while (pos_ < max_context_) {
            if (cancelled(cmd.gen)) {                              // wait-free barge-in
                out.interrupted = true;
                out.reason = blackwell::bridge::TerminationReason::BargeIn;
                stop_reason = "barge-in detected (gen superseded)";
                break;
            }
            {   // telemetry: EVERY produced token, stop/special tokens included
                const auto now = std::chrono::steady_clock::now();
                log_decode_token(next,
                                 std::chrono::duration<double, std::milli>(now - t_tok).count());
                t_tok = now;
            }
            if (tok_->is_stop(next)) {
                // THE COMMIT CASE. The model closed the thought itself; this is
                // the only exit from this loop that may reach the Cloud API.
                out.reason = blackwell::bridge::TerminationReason::Eos;
                stop_reason = "EOT token reached";
                break;
            }
            // Checked AFTER the stop test so a natural EOT landing exactly on the
            // cap is still reported as an EOT, and BEFORE emitting so the turn
            // ends at exactly `cap` streamed tokens. That ordering
            // also means a turn is only ever TokenCap when it genuinely had more
            // to say -- which is what makes dropped_token_cap() actionable.
            if (out.emitted >= cap) {
                out.reason = blackwell::bridge::TerminationReason::TokenCap;
                stop_reason = "Max turn tokens reached";
                break;
            }
            const std::string piece = tok_->decode(next, /*render_special=*/false);
            if (!piece.empty()) {
                // reply accumulates the RAW bytes (concatenation is lossless and
                // is what the commit gate dispatches); only the STREAMED chunk
                // goes through the assembler. An empty chunk means this token was
                // half a character -- emit nothing rather than an empty event.
                out.reply += piece;
                const std::string chunk = utf8_.push(piece);
                if (!chunk.empty()) {
                    cmd.sink.emit(chunk.c_str(), out.emitted, /*is_final=*/0, BRIDGE_OK);
                }
            }
            ++out.emitted;
            push_repetition_token(next);   // this turn's rolling penalty window
            if (decode_step(next, pos_, temperature, top_p, &next) !=
                blackwell::EngineStatus::Success) {
                out.faulted = true;
                out.reason = blackwell::bridge::TerminationReason::Fault;
                stop_reason = "engine fault (run_token/sample != Success)";
                break;
            }
            ++pos_;
        }
        // The turn is over: surrender any held fragment. Non-empty only when the
        // generation genuinely stopped mid-character (a cap or barge-in landing
        // between the two halves of a code point) -- dropping it silently would
        // lose a character the model did produce.
        if (const std::string tail = utf8_.flush(); !tail.empty()) {
            cmd.sink.emit(tail.c_str(), out.emitted, /*is_final=*/0, BRIDGE_OK);
        }
        log_decode_stop(stop_reason, pos_, out.emitted);
        return out;
    }

    // Hand the finished turn to the commit gate. Called on EVERY commit path,
    // including the non-dispatchable ones -- that is deliberate. "Skip dispatch"
    // must be implemented as "offer and let the gate reject", never as "do not
    // offer", or dropped_token_cap() never increments and a mis-sized
    // max_new_tokens() silently kills the cloud pathway with no telemetry.
    //
    // Returns true iff the intent was committed (i.e. reason == Eos).
    bool publish_turn(const Command& cmd, const TurnDecode& d) noexcept {
        // WHAT CROSSES THE BOUNDARY. From the ephemeral transcription session the
        // intent is the user's WORDS, not that session's output contract: its
        // "[Speech] ... | [Translation] ..." framing is an artifact of how B was
        // asked to answer, and forwarding it verbatim would make the chat session
        // (and the paid API) reason about the tags. A chat-session turn is
        // already plain text and passes through untouched.
        std::string intent =
            (isolated_ && active_->ephemeral) ? extract_transcript(d.reply, d.reply) : d.reply;
        const bool committed =
            publish_intent(d.reason, cmd.gen, std::move(intent),
                           static_cast<std::uint32_t>(d.emitted < 0 ? 0 : d.emitted));
        // Every turn reports its verdict. Without this the gate is invisible from
        // the console and the only way to tell a dispatched turn from a dropped
        // one is to watch the UI -- which is exactly the silent-failure mode the
        // whole Local Router design is trying to avoid.
        if (committed) {
            std::printf("[Commit Gate] EOS -> DISPATCHED (%d tokens)\n", d.emitted);
        } else if (d.reason == blackwell::bridge::TerminationReason::TokenCap) {
            // Loudest case: the user spoke, the local model answered, and nothing
            // was ever dispatched -- with no error raised anywhere else.
            std::printf("[Commit Gate] TRUNCATED at %d tokens (cap %d) -- NOT dispatched. "
                        "Raise the reply-length limit in Settings if this repeats.\n",
                        d.emitted, max_new_tokens());
        } else {
            std::printf("[Commit Gate] %s -- NOT dispatched (%d tokens)\n",
                        blackwell::bridge::to_string(d.reason), d.emitted);
        }
        std::fflush(stdout);
        return committed;
    }

    // The checkpoint's stop set (<|eot_id|>, <|end_of_text|>, <|im_end|>, ...),
    // resolved by the tokenizer from generation_config.json.
    //
    // NOTE ON REACH: this override serves EngineControlBridge::run_decode_loop,
    // which THIS class does not use — decode_assistant_turn is its own loop (it
    // needs the repetition-penalty window and the per-token TTFT telemetry that
    // the bridge loop has no notion of), and that loop calls tok_->is_stop
    // directly. The override exists so the two loops cannot disagree about what
    // EOS means, and so any consumer that does drive run_decode_loop against
    // this control gets the real stop set instead of the base tier's
    // fail-closed `false`.
    bool is_eos(int token_id) const noexcept override {
        return tok_ != nullptr && tok_->is_stop(token_id);
    }

    // One PENALIZED decode step. Mirrors BlackwellEngine::forward_status — the
    // same run_token then sample_top_p pair — but shapes the logits with this
    // turn's repetition window in between. The engine's public API and its own
    // decode path stay UNTOUCHED (hard rule): the extra stage sits between two
    // calls the white-box tier already owns, exactly like the step_* audio
    // injection sweep above. RUNTIME error tier: noexcept, reports by status.
    blackwell::EngineStatus decode_step(int token_id, int pos, float temperature,
                                        float top_p, int* out_token) noexcept {
        auto* core = engine_->get_impl();
        try {
            const blackwell::EngineStatus st =
                core->run_token(token_id, pos, active_seq_);
            if (st != blackwell::EngineStatus::Success) return st;
            launch_repetition_penalty_kernel(
                core->d_logits, core->m_config.vocab_size, d_penalty_ids_,
                std::min(penalty_count_, kRepetitionWindow), kRepetitionPenalty);
            *out_token = sample_top_p(core->d_logits, core->m_config.vocab_size,
                                      temperature, top_p);
            return blackwell::EngineStatus::Success;
        } catch (...) {
            // The decode hot loop never unwinds (hybrid error doctrine): the only
            // throwing callee is sample_top_p's scratch alloc / copy-back.
            return blackwell::EngineStatus::CudaRuntimeError;
        }
    }

    // Arm an empty penalty window for a new turn (lazily allocating the device
    // ring on first use). NOTHING here may throw: an exception escaping the
    // decode path skips finish()/clear_in_flight() and wedges the pipeline in
    // DECODE_TRANSLATING — the exact failure these guards exist to prevent. On a
    // failed allocation the penalty is simply off for the session (count()==0
    // makes push/launch no-ops); the hard token cap still holds the line.
    void reset_repetition_window() noexcept {
        penalty_count_ = 0;
        try {
            if (d_penalty_ids_.count() != static_cast<std::size_t>(kRepetitionWindow))
                d_penalty_ids_.allocate(static_cast<std::size_t>(kRepetitionWindow));
            penalty_ids_host_.assign(static_cast<std::size_t>(kRepetitionWindow), -1);
        } catch (...) {
            // allocate() reset the buffer before failing, so count() == 0 holds.
        }
    }

    // Record one generated id into the rolling window: a 4-byte H2D copy per
    // token (~microseconds against a ~30 ms decode step). Synchronous on purpose
    // — the source is a host member, and an async copy would race the next write.
    // A failed copy only weakens the penalty for one token; see above for why it
    // must not throw.
    void push_repetition_token(int token_id) noexcept {
        if (d_penalty_ids_.count() == 0) return;
        const std::size_t slot =
            static_cast<std::size_t>(penalty_count_ % kRepetitionWindow);
        penalty_ids_host_[slot] = token_id;
        if (cudaMemcpy(d_penalty_ids_.get() + slot, &penalty_ids_host_[slot],
                       sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
            return;
        ++penalty_count_;
    }

    // ---- raw decode telemetry (stdout diagnostics for the TRANSLATION stall) --
    // One line per generated token. The text is SPECIAL-rendered so invisible
    // tokens (<|eot_id|>, headers, hallucinated specials) are identifiable —
    // the piece the sink streams stays render_special=false as before. dt_ms is
    // the time this token took to produce; the FIRST token's clock is seeded at
    // the VAD trigger, so its line reports the true wall-clock TTFT.
    void log_decode_token(int token_id, double dt_ms) const {
        std::printf("[Decode] TokenID: %d | String: '%s' | Time: %.1f ms\n", token_id,
                    tok_->decode(token_id, /*render_special=*/true).c_str(), dt_ms);
        std::fflush(stdout);
    }
    static void log_decode_stop(const char* reason, int pos, int emitted) {
        std::printf("[Decode Stop] Reason: %s (pos=%d, emitted=%d)\n", reason, pos, emitted);
        std::fflush(stdout);
    }

    // ---- bounded-history internals (engine thread only) -----------------------

    // Headroom the NEXT turn needs below max_context_: <=188 audio soft-tokens
    // (30 s), the template framing, and a decode budget. Crossing it triggers the
    // amortized history rebuild.
    static constexpr int kTurnHeadroomTokens = 320;

    // One completed dialogue turn re-encoded as TEXT for KV rebuilds: user turn =
    // the transcript the model itself produced, assistant turn = the full
    // formatted reply. The in-place KV keeps the richer audio form until a
    // rebuild evicts it, so between rebuilds the model attends to full turns.
    struct TurnRecord {
        std::vector<int> ids;
    };

    // The decode loop breaks on is_stop BEFORE forwarding the stop token, so the
    // assistant turn is still open in the KV; feed one <|eot_id|> to close it.
    void close_assistant_turn() {
        int next = -1;
        for (const int id : tok_->encode("<|eot_id|>", /*add_special=*/false)) {
            if (pos_ >= max_context_) break;
            if (engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                blackwell::EngineStatus::Success)
                break;
            ++pos_;
        }
    }

    // Recover the user-side text -- WHAT WAS SAID -- from the transcription
    // session's tagged reply. Tolerates EVERY task mode's tag set (see
    // expected_output_format): "[Speech] X | [Translation] Y", a bare
    // "[Speech] X", or a translation-only reply with no transcript to recover.
    // Anything that does not match degrades to an "(audio)" placeholder rather
    // than carrying the model's formatting noise forward.
    //
    // This is the ISOLATION BOUNDARY's projection function: session B's output
    // is a formatted transcript, and both the chat session and the cloud
    // transport want the user's words, not B's output contract. Nothing but the
    // string this returns ever crosses from B to A.
    // `fallback` is what an UNTAGGED reply degrades to, and the two callers want
    // different things: the retained history wants a neutral "(audio)" placeholder
    // (a stray sentence of the model's formatting noise must not become the
    // conversation's memory of what the user said), while the intent the gate
    // dispatches wants the raw text -- a model that dropped the tags but still
    // transcribed correctly should not have its transcript replaced by a
    // placeholder on the way to the answer.
    static std::string extract_transcript(const std::string& reply,
                                          const std::string& fallback) {
        auto trim = [](const std::string& s) {
            const auto b = s.find_first_not_of(" \t\r\n");
            const auto e = s.find_last_not_of(" \t\r\n");
            return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
        };
        constexpr const char* kSpeech = "[Speech]";
        const auto sp = reply.find(kSpeech);
        if (sp != std::string::npos) {
            const auto beg = sp + std::char_traits<char>::length(kSpeech);
            // The transcript runs to the translation separator, or to the end of
            // the reply in transcribe-only mode.
            const auto cut = reply.find(" | [Translation]", beg);
            const std::string t = trim(cut == std::string::npos
                                           ? reply.substr(beg)
                                           : reply.substr(beg, cut - beg));
            if (!t.empty()) return t;
        }
        // Translation-only mode: the tagged translation IS the user-side text.
        constexpr const char* kTranslation = "[Translation]";
        const auto tp = reply.find(kTranslation);
        if (tp != std::string::npos) {
            const std::string t =
                trim(reply.substr(tp + std::char_traits<char>::length(kTranslation)));
            if (!t.empty()) return t;
        }
        return fallback;
    }

    std::vector<int> encode_history_turn(const std::string& reply) const {
        // WHAT THE USER ACTUALLY SAID. On a text turn we know it exactly (it was
        // handed to us), so use it; the tag parsing below is the AUDIO session's
        // recovery path, where the only record of the user's words is the
        // transcript the model just wrote. Before the sessions were split every
        // turn went through the parser, which meant a chat turn's history
        // recorded the user as "(audio)" -- the reply's tags never matched.
        const std::string user_text = pending_user_text_.empty()
                                          ? extract_transcript(reply, "(audio)")
                                          : pending_user_text_;
        const std::string turn =
            std::string(kUserHeader) + user_text + "<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n" + reply + "<|eot_id|>";
        return tok_->encode(turn, /*add_special=*/false);
    }

    // The amortized rebuild: rollback to the frozen system floor, then re-prefill
    // ONLY the retained text turns. Costs one forward_status per retained token
    // (<= the budget), which is why finalize_turn defers it until the context
    // headroom demands it.
    void rebuild_history_kv() {
        const bool dbg = std::getenv("BLACKWELL_AV_DEBUG") != nullptr;
        const int floor = static_cast<int>(system_prefix_tokens());
        kv_cache_rollback({floor, verified_prompt_tokens_},
                          "bounded-history KV rebuild (context headroom exhausted)");
        if (audio_pipeline_) audio_pipeline_->reset_history();
        int next = -1;
        for (const TurnRecord& t : turn_history_) {
            for (const int id : t.ids) {
                if (pos_ >= max_context_) break;
                if (engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                    blackwell::EngineStatus::Success)
                    break;
                ++pos_;
            }
        }
        history_base_pos_.store(pos_, std::memory_order_relaxed);
        if (dbg) {
            std::printf("[context] history rebuild: %zu turn(s), %zu text tok -> base=%d\n",
                        turn_history_.size(), history_text_tokens_, pos_);
            std::fflush(stdout);
        }
    }

    // ---- CenterSlice LIVE internals (engine thread only) -----------------------

    // The encoder's positional-embedding ceiling: mel frames beyond 30 s cannot be
    // encoded at their true absolute offset, so the hop driver stops there (the
    // whole-utterance path has the same 30 s cap).
    static constexpr int kMaxUtteranceMelFrames = 3000;

    // Is the append-only live path selected RIGHT NOW? Static capability
    // (center_slice_available) AND the UI's live toggle. Callers that may run
    // mid-utterance must OR this with utterance_open_ (the mode latch): an open
    // utterance always finishes on the center path that started it, so a live
    // flip applies from the next utterance and can never corrupt an open turn.
    bool center_streaming_live() const {
        return center_slice_available() &&
               live_center_.load(std::memory_order_acquire);
    }

    // Move every sample buffered in the ring into the utterance accumulator.
    void drain_stream_pcm(blackwell::bridge::AudioRingBuffer& ring) {
        const std::size_t avail = ring.available_samples();
        if (avail == 0) return;
        const std::size_t old = utterance_pcm_.size();
        utterance_pcm_.resize(old + avail);
        const std::size_t got = ring.read_samples(utterance_pcm_.data() + old, avail);
        utterance_pcm_.resize(old + got);
    }

    // First audio of an utterance: prefill the user header (with the CURRENT
    // forced-language instruction) so the streamed soft-tokens land inside a
    // template-valid user turn. Idempotent per utterance.
    void open_utterance_if_needed() {
        if (utterance_open_) return;
        utterance_open_ = true;
        int next = -1;
        for (const int id : make_turn_frame().user_hdr) {
            if (pos_ >= max_context_) break;
            if (engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                blackwell::EngineStatus::Success)
                break;
            ++pos_;
        }
    }

    // Recompute the utterance log-mel (whole-buffer WhisperDSP; incremental STFT
    // is a future optimization — CPU cost is ms-scale for seconds of audio) and
    // advance the sliding window by whole hops through prefill_audio_center_hop.
    // streamed_mel_end_ tracks the mel frames already covered; flush_tail also
    // commits a final short window at the commit boundary.
    void run_pending_center_hops(bool flush_tail) {
        if (utterance_pcm_.empty()) return;
        const auto& plan = engine_->get_impl()->m_runtime.audio_streaming;
        const int W   = plan.window_tokens * blackwell::audio::kMelFramesPerSoftToken;
        const int hop = plan.hop_tokens    * blackwell::audio::kMelFramesPerSoftToken;
        const int nmb = dsp_->config().n_mels;

        const whisper::LogMel mel = dsp_->process(utterance_pcm_);
        int total = std::min(mel.n_frames, kMaxUtteranceMelFrames);
        total -= total % blackwell::audio::kMelFramesPerSoftToken;   // whole soft-tokens

        const std::size_t cap = static_cast<std::size_t>(nmb) * W;
        if (d_window_stage_.count() < cap) d_window_stage_.allocate(cap);

        while (true) {
            int end;
            if (streamed_mel_end_ == 0)                     end = std::min(W, total);
            else if (streamed_mel_end_ + hop <= total)      end = streamed_mel_end_ + hop;
            else if (flush_tail && streamed_mel_end_ < total) end = total;
            else break;
            if (end <= streamed_mel_end_) break;

            const int start  = std::max(0, end - W);
            const int frames = end - start;
            // Host-side mel window slice (mel-major) -> contiguous [nmb, frames].
            mel_stage_.resize(static_cast<std::size_t>(nmb) * frames);
            for (int m = 0; m < nmb; ++m)
                std::copy(mel.data.begin() + static_cast<std::size_t>(m) * mel.n_frames + start,
                          mel.data.begin() + static_cast<std::size_t>(m) * mel.n_frames + end,
                          mel_stage_.begin() + static_cast<std::size_t>(m) * frames);
            CUDA_CHECK_THROW(cudaMemcpy(d_window_stage_.get(), mel_stage_.data(),
                                        mel_stage_.size() * sizeof(float),
                                        cudaMemcpyHostToDevice));
            prefill_audio_center_hop(d_window_stage_.get(), frames, /*mel_frame_offset=*/start);
            streamed_mel_end_ = end;
        }
    }

    // Per-utterance live state reset (turn finalized, or a fresh utterance began).
    void reset_utterance() {
        utterance_pcm_.clear();
        streamed_mel_end_ = 0;
        utterance_open_ = false;
    }

    // The CenterSlice commit: flush the mel tail, complete the phrase with the
    // right-edge commit, close the turn frame, decode. On a barge-in with the
    // pause checkpoint armed, the utterance CONTINUES: only the tentative state
    // above the checkpoint is discarded (pointer rollback) and the queued
    // do_rewind sees the open utterance and preserves the audio context.
    blackwell::EngineStatus commit_center_decode(const Command& cmd) {
        const auto t_vad = std::chrono::steady_clock::now();   // VAD trigger -> TTFT clock
        const int resident_kv = pos_;            // tokens ALREADY in the KV at the trigger
        drain_stream_pcm(cmd.stream->ring);
        if (!utterance_open_ && utterance_pcm_.empty()) {
            clear_in_flight();
            return finish(cmd, 0);
        }
        open_utterance_if_needed();
        run_pending_center_hops(/*flush_tail=*/true);
        commit_audio_right_edge();               // pause checkpoint + K edge rows

        const TurnFrame frame = make_turn_frame();
        int next = -1;
        auto prefill_ids = [&](const std::vector<int>& ids) -> bool {
            for (const int id : ids) {
                if (pos_ >= max_context_) return false;
                if (engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next) !=
                    blackwell::EngineStatus::Success)
                    return false;
                ++pos_;
            }
            return true;
        };
        if (!prefill_ids(frame.user_eot) || !prefill_ids(frame.gen_cue)) {
            finalize_turn(std::string(), /*completed=*/false);
            reset_utterance();
            clear_in_flight();
            return finish(cmd, 0);
        }
        // First decode token is sampled by the last gen-cue forward: TTFT closes.
        last_ttft_ms_.store(static_cast<float>(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_vad).count()),
                            std::memory_order_release);

        // What actually hit the GPU on the critical path: the resident streamed
        // tokens were accepted as-is; only the delta was prefilled. If the delta
        // ever balloons toward the full utterance, the fast path is broken.
        std::printf("[Prefill Start] Mode: Center-Slice | Resident KV: %d tokens | "
                    "Prefill Batch (Delta): %d tokens (tail hops + K edge + turn close)\n",
                    resident_kv, pos_ - resident_kv);
        std::fflush(stdout);

        // Seed the per-token clock at the VAD trigger: the FIRST [Decode] line
        // then reports the true wall-clock TTFT, not ~0.
        const TurnDecode d = decode_assistant_turn(cmd, next, t_vad,
                                                   temperature(), top_p());

        if (d.interrupted && pause_pending_) {
            // Speech returned before the turn finalized. WHICH kind of return it
            // is comes down to TIME, not to the fact that we were decoding: a
            // breath continues this utterance, a long gap means the user gave up
            // on whatever we were saying and started a new thought.
            float silence_s = 0.0f;
            if (is_continuation(&silence_s)) {
                // Micro-pause resume: pointer-rollback the tentative edge +
                // template + generated tokens and keep the utterance open — the
                // next warm hop overwrites the edge region with fresh,
                // context-complete center tokens.
                resume_after_pause();
                clear_in_flight();
                return finish(cmd, d.emitted);
            }
            // Fresh utterance: drop this one outright. finalize_turn's rollback to
            // history_base_pos_ resolves pause_pending_, so the do_rewind the
            // barge-in queued finds a clean slate and only has to flush the ring.
            std::printf("[Barge-in Route] FRESH UTTERANCE after %.2f s of silence — "
                        "discarding the interrupted turn (no resume)\n", silence_s);
            std::fflush(stdout);
        }
        finalize_turn(d.reply, d.completed());
        publish_turn(cmd, d);   // commit gate: dispatches iff reason == Eos
        reset_utterance();
        clear_in_flight();
        return finish(cmd, d.emitted);
    }

    // Drain the buffered utterance PCM from the stream ring (engine-thread consumer),
    // run WhisperDSP -> [128,3000] log-mel -> prefill_ultravox_turn (audio soft-tokens)
    // -> stream the assistant reply through the sink, checking cancelled(gen) before
    // every token for wait-free barge-in. Mirrors the proven --wav transcribe path.
    blackwell::EngineStatus commit_audio_decode(const Command& cmd) {
        const auto t_vad = std::chrono::steady_clock::now();   // VAD trigger -> TTFT clock
        blackwell::bridge::AudioRingBuffer& ring = cmd.stream->ring;
        const std::size_t avail = ring.available_samples();
        if (avail == 0) { clear_in_flight(); return finish(cmd, 0); }
        pcm_stage_.resize(avail);
        const std::size_t got = ring.read_samples(pcm_stage_.data(), avail);
        pcm_stage_.resize(got);

        // Announce the batch BEFORE the multi-second work so the log shows what
        // is about to hit the GPU (header + ALL audio soft-tokens + turn close).
        std::printf("[Prefill Start] Mode: Whole-Utterance | Resident KV: %d tokens | "
                    "Prefill Batch: FULL utterance (~%d audio soft-tokens + turn framing)\n",
                    pos_, audio_out_frames());
        std::fflush(stdout);

        const float* d_mel = stage_logmel(pcm_stage_);
        int next = -1;
        if (prefill_ultravox_turn(d_mel, &next) != blackwell::EngineStatus::Success) {
            finalize_turn(std::string(), /*completed=*/false);  // drop the partial prefill
            clear_in_flight();
            return finish(cmd, 0);
        }
        // The whole utterance (DSP + encoder + 188-token prefill) sat on this
        // path — recording it makes the panel readout a direct A/B vs CenterSlice.
        last_ttft_ms_.store(static_cast<float>(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_vad).count()),
                            std::memory_order_release);

        // Seed the per-token clock at the VAD trigger: the FIRST [Decode] line
        // then reports the true wall-clock TTFT, not ~0.
        const TurnDecode d = decode_assistant_turn(cmd, next, t_vad,
                                                   temperature(), top_p());
        finalize_turn(d.reply, d.completed());
        publish_turn(cmd, d);   // commit gate: dispatches iff reason == Eos
        clear_in_flight();
        return finish(cmd, d.emitted);
    }

    // PCM (any length) -> padded/trimmed 30 s -> WhisperDSP log-mel -> [n_mels,3000]
    // (mel-major, zero-pad time) -> device buffer d_mel_. Returns the device pointer.
    const float* stage_logmel(const std::vector<float>& pcm_in) {
        constexpr int kConvFrames = 3000;              // Whisper large-v3-turbo 30 s
        const int nmb = dsp_->config().n_mels;
        std::vector<float> pcm = pcm_in;
        pcm.resize(static_cast<std::size_t>(dsp_->config().sample_rate) * 30, 0.0f);
        const whisper::LogMel mel = dsp_->process(pcm);
        mel_stage_.assign(static_cast<std::size_t>(nmb) * kConvFrames, 0.0f);
        const int nf = std::min(mel.n_frames, kConvFrames);
        for (int m = 0; m < nmb; ++m)
            std::copy(mel.data.begin() + static_cast<std::size_t>(m) * mel.n_frames,
                      mel.data.begin() + static_cast<std::size_t>(m) * mel.n_frames + nf,
                      mel_stage_.begin() + static_cast<std::size_t>(m) * kConvFrames);
        if (d_mel_.count() != mel_stage_.size()) d_mel_.allocate(mel_stage_.size());
        CUDA_CHECK_THROW(cudaMemcpy(d_mel_.get(), mel_stage_.data(),
                                    mel_stage_.size() * sizeof(float), cudaMemcpyHostToDevice));
        return d_mel_.get();
    }

protected:
    // ---- state (protected, not private: the Tier-2 profiler/latency tests
    //      subclass this control and drive the SAME single-threaded flow the
    //      do_* overrides use — white-box tier, engine thread only) -----------
    blackwell::ITokenizer* tok_ = nullptr;   // non-owning
    int  max_context_ = 0;

    // THE ACTIVE SESSION'S LIVE STATE. pos_ / verified_prompt_tokens_ /
    // active_seq_ / the base's system-prefix floor / history_base_pos_ all
    // describe whichever session activate() last made current -- they are NOT
    // global. Everything below the session switch (the decode loop, the step_*
    // injection sweep, the rollback paths) reads them as before and needs no
    // notion of sessions at all, which is the whole point: one swap at the
    // entry points instead of a session argument threaded through 40 call sites.
    int  pos_ = 0;                            // logical decode position (engine thread only)
    uint32_t verified_prompt_tokens_ = 0;     // committed-prefix boundary (checkpoint/rollback)
    int  active_seq_ = 0;                     // engine seq_id of the active session

    // The two contexts (see the Session doc above). chat_ is active at
    // construction, so a control with isolation OFF behaves exactly as before.
    Session  chat_ {/*seq_id=*/0, 0, 0, 0, 0, /*ephemeral=*/false, "chat"};
    Session  audio_{/*seq_id=*/1, 0, 0, 0, 0, /*ephemeral=*/true,  "audio"};
    Session* active_ = &chat_;
    bool     isolated_ = false;               // enable_isolated_sessions() succeeded
    KVCheckpoint pause_cp_{};                 // pre-edge-commit snapshot (CenterSlice pause)
    bool pause_pending_ = false;              // an edge commit awaits finalize-or-resume
    std::atomic<uint32_t> last_rewind_keep_{0};
    std::atomic<std::size_t> turn_counter_{0};

    // ---- text-context policy (UI thread writes, engine thread reads at turn
    //      boundaries; see the ContextMode section) ----------------------------
    std::atomic<int> mode_{static_cast<int>(ContextMode::Stateless)};
    std::atomic<int> history_budget_tokens_{256};
    std::atomic<int> src_lang_{0};            // rt::kLanguages index (0 = Auto)
    std::atomic<int> tgt_lang_{0};            // rt::kLanguages index (0 = Auto)
    std::atomic<bool> task_transcribe_{true}; // emit [Speech]      (see set_tasks)
    std::atomic<bool> task_translate_{true};  // emit [Translation] (see set_tasks)

    // User-supplied audio task verb; empty = the generated phrase. Read at turn
    // boundaries (engine thread), written by the UI thread. See
    // set_audio_task_prompt for why this is a mutex and not an atomic.
    mutable std::mutex audio_task_mu_;
    std::string        audio_task_prompt_;
    // Forced spoken language, free text; empty = Auto (model-side LID). Shares
    // the mutex above: both are read together when the transcription prefix is
    // composed, so one lock is the natural granularity. See set_speech_language.
    std::string        speech_language_;
    // Restated at the END of every locally-answered user block; empty = append
    // nothing, which is what an app with no reply-format contract wants. Shares
    // the mutex for the same reason the two above do. See
    // set_reply_format_reminder for why a pinned system prefix is not enough.
    std::string        reply_format_reminder_;
    std::atomic<bool> live_center_{false};    // UI streaming toggle (utterance-latched)
    std::atomic<float> last_ttft_ms_{0.0f};   // VAD->first-token, panel readout (0 = none)

    // ---- live sampling (Settings -> Inference); see set_sampling ---------------
    // Greedy by default: this model transcribes, and sampling only costs fidelity.
    std::atomic<float> temperature_{0.0f};
    std::atomic<float> top_p_{1.0f};
    std::atomic<int>   max_new_tokens_{kDefaultMaxGeneratedTokens};

    // Heals sub-character token splits before anything leaves the decode loop.
    // Engine-thread-only, like every other decode-loop member (no atomic needed).
    blackwell::bridge::Utf8StreamAssembler utf8_;

    // Bounded-history state. The deque/counter are engine-thread-only; the base
    // position is atomic solely for the cross-thread invariant readout (the
    // engine thread is its only writer).
    std::deque<TurnRecord> turn_history_;
    std::size_t history_text_tokens_ = 0;     // sum of turn_history_ id counts
    // The user text of the turn being decoded, when it is KNOWN rather than
    // recovered from the reply (text turns: typed input, and the transcript fed
    // to the chat session). Empty on an audio turn, where encode_history_turn
    // must parse it back out. Engine thread only; cleared by finalize_turn.
    std::string pending_user_text_;
    std::atomic<int> history_base_pos_{0};    // system floor + retained turns

    // Audio frontend (owned; loaded lazily via load_audio_head). Both are driven
    // only from the engine-owning thread; the pipeline runs on its own stream.
    std::unique_ptr<blackwell::audio::AudioEmbeddingPipeline> audio_pipeline_;
    std::unique_ptr<blackwell::audio::PingPongAudioBuffer>    audio_pp_;
    long long audio_frame_ = 0;              // producer/consumer frame counter

    // Live mic log-mel (non-owning DSP + reused staging buffers, engine thread only).
    whisper::WhisperDSP*        dsp_ = nullptr;
    std::vector<float>          pcm_stage_;
    std::vector<float>          mel_stage_;
    blackwell::DeviceBuffer<float> d_mel_;
    blackwell::DeviceBuffer<float> d_window_stage_;   // contiguous sliding-window slice

    // CenterSlice live-utterance state (engine thread only): the utterance's
    // accumulated PCM, the mel frames already covered by hops, and whether the
    // user turn's header is prefilled (utterance open until finalize/rewind).
    std::vector<float> utterance_pcm_;
    int  streamed_mel_end_ = 0;
    bool utterance_open_ = false;

    // Barge-in routing clock (engine thread only): when the user last stopped
    // speaking, stamped at the VAD commit. See is_continuation().
    std::chrono::steady_clock::time_point last_speech_end_{};
    bool have_speech_end_ = false;

    // Rolling repetition-penalty window for the CURRENT turn (engine thread
    // only): the device ring the sampler reads, its host mirror, and the total
    // pushed this turn (the live length is min(count, kRepetitionWindow)).
    blackwell::DeviceBuffer<int> d_penalty_ids_;
    std::vector<int> penalty_ids_host_;
    int penalty_count_ = 0;
};

}  // namespace rt
