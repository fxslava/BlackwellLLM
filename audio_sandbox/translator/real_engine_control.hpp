#pragma once
// -----------------------------------------------------------------------------
// translator/real_engine_control.hpp — the PRODUCTION engine-assembly seam for
// audio_translator. Replaces MockEngineControl: it drives a REAL BlackwellEngine
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
//     per-turn ceiling (kMaxGeneratedTokens) that force-closes the turn cleanly.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
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
    uint32_t prefill_system_prompt(const std::string& system_prompt) {
        const std::vector<int> ids = tok_->encode_chat_prelude(system_prompt);
        int next = -1;
        for (const int id : ids) {
            if (engine_->forward_status(id, pos_, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                        cfg_.seq_id, &next) != blackwell::EngineStatus::Success)
                break;
            ++pos_;
        }
        const auto n = static_cast<uint32_t>(ids.size());
        set_system_prefix_tokens(n);   // KV rewind floor (barge-in never truncates it)
        history_base_pos_.store(pos_, std::memory_order_relaxed);  // no history yet
        return n;
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
        engine_->get_impl()->kv_mgr->rewind(cfg_.seq_id, safe_pos);

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
                                            cfg_.seq_id, &next) != blackwell::EngineStatus::Success)
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
        int next = -1;
        if (prefill_ultravox_turn(d_mel, &next) != blackwell::EngineStatus::Success)
            return std::string();
        std::string out;
        for (int i = 0; i < max_new_tokens && pos_ < max_context_; ++i) {
            if (tok_->is_stop(next)) break;
            out += tok_->decode(next, /*render_special=*/false);
            if (engine_->forward_status(next, pos_, /*temp=*/0.0f, /*top_p=*/1.0f,
                                        cfg_.seq_id, &next) != blackwell::EngineStatus::Success)
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
        const int W   = plan.window_tokens * blackwell::audio::kMelFramesPerSoftToken;
        const int hop = plan.hop_tokens    * blackwell::audio::kMelFramesPerSoftToken;

        int next = -1;
        auto prefill_ids = [&](const std::vector<int>& ids) -> bool {
            for (const int id : ids) {
                if (pos_ >= max_context_) return false;
                if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
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
            if (engine_->forward_status(next, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
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
    // Latin-transliteration failure mode on short audio hops. Any thread (reads
    // four atomics).
    //
    // ORDER IS LOAD-BEARING: language constraints, then the output-tag contract,
    // then the task phrase LAST — it ends in ": " and must abut the audio
    // soft-tokens that occupy the placeholder slot immediately after this text.
    std::string build_user_instruction() const {
        const int si = src_lang_.load(std::memory_order_acquire);
        const int ti = tgt_lang_.load(std::memory_order_acquire);
        bool transcribe = task_transcribe_.load(std::memory_order_acquire);
        const bool translate = task_translate_.load(std::memory_order_acquire);
        // Both tasks off would produce an instruction with no verb; fall back to
        // transcription — the one task that needs no target language. Matches
        // expected_output_format(), which resolves the same case to [Speech].
        if (!transcribe && !translate) transcribe = true;

        std::string s;
        if (si > 0) {
            s += "The audio language is ";
            s += rt::kLanguages[si];
            s += ". Write the transcript in ";
            s += rt::kLanguages[si];
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
        pause_pending_ = false;   // the turn is over: the edge commit is resolved
        const bool bounded = context_mode() == ContextMode::BoundedHistory;
        if (!bounded) {
            // Also covers a live Bounded -> Stateless switch: drop the retained
            // text history and re-anchor the base to the frozen system prefix.
            turn_history_.clear();
            history_text_tokens_ = 0;
            history_base_pos_.store(static_cast<int>(system_prefix_tokens()),
                                    std::memory_order_relaxed);
        }
        if (!bounded || !completed || reply.empty()) {
            kv_cache_rollback({history_base_pos_.load(std::memory_order_relaxed),
                               verified_prompt_tokens_},
                              !bounded ? "end-of-turn stateless flush (by design: audio + "
                                         "generated text leave no trace)"
                                       : "incomplete/empty turn discarded");
            if (audio_pipeline_) audio_pipeline_->reset_history();
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
        engine_->get_impl()->kv_mgr->rewind(cfg_.seq_id, static_cast<int>(keep));
        // The KV is rewound to (at most) the frozen system prefix, so no audio soft-
        // tokens survive: drop the overlap-reconciliation / center-slice history too
        // (reset_history covers both), any outstanding pause-edge commit, and the
        // live utterance accumulators (a fresh utterance starts here).
        if (audio_pipeline_) audio_pipeline_->reset_history();
        pause_pending_ = false;
        reset_utterance();

        // HARD AUDIO FLUSH: a fresh utterance starts from THIS moment's audio.
        // Everything still buffered in the ring is pre-speech-start material —
        // in auto mode the background noise accumulated since the last commit
        // (unbounded between utterances), in manual mode anything that leaked in
        // before the press — and encoding it makes the LLM hallucinate over
        // noise. Discarded HERE because this runs on the engine thread, the only
        // consumer allowed to move the ring's read cursor (SPSC contract). The
        // few ms of samples pushed between the trigger and this command
        // executing are dropped with the noise (the engine is idle at speech
        // start, so that window is command-latency-small). The mid-utterance
        // resume branch above deliberately does NOT flush: there the ring holds
        // the freshly resumed speech.
        if (cmd.stream != nullptr) {
            blackwell::bridge::AudioRingBuffer& ring = cmd.stream->ring;
            const std::size_t stale = ring.available_samples();
            if (stale != 0) {
                pcm_stage_.resize(stale);
                const std::size_t got = ring.read_samples(pcm_stage_.data(), stale);
                if (std::getenv("BLACKWELL_AV_DEBUG")) {
                    std::printf("[flush] speech start: discarded %zu stale PCM samples"
                                " (%.0f ms of pre-press audio)\n",
                                got, static_cast<double>(got) / 16.0);
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

        // Frame a user turn + assistant cue through the checkpoint's chat template,
        // then prefill it. Cycle a few prompts so the live GUI shows varied real
        // model output rather than one repeated answer.
        static constexpr const char* kPrompts[] = {
            "Briefly greet the user in Russian and English.",
            "Say one short encouraging sentence in Russian.",
            "Translate 'good morning' into Russian and use it in a short greeting.",
        };
        const std::size_t idx =
            turn_counter_.fetch_add(1, std::memory_order_relaxed) % 3;

        std::vector<int> turn = tok_->encode_chat_message(
            blackwell::ChatMessage{"user", kPrompts[idx]});
        const std::vector<int> gen = tok_->encode_generation_prompt();
        turn.insert(turn.end(), gen.begin(), gen.end());

        int next = -1;
        const auto t_turn0 = std::chrono::steady_clock::now();  // placeholder TTFT clock
        for (const int id : turn) {
            if (pos_ >= max_context_ ||
                engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
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
                                                   /*temperature=*/0.7f, /*top_p=*/0.9f);
        finalize_turn(d.reply, d.completed());
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
            core->kv_mgr->prepare_decode_step(cfg_.seq_id, pos_);

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
            core->kv_mgr->prepare_decode_step(cfg_.seq_id, pos_);
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
    //   * kMaxGeneratedTokens is the hard ceiling for when it does not. Hitting
    //     it force-closes the turn CLEANLY (counts as completed -> finalize_turn
    //     emits the EOT / stateless flush), so the pipeline returns to IDLE and
    //     the next utterance is a normal fresh start, not a wedged state.
    // The penalty window is per TURN and rolling: a penalty that leaked across
    // utterances would suppress legitimately repeated words forever, and an
    // unbounded window would suppress common words within one long reply.
    static constexpr float kRepetitionPenalty  = 1.15f;
    static constexpr int   kMaxGeneratedTokens = 256;
    static constexpr int   kRepetitionWindow   = 64;

    // What a decode loop produced. `completed` deliberately treats a max-token or
    // context-cap stop as CLEAN: only a barge-in or an engine fault leaves the
    // turn unusable.
    struct TurnDecode {
        std::string reply;
        int  emitted = 0;
        bool interrupted = false;   // barge-in superseded this generation
        bool faulted = false;       // run_token / sampling reported != Success
        bool completed() const noexcept { return !interrupted && !faulted; }
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
        auto t_tok = t_start;
        while (pos_ < max_context_) {
            if (cancelled(cmd.gen)) {                              // wait-free barge-in
                out.interrupted = true;
                stop_reason = "barge-in detected (gen superseded)";
                break;
            }
            {   // telemetry: EVERY produced token, stop/special tokens included
                const auto now = std::chrono::steady_clock::now();
                log_decode_token(next,
                                 std::chrono::duration<double, std::milli>(now - t_tok).count());
                t_tok = now;
            }
            if (tok_->is_stop(next)) { stop_reason = "EOT token reached"; break; }
            // Checked AFTER the stop test so a natural EOT landing exactly on the
            // cap is still reported as an EOT, and BEFORE emitting so the turn
            // ends at exactly kMaxGeneratedTokens streamed tokens.
            if (out.emitted >= kMaxGeneratedTokens) {
                stop_reason = "Max turn tokens reached";
                break;
            }
            const std::string piece = tok_->decode(next, /*render_special=*/false);
            if (!piece.empty()) {
                cmd.sink.emit(piece.c_str(), out.emitted, /*is_final=*/0, BRIDGE_OK);
                out.reply += piece;
            }
            ++out.emitted;
            push_repetition_token(next);   // this turn's rolling penalty window
            if (decode_step(next, pos_, temperature, top_p, &next) !=
                blackwell::EngineStatus::Success) {
                out.faulted = true;
                stop_reason = "engine fault (run_token/sample != Success)";
                break;
            }
            ++pos_;
        }
        log_decode_stop(stop_reason, pos_, out.emitted);
        return out;
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
                core->run_token(token_id, pos, cfg_.seq_id);
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
            if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
                blackwell::EngineStatus::Success)
                break;
            ++pos_;
        }
    }

    // Recover the user-side text from the model's reply. Tolerates EVERY task
    // mode's tag set (see expected_output_format): "[Speech] X | [Translation] Y",
    // a bare "[Speech] X", or a translation-only reply with no transcript to
    // recover. Anything that does not match degrades to an "(audio)" placeholder
    // user turn rather than poisoning the retained history with the model's
    // formatting noise.
    std::vector<int> encode_history_turn(const std::string& reply) const {
        auto trim = [](const std::string& s) {
            const auto b = s.find_first_not_of(" \t\r\n");
            const auto e = s.find_last_not_of(" \t\r\n");
            return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
        };
        std::string user_text = "(audio)";
        constexpr const char* kSpeech = "[Speech]";
        const auto sp = reply.find(kSpeech);
        if (sp != std::string::npos) {
            const auto beg = sp + std::char_traits<char>::length(kSpeech);
            // The transcript runs to the translation separator, or to the end of
            // the reply in transcribe-only mode.
            const auto cut = reply.find(" | [Translation]", beg);
            const std::string t =
                trim(cut == std::string::npos ? reply.substr(beg)
                                              : reply.substr(beg, cut - beg));
            if (!t.empty()) user_text = t;
        }
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
                if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
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
            if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
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
                if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
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
                                                   /*temperature=*/0.0f, /*top_p=*/1.0f);

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
                                                   /*temperature=*/0.0f, /*top_p=*/1.0f);
        finalize_turn(d.reply, d.completed());
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
    int  pos_ = 0;                            // logical decode position (engine thread only)
    uint32_t verified_prompt_tokens_ = 0;     // committed-prefix boundary (checkpoint/rollback)
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
    std::atomic<bool> live_center_{false};    // UI streaming toggle (utterance-latched)
    std::atomic<float> last_ttft_ms_{0.0f};   // VAD->first-token, panel readout (0 = none)

    // Bounded-history state. The deque/counter are engine-thread-only; the base
    // position is atomic solely for the cross-thread invariant readout (the
    // engine thread is its only writer).
    std::deque<TurnRecord> turn_history_;
    std::size_t history_text_tokens_ = 0;     // sum of turn_history_ id counts
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
