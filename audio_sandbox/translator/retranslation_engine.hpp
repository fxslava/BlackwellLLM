#pragma once
// -----------------------------------------------------------------------------
// translator/retranslation_engine.hpp — the LIVE IRetranslationEngine
// (docs/CONTINUOUS_STREAMING.md §6, T5). RetranslationSession owns the draft-and-
// commit MECHANIC; this class is the half that actually touches CUDA.
//
// WHY THIS IS NOT RealEngineControl. That class implements the SUPERSEDED
// strategies (whole-utterance and CenterSlice) on top of EngineControlBridge, and
// carries the state they need: a barge-in epoch, an SPSC command ring, a pause
// checkpoint, an injected-soft-token history for overlap reconciliation. Re-
// translation needs NONE of it — every redraft rewinds to C and re-encodes the
// utterance from its first sample, so there is no incremental history to
// reconcile, no partial window to resume and nothing to barge in on. Subclassing
// it would inherit four kinds of state this loop must not have. What IS reused is
// every proven primitive: the forward_status prefill loop, the white-box step_*
// embedding sweep, kv_mgr->rewind, and the penalized decode step.
//
// THE FIVE OPERATIONS, and what each one costs:
//
//   rewind_to(C)          O(1). VRAMArena::truncate_kv is bookkeeping — the FP32
//                         K/V slabs are position-addressed and simply overwritten
//                         by the next prefill. This is why redrafting on every
//                         Partial is affordable at all (§3).
//   prefill_turn_prefix   ~20 text tokens through forward_status.
//   prefill_audio(b,e)    THE expensive one: ring read -> log-mel -> bucketed
//                         Whisper encode -> project -> step_* sweep. Always from
//                         the utterance's FIRST sample (the growing-window
//                         invariant), never a splice of deltas.
//   prefill_turn_suffix   ~5 text tokens; its last forward_status samples the
//                         first token the decode will emit.
//   decode_draft          greedy, penalized, capped.
//   evict_head            one launch_kv_evict_head per layer + a pointer fix.
//
// POSITION BOOKKEEPING. `pos_` is the single source of truth for the KV write
// cursor and it is only ever moved by the five methods above. The KvLedger is a
// PARALLEL map maintained by RetranslationSession; the two are reconciled by
// construction because every method returns exactly the number of rows it
// appended. An adapter that returned a count it did not append would desync the
// ledger from the cache silently, which is the one failure mode that makes every
// later eviction plan a fiction — so each method counts what it DID, not what it
// intended.
//
// THREADING: engine-thread only, all of it (CLAUDE.md single-threaded doctrine).
// The audio thread reaches this class solely through AbsoluteAudioRing::write on
// the far side of the ring's mutex.
//
// ERROR TIER: RUNTIME. Nothing here throws — the CUDA_CHECK_THROW sites inside
// the reused primitives are wrapped, and a failure is reported as a short count
// or a false return, which RetranslationSession unwinds atomically. INIT-tier
// setup (load_audio_head, the engine ctor) throws as usual and runs before any
// session exists.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "whisper_dsp.h"                // whisper::WhisperDSP, whisper::LogMel

#include "blackwell/engine.h"           // BlackwellEngine, blackwell::EngineStatus
#include "blackwell/tokenizer.h"        // blackwell::ITokenizer

#include "common.h"                     // CUDA_CHECK_THROW
#include "device_buffer.h"              // blackwell::DeviceBuffer
#include "engine_impl.h"                // BlackwellEngine::Impl (white-box step_*)
#include "rope_config.h"                // rope_scaling_from(ModelConfig)
#include "kernels/kv_evict.cuh"         // launch_kv_evict_head
#include "kernels/sampling.cuh"         // sample_top_p, launch_repetition_penalty_kernel
#include "audio_embedding_pipeline.h"   // blackwell::audio::AudioEmbeddingPipeline
#include "sliding_audio_window.h"       // blackwell::audio::kMelFramesPerSoftToken

#include "absolute_audio_ring.hpp"      // blackwell::bridge::AbsoluteAudioRing
#include "retranslation_session.hpp"    // blackwell::bridge::IRetranslationEngine
#include "audio_head_loader.hpp"        // rt::load_audio_pipeline
#include "language_table.hpp"           // rt::kLanguages

namespace rt {

// Per-turn instruction knobs. These live in the USER turn, re-prefilled on every
// redraft, so a change takes effect on the very next Partial without touching the
// frozen system prefix (which is the KV floor and cannot be re-prefilled).
struct RetranslationPrompt {
    int  source_language_index = 0;   // rt::kLanguages index; 0 == Auto
    int  target_language_index = 0;
    bool transcribe = false;          // drafts are noisy enough without both tags
    bool translate  = true;
};

class RetranslationEngine final : public blackwell::bridge::IRetranslationEngine {
public:
    // engine / tok / dsp / ring are non-owning and must outlive this object. The
    // audio head is owned (load_audio_head), because its lifetime is exactly this
    // object's. max_context caps the KV; it is what context_capacity() reports and
    // what every prefill loop guards against.
    RetranslationEngine(BlackwellEngine* engine, blackwell::ITokenizer* tok,
                        whisper::WhisperDSP* dsp,
                        blackwell::bridge::AbsoluteAudioRing* ring,
                        int max_context, std::uint32_t seq_id = 0) noexcept
        : engine_(engine), tok_(tok), dsp_(dsp), ring_(ring),
          max_context_(max_context), seq_id_(seq_id) {}

    // ---- INIT tier (throws; runs before the session exists) ------------------

    // Whisper encoder + Ultravox projector from <audio_head>/model.safetensors.
    // The projector's output width is taken from the ENGINE's hidden_size, so the
    // soft tokens always match the backbone they splice into.
    void load_audio_head(const std::string& audio_head) {
        const int hidden = static_cast<int>(engine_->get_impl()->m_config.hidden_dim);
        audio_ = rt::load_audio_pipeline(audio_head, hidden);
    }
    bool audio_head_loaded() const noexcept { return audio_ != nullptr; }

    // Prefill the frozen system prompt and freeze its length as S — the rewind
    // floor no redraft and no eviction may ever cross. Returns the token count,
    // which is what the KvLedger is constructed with.
    //
    // INIT tier: THROWS on a short prefill. A partial system prompt is not a
    // degraded session, it is a corrupt one — S would name a floor in the middle
    // of the chat template, every later rewind would restore a truncated prompt,
    // and the ledger would be built on a lie. This must fail loudly at setup
    // rather than silently mis-anchor the whole run.
    std::uint32_t prefill_system_prompt(const std::string& system_prompt) {
        const std::vector<int> ids = tok_->encode_chat_prelude(system_prompt);
        int next = -1;
        std::size_t done = 0;
        for (const int id : ids) {
            if (pos_ >= max_context_) break;
            if (engine_->forward_status(id, pos_, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                        seq_id_, &next) != blackwell::EngineStatus::Success)
                break;
            ++pos_;
            ++done;
        }
        if (done != ids.size())
            throw std::runtime_error(
                "RetranslationEngine: system-prompt prefill stopped after " +
                std::to_string(done) + " of " + std::to_string(ids.size()) +
                " tokens — the engine faulted during setup, so the frozen prefix "
                "cannot be anchored.");
        frozen_prefix_ = static_cast<std::uint32_t>(pos_);
        return frozen_prefix_;
    }

    void set_prompt(const RetranslationPrompt& p) noexcept { prompt_ = p; }
    const RetranslationPrompt& prompt() const noexcept { return prompt_; }

    // ---- IRetranslationEngine (RUNTIME tier; engine thread) ------------------

    // O(1) truncation. Idempotent: rewinding to where we already are is a no-op
    // that still leaves the cursor at C, which is what lets the session call this
    // unconditionally at the top of every redraft.
    bool rewind_to(std::uint32_t target) override {
        const int p = static_cast<int>(target);
        if (p < 0 || p > pos_) {
            // Above the cursor is not a rewind — refuse rather than fabricate KV.
            return p == pos_;
        }
        if (p < static_cast<int>(frozen_prefix_)) return false;   // the floor holds
        engine_->get_impl()->kv_mgr->rewind(seq_id_, p);
        pos_ = p;
        return true;
    }

    std::uint32_t prefill_turn_prefix() override {
        const std::vector<int> ids =
            tok_->encode(std::string(kUserHeader) + build_user_instruction(),
                         /*add_special=*/false);
        return prefill_ids(ids);
    }

    // THE growing window. [begin, end) is the whole utterance every time, so this
    // re-encodes from the first sample on every Partial — that is the invariant
    // that removed the encoder worker (docs §4), not an inefficiency to optimize
    // away. Returns 0 (not a failure) when the window is too short to make one
    // soft token, or when the ring no longer holds it.
    std::uint32_t prefill_audio(std::uint64_t begin_sample, std::uint64_t end_sample) override {
        if (audio_ == nullptr || dsp_ == nullptr || ring_ == nullptr) return 0;
        if (!ring_->read(begin_sample, end_sample, &pcm_stage_)) {
            // The ring has forgotten the head of this utterance. Translating what
            // is left would confidently mistranslate a sentence missing its first
            // words, so refuse and let the session unwind (AbsoluteAudioRing §2).
            std::printf("[audio] window [%llu,%llu) no longer resident (oldest=%llu) — skipped\n",
                        static_cast<unsigned long long>(begin_sample),
                        static_cast<unsigned long long>(end_sample),
                        static_cast<unsigned long long>(ring_->available_from()));
            std::fflush(stdout);
            return 0;
        }

        const int frames = stage_logmel(pcm_stage_);
        if (frames < blackwell::audio::kMelFramesPerSoftToken) return 0;

        try {
            const auto win = audio_->encode_project_window(d_mel_.get(), frames,
                                                           /*mel_frame_offset=*/0);
            // The encode+project ran on the audio stream; the step_* sweep below
            // runs on stream 0, so the producer edge must be closed first.
            CUDA_CHECK_THROW(cudaStreamSynchronize(audio_->audio_stream()));
            if (win.num_tokens <= 0) return 0;
            return inject_embedding_rows(win.embeds, win.num_tokens);
        } catch (...) {
            return 0;   // RUNTIME tier: a short count, unwound by the session
        }
    }

    std::uint32_t prefill_turn_suffix() override {
        std::uint32_t n = prefill_ids(tok_->encode("<|eot_id|>", /*add_special=*/false));
        n += prefill_ids(tok_->encode_generation_prompt());
        return n;
    }

    // Greedy decode with the two guards RealEngineControl learned the hard way: a
    // rolling repetition penalty (which breaks an exact phrase loop
    // mathematically) and the caller's hard cap (for when it does not). Both
    // matter more here than in the whole-utterance app, because a runaway draft is
    // paid again on every Partial of the same utterance.
    std::uint32_t decode_draft(std::uint32_t max_new_tokens, std::string* out) override {
        int next = next_token_;
        std::uint32_t emitted = 0;
        reset_repetition_window();
        while (emitted < max_new_tokens && pos_ < max_context_) {
            if (next < 0 || tok_->is_stop(next)) break;
            const std::string piece = tok_->decode(next, /*render_special=*/false);
            if (out != nullptr) *out += piece;
            ++emitted;
            push_repetition_token(next);
            if (decode_step(next, pos_, &next) != blackwell::EngineStatus::Success) break;
            ++pos_;
        }
        return emitted;
    }

    // Drop [keep_from, keep_from + delta) from EVERY layer and re-phase the
    // survivors, then move the cursor down by the same delta. The kernel is fed
    // the engine's OWN theta and scaling — passing anything else would measure the
    // composed angle against a different frequency ladder than the append kernel
    // used, which is precisely the mistake T3's test is positioned to catch.
    bool evict_head(std::uint32_t keep_from, std::uint32_t delta,
                    std::uint32_t cache_len) override {
        if (delta == 0 || cache_len > static_cast<std::uint32_t>(pos_)) return false;
        if (keep_from + delta > cache_len) return false;
        if (keep_from < frozen_prefix_) return false;   // never cut into the sink

        auto* core = engine_->get_impl();
        const auto& c = core->m_config;
        for (int l = 0; l < static_cast<int>(c.num_layers); ++l) {
            launch_kv_evict_head(static_cast<float*>(core->kv_mgr->get_layer_k_ptr(l)),
                                 static_cast<float*>(core->kv_mgr->get_layer_v_ptr(l)),
                                 static_cast<int>(keep_from), static_cast<int>(delta),
                                 static_cast<int>(cache_len), c.num_key_value_heads,
                                 c.head_dim, c.rotary_dim,
                                 c.rope_pairing == RopePairing::Interleaved,
                                 core->arena.get_max_seq_len(), c.rope_theta,
                                 rope_scaling_from(c));
        }
        if (cudaGetLastError() != cudaSuccess) return false;
        if (cudaDeviceSynchronize() != cudaSuccess) return false;

        // The live region shrank: move the cursor and reconcile the arena's high
        // water marks through the same bookkeeping rewind a truncation uses.
        pos_ = static_cast<int>(cache_len - delta);
        core->kv_mgr->rewind(seq_id_, pos_);
        return true;
    }

    std::uint32_t context_capacity() const override {
        return static_cast<std::uint32_t>(max_context_ < 0 ? 0 : max_context_);
    }

    // ---- observables ---------------------------------------------------------
    int           position() const noexcept { return pos_; }
    std::uint32_t frozen_prefix() const noexcept { return frozen_prefix_; }
    int           soft_tokens_last_window() const noexcept { return last_soft_tokens_; }
    int           mel_frames_last_window() const noexcept { return last_mel_frames_; }

private:
    static constexpr const char* kUserHeader =
        "<|start_header_id|>user<|end_header_id|>\n\n";
    // Mirrors RealEngineControl's guards; see decode_draft.
    static constexpr float kRepetitionPenalty = 1.15f;
    static constexpr int   kRepetitionWindow  = 64;
    // The bucketed encoder's largest graph (30 s). A longer utterance cannot be
    // encoded in one window, and max_utterance_ms is clamped well below it.
    static constexpr int   kMaxMelFrames = 3000;

    // ORDER IS LOAD-BEARING: language constraints, then the output-tag contract,
    // then the task phrase LAST — it ends in ": " and must abut the audio soft
    // tokens that occupy the placeholder slot immediately after this text.
    std::string build_user_instruction() const {
        const int si = prompt_.source_language_index;
        const int ti = prompt_.target_language_index;
        bool transcribe = prompt_.transcribe;
        const bool translate = prompt_.translate;
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
        if (transcribe && translate)  s += "[Speech] <transcript> | [Translation] <translation>";
        else if (translate)           s += "[Translation] <translation>";
        else                          s += "[Speech] <transcript>";
        s += ". ";

        std::string into;
        if (ti > 0) {
            into = " into ";
            into += rt::kLanguages[ti];
        }
        if (transcribe && translate) s += "Translate" + into + ". Transcribe the following audio: ";
        else if (translate)          s += "Translate the following audio" + into + ": ";
        else                         s += "Transcribe the following audio: ";
        return s;
    }

    // Text prefill. Returns how many tokens actually landed — a short count on a
    // fault is what makes the session's atomic unwind correct.
    std::uint32_t prefill_ids(const std::vector<int>& ids) {
        std::uint32_t n = 0;
        for (const int id : ids) {
            if (pos_ >= max_context_) break;
            if (engine_->forward_status(id, pos_, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                        seq_id_, &next_token_) != blackwell::EngineStatus::Success)
                break;
            ++pos_;
            ++n;
        }
        return n;
    }

    // Utterance PCM -> log-mel, packed TIGHTLY as [n_mels, frames] (mel-major)
    // because that is what the bucketed encoder reads. Trimmed to a whole number
    // of soft tokens so the projector's stack-by-8 never sees a ragged tail.
    // Returns the frame count staged into d_mel_.
    int stage_logmel(const std::vector<float>& pcm) {
        const int nmb = dsp_->config().n_mels;
        const whisper::LogMel mel = dsp_->process(pcm);
        int frames = std::min(mel.n_frames, kMaxMelFrames);
        frames -= frames % blackwell::audio::kMelFramesPerSoftToken;
        if (frames <= 0) { last_mel_frames_ = 0; return 0; }

        mel_stage_.resize(static_cast<std::size_t>(nmb) * frames);
        for (int m = 0; m < nmb; ++m)
            std::copy(mel.data.begin() + static_cast<std::size_t>(m) * mel.n_frames,
                      mel.data.begin() + static_cast<std::size_t>(m) * mel.n_frames + frames,
                      mel_stage_.begin() + static_cast<std::size_t>(m) * frames);

        const std::size_t cap = static_cast<std::size_t>(nmb) * kMaxMelFrames;
        if (d_mel_.count() < cap) d_mel_.allocate(cap);
        if (cudaMemcpy(d_mel_.get(), mel_stage_.data(), mel_stage_.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            last_mel_frames_ = 0;
            return 0;
        }
        last_mel_frames_ = frames;
        return frames;
    }

    // The white-box step_* sweep: one projector row becomes one KV column. This is
    // the ONLY way embeddings enter the cache; the engine's public API and its
    // decode loop stay untouched (src/audio/README.md hard rule #1).
    std::uint32_t inject_embedding_rows(const float* d_embeds, int count) {
        auto* core = engine_->get_impl();
        const int hidden = static_cast<int>(core->m_config.hidden_dim);
        const int num_layers = static_cast<int>(core->m_config.num_layers);
        std::uint32_t injected = 0;
        using S = blackwell::EngineStatus;
        for (int a = 0; a < count && pos_ < max_context_; ++a) {
            if (cudaMemcpyAsync(core->d_X_accum,
                                d_embeds + static_cast<std::size_t>(a) * hidden,
                                static_cast<std::size_t>(hidden) * sizeof(float),
                                cudaMemcpyDeviceToDevice, /*stream 0*/ nullptr) != cudaSuccess)
                break;
            core->kv_mgr->prepare_decode_step(seq_id_, pos_);
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
            if (!ok) break;
            ++pos_;
            ++injected;
        }
        last_soft_tokens_ = static_cast<int>(injected);
        // The audio rows carry no sampled logits of their own; the next text token
        // the decode starts from is sampled by prefill_turn_suffix's last forward.
        return injected;
    }

    // One PENALIZED decode step: run_token, shape the logits with this draft's
    // repetition window, then sample. Mirrors forward_status with the extra stage
    // between two calls the white-box tier already owns. noexcept by contract —
    // the decode hot loop never unwinds (hybrid error doctrine).
    blackwell::EngineStatus decode_step(int token_id, int pos, int* out_token) noexcept {
        auto* core = engine_->get_impl();
        try {
            const blackwell::EngineStatus st = core->run_token(token_id, pos, seq_id_);
            if (st != blackwell::EngineStatus::Success) return st;
            launch_repetition_penalty_kernel(core->d_logits, core->m_config.vocab_size,
                                             d_penalty_ids_,
                                             std::min(penalty_count_, kRepetitionWindow),
                                             kRepetitionPenalty);
            *out_token = sample_top_p(core->d_logits, core->m_config.vocab_size,
                                      /*temperature=*/0.0f, /*top_p=*/1.0f);
            return blackwell::EngineStatus::Success;
        } catch (...) {
            return blackwell::EngineStatus::CudaRuntimeError;
        }
    }

    // Per-DRAFT window, not per-utterance: a penalty leaking across redrafts would
    // suppress the very words the previous draft got right. Must not throw (see
    // RealEngineControl::reset_repetition_window for why).
    void reset_repetition_window() noexcept {
        penalty_count_ = 0;
        try {
            if (d_penalty_ids_.count() != static_cast<std::size_t>(kRepetitionWindow))
                d_penalty_ids_.allocate(static_cast<std::size_t>(kRepetitionWindow));
            penalty_ids_host_.assign(static_cast<std::size_t>(kRepetitionWindow), -1);
        } catch (...) {
            // allocate() reset the buffer before failing, so count() == 0 holds and
            // the penalty is simply off; the hard cap still holds the line.
        }
    }

    void push_repetition_token(int token_id) noexcept {
        if (d_penalty_ids_.count() == 0) return;
        const std::size_t slot = static_cast<std::size_t>(penalty_count_ % kRepetitionWindow);
        penalty_ids_host_[slot] = token_id;
        if (cudaMemcpy(d_penalty_ids_.get() + slot, &penalty_ids_host_[slot], sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return;
        ++penalty_count_;
    }

    // ---- state (engine thread only) ------------------------------------------
    BlackwellEngine*                      engine_ = nullptr;   // non-owning
    blackwell::ITokenizer*                tok_    = nullptr;   // non-owning
    whisper::WhisperDSP*                  dsp_    = nullptr;   // non-owning
    blackwell::bridge::AbsoluteAudioRing* ring_   = nullptr;   // non-owning

    int           max_context_   = 0;
    std::uint32_t seq_id_        = 0;
    int           pos_           = 0;   // THE KV write cursor
    std::uint32_t frozen_prefix_ = 0;   // S: the rewind floor
    int           next_token_    = -1;  // sampled by the last prefill forward
    RetranslationPrompt prompt_{};

    // Audio frontend (owned; its lifetime is exactly this object's).
    std::unique_ptr<blackwell::audio::AudioEmbeddingPipeline> audio_;

    std::vector<float>             pcm_stage_;   // ring read target
    std::vector<float>             mel_stage_;   // tightly packed [n_mels, frames]
    blackwell::DeviceBuffer<float> d_mel_;
    int last_mel_frames_  = 0;
    int last_soft_tokens_ = 0;

    blackwell::DeviceBuffer<int> d_penalty_ids_;
    std::vector<int>             penalty_ids_host_;
    int                          penalty_count_ = 0;
};

}  // namespace rt
