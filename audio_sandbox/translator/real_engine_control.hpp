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
// THE AUDIO PATH (encoder + double-buffered pipeline now real; feed is the seam)
//   The full multimodal path is  live PCM -> Whisper log-mel (whisper_dsp) ->
//   Whisper ENCODER (audio_tower) -> Ultravox projector -> prefill-from-embeddings
//   -> decode. The encoder (src/audio/whisper_encoder.*) and the async
//   double-buffered pipeline (AudioEmbeddingPipeline + PingPongAudioBuffer) are
//   now implemented and unit-tested (tests/integration/test_audio_pingpong_
//   pipeline). prefill_audio_embeddings() below is the CONSUMER edge: it gates the
//   engine's stream on the producer's ready event (cudaStreamWaitEvent, no host
//   sync) and prefills the staged [num_audio, hidden] embeddings into the KV via
//   the white-box step_* sweep. The engine's public API + decode loop stay
//   untouched (hard rule). The ONE remaining seam is the live wiring in main.cpp:
//   the DSP log-mel frames are not yet fed into the pipeline, so do_commit_decode
//   still runs a REAL text decode from the frozen system prefix (genuine model
//   output). Everything except that feed -- encode, project, double-buffer
//   handoff, embedding prefill, decode, streaming, barge-in -- is real.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "whisper_dsp.h"                // whisper::WhisperDSP (live mic log-mel)

#include "blackwell/engine.h"           // BlackwellEngine, blackwell::EngineStatus
#include "blackwell/tokenizer.h"        // blackwell::ITokenizer, ChatMessage

#include "common.h"                     // CUDA_CHECK_THROW
#include "engine_impl.h"                // BlackwellEngine::Impl (white-box step_* + d_X_accum)
#include "ping_pong_audio_buffer.h"     // blackwell::audio::PingPongAudioBuffer
#include "audio_embedding_pipeline.h"   // blackwell::audio::AudioEmbeddingPipeline
#include "audio_head_loader.hpp"        // rt::load_audio_pipeline

#include "engine_control_bridge.hpp"    // EngineControlBridge, Command
#include "bridge/engine_api.h"          // BRIDGE_OK for the token sink

namespace rt {

class RealEngineControl : public blackwell::bridge::EngineControlBridge {
public:
    using Config = blackwell::bridge::EngineControlBridge::Config;

    // engine + tok are non-owning; both outlive this object (single-thread
    // doctrine). max_context caps the persistent KV growth across utterances.
    RealEngineControl(BlackwellEngine* engine, blackwell::ITokenizer* tok,
                      int max_context, const Config& cfg = {})
        : blackwell::bridge::EngineControlBridge(engine, cfg),
          tok_(tok), max_context_(max_context) {}

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
    void kv_cache_rollback(const KVCheckpoint& cp) {
        const int floor    = static_cast<int>(effective_keep_tokens(0));
        const int safe_pos = std::max(cp.pos, floor);
        const bool dbg     = std::getenv("BLACKWELL_AV_DEBUG") != nullptr;
        const auto t0      = std::chrono::steady_clock::now();

        pos_ = safe_pos;
        verified_prompt_tokens_ = cp.verified_tokens;
        // Continuous KV rewind() delegates to VRAMArena::truncate_kv; safe below the
        // engine's public API (single-thread doctrine) and a no-op for a fully
        // resident model (nothing offloaded), so regular decode is unaffected.
        engine_->get_impl()->kv_mgr->rewind(cfg_.seq_id, safe_pos);

        if (dbg) {
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("[kv-ckpt] rollback pos %d -> %d (floor=%d) verified_tokens=%u  %.1f us\n",
                        cp.pos, safe_pos, floor, cp.verified_tokens, us);
            std::fflush(stdout);
        }
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
        // Strict Llama-3 template with an explicit instruction; the 188 audio
        // soft-tokens occupy the <|audio|> placeholder slot (injected as embeddings,
        // not as a literal token). Special-token strings are recognized by encode()
        // (add_special=false: the BOS already lives in the frozen system prefix).
        const std::string user_prefix =
            "<|start_header_id|>user<|end_header_id|>\n\nTranscribe the following audio: ";
        const std::string user_suffix = "<|eot_id|>";
        const std::vector<int> user_hdr = tok_->encode(user_prefix, /*add_special=*/false);
        const std::vector<int> user_eot = tok_->encode(user_suffix, false);
        const std::vector<int> gen_cue  = tok_->encode_generation_prompt();

        std::printf("[prompt] <sys:%d tok>%s<|audio x%d|>%s%s\n", pos_,
                    user_prefix.c_str(), audio_pipeline_->out_frames(),
                    user_suffix.c_str(), tok_->decode(gen_cue, /*render_special=*/true).c_str());
        std::fflush(stdout);

        if (!prefill_ids(user_hdr)) return blackwell::EngineStatus::StateMismatch;
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

        if (!prefill_ids(user_eot)) return blackwell::EngineStatus::StateMismatch;
        if (!prefill_ids(gen_cue))  return blackwell::EngineStatus::StateMismatch;

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

protected:
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
        const uint32_t keep = effective_keep_tokens(cmd.keep_prompt_tokens);
        last_rewind_keep_.store(keep, std::memory_order_release);
        pos_ = static_cast<int>(keep);
        // Full host-mirror reconciliation (Phase 0 truncate_kv), safe below the
        // engine's public API on the single engine-owning thread.
        engine_->get_impl()->kv_mgr->rewind(cfg_.seq_id, static_cast<int>(keep));

        if (std::getenv("BLACKWELL_AV_DEBUG")) {
            std::printf("[kv-ckpt] barge-in rewind -> keep=%u (floor=%u)\n",
                        keep, system_prefix_tokens());
            std::fflush(stdout);
        }
        return blackwell::EngineStatus::Success;
    }

    // Speculative warm-prefill: the real path grows the KV over buffered audio via
    // the projector splice. That needs the (unimplemented) Whisper encoder, so it
    // is a documented no-op here — report Success so the state machine keeps
    // warming without faulting the stream.
    blackwell::EngineStatus do_warm_prefill(const Command& /*cmd*/) override {
        return blackwell::EngineStatus::Success;
    }

    // Commit + decode: run a REAL greedy-ish decode on the engine and stream each
    // detokenized piece through the sink, checking cancelled(gen) BEFORE every
    // token so a barge-in aborts within one token. The user turn is a placeholder
    // (the audio-derived transcript is the marked seam — see the header preamble);
    // everything downstream is the genuine GPU decode path.
    blackwell::EngineStatus do_commit_decode(const Command& cmd) override {
        // LIVE AUDIO PATH: with an audio head + DSP bound, transcribe the buffered
        // utterance (the SAME encode->project->prefill_audio->decode path the --wav
        // mode proves) instead of the text placeholder below.
        if (audio_head_loaded() && dsp_ != nullptr && cmd.stream != nullptr)
            return commit_audio_decode(cmd);

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
        for (const int id : turn) {
            if (pos_ >= max_context_) { clear_in_flight(); return finish(cmd, 0); }
            if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
                blackwell::EngineStatus::Success) {
                clear_in_flight();
                return finish(cmd, 0);
            }
            ++pos_;
        }

        // Decode the assistant turn, streaming real detokenized tokens.
        int emitted = 0;
        blackwell::EngineStatus st = blackwell::EngineStatus::Success;
        while (pos_ < max_context_) {
            if (cancelled(cmd.gen)) break;          // wait-free barge-in abort
            if (tok_->is_stop(next)) break;
            const std::string piece = tok_->decode(next, /*render_special=*/false);
            if (!piece.empty())
                cmd.sink.emit(piece.c_str(), emitted, /*is_final=*/0, BRIDGE_OK);
            ++emitted;
            st = engine_->forward_status(next, pos_, /*temperature=*/0.7f, /*top_p=*/0.9f,
                                         cfg_.seq_id, &next);
            if (st != blackwell::EngineStatus::Success) break;
            ++pos_;
        }

        clear_in_flight();
        return finish(cmd, emitted);
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

private:
    // Emit the final callback (clean supersede or cap/fault both report OK to the
    // stream) and return Success — a barge-in is not an error.
    blackwell::EngineStatus finish(const Command& cmd, int emitted) {
        cmd.sink.emit("", emitted, /*is_final=*/1, BRIDGE_OK);
        return blackwell::EngineStatus::Success;
    }

    // Drain the buffered utterance PCM from the stream ring (engine-thread consumer),
    // run WhisperDSP -> [128,3000] log-mel -> prefill_ultravox_turn (audio soft-tokens)
    // -> stream the assistant reply through the sink, checking cancelled(gen) before
    // every token for wait-free barge-in. Mirrors the proven --wav transcribe path.
    blackwell::EngineStatus commit_audio_decode(const Command& cmd) {
        blackwell::bridge::AudioRingBuffer& ring = cmd.stream->ring;
        const std::size_t avail = ring.available_samples();
        if (avail == 0) { clear_in_flight(); return finish(cmd, 0); }
        pcm_stage_.resize(avail);
        const std::size_t got = ring.read_samples(pcm_stage_.data(), avail);
        pcm_stage_.resize(got);

        const float* d_mel = stage_logmel(pcm_stage_);
        int next = -1;
        if (prefill_ultravox_turn(d_mel, &next) != blackwell::EngineStatus::Success) {
            clear_in_flight();
            return finish(cmd, 0);
        }

        int emitted = 0;
        while (pos_ < max_context_) {
            if (cancelled(cmd.gen)) break;              // wait-free barge-in abort
            if (tok_->is_stop(next)) break;
            const std::string piece = tok_->decode(next, /*render_special=*/false);
            if (!piece.empty()) cmd.sink.emit(piece.c_str(), emitted, /*is_final=*/0, BRIDGE_OK);
            ++emitted;
            if (engine_->forward_status(next, pos_, /*temp=*/0.0f, /*top_p=*/1.0f,
                                        cfg_.seq_id, &next) != blackwell::EngineStatus::Success)
                break;
            ++pos_;
        }
        clear_in_flight();
        return finish(cmd, emitted);
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

    blackwell::ITokenizer* tok_ = nullptr;   // non-owning
    int  max_context_ = 0;
    int  pos_ = 0;                            // logical decode position (engine thread only)
    uint32_t verified_prompt_tokens_ = 0;     // committed-prefix boundary (checkpoint/rollback)
    std::atomic<uint32_t> last_rewind_keep_{0};
    std::atomic<std::size_t> turn_counter_{0};

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
};

}  // namespace rt
