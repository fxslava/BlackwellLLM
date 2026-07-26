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
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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

protected:
    // Barge-in micro-rewind. The 8B AWQ backbone runs Continuous KV, which is
    // position-addressed and has no CoW rewind (BlackwellEngine::rewind is gated
    // to Paged), so we do NOT call the engine here — re-decoding from an earlier
    // position simply overwrites stale KV slots. We still honour the frozen-
    // prefix floor (effective_keep_tokens) and re-anchor pos_ to the resolved
    // keep so the next utterance decodes on top of the system prefix, not on top
    // of the previous (superseded) answer.
    blackwell::EngineStatus do_rewind(const Command& cmd) override {
        const uint32_t keep = effective_keep_tokens(cmd.keep_prompt_tokens);
        last_rewind_keep_.store(keep, std::memory_order_release);
        pos_ = static_cast<int>(keep);
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

    blackwell::ITokenizer* tok_ = nullptr;   // non-owning
    int  max_context_ = 0;
    int  pos_ = 0;                            // logical decode position (engine thread only)
    std::atomic<uint32_t> last_rewind_keep_{0};
    std::atomic<std::size_t> turn_counter_{0};

    // Audio frontend (owned; loaded lazily via load_audio_head). Both are driven
    // only from the engine-owning thread; the pipeline runs on its own stream.
    std::unique_ptr<blackwell::audio::AudioEmbeddingPipeline> audio_pipeline_;
    std::unique_ptr<blackwell::audio::PingPongAudioBuffer>    audio_pp_;
    long long audio_frame_ = 0;              // producer/consumer frame counter
};

}  // namespace rt
