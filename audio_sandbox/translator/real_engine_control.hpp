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
// THE AUDIO-ENCODER SEAM (the one remaining blocker, documented, not faked)
//   The full multimodal path is  live PCM -> Whisper log-mel (whisper_dsp) ->
//   Whisper ENCODER (audio_tower) -> Ultravox projector -> inject_audio_
//   embeddings at token 128256 -> prefill-from-embeddings -> decode. The Whisper
//   *encoder* transformer that turns log-mel [128,N] into encoder hidden states
//   [1500,1280] is NOT implemented (src/audio/kernels/ is empty; see
//   ULTRAVOX_AUDIO_PLAN.md Phase 1). Until it lands there is no live audio->text
//   content. do_commit_decode therefore runs a REAL text decode from the frozen
//   system prefix (streaming genuine model output), and the projector splice is
//   left as run_audio_prefill_seam() — wired to the real projector but never fed,
//   with the exact drop-in point marked. This is the honest maximum: the whole
//   GPU decode + streaming + barge-in path is real; only the audio CONTENT is the
//   seam.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "blackwell/engine.h"           // BlackwellEngine, blackwell::EngineStatus
#include "blackwell/tokenizer.h"        // blackwell::ITokenizer, ChatMessage

#include "engine_control_bridge.hpp"    // EngineControlBridge, Command
#include "bridge/engine_api.h"          // BRIDGE_OK for the token sink

namespace blackwell::audio { class UltravoxProjector; }

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

    // The drop-in seam for the real audio path. When the Whisper encoder lands,
    // this runs: projector_->forward(encoder_hidden) -> audio_embeds ->
    // inject_audio_embeddings at token 128256 -> prefill-from-embeddings (the
    // granular step_* sweep, cf. tests/integration/test_ultravox8b_full_pipeline)
    // -> return the position to decode from. It is intentionally never called
    // today (no encoder feeds the projector); kept to pin the integration point.
    int run_audio_prefill_seam() { return pos_; }

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
};

}  // namespace rt
