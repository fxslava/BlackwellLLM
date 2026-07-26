#pragma once
// -----------------------------------------------------------------------------
// AudioEmbeddingPipeline — the Stream-1 (audio) producer of the double-buffered
// translator pipeline. Chains, all on the Whisper encoder's private stream:
//
//   log-mel [num_mel_bins, conv_frames]
//     -> WhisperEncoder (graph-captured)   -> encoder hidden [1500, 1280]
//     -> UltravoxProjector                 -> audio embeds  [out_frames, text_hidden]
//     -> cudaMemcpyAsync into a ping-pong slot + record its ready event
//
// Everything is enqueued on audio_stream() with NO host sync: the encoder already
// returns without synchronising, the projector runs on the SAME stream (so it is
// correctly ordered after the encoder graph), and the slot copy + ready event
// close the producer edge. The consumer (LLM stream) gates on the ping-pong event.
//
// DOCTRINE: single owning thread; INIT-tier ctor/load throw; forward path
// allocates nothing and never calls cudaDeviceSynchronize.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"                        // CUDA_CHECK_THROW
#include "blackwell/runtime_config.h"      // blackwell::AudioStreamingPlan (resolved tier-3)
#include "whisper_encoder.h"               // WhisperEncoder{,Config}, WhisperWeights
#include "ultravox_projector_pipeline.cuh" // UltravoxProjector, ProjectorConfig
#include "ping_pong_audio_buffer.h"        // PingPongAudioBuffer

namespace blackwell::audio {

// What a streaming hop should do with the KV after re-encoding its window. When
// `diverged`, the caller rolls the KV back to `rewind_to_pos` (Phase-0
// kv_cache_rollback) and re-injects the NEW window's soft-tokens from index
// `refill_from` (the corrected overlap tail + the new hop); otherwise it injects
// only the tail delta (hop_tokens). See AudioEmbeddingPipeline::reconcile.
struct ReconciliationPlan {
    bool diverged      = false;
    int  rewind_to_pos = 0;   // KV position to roll back to before re-injecting
    int  refill_from   = 0;   // start index into the NEW window's soft-tokens
};

class AudioEmbeddingPipeline {
public:
    AudioEmbeddingPipeline(const WhisperEncoderConfig& enc_cfg,
                           const ProjectorConfig& proj_cfg)
        : encoder_(enc_cfg),
          projector_(proj_cfg, enc_cfg.max_source_positions),
          num_input_frames_(enc_cfg.max_source_positions) {}

    // One-time weight upload (host fp32). Not on the forward path.
    void load_weights(const WhisperWeights& enc_w,
                      const std::vector<float>& ln_pre,
                      const std::vector<float>& linear_1,
                      const std::vector<float>& ln_mid,
                      const std::vector<float>& linear_2) {
        encoder_.load_weights(enc_w);
        projector_.load_weights(ln_pre, linear_1, ln_mid, linear_2);
    }

    // The audio (Stream-1) stream every enqueue targets.
    cudaStream_t audio_stream() const { return encoder_.stream(); }

    int out_frames() const { return projector_.out_frames(num_input_frames_); }
    int text_hidden() const { return projector_.text_hidden(); }
    // Elements in one embeddings frame == a ping-pong slot's required size.
    std::size_t embed_elems() const {
        return static_cast<std::size_t>(out_frames()) * text_hidden();
    }

    // Encode + project d_mel [num_mel_bins, conv_frames] and stage the resulting
    // [out_frames, text_hidden] embeddings into pp.slot(frame) -- all on the audio
    // stream. Closes the producer edge (WAR-acquire on the reused slot, then
    // publish the ready event). Fully async; the caller drives the consumer.
    void process_frame(const float* d_mel, PingPongAudioBuffer& pp, long long frame) {
        const float* d_hidden = encoder_.forward(d_mel);  // Stream-1, graph launch
        const float* d_embeds =
            projector_.forward(d_hidden, num_input_frames_, audio_stream());
        pp.producer_acquire(frame, audio_stream());       // slot free (frame-2 read)?
        CUDA_CHECK_THROW(cudaMemcpyAsync(pp.slot(frame), d_embeds, pp.bytes(),
                                         cudaMemcpyDeviceToDevice, audio_stream()));
        pp.producer_publish(frame, audio_stream());       // slot ready for consumer
    }

    // ---- Streaming path: dynamic overlap reconciliation --------------------------

    // Bind the resolved tier-3 streaming plan (engine-thread setup). When
    // plan.enabled, reconcile()/record_injected() maintain the injected-token history
    // and dynamic tail rewrite; otherwise they are inert (Phase-1 tail-slicing).
    void configure_streaming(const blackwell::AudioStreamingPlan& plan) { plan_ = plan; }
    const blackwell::AudioStreamingPlan& streaming_plan() const { return plan_; }

    // Bucketed encode + project of a `mel_frames`-long window (mel-major d_mel):
    // returns the projector output [num_tokens, text_hidden] device pointer (owned by
    // the projector) and the VALID soft-token count. All on audio_stream(); the caller
    // must gate/sync before consuming. No ping-pong staging (the streaming consumer
    // reads the pointer directly).
    struct ProjectedWindow { const float* embeds; int num_tokens; };
    ProjectedWindow encode_project_window(const float* d_mel, int mel_frames,
                                          int mel_frame_offset = 0) {
        // conv2 (stride 2) halves the time axis, so a window starting at absolute mel
        // frame `mel_frame_offset` begins at absolute encoder position offset/2. Passing
        // it keeps the streamed soft-tokens' positional embeddings monotonic across
        // windows (0 == whole-clip encode).
        const float* d_hidden = encoder_.forward(d_mel, mel_frames, mel_frame_offset / 2);
        const int valid = encoder_.valid_output_frames(mel_frames);   // unpadded frames
        const float* d_embeds = projector_.forward(d_hidden, valid, audio_stream());
        return { d_embeds, projector_.out_frames(valid) };
    }

    // Cosine similarity over one text_hidden-dim embedding pair, double-accumulated.
    // In 4096-d, similar embeddings sit at cos ~ 1.0; a drop below the (strict) tier-3
    // threshold marks a genuine acoustic-meaning shift, not float noise.
    static double cosine_similarity(const float* a, const float* b, int n) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < n; ++i) {
            const double x = a[i], y = b[i];
            dot += x * y; na += x * x; nb += y * y;
        }
        const double den = std::sqrt(na) * std::sqrt(nb);
        return den > 1e-12 ? dot / den : 0.0;
    }

    // Scan the TAIL of the overlap (the newest max_rewind_tokens rows, nearest the
    // just-arrived audio) against the historically INJECTED tokens for the same span,
    // and at the first row whose cosine < threshold return a rewind/refill plan
    // (rewind_to_pos = KV position to roll back to; refill_from = NEW-window index to
    // resume injection from). Returns {diverged=false} when the tail is stable (inject
    // only the delta) or when streaming/history is not yet available.
    //
    // WHY ONLY THE TAIL: on a forward slide, the FRONT overlap tokens LOSE left-context
    // in the bidirectional encoder (their window no longer reaches as far back), so
    // they diverge — but they were originally encoded with MORE context, and rewriting
    // them with the lower-context representation DEGRADES them. Only the tail tokens
    // legitimately improve (they gain right-context from the new audio). Freezing the
    // front (never scanning it) preserves the high-context tokens; scanning only the
    // last max_rewind_tokens also makes the rewind depth intrinsically bounded.
    // Does ONE small D2H of just the tail region (off the per-token hot path).
    ReconciliationPlan reconcile(const float* d_new_proj, int num_new_tokens) {
        if (!plan_.enabled) return {};
        const int H = text_hidden();
        const int overlap = std::min({plan_.overlap_tokens,
                                      static_cast<int>(injected_history_.size()),
                                      num_new_tokens});
        const int scan_start = std::max(0, overlap - plan_.max_rewind_tokens);
        const int scan_n = overlap - scan_start;   // == min(overlap, max_rewind_tokens)
        if (scan_n <= 0) return {};                 // no overlap, or cap 0 (reconcile off)

        recon_stage_.resize(static_cast<std::size_t>(scan_n) * H);
        CUDA_CHECK_THROW(cudaMemcpy(recon_stage_.data(),
                                    d_new_proj + static_cast<std::size_t>(scan_start) * H,
                                    recon_stage_.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        const int base = static_cast<int>(injected_history_.size()) - overlap;
        const int cur_pos = injected_history_.back().kv_pos + 1;   // next free KV slot
        for (int j = 0; j < scan_n; ++j) {
            const int i = scan_start + j;   // overlap index (front frozen: i >= scan_start)
            const InjectedToken& hist = injected_history_[static_cast<std::size_t>(base + i)];
            const double cos = cosine_similarity(&recon_stage_[static_cast<std::size_t>(j) * H],
                                                 hist.embed.data(), H);
            if (cos < plan_.reconciliation_threshold) {
                // Rewrite from the first diverging TAIL token to the end of the overlap;
                // depth = overlap - i <= max_rewind_tokens by construction.
                return { /*diverged=*/true,
                         /*rewind_to_pos=*/cur_pos - (overlap - i),
                         /*refill_from=*/ i };
            }
        }
        return {};   // tail stable within threshold -> inject only the delta
    }

    // Record the soft-tokens actually injected into the KV this hop: `num_tokens`
    // rows of d_embeds committed at KV positions [start_pos, start_pos+num_tokens).
    // First evicts any history at/after start_pos (mirrors a preceding rollback), so
    // the mirror stays consistent with the KV; then appends, capped to window_tokens
    // (only the overlap can ever be rewound). Inert unless streaming is enabled.
    void record_injected(const float* d_embeds, int start_pos, int num_tokens) {
        if (!plan_.enabled || num_tokens <= 0) return;
        while (!injected_history_.empty() && injected_history_.back().kv_pos >= start_pos)
            injected_history_.pop_back();

        const int H = text_hidden();
        recon_stage_.resize(static_cast<std::size_t>(num_tokens) * H);
        CUDA_CHECK_THROW(cudaMemcpy(recon_stage_.data(), d_embeds,
                                    recon_stage_.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_tokens; ++i) {
            InjectedToken t;
            t.kv_pos = start_pos + i;
            t.embed.assign(recon_stage_.begin() + static_cast<std::size_t>(i) * H,
                           recon_stage_.begin() + static_cast<std::size_t>(i + 1) * H);
            injected_history_.push_back(std::move(t));
            if (static_cast<int>(injected_history_.size()) > plan_.window_tokens)
                injected_history_.pop_front();
        }
    }

    // Drop the whole injected-token history (barge-in: the KV is rewound to the
    // frozen system prefix, so no audio history survives).
    void reset_history() noexcept { injected_history_.clear(); }
    int  history_size() const noexcept { return static_cast<int>(injected_history_.size()); }

private:
    WhisperEncoder encoder_;
    UltravoxProjector projector_;
    int num_input_frames_;

    // Injected-token history (host mirror) for overlap reconciliation: the soft-token
    // embeddings already committed to the KV, newest last, each tagged with its KV
    // position. Bounded to plan_.window_tokens rows.
    blackwell::AudioStreamingPlan plan_{};
    struct InjectedToken { int kv_pos; std::vector<float> embed; };  // embed: [text_hidden]
    std::deque<InjectedToken> injected_history_;
    std::vector<float> recon_stage_;   // reused D2H staging (overlap prefix / injected rows)
};

}  // namespace blackwell::audio
