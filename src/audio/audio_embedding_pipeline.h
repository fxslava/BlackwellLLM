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
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"                        // CUDA_CHECK_THROW
#include "whisper_encoder.h"               // WhisperEncoder{,Config}, WhisperWeights
#include "ultravox_projector_pipeline.cuh" // UltravoxProjector, ProjectorConfig
#include "ping_pong_audio_buffer.h"        // PingPongAudioBuffer

namespace blackwell::audio {

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

private:
    WhisperEncoder encoder_;
    UltravoxProjector projector_;
    int num_input_frames_;
};

}  // namespace blackwell::audio
