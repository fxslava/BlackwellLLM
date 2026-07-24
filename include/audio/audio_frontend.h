#pragma once
// -----------------------------------------------------------------------------
// Ultravox audio frontend — PUBLIC (white-box) surface.
//
// Target: fixie-ai/ultravox-v0_5-llama-3_2-1b (whisper-large-v3-turbo encoder +
// Ultravox SwiGLU projector + Llama-3.2-1B backbone).
//
// A self-contained multimodal PRE-processor: raw mono audio -> *audio embeddings*
// (rows in the TEXT model's embedding space) that the engine splices into the
// `<|audio|>` placeholder slots (token id 128256) before the ordinary text decode
// runs. It NEVER touches the text LLM kernels, KV cache, or decode loop.
//
// DECOUPLING CONTRACT (load-bearing — future speculative decoding):
//   The frontend is a PURE FUNCTION of audio. It holds NO engine/KV/sequence
//   state and no reference to any BlackwellEngine. The pipeline is split at the
//   encoder/projector boundary so the EXPENSIVE Whisper encoder runs ONCE and the
//   CHEAP projector runs per text model: a Llama-3.2-1B draft (hidden 2048) and an
//   ~8B target (hidden 4096) consume the SAME encoder features via TWO projectors.
//   Do not fuse encode_features() and project() — the split is the spec-dec seam.
//
// Style: light header (forward decls, NO CUDA / no engine includes); device work
// lives behind the PIMPL in src/audio/. Mirrors include/blackwell/engine.h.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace blackwell::audio {

// Resolved geometry, populated by the weight loader from the checkpoint's
// audio_config / projector config (verify against the installed transformers /
// Ultravox source — see ULTRAVOX_AUDIO_PLAN.md §1). Defaults are the published
// v0_5-llama-3_2-1b values and are a sanity baseline, NOT a substitute for
// reading the checkpoint config.
struct AudioEncoderGeometry {
    // --- whisper-large-v3-turbo encoder (Ultravox uses the encoder only) ---
    int num_mel_bins = 128;          // log-mel feature bins
    int d_model = 1280;              // encoder hidden width
    int num_layers = 32;             // encoder layers (turbo shrinks the DECODER, not this)
    int num_heads = 20;              // bidirectional self-attention heads
    int head_dim = 64;               // d_model / num_heads
    int ffn_dim = 5120;              // encoder MLP intermediate
    int conv_downsample = 2;         // conv2 stride (time /2)
    // --- Ultravox projector ---
    int stack_factor = 8;            // StackAudioFrames time /8; dim_in = d_model*stack_factor
    int projector_hidden = 4096;     // linear_1 output; swiglu halves it to 2048
    bool projector_ln_mid = true;    // v0.5: RMSNorm after linear_1 (ln_post=Identity)
    int text_hidden_dim = 0;         // projector linear_2 output == target LLM hidden size
    int audio_token_id = 128256;     // `<|audio|>` placeholder replaced by embeddings
    // --- feature_extractor (DSP); verify against the processor ---
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
};

// Whisper-space encoder output: [num_frames, d_model], row-major. MODEL-AGNOSTIC
// and expensive — compute once, reuse across projectors (spec-dec draft+target).
struct AudioFeatures {
    std::vector<float> data;   // [num_frames * d_model]
    size_t num_frames = 0;
    size_t d_model = 0;
};

// Projected embeddings in ONE text model's space: [num_tokens, hidden_dim],
// row-major, ready to scatter into the `<|audio|>` slots of that model's prompt.
struct AudioEmbeddings {
    std::vector<float> data;   // [num_tokens * hidden_dim]
    size_t num_tokens = 0;
    size_t hidden_dim = 0;
};

// Bound to ONE text model: owns that model's projector weights (linear_1/2 +
// RMSNorms) and knows its text_hidden. Speculative decoding builds two of these
// (draft + target) over shared AudioFeatures. Not thread-safe; engine-owning
// thread only.
class AudioProjector {
public:
    virtual ~AudioProjector() = default;
    // StackAudioFrames -> ln_pre -> linear_1 -> swiglu -> ln_mid -> linear_2.
    virtual AudioEmbeddings project(const AudioFeatures& features) = 0;
    virtual int text_hidden_dim() const = 0;
};

// The encoder half: DSP -> conv subsample -> 32 bidirectional layers -> final LN.
// Owns encoder weights + device scratch. Stateless w.r.t. the LLM. Not
// thread-safe; construct + call from the engine-owning thread only.
//
// NOTE: SCAFFOLD contract. Bodies land per ULTRAVOX_AUDIO_PLAN.md (Phases 1-6).
class AudioFrontend {
public:
    virtual ~AudioFrontend() = default;

    // Predicted audio-embedding count for `num_samples` mono PCM samples:
    // conv2 (/2) then StackAudioFrames (/8), i.e. ceil((mel_frames/2)/stack_factor).
    // Lets the engine reserve `<|audio|>` placeholder slots before encoding.
    virtual size_t predict_num_tokens(size_t num_samples) const = 0;

    // The SHARED, expensive step: run once per clip (16 kHz mono PCM in [-1,1]).
    virtual AudioFeatures encode_features(const float* pcm, size_t num_samples) = 0;

    // Build a projector bound to a text model of the given hidden size, loading
    // that model's projector weights from `projector_weights_dir`. Spec-dec calls
    // this twice (draft 2048, target 4096).
    virtual std::unique_ptr<AudioProjector> make_projector(
        const std::string& projector_weights_dir, int text_hidden_dim) = 0;

    const AudioEncoderGeometry& geometry() const { return geo_; }

protected:
    AudioEncoderGeometry geo_;
};

// Factory: loads the Ultravox encoder from `weights_dir`. `text_hidden_dim` is the
// primary consumer LLM's hidden size (used to preload its projector for the
// single-model convenience path). INIT tier: throws on failure (Hybrid doctrine).
std::unique_ptr<AudioFrontend> create_audio_frontend(const std::string& weights_dir,
                                                     int text_hidden_dim);

}  // namespace blackwell::audio
