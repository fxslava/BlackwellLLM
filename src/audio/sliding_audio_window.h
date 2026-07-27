#pragma once
// -----------------------------------------------------------------------------
// SlidingAudioWindow — Phase 1 streaming acoustic-context window + delta soft-token
// extraction (header-only, like ping_pong_audio_buffer.h / audio_embedding_pipeline.h).
//
// The live mic path decodes incrementally: every ~300 ms a new PCM/log-mel chunk
// arrives, but the Whisper encoder needs surrounding acoustic context to encode it
// well. So we keep a rolling window of [ HISTORY (~2 s) ++ NEW CHUNK (~300 ms) ],
// run the encoder + projector over the WHOLE window (Feature-1 bucket sized to the
// window), and append ONLY the soft-tokens for the new chunk (the "delta") to
// Llama's KV. The history soft-tokens are recomputed each hop for context and
// discarded, keeping Llama's KV strictly monotonic (grows by hop_tokens per hop).
//
//   window (mel):  [ ....... history ....... | new chunk ]
//   projector   :  [ s0 s1 ...        s_{k-2} | s_{k-1} s_k ]   (soft-tokens)
//   appended    :                             ^^^^^^^^^^^^^     (delta = tail)
//
// GRANULARITY: one Ultravox soft-token spans conv_stride(2) * stack_factor(8) = 16
// mel frames = 160 ms. Expressing window/hop in whole soft-tokens (multiples of 16
// mel frames) makes the delta an exact TAIL slice of the projector output.
//
// SEEDING: the FIRST utterance is committed by the existing full prefill
// (RealEngineControl::prefill_ultravox_turn); this window drives the subsequent
// incremental hops, so the delta is always the newest hop_tokens rows.
//
// DOCTRINE: single owning thread (engine thread); INIT-tier ctor throws; the hop
// path allocates nothing and issues no host sync (pure enqueue on the given stream).
// The window is double-buffered so the per-hop shift has no overlapping-copy UB.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "common.h"          // CUDA_CHECK_THROW
#include "device_buffer.h"   // blackwell::DeviceBuffer

namespace blackwell::audio {

// Temporal constants of the Ultravox front-end, in mel-frame units. One mel frame
// is 10 ms (Whisper hop 160 @ 16 kHz). One soft-token spans kMelFramesPerSoftToken
// mel frames after the conv-stride-2 downsample and the stack_factor-8 compression.
inline constexpr int kConvStride            = 2;
inline constexpr int kStackFactor           = 8;
inline constexpr int kMelFramesPerSoftToken = kConvStride * kStackFactor;  // 16 == 160 ms
inline constexpr int kMelFramesPerSecond    = 100;                          // 10 ms hop

// Sliding-window geometry in soft-tokens. In the LIVE path these are NOT hardcoded:
// they are derived from the validated tier-3 AudioStreamingPlan via
// sliding_window_from_plan() below (window_tokens / hop_tokens resolved from the
// InferenceConfig ms request). The defaults here are an illustrative 2.24 s / 320 ms
// preset for standalone tests only.
struct SlidingWindowConfig {
    int num_mel_bins   = 128;   // Whisper log-mel channels (encoder conv1 in-channels)
    int history_tokens = 12;    // acoustic history kept per hop (12 * 160 ms = 1.92 s)
    int hop_tokens     = 2;     // new soft-tokens produced+appended per hop (320 ms)

    int window_tokens()     const { return history_tokens + hop_tokens; }
    int window_mel_frames() const { return window_tokens() * kMelFramesPerSoftToken; }
};

// Build the window geometry from the resolved plan's soft-token counts (retires the
// magic numbers: the live path passes plan.window_tokens / plan.hop_tokens). Kept as
// scalars so this header stays free of the core config dependency.
inline SlidingWindowConfig sliding_window_from_plan(int window_tokens, int hop_tokens,
                                                    int num_mel_bins) {
    SlidingWindowConfig c;
    c.num_mel_bins   = num_mel_bins;
    c.hop_tokens     = hop_tokens;
    c.history_tokens = window_tokens - hop_tokens;
    return c;
}

// ---- CenterSlice geometry (pure arithmetic, Tier-1 testable) -----------------
// Which rows of a projected window are NEW stable center tokens. All coordinates
// are ABSOLUTE soft-token indices from the utterance start except the returned
// row range, which is window-relative.
//
//   committed_abs     tokens already injected into the KV ([0, committed_abs))
//   window_start_abs  the window's first token (mel_frame_offset / 16)
//   num_tokens        valid soft-tokens the projector produced for this window
//   right_edge_tokens K — the unstable future edge withheld during speech
//
// The stable span is [start_row, num_tokens - K): everything not yet committed,
// stopping K short of the window's right edge. The LEFT edge needs no explicit
// term here: the plan validator enforces left + right + hop <= window, which
// guarantees committed_abs >= window_start_abs + left_edge on every warm hop —
// so the returned rows always sit >= left_edge tokens from the window's left
// boundary (full acoustic left context). On the COLD first window committed_abs
// is 0 and the slice starts at row 0: the utterance opening has no earlier audio
// to lose context from, and no later window will ever re-cover it.
struct CenterSlicePlan {
    int start_row = 0;   // first NEW row (window-relative)
    int count     = 0;   // rows to inject (0 = nothing new stable yet)
};
inline CenterSlicePlan center_slice_plan(int committed_abs, int window_start_abs,
                                         int num_tokens, int right_edge_tokens) {
    const int stable_end = num_tokens - right_edge_tokens;              // window-relative
    const int start_row  = std::max(0, committed_abs - window_start_abs);
    return { start_row, std::max(0, stable_end - start_row) };
}

// The extracted delta: a view into the projector output plus how many soft-token
// rows (each text_hidden wide) belong to the new chunk. These — and only these —
// are prefilled into Llama's KV.
struct DeltaTokens {
    const float* data        = nullptr;  // -> first delta soft-token row [count, text_hidden]
    int          count        = 0;       // soft-token rows to append (<= hop_tokens)
    int          text_hidden = 0;        // row width (backbone hidden), the KV prefill stride
};

class SlidingAudioWindow {
public:
    // text_hidden is the projector output width (== backbone hidden). INIT tier:
    // allocates the double-buffered device mel window (zeroed) and throws on OOM.
    SlidingAudioWindow(const SlidingWindowConfig& cfg, int text_hidden)
        : cfg_(cfg), text_hidden_(text_hidden) {
        const std::size_t w = static_cast<std::size_t>(cfg_.num_mel_bins) *
                              cfg_.window_mel_frames();
        for (int i = 0; i < 2; ++i) {
            buf_[i].allocate(w);
            CUDA_CHECK_THROW(cudaMemset(buf_[i].get(), 0, w * sizeof(float)));
        }
    }

    SlidingAudioWindow(const SlidingAudioWindow&) = delete;
    SlidingAudioWindow& operator=(const SlidingAudioWindow&) = delete;

    // Append a new mel chunk [num_mel_bins, chunk_frames] (device ptr, mel-major),
    // shifting the window left by chunk_frames so the freshest audio lands at the
    // tail (which is where the delta is sliced from). Builds the next window in the
    // idle buffer (no overlapping copy) and swaps. Enqueued on `stream`; no host sync.
    void push_chunk(const float* d_chunk_mel, int chunk_frames, cudaStream_t stream) {
        const int W    = cfg_.window_mel_frames();
        const int mel  = cfg_.num_mel_bins;
        const int newc = std::min(chunk_frames, W);        // if a chunk exceeds W, keep its tail
        const int keep = W - newc;                         // history frames retained
        const std::size_t pitch = static_cast<std::size_t>(W) * sizeof(float);

        float* dst = buf_[cur_ ^ 1].get();
        const float* src = buf_[cur_].get();

        // nxt[m, 0:keep] = cur[m, newc : newc+keep]  (drop the oldest `newc` frames).
        if (keep > 0) {
            CUDA_CHECK_THROW(cudaMemcpy2DAsync(
                dst, pitch,
                src + newc, pitch,
                static_cast<std::size_t>(keep) * sizeof(float), static_cast<std::size_t>(mel),
                cudaMemcpyDeviceToDevice, stream));
        }
        // nxt[m, keep:W] = chunk[m, chunk_frames-newc : chunk_frames]  (append newest).
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(
            dst + keep, pitch,
            d_chunk_mel + (chunk_frames - newc), static_cast<std::size_t>(chunk_frames) * sizeof(float),
            static_cast<std::size_t>(newc) * sizeof(float), static_cast<std::size_t>(mel),
            cudaMemcpyDeviceToDevice, stream));

        cur_ ^= 1;                                          // the freshly built window is live
        filled_mel_ = std::min(W, filled_mel_ + chunk_frames);
        ++hops_;
    }

    // The contiguous [num_mel_bins, window_mel_frames()] view the encoder consumes.
    const float* window_mel() const { return buf_[cur_].get(); }
    int          window_mel_frames() const { return cfg_.window_mel_frames(); }

    // Slice the delta soft-tokens out of a projector output of `out_soft_tokens` rows
    // (== projector.out_frames(encoder.valid_output_frames(window_mel_frames()))): the
    // newest hop_tokens rows (the tail). The tail is the freshest audio regardless of
    // conv/stack rounding, so this is robust cold and warm. Returns a VIEW (no copy).
    DeltaTokens extract_delta(const float* d_projector_out, int out_soft_tokens) const {
        const int count = std::min(cfg_.hop_tokens, std::max(0, out_soft_tokens));
        const int start = out_soft_tokens - count;
        return DeltaTokens{d_projector_out + static_cast<std::size_t>(start) * text_hidden_,
                           count, text_hidden_};
    }

    // Soft-tokens a normal (warm) hop appends. The caller advances Llama's pos_ by
    // exactly this on an accepted hop (checkpoint/rollback boundary — Phase 0).
    int hop_tokens() const { return cfg_.hop_tokens; }

    // True once history_tokens of real audio have accumulated (the window is full and
    // now genuinely sliding). Informational; the tail-delta rule holds either way.
    bool warm() const { return filled_mel_ >= cfg_.window_mel_frames(); }

private:
    SlidingWindowConfig cfg_;
    int text_hidden_ = 0;
    DeviceBuffer<float> buf_[2];    // double-buffered [num_mel_bins, window_mel_frames()]
    int cur_ = 0;                   // index of the live window
    int filled_mel_ = 0;            // real mel frames accumulated (cold-start gate for warm())
    long long hops_ = 0;            // pushes so far
};

}  // namespace blackwell::audio
