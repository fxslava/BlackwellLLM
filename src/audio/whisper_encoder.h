#pragma once
// -----------------------------------------------------------------------------
// WhisperEncoder — GPU forward pass of the Ultravox audio_tower (Phase 3).
//
// This is the whisper-large-v3-turbo ENCODER (the "audio_tower" in the Ultravox
// checkpoint): raw log-mel spectrogram -> [max_source_positions, d_model] audio
// features. It is the stage that feeds UltravoxProjector (ultravox_projector*).
//
// Architecture (config values are the turbo defaults):
//   input  log-mel [num_mel_bins, conv_frames]            (128 x 3000, 30 s)
//   conv1  Conv1d(128->1280, k=3, s=1, pad=1) + GELU      -> [1280, 3000]
//   conv2  Conv1d(1280->1280, k=3, s=2, pad=1) + GELU     -> [1280, 1500]
//   permute                                               -> [1500, 1280]
//   + embed_positions[1500,1280]
//   32x EncoderLayer (pre-LN, bidirectional MHA, GELU MLP, residuals)
//   final LayerNorm                                       -> [1500, 1280]
//
// PERFORMANCE (Phase 3, opt pass): the projections and the batched attention run
// on cuBLAS with TF32 tensor cores; LayerNorm/softmax and the bias+GELU /
// bias+residual epilogues are fused custom kernels. Because the input shape is
// FIXED at inference, the entire 32-layer pass is captured ONCE into a
// cudaGraphExec_t (built on the first forward()); every subsequent forward() is a
// single cudaGraphLaunch, eliminating per-kernel CPU launch overhead. This takes
// the pass from ~9.7 s (naive fp32) to a few ms. Parity vs. the fp32 PyTorch
// reference stays well within the bar (TF32 accumulates in fp32).
//
// DOCTRINE (inherited from the engine, see src/audio/README.md):
//   * Single owning thread: construct + call from the engine-owning thread only.
//     The private stream / cuBLAS handle / graph are NOT thread-safe.
//   * INIT tier (ctor / load_weights): CUDA_CHECK_THROW -> blackwell::cuda_error
//     (and a throw on cuBLAS/graph setup failure); DeviceBuffer + the dtor make
//     the throwing ctor leak-free.
//   * DeviceBuffer<T> owns every device allocation; forward() allocates NOTHING
//     and issues NO host-side sync (pure async enqueue on stream()).
//   * The class is core-free: it takes HOST fp32 weights (WhisperWeights). The
//     caller (test / audio_frontend) reads + de-quantises the checkpoint's BF16
//     audio_tower.* tensors.
// -----------------------------------------------------------------------------
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device_buffer.h"  // blackwell::DeviceBuffer

namespace blackwell::audio {

struct WhisperEncoderConfig {
    int num_mel_bins = 128;           // conv1 in-channels
    int d_model = 1280;               // hidden size (conv out-channels)
    int num_layers = 32;              // encoder_layers
    int num_heads = 20;               // encoder_attention_heads
    int ffn_dim = 5120;               // encoder_ffn_dim (fc1 out / fc2 in)
    int max_source_positions = 1500;  // encoder output frames (== seq len)
    int conv_frames = 3000;           // mel time frames fed to conv1 (30 s); the
                                      // LARGEST bucket + the workspace-sizing bound
    float ln_eps = 1e-5f;             // nn.LayerNorm default eps

    // CUDA-graph buckets (conv-frame counts) the encoder may capture, ascending. A
    // forward(d_mel, mel_frames) rounds its input UP to the smallest covering bucket
    // and captures that graph lazily, so a short chunk is NOT padded to 30 s. Any
    // value > conv_frames is dropped and conv_frames is always included (it is the
    // parity reference + the buffer-sizing bound). Default: 1.5 s / 3 s / 5 s / 30 s.
    std::vector<int> graph_buckets = {150, 300, 500, 3000};

    int head_dim() const { return d_model / num_heads; }
    // conv2 halves the time axis (stride 2, k=3, pad=1): out == max_source_positions.
    int conv_out_frames() const { return conv_out_frames_of(conv_frames); }
    // Encoder frames produced for `cf` mel frames (shared by the buckets + the valid-
    // prefix accounting the sliding window uses inside a padded bucket).
    static int conv_out_frames_of(int cf) { return (cf + 2 * 1 - 3) / 2 + 1; }
};

// Host-side fp32 weights for ONE encoder layer. Linear weights keep the PyTorch
// [out, in] row-major layout (y = x @ W^T + b). k_proj has NO bias in Whisper.
struct WhisperLayerWeights {
    std::vector<float> self_attn_layer_norm_w, self_attn_layer_norm_b;  // [d_model]
    std::vector<float> q_w, q_b;   // [d_model,d_model] , [d_model]
    std::vector<float> k_w;        // [d_model,d_model]  (no bias)
    std::vector<float> v_w, v_b;   // [d_model,d_model] , [d_model]
    std::vector<float> out_w, out_b;                                    // [d_model,*]
    std::vector<float> final_layer_norm_w, final_layer_norm_b;         // [d_model]
    std::vector<float> fc1_w, fc1_b;  // [ffn,d_model] , [ffn]
    std::vector<float> fc2_w, fc2_b;  // [d_model,ffn] , [d_model]
};

// Host-side fp32 weights for the whole encoder.
struct WhisperWeights {
    std::vector<float> conv1_w, conv1_b;  // [d_model,num_mel_bins,3] , [d_model]
    std::vector<float> conv2_w, conv2_b;  // [d_model,d_model,3]      , [d_model]
    std::vector<float> embed_positions;   // [max_source_positions, d_model]
    std::vector<WhisperLayerWeights> layers;  // size == num_layers
    std::vector<float> layer_norm_w, layer_norm_b;  // final LayerNorm [d_model]
};

class WhisperEncoder {
public:
    explicit WhisperEncoder(const WhisperEncoderConfig& cfg);
    ~WhisperEncoder();

    WhisperEncoder(const WhisperEncoder&) = delete;
    WhisperEncoder& operator=(const WhisperEncoder&) = delete;

    // Uploads all host weights to device (INIT tier: throws on OOM / size
    // mismatch). Must be called once before forward().
    void load_weights(const WhisperWeights& w);

    // Runs the full encoder over the FULL 30 s bucket. d_mel is a device pointer to
    // the log-mel spectrogram [num_mel_bins, conv_frames] (row-major, mel-major).
    // Back-compat overload of the bucketed forward below (mel_frames == conv_frames):
    // returns [max_source_positions, d_model]. Kept bit-identical for the --wav path
    // and the golden-parity tests.
    const float* forward(const float* d_mel);

    // Bucketed forward. d_mel is [num_mel_bins, mel_frames] (mel-major); the encode
    // is sized to the smallest captured bucket whose frame count covers mel_frames
    // (padded with zeros to the bucket width), so a short chunk pays a short encode
    // instead of the full 30 s. On the first use of a bucket its graph is built
    // (warm-up + capture); every call issues an async input pack + cudaGraphLaunch on
    // stream() and returns without host sync. Returns a device pointer to the output
    // [output_frames(), d_model], owned by this object; output_frames() reflects the
    // SELECTED bucket after the call. Synchronize stream() before reading.
    //
    // enc_pos_offset shifts the absolute positional embeddings: the encoder adds
    // embed_positions[enc_pos_offset + j] to output frame j instead of [j]. A
    // STREAMING slice starting at absolute encoder frame `off` passes enc_pos_offset
    // = off so its soft-tokens carry utterance-absolute positions and splice
    // coherently after earlier windows (0 == the default whole-clip encode).
    const float* forward(const float* d_mel, int mel_frames, int enc_pos_offset = 0);

    // The stream every forward() enqueues on (input copy + graph launch). Bracket
    // timing events on THIS stream and synchronize it before reading output().
    cudaStream_t stream() const { return stream_; }

    // Persistent post-conv activation [max_source_positions, d_model] (post-GELU,
    // permuted; == conv_out.bin), from the last forward(). Diagnostic seam.
    const float* conv_out() const { return conv_out_.get(); }

    const WhisperEncoderConfig& config() const { return cfg_; }
    // Valid output frames of the LAST forward() (the selected bucket's seq); falls
    // back to the full 30 s length before the first call.
    int output_frames() const { return last_seq_ ? last_seq_ : cfg_.max_source_positions; }
    // Real (unpadded) encoder frames for `mel_frames` of input — the valid prefix the
    // projector / sliding-window delta-slicer should trust inside a padded bucket.
    int valid_output_frames(int mel_frames) const {
        return WhisperEncoderConfig::conv_out_frames_of(mel_frames);
    }

private:
    // One captured graph for a given input length. The workspace buffers are sized
    // for the largest bucket, so a smaller bucket is a prefix of the same allocations.
    struct GraphBucket {
        int             conv_frames = 0;   // input mel frames this graph is captured for
        int             seq = 0;           // conv_out_frames_of(conv_frames) == output rows
        cudaGraph_t     graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool            ready = false;      // captured yet? (lazy on first covering call)
    };

    // WEIGHT PRECISION: the six GEMM matrices are fp16, everything else is fp32.
    //
    // WHY THIS IS NOT A NUMERICAL CHANGE. kComputeType is already
    // CUBLAS_COMPUTE_32F_FAST_16F (whisper_encoder.cu), which rounds BOTH
    // operands to fp16 before the tensor-core multiply and accumulates in fp32.
    // Storing these matrices as fp32 therefore bought nothing: cuBLAS threw the
    // low mantissa bits away on every call anyway. Holding them as __half feeds
    // the same numbers to the same kernel -- the rounding just happens once at
    // upload instead of once per GEMM.
    //
    // WHY IT MATTERS. These six are the model: 4 x [D,D] + [ffn,D] + [D,ffn] is
    // ~78.6 MB/layer in fp32, x32 layers = ~2.52 GB, and the Ultravox checkpoint
    // is BF16 on disk (1.375 GB) -- so the fp32 upload was DOUBLING a file that
    // never had those bits. Halving it returns ~1.26 GB, which is the difference
    // between fitting under the WDDM paging cliff on a 12 GB card and not.
    //
    // The bias/LayerNorm vectors stay fp32 deliberately: they are [D] or [ffn],
    // i.e. ~0.1% of the layer, and they are consumed by the fused epilogue
    // kernels rather than by cuBLAS -- halving them would save nothing
    // measurable and would put a rounding step in front of the residual adds,
    // which is where precision actually shows up.
    struct DeviceLayer {
        DeviceBuffer<float>  attn_ln_w, attn_ln_b;
        DeviceBuffer<__half> q_w, k_w, v_w, out_w;
        DeviceBuffer<float>  q_b, v_b, out_b;
        DeviceBuffer<float>  final_ln_w, final_ln_b;
        DeviceBuffer<__half> fc1_w, fc2_w;
        DeviceBuffer<float>  fc1_b, fc2_b;
    };

    // Issues the full forward op sequence on `stream_` for a `conv_frames`-frame
    // input (no input copy, no sync). Every extent derives from conv_frames, so the
    // eager warm-up and the graph capture are bit-identical for a given bucket.
    void record(cudaStream_t s, int conv_frames);
    // Y[T,O] = X[T,K] @ W[O,K]^T on cuBLAS (row-major via a transposed col-major
    // formulation); no bias (added by a fused epilogue).
    // The activation operand arrives fp32 and is narrowed into act_h_ first --
    // cublasGemmEx requires Atype == Btype, so an fp16 weight forces an fp16 X.
    // That narrowing is free in accuracy terms (see DeviceLayer) and costs one
    // elementwise kernel over at most [seq, ffn].
    void gemm_linear(const float* X, const __half* W, float* Y, int T, int K, int O,
                     cudaStream_t s);
    // C[M,N] = A[M,K] @ B[K,N] on cuBLAS (both row-major); the im2col conv GEMM.
    // A is the fp16 conv weight, B the fp32 im2col columns (narrowed as above).
    void gemm_ab(const __half* A, const float* B, float* C, int M, int N, int K,
                 cudaStream_t s);
    // Eager warm-up (resolve cuBLAS algorithms/workspace) at b.conv_frames, then
    // capture the graph into b.exec. Capture forbids new allocations, so the warm-up
    // must run first. Populates b.{graph,exec,ready}.
    void build_graph(GraphBucket& b);
    // Smallest captured bucket whose conv_frames covers mel_frames (clamped to the
    // largest); builds it lazily. buckets_ is sorted ascending.
    GraphBucket& select_bucket(int mel_frames);

    WhisperEncoderConfig cfg_;

    // GPU control objects (raw handles; freed in the dtor).
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    std::vector<GraphBucket> buckets_;  // sorted ascending by conv_frames; captured lazily
    int last_seq_ = 0;                  // valid output length of the last forward()

    // Weights (device). conv1/conv2 weights are GEMM operands (im2col lowering),
    // so they follow the same fp16 rule as the layer matrices; the biases do not.
    DeviceBuffer<__half> conv1_w_, conv2_w_;
    DeviceBuffer<float>  conv1_b_, conv2_b_;
    DeviceBuffer<float> embed_positions_;   // [max_source_positions, d_model] (full table)
    // Per-launch positional window the captured graph reads from: forward() copies
    // embed_positions_[enc_pos_offset .. +seq) into this fixed buffer before the graph
    // launch, so the absolute-position offset is runtime-selectable WITHOUT recapturing
    // (the graph baked a fixed pointer, not a fixed offset).
    DeviceBuffer<float> d_pos_window_;
    std::vector<DeviceLayer> dlayers_;
    DeviceBuffer<float> ln_post_w_, ln_post_b_;

    // Workspace (allocated in the ctor from the resolved geometry; forward()
    // never allocates). See whisper_encoder.cu for the role of each buffer.
    DeviceBuffer<float> d_input_;     // [num_mel_bins, conv_frames] (graph input)
    DeviceBuffer<float> cols1_;       // [num_mel_bins*3, conv_frames]  (im2col conv1)
    DeviceBuffer<float> cols2_;       // [d_model*3, conv_out_frames]   (im2col conv2)
    DeviceBuffer<float> conv1_out_;   // [d_model, conv_frames]
    DeviceBuffer<float> conv2_out_;   // [d_model, conv_out_frames]
    DeviceBuffer<float> conv_out_;    // [seq, d_model]  (permuted post-conv)
    DeviceBuffer<float> hidden_;      // [seq, d_model]  (residual stream)
    DeviceBuffer<float> normed_;      // [seq, d_model]  (LN output scratch)
    DeviceBuffer<float> q_, k_, v_;   // [seq, d_model]
    DeviceBuffer<float> attn_ctx_;    // [seq, d_model]  (attention context)
    DeviceBuffer<float> attn_proj_;   // [seq, d_model]  (out_proj result)
    DeviceBuffer<float> scores_;      // [num_heads, seq, seq]  (attention logits/probs)
    DeviceBuffer<float> mlp_hidden_;  // [seq, ffn_dim]
    DeviceBuffer<float> mlp_out_;     // [seq, d_model]
    DeviceBuffer<float> output_;      // [seq, d_model]  (post final LayerNorm)
    // Narrowed activation operand, shared by EVERY GEMM. One buffer is safe
    // because the convert and the GEMM that consumes it are consecutive on a
    // single stream, so the next convert cannot outrun the previous multiply --
    // the same ordering guarantee the captured graph already relies on. Sized
    // for the widest operand any GEMM presents: max(seq*ffn, d_model*3*seq).
    DeviceBuffer<__half> act_h_;
    DeviceBuffer<char> blas_ws_;      // cuBLAS workspace (fixed, capture-safe)

    bool weights_loaded_ = false;
};

}  // namespace blackwell::audio
