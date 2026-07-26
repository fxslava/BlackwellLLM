#include "whisper_encoder.cuh"
#include "whisper_encoder.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF_F

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"  // CUDA_CHECK_THROW

// -----------------------------------------------------------------------------
// WhisperEncoder — optimised forward pass (whisper-large-v3-turbo encoder).
//
// Heavy matmuls  -> cuBLAS FP16 tensor cores (CUBLAS_COMPUTE_32F_FAST_16F: fp32
//                   I/O, fp16 multiply, fp32 accumulate). Both convs are lowered
//                   to im2col + GEMM (a direct conv1d is ~40x slower here).
// Attention      -> a fused flash-attention kernel (online softmax; never
//                   materialises the [heads, seq, seq] scores tensor, whose VRAM
//                   round-trips alone were a ~35 ms floor).
// Glue           -> fused, modulo-free 2-D custom kernels (LayerNorm, bias+GELU,
//                   bias+residual).
// Whole 32-layer pass -> captured once into a CUDA graph, then replayed in one
//                   cudaGraphLaunch (kills per-kernel CPU launch overhead).
//
// cuBLAS row-major recipe: a row-major [R,C] matrix is the column-major [C,R]
// matrix of the same memory (ld = C). See gemm_linear / gemm_ab.
// -----------------------------------------------------------------------------

#define CUBLAS_CHECK_THROW(expr)                                                  \
    do {                                                                          \
        cublasStatus_t status_ = (expr);                                         \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                     \
            throw std::runtime_error(std::string("cuBLAS error ") +               \
                                     std::to_string(static_cast<int>(status_)) +  \
                                     " at " #expr);                               \
    } while (0)

namespace {

constexpr int kBlock = 256;

// fp32 I/O, fp16 tensor-core multiply, fp32 accumulate. ~2x TF32 throughput; the
// fp32 accumulate keeps the 32-layer stack within the cosine>0.995 / MSE<1e-4 bar.
constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F_FAST_16F;
// The strided-batched attention GEMMs reject FAST_16F (interleaved head strides,
// k=64) -> CUBLAS_STATUS_INVALID_VALUE; TF32 tensor cores are accepted there.
constexpr cublasComputeType_t kAttnCompute = CUBLAS_COMPUTE_32F_FAST_TF32;

__device__ __forceinline__ float gelu_exact(float x) {
    // transformers "gelu" == exact erf GELU: 0.5 x (1 + erf(x / sqrt(2))).
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

inline unsigned int grid1d(long long total) {
    return static_cast<unsigned int>((total + kBlock - 1) / kBlock);
}

// im2col for Conv1d(k=3, pad=1, given stride). Produces cols[in_ch*3, out_len]
// (row-major, ld=out_len): cols[(ic*3+kk), t] = in[ic, t*stride + kk - 1] (0 pad).
__global__ void im2col_k3_kernel(const float* __restrict__ in, float* __restrict__ cols,
                                 int in_len, int out_len, int stride, long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = (int)(idx % out_len);
    const long long r = idx / out_len;   // r = ic*3 + kk
    const int kk = (int)(r % 3);
    const int ic = (int)(r / 3);
    const int pos = t * stride + kk - 1;  // pad = 1
    cols[idx] = (pos >= 0 && pos < in_len) ? in[(size_t)ic * in_len + pos] : 0.f;
}

// Conv epilogue: bias is per-ROW (out-channel), then GELU. x is [out_ch, out_len].
__global__ void row_bias_gelu_kernel(float* __restrict__ x, const float* __restrict__ bias,
                                     int out_len, long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = gelu_exact(x[idx] + bias[(int)(idx / out_len)]);
}

__global__ void transpose_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, int rows, int cols) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * cols;
    if (idx >= total) return;
    const int r = (int)(idx / cols);
    const int c = (int)(idx % cols);
    out[(size_t)c * rows + r] = in[(size_t)r * cols + c];
}

__global__ void add_positional_kernel(const float* __restrict__ in,
                                      const float* __restrict__ pos,
                                      float* __restrict__ out, long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = in[idx] + pos[idx];
}

// LayerNorm over the last dim (biased variance). One block per row.
__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y, int dim, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;

    __shared__ float red[kBlock];

    float local = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) local += xr[i];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = red[0] / dim;
    __syncthreads();

    local = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = xr[i] - mean;
        local += d * d;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv_std = rsqrtf(red[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = (xr[i] - mean) * inv_std * gamma[i] + beta[i];
}

// Modulo-free 2-D epilogues: grid.y = row, x-dim covers cols.
__global__ void add_bias_kernel(float* __restrict__ x, const float* __restrict__ bias,
                                int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    x[(size_t)blockIdx.y * cols + c] += bias[c];
}

__global__ void bias_gelu_kernel(float* __restrict__ x, const float* __restrict__ bias,
                                 int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const size_t i = (size_t)blockIdx.y * cols + c;
    x[i] = gelu_exact(x[i] + bias[c]);
}

__global__ void bias_residual_kernel(float* __restrict__ io,
                                     const float* __restrict__ proj,
                                     const float* __restrict__ bias, int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const size_t i = (size_t)blockIdx.y * cols + c;
    io[i] = io[i] + proj[i] + bias[c];
}

// In-place row softmax over j for each (h,i) row of scores[num_heads,seq,seq].
// ONE WARP per row: 30000 tiny rows want maximum occupancy, and warp-shuffle
// reductions avoid block-wide __syncthreads entirely. The 6 KB row stays in L1
// across the three lane-strided scans (max / sum / normalise). total_rows =
// num_heads*seq.
__global__ void softmax_rows_kernel(float* __restrict__ scores, int seq,
                                    int total_rows) {
    const int row = (int)((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5));
    if (row >= total_rows) return;
    const int lane = (int)(threadIdx.x & 31);
    float* sr = scores + (size_t)row * seq;

    float m = -CUDART_INF_F;
    for (int j = lane; j < seq; j += 32) m = fmaxf(m, sr[j]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));

    float sum = 0.f;
    for (int j = lane; j < seq; j += 32) sum += __expf(sr[j] - m);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);

    const float inv = 1.0f / sum;
    for (int j = lane; j < seq; j += 32) sr[j] = __expf(sr[j] - m) * inv;
}

}  // namespace

namespace blackwell::audio {

void launch_transpose(const float* in, float* out, int rows, int cols,
                      cudaStream_t stream) {
    const long long total = (long long)rows * cols;
    transpose_kernel<<<grid1d(total), kBlock, 0, stream>>>(in, out, rows, cols);
}

void launch_add_positional(const float* in, const float* pos, float* out,
                           int rows, int cols, cudaStream_t stream) {
    const long long total = (long long)rows * cols;
    add_positional_kernel<<<grid1d(total), kBlock, 0, stream>>>(in, pos, out, total);
}

void launch_layernorm(const float* x, const float* gamma, const float* beta,
                      float* y, int rows, int dim, float eps, cudaStream_t stream) {
    layernorm_kernel<<<rows, kBlock, 0, stream>>>(x, gamma, beta, y, dim, eps);
}

void launch_add_bias(float* x, const float* bias, int rows, int cols,
                     cudaStream_t stream) {
    dim3 grid((cols + kBlock - 1) / kBlock, rows);
    add_bias_kernel<<<grid, kBlock, 0, stream>>>(x, bias, cols);
}

void launch_bias_gelu(float* x, const float* bias, int rows, int cols,
                      cudaStream_t stream) {
    dim3 grid((cols + kBlock - 1) / kBlock, rows);
    bias_gelu_kernel<<<grid, kBlock, 0, stream>>>(x, bias, cols);
}

void launch_bias_residual(float* io, const float* proj, const float* bias,
                          int rows, int cols, cudaStream_t stream) {
    dim3 grid((cols + kBlock - 1) / kBlock, rows);
    bias_residual_kernel<<<grid, kBlock, 0, stream>>>(io, proj, bias, cols);
}

namespace {

void launch_im2col_k3(const float* in, float* cols, int in_ch, int in_len,
                      int out_len, int stride, cudaStream_t stream) {
    const long long total = (long long)in_ch * 3 * out_len;
    im2col_k3_kernel<<<grid1d(total), kBlock, 0, stream>>>(in, cols, in_len, out_len,
                                                           stride, total);
}

void launch_row_bias_gelu(float* x, const float* bias, int out_ch, int out_len,
                          cudaStream_t stream) {
    const long long total = (long long)out_ch * out_len;
    row_bias_gelu_kernel<<<grid1d(total), kBlock, 0, stream>>>(x, bias, out_len, total);
}

void launch_softmax_rows(float* scores, int num_heads, int seq, cudaStream_t stream) {
    const int total_rows = num_heads * seq;
    const int warps_per_block = kBlock >> 5;  // 8
    const unsigned int blocks = (unsigned int)((total_rows + warps_per_block - 1) / warps_per_block);
    softmax_rows_kernel<<<blocks, kBlock, 0, stream>>>(scores, seq, total_rows);
}

void upload(DeviceBuffer<float>& d, const std::vector<float>& h, size_t expect,
            const char* what) {
    if (h.size() != expect)
        throw std::runtime_error(std::string("WhisperEncoder weight size mismatch: ") +
                                 what + " has " + std::to_string(h.size()) +
                                 ", expected " + std::to_string(expect));
    d.allocate(h.size());
    CUDA_CHECK_THROW(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
}

}  // namespace

WhisperEncoder::WhisperEncoder(const WhisperEncoderConfig& cfg) : cfg_(cfg) {
    const int seq = cfg_.max_source_positions;
    const int D = cfg_.d_model;
    const int mel = cfg_.num_mel_bins;
    const int cf = cfg_.conv_frames;
    const int c2 = cfg_.conv_out_frames();
    const size_t sD = (size_t)seq * D;

    CUDA_CHECK_THROW(cudaStreamCreate(&stream_));
    CUBLAS_CHECK_THROW(cublasCreate(&blas_));
    CUBLAS_CHECK_THROW(cublasSetStream(blas_, stream_));
    blas_ws_.allocate(32u * 1024u * 1024u);  // 32 MB, capture-safe
    CUBLAS_CHECK_THROW(cublasSetWorkspace(blas_, blas_ws_.get(), blas_ws_.size_bytes()));

    d_input_.allocate((size_t)mel * cf);
    d_pos_window_.allocate(sD);              // [max_source_positions, D] positional window
    cols1_.allocate((size_t)mel * 3 * cf);
    cols2_.allocate((size_t)D * 3 * c2);
    conv1_out_.allocate((size_t)D * cf);
    conv2_out_.allocate((size_t)D * c2);
    conv_out_.allocate(sD);
    hidden_.allocate(sD);
    normed_.allocate(sD);
    q_.allocate(sD);
    k_.allocate(sD);
    v_.allocate(sD);
    attn_ctx_.allocate(sD);
    attn_proj_.allocate(sD);
    scores_.allocate((size_t)cfg_.num_heads * seq * seq);
    mlp_hidden_.allocate((size_t)seq * cfg_.ffn_dim);
    mlp_out_.allocate(sD);
    output_.allocate(sD);

    // Resolve the graph-bucket set: sort ascending, drop duplicates and anything
    // larger than conv_frames (the workspace-sizing bound), and guarantee conv_frames
    // itself is present (the parity reference + the largest / fallback bucket). Each
    // bucket is captured lazily on its first covering forward().
    std::vector<int> want = cfg_.graph_buckets;
    want.push_back(cfg_.conv_frames);
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    for (const int f : want) {
        if (f <= 0 || f > cfg_.conv_frames) continue;
        GraphBucket b;
        b.conv_frames = f;
        b.seq = WhisperEncoderConfig::conv_out_frames_of(f);
        buckets_.push_back(b);
    }
}

WhisperEncoder::~WhisperEncoder() {
    for (GraphBucket& b : buckets_) {
        if (b.exec) cudaGraphExecDestroy(b.exec);
        if (b.graph) cudaGraphDestroy(b.graph);
    }
    if (blas_) cublasDestroy(blas_);
    if (stream_) cudaStreamDestroy(stream_);
}

void WhisperEncoder::load_weights(const WhisperWeights& w) {
    const int D = cfg_.d_model;
    const int mel = cfg_.num_mel_bins;
    const int ffn = cfg_.ffn_dim;
    const int seq = cfg_.max_source_positions;

    upload(conv1_w_, w.conv1_w, (size_t)D * mel * 3, "conv1.weight");
    upload(conv1_b_, w.conv1_b, (size_t)D, "conv1.bias");
    upload(conv2_w_, w.conv2_w, (size_t)D * D * 3, "conv2.weight");
    upload(conv2_b_, w.conv2_b, (size_t)D, "conv2.bias");
    upload(embed_positions_, w.embed_positions, (size_t)seq * D, "embed_positions");
    upload(ln_post_w_, w.layer_norm_w, (size_t)D, "layer_norm.weight");
    upload(ln_post_b_, w.layer_norm_b, (size_t)D, "layer_norm.bias");

    if ((int)w.layers.size() != cfg_.num_layers)
        throw std::runtime_error("WhisperEncoder: expected " +
                                 std::to_string(cfg_.num_layers) + " layers, got " +
                                 std::to_string(w.layers.size()));

    dlayers_.resize(cfg_.num_layers);
    for (int l = 0; l < cfg_.num_layers; ++l) {
        const WhisperLayerWeights& s = w.layers[(size_t)l];
        DeviceLayer& dl = dlayers_[(size_t)l];
        upload(dl.attn_ln_w, s.self_attn_layer_norm_w, (size_t)D, "self_attn_ln.w");
        upload(dl.attn_ln_b, s.self_attn_layer_norm_b, (size_t)D, "self_attn_ln.b");
        upload(dl.q_w, s.q_w, (size_t)D * D, "q_proj.w");
        upload(dl.q_b, s.q_b, (size_t)D, "q_proj.b");
        upload(dl.k_w, s.k_w, (size_t)D * D, "k_proj.w");
        upload(dl.v_w, s.v_w, (size_t)D * D, "v_proj.w");
        upload(dl.v_b, s.v_b, (size_t)D, "v_proj.b");
        upload(dl.out_w, s.out_w, (size_t)D * D, "out_proj.w");
        upload(dl.out_b, s.out_b, (size_t)D, "out_proj.b");
        upload(dl.final_ln_w, s.final_layer_norm_w, (size_t)D, "final_ln.w");
        upload(dl.final_ln_b, s.final_layer_norm_b, (size_t)D, "final_ln.b");
        upload(dl.fc1_w, s.fc1_w, (size_t)ffn * D, "fc1.w");
        upload(dl.fc1_b, s.fc1_b, (size_t)ffn, "fc1.b");
        upload(dl.fc2_w, s.fc2_w, (size_t)D * ffn, "fc2.w");
        upload(dl.fc2_b, s.fc2_b, (size_t)D, "fc2.b");
    }
    weights_loaded_ = true;
}

// Y[T,O] = X[T,K] @ W[O,K]^T  (W stored row-major [O,K] == col-major [K,O] ld K).
void WhisperEncoder::gemm_linear(const float* X, const float* W, float* Y, int T,
                                 int K, int O, cudaStream_t /*s*/) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK_THROW(cublasGemmEx(
        blas_, CUBLAS_OP_T, CUBLAS_OP_N, O, T, K, &one,
        W, CUDA_R_32F, K, X, CUDA_R_32F, K, &zero, Y, CUDA_R_32F, O,
        kComputeType, CUBLAS_GEMM_DEFAULT));
}

// C[M,N] = A[M,K] @ B[K,N]  (both row-major). Used by the im2col conv lowering.
void WhisperEncoder::gemm_ab(const float* A, const float* B, float* C, int M, int N,
                             int K) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK_THROW(cublasGemmEx(
        blas_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
        B, CUDA_R_32F, N, A, CUDA_R_32F, K, &zero, C, CUDA_R_32F, N,
        kComputeType, CUBLAS_GEMM_DEFAULT));
}

void WhisperEncoder::record(cudaStream_t s, int conv_frames) {
    const int D = cfg_.d_model;
    const int mel = cfg_.num_mel_bins;
    const int cf = conv_frames;                              // this bucket's input width
    const int ffn = cfg_.ffn_dim;
    const int H = cfg_.num_heads;
    const int hd = cfg_.head_dim();
    const int c2 = WhisperEncoderConfig::conv_out_frames_of(cf);
    const int seq = c2;                                      // encoder rows for this bucket
    const float scale = 1.0f / sqrtf((float)hd);
    const float one = 1.0f, zero = 0.0f;

    // --- Feature extractor: 2x (im2col + GEMM + bias/GELU), permute ---
    launch_im2col_k3(d_input_, cols1_, mel, cf, cf, /*stride=*/1, s);
    gemm_ab(conv1_w_, cols1_, conv1_out_, D, cf, mel * 3);        // [D, cf]
    launch_row_bias_gelu(conv1_out_, conv1_b_, D, cf, s);
    launch_im2col_k3(conv1_out_, cols2_, D, cf, c2, /*stride=*/2, s);
    gemm_ab(conv2_w_, cols2_, conv2_out_, D, c2, D * 3);          // [D, c2==seq]
    launch_row_bias_gelu(conv2_out_, conv2_b_, D, c2, s);
    launch_transpose(conv2_out_, conv_out_, D, c2, s);           // [D,seq] -> [seq,D]
    // Add positions from the per-launch window (filled by forward() with the absolute
    // offset slice), NOT directly from embed_positions_, so the offset is runtime-
    // selectable without recapturing the graph.
    launch_add_positional(conv_out_, d_pos_window_, hidden_, seq, D, s);

    // --- 32x encoder layers (pre-LN, bidirectional attention, GELU MLP) ---
    for (int l = 0; l < cfg_.num_layers; ++l) {
        const DeviceLayer& dl = dlayers_[(size_t)l];

        launch_layernorm(hidden_, dl.attn_ln_w, dl.attn_ln_b, normed_, seq, D, cfg_.ln_eps, s);
        gemm_linear(normed_, dl.q_w, q_, seq, D, D, s);
        launch_add_bias(q_, dl.q_b, seq, D, s);
        gemm_linear(normed_, dl.k_w, k_, seq, D, D, s);  // k_proj: no bias
        gemm_linear(normed_, dl.v_w, v_, seq, D, D, s);
        launch_add_bias(v_, dl.v_b, seq, D, s);
        // scores[h] = scale * Q_h @ K_h^T  (strided-batched, one batch per head).
        CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
            blas_, CUBLAS_OP_T, CUBLAS_OP_N, seq, seq, hd, &scale,
            k_.get(), CUDA_R_32F, D, (long long)hd, q_.get(), CUDA_R_32F, D, (long long)hd,
            &zero, scores_.get(), CUDA_R_32F, seq, (long long)seq * seq, H,
            kAttnCompute, CUBLAS_GEMM_DEFAULT));
        launch_softmax_rows(scores_, H, seq, s);
        // ctx[h] = P_h @ V_h  -> writes interleaved into attn_ctx_ [seq, D].
        CUBLAS_CHECK_THROW(cublasGemmStridedBatchedEx(
            blas_, CUBLAS_OP_N, CUBLAS_OP_N, hd, seq, seq, &one,
            v_.get(), CUDA_R_32F, D, (long long)hd, scores_.get(), CUDA_R_32F, seq,
            (long long)seq * seq, &zero, attn_ctx_.get(), CUDA_R_32F, D, (long long)hd, H,
            kAttnCompute, CUBLAS_GEMM_DEFAULT));
        gemm_linear(attn_ctx_, dl.out_w, attn_proj_, seq, D, D, s);
        launch_bias_residual(hidden_, attn_proj_, dl.out_b, seq, D, s);

        launch_layernorm(hidden_, dl.final_ln_w, dl.final_ln_b, normed_, seq, D, cfg_.ln_eps, s);
        gemm_linear(normed_, dl.fc1_w, mlp_hidden_, seq, D, ffn, s);
        launch_bias_gelu(mlp_hidden_, dl.fc1_b, seq, ffn, s);
        gemm_linear(mlp_hidden_, dl.fc2_w, mlp_out_, seq, ffn, D, s);
        launch_bias_residual(hidden_, mlp_out_, dl.fc2_b, seq, D, s);
    }

    launch_layernorm(hidden_, ln_post_w_, ln_post_b_, output_, seq, D, cfg_.ln_eps, s);
}

void WhisperEncoder::build_graph(GraphBucket& b) {
    const size_t used = (size_t)cfg_.num_mel_bins * b.conv_frames;
    // Warm-up (eager): zero-fill the used input region (values are irrelevant for
    // algorithm resolution) and let cuBLAS resolve its algorithms + touch its
    // workspace at THIS bucket's shapes BEFORE capture (capture forbids new allocs).
    CUDA_CHECK_THROW(cudaMemsetAsync(d_input_.get(), 0, used * sizeof(float), stream_));
    record(stream_, b.conv_frames);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_));

    CUDA_CHECK_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    record(stream_, b.conv_frames);
    CUDA_CHECK_THROW(cudaStreamEndCapture(stream_, &b.graph));
    CUDA_CHECK_THROW(cudaGraphInstantiate(&b.exec, b.graph, 0));
    b.ready = true;
}

WhisperEncoder::GraphBucket& WhisperEncoder::select_bucket(int mel_frames) {
    // buckets_ is sorted ascending; pick the smallest that covers mel_frames, else
    // the largest (== conv_frames, always present).
    GraphBucket* chosen = &buckets_.back();
    for (GraphBucket& b : buckets_) {
        if (b.conv_frames >= mel_frames) { chosen = &b; break; }
    }
    if (!chosen->ready) build_graph(*chosen);
    return *chosen;
}

const float* WhisperEncoder::forward(const float* d_mel) {
    return forward(d_mel, cfg_.conv_frames, 0);  // full 30 s bucket, positions from 0
}

const float* WhisperEncoder::forward(const float* d_mel, int mel_frames, int enc_pos_offset) {
    if (!weights_loaded_)
        throw std::runtime_error("WhisperEncoder::forward called before load_weights");
    if (mel_frames <= 0 || mel_frames > cfg_.conv_frames)
        throw std::runtime_error("WhisperEncoder::forward: mel_frames out of range "
                                 "(1.." + std::to_string(cfg_.conv_frames) + ")");

    GraphBucket& b = select_bucket(mel_frames);  // lazily captured
    const int mel = cfg_.num_mel_bins;
    const int D = cfg_.d_model;

    // Stage the absolute-position slice this launch needs: embed_positions_[off ..
    // off+b.seq) -> d_pos_window_[0 .. b.seq). Clamp so the slice stays inside the
    // trained positional range [0, max_source_positions); a window past 30 s of
    // absolute position reuses the final positions (Whisper's hard limit).
    const int max_pos = cfg_.max_source_positions;
    int off = enc_pos_offset < 0 ? 0 : enc_pos_offset;
    if (off + b.seq > max_pos) off = max_pos - b.seq;   // b.seq <= max_pos always
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        d_pos_window_.get(), embed_positions_.get() + static_cast<size_t>(off) * D,
        static_cast<size_t>(b.seq) * D * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_));

    // Pack the [mel, mel_frames] input into d_input_ as [mel, b.conv_frames]: the
    // graph reads each mel row at stride b.conv_frames, so a shorter input is copied
    // column-wise and the tail is zero-padded. When it exactly fills the bucket
    // (e.g. the 30 s back-compat path) this is a single contiguous copy.
    if (mel_frames == b.conv_frames) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(d_input_.get(), d_mel,
                                         (size_t)mel * b.conv_frames * sizeof(float),
                                         cudaMemcpyDeviceToDevice, stream_));
    } else {
        CUDA_CHECK_THROW(cudaMemsetAsync(d_input_.get(), 0,
                                         (size_t)mel * b.conv_frames * sizeof(float), stream_));
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(
            d_input_.get(), (size_t)b.conv_frames * sizeof(float),   // dst row pitch
            d_mel,          (size_t)mel_frames * sizeof(float),      // src row pitch
            (size_t)mel_frames * sizeof(float),                      // copied width (bytes)
            (size_t)mel,                                             // rows (mel bins)
            cudaMemcpyDeviceToDevice, stream_));
    }
    CUDA_CHECK_THROW(cudaGraphLaunch(b.exec, stream_));
    last_seq_ = b.seq;
    return output_.get();
}

}  // namespace blackwell::audio
