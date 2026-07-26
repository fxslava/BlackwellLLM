#include "ultravox_projector_pipeline.cuh"

#include <cuda_runtime.h>

#include "common.h"   // CUDA_CHECK_THROW

// FP32 RMSNorm + Linear helpers for the isolated-projector-weights parity path
// (STRICT rule #3's float32 escape hatch). The verified StackAudioFrames / SwiGLU
// kernels in ultravox_projector.cu are reused unchanged (STRICT rule #2).

namespace {

constexpr int kBlock = 256;

// LlamaRMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight. One block per row.
__global__ void rmsnorm_f32_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   float* __restrict__ y,
                                   int rows, int H, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * H;
    float* yr = y + (size_t)row * H;

    __shared__ float red[kBlock];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = xr[i];
        local += v * v;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(red[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yr[i] = xr[i] * inv_rms * w[i];
}

// Y[T,O] = X[T,K] @ W[O,K]^T  (F.linear with bias-free weight [O,K]). One thread
// per output element; float4 over the contiguous K (K % 4 == 0). Accumulates in
// float to mirror the FP32 PyTorch reference.
__global__ void linear_wt_f32_kernel(const float* __restrict__ X,
                                     const float* __restrict__ W,
                                     float* __restrict__ Y,
                                     int T, int K, int O) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (o >= O || t >= T) return;

    const float4* xr = reinterpret_cast<const float4*>(X + (size_t)t * K);
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)o * K);
    const int K4 = K >> 2;
    float acc = 0.f;
    for (int k = 0; k < K4; ++k) {
        const float4 a = xr[k], b = wr[k];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    Y[(size_t)t * O + o] = acc;
}

void launch_rmsnorm_f32(const float* d_x, const float* d_w, float* d_y,
                        int rows, int H, float eps, cudaStream_t stream) {
    rmsnorm_f32_kernel<<<rows, kBlock, 0, stream>>>(d_x, d_w, d_y, rows, H, eps);
}

void launch_linear_wt_f32(const float* d_x, const float* d_w, float* d_y,
                          int T, int K, int O, cudaStream_t stream) {
    dim3 grid((O + kBlock - 1) / kBlock, T);
    linear_wt_f32_kernel<<<grid, kBlock, 0, stream>>>(d_x, d_w, d_y, T, K, O);
}

}  // namespace

namespace blackwell::audio {

void ProjectorWorkspace::allocate(const ProjectorConfig& c, int max_out_frames) {
    stacked.allocate((size_t)max_out_frames * c.stacked_dim());
    norm0.allocate((size_t)max_out_frames * c.stacked_dim());
    linear1.allocate((size_t)max_out_frames * c.proj_hidden);
    swiglu.allocate((size_t)max_out_frames * c.swiglu_out());
    norm1.allocate((size_t)max_out_frames * c.swiglu_out());
    out.allocate((size_t)max_out_frames * c.text_hidden);
}

UltravoxProjector::UltravoxProjector(const ProjectorConfig& cfg, int max_input_frames)
    : cfg_(cfg), max_input_frames_(max_input_frames) {
    ws_.allocate(cfg_, out_frames(max_input_frames));
}

void UltravoxProjector::load_weights(const std::vector<float>& ln_pre,
                                     const std::vector<float>& linear_1,
                                     const std::vector<float>& ln_mid,
                                     const std::vector<float>& linear_2) {
    auto upload = [](DeviceBuffer<float>& d, const std::vector<float>& h) {
        d.allocate(h.size());
        CUDA_CHECK_THROW(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
    };
    upload(w_ln_pre_, ln_pre);
    upload(w_linear_1_, linear_1);
    upload(w_ln_mid_, ln_mid);
    upload(w_linear_2_, linear_2);
}

const float* UltravoxProjector::forward(const float* d_whisper_out, int num_frames,
                                        cudaStream_t stream) {
    const int T = out_frames(num_frames);
    const int stacked = cfg_.stacked_dim();
    const int mid = cfg_.swiglu_out();

    // 1. StackAudioFrames  [num_frames,1280] -> [T,10240]
    launch_stack_audio_frames(d_whisper_out, ws_.stacked, num_frames,
                              cfg_.hidden_dim, cfg_.stack_factor, stream);
    // 2. RMSNorm(ln_pre)   -> [T,10240]
    launch_rmsnorm_f32(ws_.stacked, w_ln_pre_, ws_.norm0, T, stacked, cfg_.eps, stream);
    // 3. Linear_1          -> [T,4096]
    launch_linear_wt_f32(ws_.norm0, w_linear_1_, ws_.linear1, T, stacked, cfg_.proj_hidden, stream);
    // 4. SwiGLU            -> [T,2048]
    launch_swiglu(ws_.linear1, ws_.swiglu, T, cfg_.proj_hidden, stream);
    // 5. RMSNorm(ln_mid)   -> [T,2048]
    launch_rmsnorm_f32(ws_.swiglu, w_ln_mid_, ws_.norm1, T, mid, cfg_.eps, stream);
    // 6. Linear_2          -> [T, text_hidden]  (4096 for the 8B default, 2048 for 1B)
    launch_linear_wt_f32(ws_.norm1, w_linear_2_, ws_.out, T, mid, cfg_.text_hidden, stream);

    return ws_.out.get();
}

}  // namespace blackwell::audio
