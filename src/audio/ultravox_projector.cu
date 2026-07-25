#include "ultravox_projector.cuh"

#include <cuda_runtime.h>

// Ultravox projector kernels. Pure FP32 (no BF16 truncation like the text
// swiglu.cu) because the golden reference is the FP32 PyTorch pass. All buffers
// are caller-owned; nothing is allocated here.

namespace {

constexpr int kBlock = 256;

// StackAudioFrames. One thread copies one float4 (4 contiguous channels) of the
// output. The output float4 index equals the flattened thread index by
// construction (each output row is cf*hidden_dim floats = cf*vecH float4s), so
// the decode below only needs to recover which input frame feeds it.
__global__ void stack_audio_frames_kernel(const float4* __restrict__ in,
                                          float4* __restrict__ out,
                                          int num_frames,
                                          int vecH,          // hidden_dim / 4
                                          int cf,            // compression factor
                                          long long total_vec) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_vec) return;

    int d4 = (int)(i % vecH);          // channel-quad within the frame
    long long rem = i / vecH;
    int c = (int)(rem % cf);           // which of the cf stacked frames
    long long tprime = rem / cf;       // output frame
    long long in_frame = tprime * cf + c;

    if (in_frame < num_frames)
        out[i] = in[in_frame * vecH + d4];
    else
        out[i] = make_float4(0.f, 0.f, 0.f, 0.f);   // pad tail
}

__device__ __forceinline__ float silu(float v) {
    // SiLU(v) = v * sigmoid(v) = v / (1 + e^-v). __expf matches the fast-math
    // path the kernels build with; parity is well within the 0.999 bar.
    return v / (1.0f + __expf(-v));
}

// SwiGLU (Ultravox: out = SiLU(second_half) * first_half). One thread produces
// one float4 of the output; it reads the matching float4 from each half of the
// same input row.
__global__ void swiglu_kernel(const float4* __restrict__ in,
                              float4* __restrict__ out,
                              int input_dim_vec,   // input_dim / 4
                              int mid_vec,         // (input_dim/2) / 4
                              long long total_vec) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_vec) return;

    int j4 = (int)(i % mid_vec);       // channel-quad within the output row
    long long t = i / mid_vec;         // token
    const float4* row = in + t * input_dim_vec;

    float4 x = row[j4];                // first half  -> multiplicand
    float4 g = row[mid_vec + j4];      // second half -> gate

    float4 o;
    o.x = silu(g.x) * x.x;
    o.y = silu(g.y) * x.y;
    o.z = silu(g.z) * x.z;
    o.w = silu(g.w) * x.w;
    out[i] = o;
}

inline unsigned int grid_for(long long total_vec) {
    return (unsigned int)((total_vec + kBlock - 1) / kBlock);
}

}  // namespace

void launch_stack_audio_frames(const float* d_in,
                               float* d_out,
                               int num_frames,
                               int hidden_dim,
                               int compression_factor) {
    const int out_frames = (num_frames + compression_factor - 1) / compression_factor;
    const int vecH = hidden_dim / 4;
    const long long total_vec = (long long)out_frames * compression_factor * vecH;

    stack_audio_frames_kernel<<<grid_for(total_vec), kBlock>>>(
        reinterpret_cast<const float4*>(d_in),
        reinterpret_cast<float4*>(d_out),
        num_frames, vecH, compression_factor, total_vec);
}

void launch_swiglu(const float* d_in,
                   float* d_out,
                   int num_tokens,
                   int input_dim) {
    const int mid = input_dim / 2;
    const int input_dim_vec = input_dim / 4;
    const int mid_vec = mid / 4;
    const long long total_vec = (long long)num_tokens * mid_vec;

    swiglu_kernel<<<grid_for(total_vec), kBlock>>>(
        reinterpret_cast<const float4*>(d_in),
        reinterpret_cast<float4*>(d_out),
        input_dim_vec, mid_vec, total_vec);
}
