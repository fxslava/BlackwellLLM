#pragma once
// -----------------------------------------------------------------------------
// UltravoxProjector — full forward graph (audio frontend, preprocessor-only).
//
//   whisper_out [T,1280]
//     -> StackAudioFrames  [T', 10240]        (T' = ceil(T/stack_factor))
//     -> RMSNorm(ln_pre)   [T', 10240]        (eps 1e-6)
//     -> Linear_1          [T', 4096]         (W1 = [4096, 10240], bias-free)
//     -> SwiGLU            [T', 2048]         (Ultravox: gate = 2nd half)
//     -> RMSNorm(ln_mid)   [T', 2048]         (eps 1e-6, projector_ln_mid=True)
//     -> Linear_2          [T', 2048]         (W2 = [2048, 2048], bias-free)
//
// Geometry note: proj_hidden = 4096 (linear_1 out); SwiGLU halves it to 2048;
// linear_2 maps 2048 -> text_hidden (2048 for Llama-3.2-1B). This matches the
// verified weight shapes and audio_embeds.bin [188, 2048].
//
// STRICT: zero allocation in forward(). Every intermediate lives in
// ProjectorWorkspace, allocated once at construction; weights are uploaded once
// via load_weights(). forward() only launches kernels over pre-owned buffers.
//
// Numerics: the isolated projector weights are FP32, so RMSNorm and both linears
// run in FP32 (F.linear-equivalent, accumulate in float) to match the FP32
// PyTorch reference — the engine's quantized Linear kernels take bf16/fp8/int4
// weights, not these. Reuses the verified launch_stack_audio_frames / launch_swiglu.
// -----------------------------------------------------------------------------
#include <vector>

#include "device_buffer.h"          // blackwell::DeviceBuffer
#include "ultravox_projector.cuh"   // launch_stack_audio_frames, launch_swiglu

namespace blackwell::audio {

struct ProjectorConfig {
    int hidden_dim = 1280;      // whisper d_model
    int stack_factor = 8;
    int proj_hidden = 4096;     // linear_1 output
    int text_hidden = 2048;     // linear_2 output (== host LLM hidden)
    float eps = 1e-6f;          // LlamaRMSNorm epsilon

    int stacked_dim() const { return hidden_dim * stack_factor; }  // 10240
    int swiglu_out() const { return proj_hidden / 2; }             // 2048
};

// Pre-allocated intermediate activation buffers (STRICT rule #1).
struct ProjectorWorkspace {
    DeviceBuffer<float> stacked;   // [T', stacked_dim]
    DeviceBuffer<float> norm0;     // [T', stacked_dim]
    DeviceBuffer<float> linear1;   // [T', proj_hidden]
    DeviceBuffer<float> swiglu;    // [T', swiglu_out]
    DeviceBuffer<float> norm1;     // [T', swiglu_out]
    DeviceBuffer<float> out;       // [T', text_hidden]

    void allocate(const ProjectorConfig& c, int max_out_frames);
};

class UltravoxProjector {
public:
    // Allocates workspace sized for up to `max_input_frames` whisper frames.
    UltravoxProjector(const ProjectorConfig& cfg, int max_input_frames);

    // One-time weight upload (host FP32 -> device). Not on the forward path.
    void load_weights(const std::vector<float>& ln_pre,     // [stacked_dim]
                      const std::vector<float>& linear_1,   // [proj_hidden, stacked_dim]
                      const std::vector<float>& ln_mid,     // [swiglu_out]
                      const std::vector<float>& linear_2);  // [text_hidden, swiglu_out]

    // Runs the 6-stage graph for `num_frames` input frames (<= max_input_frames).
    // Returns a device pointer to [out_frames(num_frames), text_hidden]. Launches
    // only; allocates nothing.
    const float* forward(const float* d_whisper_out, int num_frames);

    int out_frames(int num_frames) const {
        return (num_frames + cfg_.stack_factor - 1) / cfg_.stack_factor;
    }
    int text_hidden() const { return cfg_.text_hidden; }

private:
    ProjectorConfig cfg_;
    int max_input_frames_;
    DeviceBuffer<float> w_ln_pre_, w_linear_1_, w_ln_mid_, w_linear_2_;
    ProjectorWorkspace ws_;
};

}  // namespace blackwell::audio
