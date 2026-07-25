#include "prompt_injector.cuh"

#include <cuda_runtime.h>

// One thread copies one float4 of one output row; the row index selects its
// source region (text prefix / audio block / text suffix). Pure gather-copy, so
// it is exact -- cosine parity is 1.0 by construction.

namespace {

constexpr int kBlock = 256;

__global__ void splice_audio_embeddings_kernel(const float4* __restrict__ text,
                                               const float4* __restrict__ audio,
                                               float4* __restrict__ out,
                                               int audio_pos,
                                               int num_audio,
                                               int vecH,          // hidden / 4
                                               long long total_vec) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_vec) return;

    const int d4 = (int)(i % vecH);
    const long long r = i / vecH;                 // output row

    const float4* src;
    if (r < audio_pos)                            // text prefix [0, audio_pos)
        src = text + (long long)r * vecH;
    else if (r < (long long)audio_pos + num_audio)  // audio block
        src = audio + (r - audio_pos) * vecH;
    else                                          // text suffix (skip the placeholder)
        src = text + (r - num_audio + 1) * vecH;

    out[i] = src[d4];
}

}  // namespace

void inject_audio_embeddings(const float* d_text,
                             const float* d_audio,
                             float* d_out,
                             int seq_len,
                             int audio_pos,
                             int num_audio,
                             int hidden) {
    const int vecH = hidden / 4;
    const int out_rows = seq_len - 1 + num_audio;
    const long long total_vec = (long long)out_rows * vecH;
    const unsigned int blocks = (unsigned int)((total_vec + kBlock - 1) / kBlock);

    splice_audio_embeddings_kernel<<<blocks, kBlock>>>(
        reinterpret_cast<const float4*>(d_text),
        reinterpret_cast<const float4*>(d_audio),
        reinterpret_cast<float4*>(d_out),
        audio_pos, num_audio, vecH, total_vec);
}
