// =============================================================================
// bridge/prompt_injector.cu — splice audio embeddings into the text stream.
//
// Replaces the single <|audio|> row at audio_token_pos with the M projected
// audio rows, producing a [seq_len - 1 + M, hidden_dim] matrix:
//
//   dst row r  <-  src_text row r                 for r <  pos           (head)
//   dst row r  <-  audio    row (r - pos)         for pos <= r < pos+M   (audio)
//   dst row r  <-  src_text row (r - M + 1)       for r >= pos+M         (tail)
//
// Vectorised over float4 (hidden_dim % 4 == 0, and every row is 16-byte aligned
// because DeviceBuffer/cudaMalloc bases are). ZERO allocation: the caller owns
// dst; this launcher only validates + launches (Architectural Rule #3).
// =============================================================================
#include "bridge/prompt_injector.hpp"

#include <cuda_runtime.h>
#include <iostream>

namespace blackwell::bridge {
namespace {

// Runtime-tier CUDA check (Hybrid error doctrine): no throw across the decode
// path — log once and return an EngineStatus. Mirrors engine.cpp's macro.
#define BRIDGE_CUDA_RETURN(call)                                                       \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            std::cerr << "[blackwell_bridge] CUDA error: " << cudaGetErrorString(err__)\
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";                \
            return (err__ == cudaErrorMemoryAllocation)                                \
                       ? blackwell::EngineStatus::OutOfVram                            \
                       : blackwell::EngineStatus::CudaRuntimeError;                    \
        }                                                                              \
    } while (0)

// One float4 (== 4 contiguous embedding scalars) per logical element. A single
// grid-stride loop covers all new_seq_len * hidden_vec elements; the per-element
// div/mod resolves the destination row and selects its source region. Branch
// divergence is confined to the ~1 warp straddling each region boundary.
__global__ void splice_audio_embeddings_kernel(const float4* __restrict__ src_text,  // [seq_len, hidden_vec]
                                               const float4* __restrict__ audio,      // [M, hidden_vec]
                                               float4* __restrict__ dst,              // [new_seq_len, hidden_vec]
                                               int seq_len,
                                               int m_audio,
                                               int audio_token_pos,
                                               int hidden_vec,
                                               long long total_vec,
                                               int* __restrict__ out_seq_len) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec; i += stride) {
        const int row = static_cast<int>(i / hidden_vec);
        const int col = static_cast<int>(i - static_cast<long long>(row) * hidden_vec);

        long long src_index;
        if (row < audio_token_pos) {
            // head: verbatim text row
            src_index = static_cast<long long>(row) * hidden_vec + col;
            dst[i] = src_text[src_index];
        } else if (row < audio_token_pos + m_audio) {
            // audio block: projected row (row - pos)
            src_index = static_cast<long long>(row - audio_token_pos) * hidden_vec + col;
            dst[i] = audio[src_index];
        } else {
            // tail: text rows shifted by (M - 1) — the placeholder is consumed
            src_index = static_cast<long long>(row - m_audio + 1) * hidden_vec + col;
            dst[i] = src_text[src_index];
        }
    }

    // Publish the resulting length without a host round-trip (zero host alloc).
    if (out_seq_len != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
        *out_seq_len = seq_len - 1 + m_audio;
    }
    (void)seq_len;
}

}  // namespace

blackwell::EngineStatus inject_audio_embeddings(const AudioSpliceDesc& desc,
                                                const AudioEmbeddingsView& view,
                                                cudaStream_t stream) noexcept {
    using blackwell::EngineStatus;

    // ---- Preconditions (runtime tier: status, not asserts/allocation) -------
    if (!view.valid() || desc.d_text_embeddings == nullptr || desc.d_out_embeddings == nullptr) {
        return EngineStatus::InvalidArgument;
    }
    if (desc.hidden_dim <= 0 || (desc.hidden_dim & 3) != 0) {   // float4 contract
        return EngineStatus::InvalidArgument;
    }
    if (view.embed_dim != desc.hidden_dim) {                    // projection-space match
        return EngineStatus::InvalidArgument;
    }
    if (desc.seq_len <= 0 || desc.audio_token_pos < 0 || desc.audio_token_pos >= desc.seq_len) {
        return EngineStatus::InvalidArgument;
    }

    const int m_audio = view.num_tokens;
    const int new_seq_len = desc.seq_len - 1 + m_audio;         // grows by (M - 1)
    if (m_audio <= 0 || desc.out_capacity_rows < new_seq_len) {  // never grows the output
        return EngineStatus::InvalidArgument;
    }

    // ---- Launch (no allocation; fully stream-ordered) -----------------------
    const int hidden_vec = desc.hidden_dim >> 2;               // hidden_dim / 4
    const long long total_vec = static_cast<long long>(new_seq_len) * hidden_vec;

    constexpr int kBlock = 256;
    long long blocks = (total_vec + kBlock - 1) / kBlock;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;                        // grid-stride covers the rest

    splice_audio_embeddings_kernel<<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(desc.d_text_embeddings),
        reinterpret_cast<const float4*>(view.data),
        reinterpret_cast<float4*>(desc.d_out_embeddings),
        desc.seq_len, m_audio, desc.audio_token_pos, hidden_vec, total_vec,
        desc.d_out_seq_len);

    BRIDGE_CUDA_RETURN(cudaGetLastError());  // launch config / validity
    return EngineStatus::Success;            // async; caller syncs on `stream`
}

}  // namespace blackwell::bridge
