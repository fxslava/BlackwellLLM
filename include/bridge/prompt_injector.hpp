#pragma once
// =============================================================================
// bridge/prompt_injector.hpp
//
// Splices projected audio embeddings into the text embedding matrix before
// Prefill: the single `<|audio|>` placeholder row is REPLACED in-sequence by the
// M rows the Ultravox projector produced, growing the sequence by (M - 1).
//
//   tokens:  [ t0 t1 ... <|audio|> ... tn ]          (seq_len)
//   embeds:  [ e0 e1 ...   E_audio  ... en ]          [seq_len, hidden_dim]
//                            │
//                            ▼  inject_audio_embeddings
//   spliced: [ e0 e1 ... a0 a1 ... a(M-1) ... en ]    [seq_len - 1 + M, hidden_dim]
//
// SPECULATIVE-DECODING SHARING (Architectural Rule #4)
//   The audio encoder + projector is the expensive multimodal step; it must run
//   ONCE per utterance and be consumed by BOTH the Draft (1B) and Target (8B)
//   prefill passes WITHOUT re-execution and WITHOUT copying the device buffer.
//   The design separates ownership from consumption:
//     * AudioInjectionContext OWNS the projected embeddings (one DeviceBuffer,
//       filled once by the projector).
//     * AudioEmbeddingsView is a trivially-copyable, non-owning borrow of that
//       buffer. Draft and Target each take a view and splice from the SAME
//       device memory. No shared_ptr atomics on the hot path; lifetime is
//       structural — the context is owned by the engine thread for the whole
//       generation and outlives both prefill passes.
//   Draft and Target may only share a view when they consume the same embedding
//   space. projection_space_id captures that (model-family + projector-version +
//   hidden_dim). If it differs (e.g. 1B hidden=2048 vs 8B hidden=4096 with
//   distinct projectors), the two are NOT interchangeable: the caller must
//   produce one context per space from a single ENCODER pass. inject_* refuses a
//   mismatched splice (capability gating) rather than silently mis-writing.
//
// ZERO-ALLOCATION (Architectural Rule #2)
//   inject_audio_embeddings performs NO device allocation. The spliced output is
//   written into a caller-owned workspace pre-sized for (max_seq_len - 1 +
//   max_audio_tokens) * hidden_dim. It runs on the caller's cudaStream_t and
//   reports via EngineStatus (RUNTIME tier — no throw across the decode path).
// =============================================================================
#include <cstdint>

#include <cuda_runtime.h>  // cudaStream_t

#include "device_buffer.h"            // blackwell::DeviceBuffer<T>  (src/core, white-box tier)
#include "blackwell/engine_status.h"  // blackwell::EngineStatus

namespace blackwell::bridge {

// Default id of the `<|audio|>` placeholder token in the Ultravox/Llama-3
// tokenizer (first id past the 128k base vocab). ABI-stable default; the real
// value is resolved from the loaded tokenizer and may override this.
inline constexpr int kDefaultAudioTokenId = 128256;

// Identifies the text embedding space that a set of audio embeddings targets.
// Two models may share ONE set of embeddings iff their ids are equal. Compose it
// from (model_family_hash << 32) | (projector_version << 16) | hidden_dim, or
// any scheme that is stable per-space and distinct across spaces. 0 == "unset".
using ProjectionSpaceId = uint64_t;

// Non-owning, trivially-copyable borrow of projected audio embeddings. This is
// the object Draft and Target each hold; copying it does NOT copy device memory.
// Valid only while the owning AudioInjectionContext is alive.
struct AudioEmbeddingsView {
    const float* data = nullptr;              // device ptr, [num_tokens, embed_dim] row-major
    int num_tokens = 0;                       // M (post-pool/-stack projector row count)
    int embed_dim = 0;                        // == target model hidden_dim (e.g. 2048)
    ProjectionSpaceId projection_space_id = 0;

    bool valid() const noexcept { return data != nullptr && num_tokens > 0 && embed_dim > 0; }
};

// Owns the projector output for one audio segment. Produced once on the engine
// thread by the audio pipeline; borrowed (view()) by every model that shares its
// projection space. Move-only (it holds a DeviceBuffer).
struct AudioInjectionContext {
    // Placeholder token this context replaces, and where it sits in the prompt.
    int audio_token_id = kDefaultAudioTokenId;
    int audio_token_position = -1;            // index in the [seq_len] token stream

    // Projected embeddings, shape [num_audio_tokens, embed_dim], row-major,
    // FP32 (matches the projector + engine activation dtype). Owned here; the
    // projector fills it exactly once (no re-execution for Draft vs Target).
    blackwell::DeviceBuffer<float> audio_embeddings;
    int num_audio_tokens = 0;                 // M — the 2D row count of audio_embeddings
    int embed_dim = 0;                         // second dim (e.g. 2048)
    ProjectionSpaceId projection_space_id = 0;

    // Non-owning borrow for the (possibly speculative) consumers. Both the 1B
    // Draft and 8B Target splice from this SAME device buffer.
    AudioEmbeddingsView view() const noexcept {
        return AudioEmbeddingsView{audio_embeddings.get(), num_audio_tokens, embed_dim,
                                   projection_space_id};
    }
};

// I/O descriptor for the splice (the D3D "DESC" idiom: a POD of borrowed device
// pointers + shapes; the callee owns none of it). All pointers are device
// pointers on the same stream/context as `stream`.
struct AudioSpliceDesc {
    // ---- inputs ----
    const int32_t* d_prompt_tokens = nullptr;   // [seq_len]                (device)
    const float*   d_text_embeddings = nullptr; // [seq_len, hidden_dim]    (device, row-major)
    int seq_len = 0;
    int hidden_dim = 0;                          // MUST equal view.embed_dim
    int audio_token_pos = -1;                    // splice row (== ctx.audio_token_position)

    // ---- output (caller-owned, pre-allocated workspace; NOT allocated here) ----
    float* d_out_embeddings = nullptr;           // [out_capacity_rows, hidden_dim] (device)
    int out_capacity_rows = 0;                    // >= seq_len - 1 + M, else InvalidArgument
    int* d_out_seq_len = nullptr;                 // optional: receives (seq_len - 1 + M)
};

// Splice contract — dynamically resizes/reorders the embedding matrix by
// replacing the `<|audio|>` row at ctx.audio_token_position with the M audio
// rows, streaming three regions into the pre-allocated output:
//   rows [0, pos)                 <- text embeddings [0, pos)
//   rows [pos, pos + M)           <- audio view rows [0, M)
//   rows [pos + M, pos + M + tail)<- text embeddings [pos + 1, seq_len)
// so the result length is (seq_len - 1 + M). Enqueued on `stream`; no host sync.
//
// PRECONDITIONS (checked -> EngineStatus, NOT asserts, per the runtime tier):
//   * view.valid() and view.embed_dim == desc.hidden_dim
//         else EngineStatus::InvalidArgument.
//   * desc.hidden_dim % 4 == 0 (float4 vectorised copy)     else InvalidArgument.
//   * 0 <= desc.audio_token_pos < desc.seq_len              else InvalidArgument.
//   * desc.out_capacity_rows >= desc.seq_len - 1 + view.num_tokens
//         else EngineStatus::InvalidArgument   (zero-alloc: never grows output).
//   * exactly ONE audio placeholder is spliced (the one at audio_token_position);
//     any further `<|audio|>` ids in the prompt are treated as ordinary text.
// The kernel performs NO allocation and returns EngineStatus::CudaRuntimeError if
// a launch/stream error is observed. `noexcept`.
blackwell::EngineStatus inject_audio_embeddings(const AudioSpliceDesc& desc,
                                                const AudioEmbeddingsView& view,
                                                cudaStream_t stream) noexcept;

// Convenience overload: borrow the view + splice position from an owning
// context. Copies the small POD desc so the caller need not restate the
// position that already lives on the context.
inline blackwell::EngineStatus inject_audio_embeddings(const AudioSpliceDesc& desc,
                                                       const AudioInjectionContext& ctx,
                                                       cudaStream_t stream) noexcept {
    AudioSpliceDesc d = desc;
    d.audio_token_pos = ctx.audio_token_position;
    return inject_audio_embeddings(d, ctx.view(), stream);
}

}  // namespace blackwell::bridge
