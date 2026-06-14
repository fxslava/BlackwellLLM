#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// Paged Flash Attention — launcher contracts
// ============================================================================
// KV is stored in fixed-size PHYSICAL PAGES of PAGE_SIZE tokens. A sequence's
// logical token stream is mapped to scattered physical pages by a block table
// (int32 page ids). The kernel gathers K/V tiles through that block table and
// runs an exact FlashAttention online-softmax pass using HW Tensor Cores
// (nvcuda::wmma, bf16 inputs, fp32 accumulate).
//
// Storage dtype is bf16 (half the bytes of the legacy FP32 cache, and the
// natural Tensor-Core operand). This is a deliberate precision change vs. the
// FP32 contiguous cache — see the integration notes.
namespace blackwell { namespace paging {
    using kv_t = __nv_bfloat16;
    constexpr int PAGE_SIZE = 16;   // tokens per physical page (== Tensor-Core N tile)
}} // namespace blackwell::paging

// ---------------------------------------------------------------------------
// Pool layout (per layer):  kv_t[total_pages][num_kv_heads][PAGE_SIZE][head_dim]
// The block table is shared across ALL layers (one page id indexes the same
// physical slot in every layer's pool), exactly like vLLM.
// ---------------------------------------------------------------------------

// HARD LAUNCH CONTRACT (not diagnosed at runtime):
//   * head_dim in {16,32,...,128}, i.e. head_dim % 16 == 0 && head_dim <= 128.
//   * num_q_heads % num_kv_heads == 0 (GQA ratio).
//   * d_block_table holds ceil(seq_len/PAGE_SIZE) valid page ids on device.

// Single-token decode: Q is [num_q_heads, head_dim], one query at pos seq_len-1.
// (Tensor Cores are underutilised here — only 1 of BLOCK_M rows is live — but
// the path is exact and shares the prefill kernel; see notes for the batched
// decode variant.)
void launch_paged_flash_attention_decode(
    const float* d_Q,
    const blackwell::paging::kv_t* d_k_pool_layer,
    const blackwell::paging::kv_t* d_v_pool_layer,
    float* d_O,
    const int32_t* d_block_table,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0);

// Chunked prefill / speculative verification: Q is [num_q_tokens, num_q_heads,
// head_dim] occupying logical positions [0, num_q_tokens) (causal). This is the
// full Tensor-Core path (BLOCK_M=16 live query rows per tile).
void launch_paged_flash_attention_prefill(
    const float* d_Q,
    const blackwell::paging::kv_t* d_k_pool_layer,
    const blackwell::paging::kv_t* d_v_pool_layer,
    float* d_O,
    const int32_t* d_block_table,
    int seq_len,            // total tokens resident for the sequence (== num_q_tokens for a fresh prefill)
    int num_q_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0);

// Scatter one post-RoPE token's K/V into (page, slot) for a single layer.
// d_K / d_V are [num_kv_heads, head_dim] (the engine's d_K / d_V buffers).
void launch_paged_kv_append(
    const float* d_K,
    const float* d_V,
    blackwell::paging::kv_t* d_k_pool_layer,
    blackwell::paging::kv_t* d_v_pool_layer,
    int page,
    int slot,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0);

// Copy-on-Write: duplicate one physical page across EVERY layer (src -> dst).
// Invoked by SequenceManager the first time a fork-shared partial page is
// written. Pools are the full [num_layers][total_pages]... arenas.
void launch_cow_copy_page(
    blackwell::paging::kv_t* d_k_pool,
    blackwell::paging::kv_t* d_v_pool,
    int src_page,
    int dst_page,
    int num_layers,
    int total_pages,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0);
