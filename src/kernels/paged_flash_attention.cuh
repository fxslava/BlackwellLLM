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
// head_dim]. The num_q_tokens query rows occupy the LAST num_q_tokens logical
// positions of the sequence, i.e. [seq_len - num_q_tokens, seq_len) (causal over
// the whole seq_len-length prefix through the block table). A fresh full prefill
// passes seq_len == num_q_tokens (rows at [0, num_q_tokens)); an append/chunked
// prefill passes seq_len > num_q_tokens so the rows start at seq_len-num_q_tokens
// and attend over the already-resident prefix. Full Tensor-Core path (BLOCK_M=16).
void launch_paged_flash_attention_prefill(
    const float* d_Q,
    const blackwell::paging::kv_t* d_k_pool_layer,
    const blackwell::paging::kv_t* d_v_pool_layer,
    float* d_O,
    const int32_t* d_block_table,
    int seq_len,            // total tokens resident for the sequence
    int num_q_tokens,       // query rows, occupying [seq_len - num_q_tokens, seq_len)
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

// Batched scatter: num_tokens post-RoPE K/V rows into their physical page slots
// for a single layer. d_K / d_V are [num_tokens, num_kv_heads, head_dim]; token t
// lands at logical position (start_pos + t), whose page is d_block_table[pos /
// PAGE_SIZE] and slot is pos % PAGE_SIZE. The block table must already name an
// allocated page for every one of those positions (prepare_prefill_step reserves
// them). The batched-prefill counterpart of launch_paged_kv_append.
void launch_paged_kv_append_batched(
    const float* d_K,
    const float* d_V,
    blackwell::paging::kv_t* d_k_pool_layer,
    blackwell::paging::kv_t* d_v_pool_layer,
    const int32_t* d_block_table,
    int start_pos,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0);

// TRUE-batch (multi-sequence) scatter: batch_size post-RoPE K/V rows, one per
// INDEPENDENT sequence, into their resolved page slots for a single layer. Unlike
// launch_paged_kv_append_batched (consecutive positions of ONE sequence through a
// shared block table), each row b lands at (d_pages[b], d_slots[b]) — the append
// slot the control plane reserved for sequence b this step. d_K / d_V are
// [batch_size, num_kv_heads, head_dim]; d_pages / d_slots are device arrays of
// batch_size ints. The sequence-aware sibling of the batched-prefill append.
void launch_paged_kv_append_batched_seqs(
    const float* d_K,
    const float* d_V,
    blackwell::paging::kv_t* d_k_pool_layer,
    blackwell::paging::kv_t* d_v_pool_layer,
    const int32_t* d_pages,     // [batch_size] resolved physical page per sequence
    const int32_t* d_slots,     // [batch_size] resolved slot within that page
    int batch_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = 0);

// TRUE-batch paged flash-attention DECODE: one query per INDEPENDENT sequence,
// batch_size sequences processed in parallel in a single launch. Q is
// [batch_size, num_q_heads, head_dim] (sequence b's single query at its own last
// position). Each sequence carries its OWN block table and length:
//   * d_block_tables is a FLATTENED [batch_size, max_blocks] int32 array (row b is
//     sequence b's page ids; block_table_stride == max_blocks selects the row).
//   * d_seq_lens[b] is sequence b's resident token count (its query sits at
//     position seq_len-1 and attends causally over [0, seq_len)).
// Output O is [batch_size, num_q_heads, head_dim]. Grid maps the batch onto
// blockIdx.z so sequences run concurrently; the per-sequence online-softmax math
// is bit-for-bit the single-sequence launch_paged_flash_attention_decode, so a
// batched step reproduces the sequential decode loop exactly (see the
// BatchedDecodeMatchesSequential parity test). This is a NEW kernel beside the
// single-sequence decode path (Dual-Path doctrine): the latency-critical batch=1
// kernel is left untouched.
void launch_batched_paged_flash_attention_decode(
    const float* d_Q,
    const blackwell::paging::kv_t* d_k_pool_layer,
    const blackwell::paging::kv_t* d_v_pool_layer,
    float* d_O,
    const int32_t* d_block_tables,   // [batch_size, max_blocks], row-major
    const int32_t* d_seq_lens,       // [batch_size]
    int batch_size,
    int block_table_stride,          // == max_blocks (row stride of d_block_tables)
    int num_q_heads,
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
