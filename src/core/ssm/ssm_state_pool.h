#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

#include "common.h"          // CUDA_CHECK
#include "blackwell/config.h"   // ModelConfig::LinearAttnConfig

// ============================================================================
// SsmStatePool — recurrent-state allocator for hybrid linear-attention models.
// ============================================================================
// Unlike the paged KV cache, the linear-attention (SSM) layers carry a state
// whose size is INDEPENDENT of context length: a fixed [Dk x Dv] recurrent
// matrix per head plus a tiny causal-conv1d ring buffer. We deliberately drop
// Copy-on-Write here (see ModelCapabilities::supports_cow_branching == false for
// hybrid models): each active sequence owns one flat, contiguous VRAM block that
// evolves strictly linearly. No fork, no ref-counting, no page table.
//
// Device layout, one flat cudaMalloc, per sequence stride = bytes_per_seq():
//   [seq 0 | seq 1 | ... | seq N-1]
// within a sequence, packed by linear-layer index l in [0, num_linear_layers):
//   recurrent block:  float[num_linear_layers][H][Dk][Dv]   (row-major Dk x Dv)
//   conv block:       float[num_linear_layers][conv_dim][K-1]
// The two blocks are separate sub-arenas so the scan and conv kernels each get a
// dense, single-stride view of their own state.
//
// All state is fp32 (config: mamba_ssm_dtype == "float32"); decode reads/writes
// it in place. reset() zeroes a sequence (zero recurrent state == empty history).
namespace blackwell { namespace ssm {

struct SsmGeometry {
    int num_linear_layers;   // count of AttnKind::Linear layers
    int num_heads;           // value heads (state heads), e.g. 32
    int key_head_dim;        // Dk, e.g. 128
    int value_head_dim;      // Dv, e.g. 128
    int conv_dim;            // channels carried through causal conv1d
    int conv_kernel_dim;     // K, e.g. 4  (ring buffer holds K-1 past inputs)

    // Derive from a parsed ModelConfig. conv_dim defaults to the mixed qkv width
    // (q + k + v projection fan-out) the in_proj_qkv conv operates on:
    //   q,k = num_key_heads * key_head_dim each; v = num_value_heads * value_head_dim.
    static SsmGeometry from_config(const ModelConfig& cfg) {
        int linear = 0;
        for (AttnKind k : cfg.layer_types) if (k == AttnKind::Linear) ++linear;
        const auto& L = cfg.linear;
        const int q_dim = static_cast<int>(L.num_key_heads   * L.key_head_dim);
        const int v_dim = static_cast<int>(L.num_value_heads * L.value_head_dim);
        SsmGeometry g{};
        g.num_linear_layers = linear;
        g.num_heads         = static_cast<int>(L.num_value_heads);
        g.key_head_dim      = static_cast<int>(L.key_head_dim);
        g.value_head_dim    = static_cast<int>(L.value_head_dim);
        g.conv_dim          = 2 * q_dim + v_dim;   // q + k + v
        g.conv_kernel_dim   = static_cast<int>(L.conv_kernel_dim);
        return g;
    }

    size_t rec_elems_per_layer()  const { return (size_t)num_heads * key_head_dim * value_head_dim; }
    size_t conv_elems_per_layer() const { return (size_t)conv_dim * (conv_kernel_dim > 0 ? conv_kernel_dim - 1 : 0); }
    size_t rec_elems_per_seq()    const { return (size_t)num_linear_layers * rec_elems_per_layer(); }
    size_t conv_elems_per_seq()   const { return (size_t)num_linear_layers * conv_elems_per_layer(); }
};

class SsmStatePool {
public:
    SsmStatePool(const SsmGeometry& geo, int max_sequences)
        : m_geo(geo), m_max_seqs(max_sequences) {
        if (max_sequences <= 0) throw std::invalid_argument("SsmStatePool: max_sequences must be > 0");
        m_rec_stride  = m_geo.rec_elems_per_seq();
        m_conv_stride = m_geo.conv_elems_per_seq();
        const size_t rec_bytes  = (size_t)m_max_seqs * m_rec_stride  * sizeof(float);
        const size_t conv_bytes = (size_t)m_max_seqs * m_conv_stride * sizeof(float);
        // One allocation per sub-arena; zero-init == empty history for every seq.
        if (rec_bytes) {
            CUDA_CHECK(cudaMalloc(&m_d_rec, rec_bytes));
            CUDA_CHECK(cudaMemset(m_d_rec, 0, rec_bytes));
        }
        if (conv_bytes) {
            CUDA_CHECK(cudaMalloc(&m_d_conv, conv_bytes));
            CUDA_CHECK(cudaMemset(m_d_conv, 0, conv_bytes));
        }
    }
    ~SsmStatePool() { cudaFree(m_d_rec); cudaFree(m_d_conv); }
    SsmStatePool(const SsmStatePool&) = delete;
    SsmStatePool& operator=(const SsmStatePool&) = delete;

    // Recurrent [Dk x Dv] state for (sequence, linear-layer). The kernel indexes
    // heads/dims inside this dense [H][Dk][Dv] block.
    float* rec_state(int seq, int linear_layer) const {
        check(seq, linear_layer);
        return m_d_rec + (size_t)seq * m_rec_stride
                       + (size_t)linear_layer * m_geo.rec_elems_per_layer();
    }
    // Causal-conv1d ring buffer [conv_dim][K-1] for (sequence, linear-layer).
    float* conv_state(int seq, int linear_layer) const {
        check(seq, linear_layer);
        return m_d_conv + (size_t)seq * m_conv_stride
                        + (size_t)linear_layer * m_geo.conv_elems_per_layer();
    }

    // Zero a single sequence's entire state (both sub-arenas) — used on sequence
    // (re)start. Stream-ordered so it composes with the decode stream.
    void reset(int seq, cudaStream_t stream = 0) {
        check(seq, 0);
        if (m_d_rec)
            CUDA_CHECK(cudaMemsetAsync(rec_state(seq, 0), 0,
                                       m_rec_stride * sizeof(float), stream));
        if (m_d_conv)
            CUDA_CHECK(cudaMemsetAsync(conv_state(seq, 0), 0,
                                       m_conv_stride * sizeof(float), stream));
    }

    const SsmGeometry& geometry() const { return m_geo; }
    // Total resident VRAM (both sub-arenas, all sequences) — for budgeting/tests.
    size_t bytes_resident() const {
        return ((size_t)m_max_seqs * (m_rec_stride + m_conv_stride)) * sizeof(float);
    }

private:
    void check(int seq, int layer) const {
        if (seq < 0 || seq >= m_max_seqs)
            throw std::out_of_range("SsmStatePool: sequence index out of range");
        if (layer < 0 || layer >= m_geo.num_linear_layers)
            throw std::out_of_range("SsmStatePool: linear-layer index out of range");
    }

    SsmGeometry m_geo;
    int    m_max_seqs;
    size_t m_rec_stride = 0, m_conv_stride = 0;
    float* m_d_rec  = nullptr;
    float* m_d_conv = nullptr;
};

}} // namespace blackwell::ssm
