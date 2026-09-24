#include "kv_evict.cuh"

// ============================================================================
// Fused compaction + re-phase. One thread owns one (kv_head, row, channel pair):
// it moves that pair's K down by `delta` rows WHILE rotating it by -delta*freq,
// and copies the matching V pair unrotated.
//
// The rotation is written in exactly the shape the append kernels use (same freq
// ladder, same pairing, same sincosf) so the two can be read side by side. The
// ONLY difference is the angle: an append uses +pos*freq, this uses -delta*freq,
// and composing them gives (pos - delta)*freq.
//
// PAIRING. `interleaved` selects which two channels pair k holds, and it MUST
// match the append kernel that wrote the cache: half-split (k, k + rotary_dim/2)
// for rope_kv_append_kernel, adjacent (2k, 2k+1) for GLM-4's
// rope_interleaved_partial_qk_kernel.
//
// PARTIAL ROTARY. Only channels [0, rotary_dim) carry phase. The pass-through
// tail [rotary_dim, head_dim) -- 64 of GLM-4's 128 channels -- and the whole of V
// are plain copies. For a full-rotary model (rotary_dim == head_dim) the tail
// loop is empty and every value written is bit-identical to the pre-split kernel.
// ============================================================================
__global__ void kv_evict_head_kernel(float* __restrict__ K_cache,
                                     float* __restrict__ V_cache,
                                     int dst_begin,      // first destination row
                                     int delta,          // rows dropped == src-dst gap
                                     int rows,           // rows this launch moves
                                     size_t head_dim,
                                     size_t rotary_dim,
                                     bool interleaved,
                                     size_t max_seq_len,
                                     float rope_theta,
                                     RopeScaling scaling)
{
    const size_t row      = blockIdx.x;   // 0 .. rows-1
    const size_t head_idx = blockIdx.y;   // 0 .. kv_heads-1
    const size_t k        = threadIdx.x;  // channel pair

    const size_t half     = rotary_dim / 2;   // rotated pairs
    if (row >= static_cast<size_t>(rows)) return;

    const size_t dst = static_cast<size_t>(dst_begin) + row;
    const size_t src = dst + static_cast<size_t>(delta);

    float*       k_dst = K_cache + (head_idx * max_seq_len + dst) * head_dim;
    const float* k_src = K_cache + (head_idx * max_seq_len + src) * head_dim;
    float*       v_dst = V_cache + (head_idx * max_seq_len + dst) * head_dim;
    const float* v_src = V_cache + (head_idx * max_seq_len + src) * head_dim;

    if (k < half) {
        // Identical frequency ladder to rope_kv_append_kernel -- if these two ever
        // diverge, the composed angle is measured against the wrong ladder and the
        // survivors land at positions nothing else agrees with. Note the divisor is
        // the ROTARY span, which is what both kernels index the ladder by.
        float freq = __fdividef(1.0f, powf(rope_theta, __fdividef(static_cast<float>(2 * k),
                                                                  static_cast<float>(rotary_dim))));
        freq = apply_rope_scaling(freq, scaling);

        // THE COMPOSITION. A key cached at absolute p carries +p*freq; adding
        // -delta*freq leaves it carrying (p-delta)*freq, which is what a key cached
        // at its NEW row would have had.
        const float angle = -static_cast<float>(delta) * freq;

        float sin_val, cos_val;
        sincosf(angle, &sin_val, &cos_val);

        // WHICH two channels form pair k must match the pairing the APPEND kernel
        // used -- half-split (k, k + rotary_dim/2) for Llama/Qwen, adjacent
        // (2k, 2k+1) for GLM-4. Re-phasing the wrong couple mixes two different
        // frequencies' phases and is as wrong as not re-phasing at all.
        const size_t c0 = interleaved ? (2 * k)     : k;
        const size_t c1 = interleaved ? (2 * k + 1) : (k + half);

        // Read both channels BEFORE writing: k_dst and k_src are distinct rows within
        // a launch (rows <= delta), but the two channels of one pair alias each
        // other's inputs through the rotation.
        const float k0 = k_src[c0];
        const float k1 = k_src[c1];

        k_dst[c0] = k0 * cos_val - k1 * sin_val;
        k_dst[c1] = k0 * sin_val + k1 * cos_val;
    }

    // The non-rotary tail of K carries no phase: move it verbatim. Empty when
    // rotary_dim == head_dim.
    for (size_t c = rotary_dim + k; c < head_dim; c += blockDim.x)
        k_dst[c] = k_src[c];

    // V is stored unrotated by the append kernel, so the whole row is a copy.
    for (size_t c = k; c < head_dim; c += blockDim.x)
        v_dst[c] = v_src[c];
}

void launch_kv_evict_head(float* d_K_cache,
                          float* d_V_cache,
                          int keep_from,
                          int delta,
                          int cache_len,
                          size_t kv_heads,
                          size_t head_dim,
                          size_t rotary_dim,
                          bool rope_interleaved,
                          size_t max_seq_len,
                          float rope_theta,
                          RopeScaling scaling,
                          cudaStream_t stream)
{
    if (delta <= 0 || keep_from < 0 || keep_from + delta > cache_len) return;

    // Rows that survive and therefore have to move. Zero is legal and means the
    // eviction drops everything above the prefix: pure bookkeeping, no work.
    const int rows = cache_len - keep_from - delta;
    if (rows <= 0 || kv_heads == 0 || head_dim < 2) return;
    if (rotary_dim < 2 || rotary_dim > head_dim) return;

    const dim3 threads(static_cast<unsigned>(head_dim / 2));

    // OVERLAP SPLIT. src == dst + delta, so a single launch is only safe while
    // no destination row is another thread's source, i.e. while it moves at most
    // `delta` rows. Chunks are stream-ordered, and chunk j's sources are chunk
    // j+1's destinations, so ascending order is the correct one: by the time a
    // chunk overwrites a row, the chunk that reads it has already completed.
    for (int done = 0; done < rows; done += delta) {
        const int n = (rows - done < delta) ? (rows - done) : delta;
        const dim3 blocks(static_cast<unsigned>(n), static_cast<unsigned>(kv_heads));
        kv_evict_head_kernel<<<blocks, threads, 0, stream>>>(
            d_K_cache, d_V_cache, keep_from + done, delta, n,
            head_dim, rotary_dim, rope_interleaved, max_seq_len, rope_theta, scaling);
    }
}
