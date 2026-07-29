#include "kv_evict.cuh"

// ============================================================================
// Fused compaction + re-phase. One thread owns one (kv_head, row, channel pair):
// it moves that pair's K down by `delta` rows WHILE rotating it by -delta*freq,
// and copies the matching V pair unrotated.
//
// The rotation is written in exactly the shape rope_kv_append_kernel uses
// (same freq ladder, same rotate_half pairing of channel k with k + head_dim/2,
// same sincosf) so the two can be read side by side. The ONLY difference is the
// angle: the append kernel uses +pos*freq, this uses -delta*freq, and composing
// them gives (pos - delta)*freq.
// ============================================================================
__global__ void kv_evict_head_kernel(float* __restrict__ K_cache,
                                     float* __restrict__ V_cache,
                                     int dst_begin,      // first destination row
                                     int delta,          // rows dropped == src-dst gap
                                     int rows,           // rows this launch moves
                                     size_t head_dim,
                                     size_t max_seq_len,
                                     float rope_theta,
                                     RopeScaling scaling)
{
    const size_t row      = blockIdx.x;   // 0 .. rows-1
    const size_t head_idx = blockIdx.y;   // 0 .. kv_heads-1
    const size_t k        = threadIdx.x;  // channel pair

    const size_t half = head_dim / 2;
    if (k >= half || row >= static_cast<size_t>(rows)) return;

    const size_t dst = static_cast<size_t>(dst_begin) + row;
    const size_t src = dst + static_cast<size_t>(delta);

    float*       k_dst = K_cache + (head_idx * max_seq_len + dst) * head_dim;
    const float* k_src = K_cache + (head_idx * max_seq_len + src) * head_dim;
    float*       v_dst = V_cache + (head_idx * max_seq_len + dst) * head_dim;
    const float* v_src = V_cache + (head_idx * max_seq_len + src) * head_dim;

    // Identical frequency ladder to rope_kv_append_kernel -- if these two ever
    // diverge, the composed angle is measured against the wrong ladder and the
    // survivors land at positions nothing else agrees with.
    float freq = __fdividef(1.0f, powf(rope_theta, __fdividef(static_cast<float>(2 * k),
                                                              static_cast<float>(head_dim))));
    freq = apply_rope_scaling(freq, scaling);

    // THE COMPOSITION. A key cached at absolute p carries +p*freq; adding
    // -delta*freq leaves it carrying (p-delta)*freq, which is what a key cached
    // at its NEW row would have had.
    const float angle = -static_cast<float>(delta) * freq;

    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    // Read both halves BEFORE writing: k_dst and k_src are distinct rows within a
    // launch (rows <= delta), but the two halves of one row alias each other's
    // inputs through the rotation.
    const float k0 = k_src[k];
    const float k1 = k_src[k + half];

    k_dst[k]        = k0 * cos_val - k1 * sin_val;
    k_dst[k + half] = k0 * sin_val + k1 * cos_val;

    // V is stored unrotated by the append kernel, so it carries no phase to fix.
    v_dst[k]        = v_src[k];
    v_dst[k + half] = v_src[k + half];
}

void launch_kv_evict_head(float* d_K_cache,
                          float* d_V_cache,
                          int keep_from,
                          int delta,
                          int cache_len,
                          size_t kv_heads,
                          size_t head_dim,
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
            head_dim, max_seq_len, rope_theta, scaling);
    }
}
