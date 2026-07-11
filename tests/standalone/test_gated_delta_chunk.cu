// ============================================================================
// Standalone sandbox: CHUNKED DELTA RULE for GatedDeltaNet batched prefill.
// ============================================================================
// This is the real primitive behind True Batched Prefill for the hybrid model's
// AttnKind::Linear layers -- the one the scalar block scan (test_ssm_block_scan)
// could NOT express. The engine's per-token step (selective_scan_update_kernel,
// src/kernels/ssm_kernels.cu) is the GatedDeltaNet gated delta rule, whose state
// S in R^{Dk x Dv} advances by a MATRIX transition:
//
//     S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T          (per head)
//
//   with, exactly as the engine kernel:
//     a_t     = decay/gate scalar in (0,1)      (= exp(dt*(-exp(A_log))))
//     b_t     = write strength in (0,1)         (= sigmoid(in_proj_b))
//     Sk[d]   = sum_i (a_t*S[i,d]) k_t[i]        (gate applied BEFORE the delta)
//     S[i,d] := a_t*S[i,d] + b_t*(v_t[d]-Sk[d]) k_t[i]
//     o_t[d]  = sum_i S[i,d] q_t[i]              (readout uses the POST-write S_t)
//
// Because the transition is a Dk x Dk matrix (not a scalar), a plain associative
// scan is out. The chunked delta rule instead parallelizes a chunk of C tokens:
//
//   (1) GATE ABSORPTION. Let b_t = prod_{j<=t} a_j be the chunk-local cumulative
//       decay (an inclusive prefix product -- our proven scan primitive). Rescale
//       S_hat_t = S_t / b_t; the decay drops out of the transition, leaving the
//       plain (ungated) delta rule on rescaled values v~_t = v_t / b_t:
//           S_hat_t = (I - b_t k_t k_t^T) S_hat_{t-1} + b_t k_t v~_t^T.
//
//   (2) INTRA-CHUNK UT TRANSFORM. The rank-1 delta writes form a UNIT lower-
//       triangular system in the per-token "corrected writes" U (rows u_t in R^Dv):
//           (I + A) U = B (V~ - K S_0),   A[t,s] = b_t (k_t . k_s) for s<t,
//       with B=diag(b_t), K the chunk keys, S_0 the incoming state. One forward
//       substitution (sequential in t, parallel in d) yields all of U.
//
//   (3) OUTPUTS + STATE CARRY (masked matmuls):
//           o_t   = b_t [ (q_t^T S_0) + sum_{s<=t} (q_t.k_s) u_s ]   (inclusive)
//           S_C   = b_C ( S_0 + K^T U )                              (-> h_out)
//
// Verified analytically at C=1: the whole pipeline collapses back to the single
// token recurrence above (both S_C and o_1). This sandbox pins it NUMERICALLY,
// multi-chunk, against a strict sequential reference that mirrors the engine.
//
// Build (x64 Native Tools prompt, CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 tests/standalone/test_gated_delta_chunk.cu ^
//        -o test_gated_delta_chunk.exe
//   ./test_gated_delta_chunk.exe
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 99; } } while(0)

// ---- sandbox geometry (maps to Qwen3.5 linear layers by scaling H and dims) --
static constexpr int H  = 4;      // heads (grid.x)
static constexpr int Dk = 128;    // key head dim   (Qwen3.5 linear)
static constexpr int Dv = 128;    // value head dim
static constexpr int C  = 64;     // chunk length
static constexpr int NCHUNK = 4;  // chunks per sequence
static constexpr int L  = C * NCHUNK;   // 256 tokens

// index helpers (row-major, head-major)
__host__ __device__ inline size_t idxKD(int h,int t,int i){ return ((size_t)h*L+t)*Dk+i; } // k,q
__host__ __device__ inline size_t idxVD(int h,int t,int d){ return ((size_t)h*L+t)*Dv+d; } // v,o
__host__ __device__ inline size_t idxS (int h,int i,int d){ return ((size_t)h*Dk+i)*Dv+d; } // state
__host__ __device__ inline size_t idxU (int h,int t,int d){ return ((size_t)h*C +t)*Dv+d; } // chunk writes

// ============================================================================
// GPU: chunked delta rule. One block per head; chunks processed sequentially
// (state carry), the O(C*Dk*Dv) intra-chunk work parallel across the block.
// ============================================================================
__global__ void gated_delta_chunk_kernel(const float* __restrict__ k,
                                         const float* __restrict__ q,
                                         const float* __restrict__ v,
                                         const float* __restrict__ alpha,
                                         const float* __restrict__ beta,
                                         const float* __restrict__ h_prev,
                                         float* __restrict__ o_out,
                                         float* __restrict__ h_out,
                                         float* __restrict__ u_scratch) {
    const int h  = blockIdx.x;
    const int tx = threadIdx.x;
    const int NT = blockDim.x;

    __shared__ float s_alpha[C];
    __shared__ float s_beta[C];
    __shared__ float s_b[C];          // inclusive cumulative decay within chunk
    __shared__ float s_M[C * C];      // A = strictly-lower-tri, unit diagonal

    // Persistent per-head state lives in h_out (seeded from h_prev, carried,
    // finally IS the chunk-C state). Seed once.
    for (int e = tx; e < Dk * Dv; e += NT) h_out[(size_t)h*Dk*Dv + e] = h_prev[(size_t)h*Dk*Dv + e];
    __syncthreads();
    float* __restrict__ S = h_out + (size_t)h * Dk * Dv;   // S(i,d) = S[i*Dv+d]

    for (int c = 0; c < NCHUNK; ++c) {
        const int base = c * C;

        // -- load per-token scalars, build cumulative decay b_t (prefix product)
        for (int t = tx; t < C; t += NT) {
            s_alpha[t] = alpha[(size_t)h*L + base + t];
            s_beta[t]  = beta [(size_t)h*L + base + t];
        }
        __syncthreads();
        if (tx == 0) {                       // C=64: trivially serial, no scan bug risk
            float acc = 1.f;
            for (int t = 0; t < C; ++t) { acc *= s_alpha[t]; s_b[t] = acc; }
        }
        __syncthreads();

        // -- build the unit lower-tri system A[t,s] = b_t (k_t . k_s), s<t
        for (int p = tx; p < C * C; p += NT) {
            const int t = p / C, s = p % C;
            float m;
            if (s > t)       m = 0.f;
            else if (s == t) m = 1.f;        // unit diagonal (I + A)
            else {
                float dot = 0.f;
                for (int i = 0; i < Dk; ++i) dot += k[idxKD(h,base+t,i)] * k[idxKD(h,base+s,i)];
                m = s_beta[t] * dot;
            }
            s_M[p] = m;
        }
        __syncthreads();

        // -- RHS of the UT system into U:  u_t = b_t (v~_t - k_t^T S_0),  v~=v/b_t
        for (int p = tx; p < C * Dv; p += NT) {
            const int t = p / Dv, d = p % Dv;
            float kS = 0.f;
            for (int i = 0; i < Dk; ++i) kS += k[idxKD(h,base+t,i)] * S[idxS(0,i,d)];
            const float vt = v[idxVD(h,base+t,d)] / s_b[t];
            u_scratch[idxU(h,t,d)] = s_beta[t] * (vt - kS);
        }
        __syncthreads();

        // -- forward substitution: u_t -= sum_{s<t} A[t,s] u_s  (sequential in t)
        for (int t = 1; t < C; ++t) {
            for (int d = tx; d < Dv; d += NT) {
                float acc = 0.f;
                for (int s = 0; s < t; ++s) acc += s_M[t*C + s] * u_scratch[idxU(h,s,d)];
                u_scratch[idxU(h,t,d)] -= acc;
            }
            __syncthreads();                 // u_t final before row t+1 reads it
        }

        // -- outputs: o_t = b_t [ q_t^T S_0 + sum_{s<=t} (q_t.k_s) u_s ]
        for (int p = tx; p < C * Dv; p += NT) {
            const int t = p / Dv, d = p % Dv;
            float qS = 0.f;
            for (int i = 0; i < Dk; ++i) qS += q[idxKD(h,base+t,i)] * S[idxS(0,i,d)];
            float cross = 0.f;
            for (int s = 0; s <= t; ++s) {
                float qk = 0.f;
                for (int i = 0; i < Dk; ++i) qk += q[idxKD(h,base+t,i)] * k[idxKD(h,base+s,i)];
                cross += qk * u_scratch[idxU(h,s,d)];
            }
            o_out[idxVD(h,base+t,d)] = s_b[t] * (qS + cross);
        }
        __syncthreads();                     // outputs consumed S_0 before we overwrite it

        // -- state carry: S_C = b_C (S_0 + K^T U),   b_C = s_b[C-1]
        const float bC = s_b[C-1];
        for (int p = tx; p < Dk * Dv; p += NT) {
            const int i = p / Dv, d = p % Dv;
            float ku = 0.f;
            for (int s = 0; s < C; ++s) ku += k[idxKD(h,base+s,i)] * u_scratch[idxU(h,s,d)];
            S[idxS(0,i,d)] = bC * (S[idxS(0,i,d)] + ku);
        }
        __syncthreads();                     // S_C visible as S_0 for the next chunk
    }
}

// ============================================================================
// CPU reference -- token-by-token, byte-for-byte the engine's two-pass math
// (selective_scan_update_kernel): gate BEFORE delta, readout on POST-write S_t.
// ============================================================================
static void cpu_reference(const std::vector<float>& k, const std::vector<float>& q,
                          const std::vector<float>& v, const std::vector<float>& alpha,
                          const std::vector<float>& beta, const std::vector<float>& h_prev,
                          std::vector<float>& O, std::vector<float>& Sfin) {
    for (int h = 0; h < H; ++h) {
        std::vector<float> S(Dk*Dv);
        for (int i = 0; i < Dk; ++i) for (int d = 0; d < Dv; ++d)
            S[i*Dv+d] = h_prev[idxS(h,i,d)];
        for (int t = 0; t < L; ++t) {
            const float a = alpha[(size_t)h*L+t], b = beta[(size_t)h*L+t];
            std::vector<float> Sk(Dv, 0.f);
            for (int d = 0; d < Dv; ++d)
                for (int i = 0; i < Dk; ++i) Sk[d] += (a * S[i*Dv+d]) * k[idxKD(h,t,i)];
            for (int d = 0; d < Dv; ++d) {
                const float corr = b * (v[idxVD(h,t,d)] - Sk[d]);
                for (int i = 0; i < Dk; ++i) S[i*Dv+d] = a * S[i*Dv+d] + corr * k[idxKD(h,t,i)];
            }
            for (int d = 0; d < Dv; ++d) {
                float o = 0.f;
                for (int i = 0; i < Dk; ++i) o += q[idxKD(h,t,i)] * S[i*Dv+d];
                O[idxVD(h,t,d)] = o;
            }
        }
        for (int i = 0; i < Dk; ++i) for (int d = 0; d < Dv; ++d)
            Sfin[idxS(h,i,d)] = S[i*Dv+d];
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i]-b[i])); return m;
}

int main() {
    int dev = 0;
    if (cudaSetDevice(dev) != cudaSuccess) { printf("No CUDA device.\n"); return 99; }
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("Geometry: H=%d Dk=%d Dv=%d chunk C=%d  x %d chunks = %d tokens/head\n",
           H, Dk, Dv, C, NCHUNK, L);

    // ------------------------------------------------------------- inputs
    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> Ua(0.90f, 0.99f);   // decay, stable
    std::uniform_real_distribution<float> Ur(-1.0f, 1.0f);
    std::uniform_real_distribution<float> Us(-0.1f, 0.1f);    // seed state h_prev

    std::vector<float> k((size_t)H*L*Dk), q((size_t)H*L*Dk), v((size_t)H*L*Dv);
    std::vector<float> alpha((size_t)H*L), beta((size_t)H*L), h_prev((size_t)H*Dk*Dv);
    for (auto& e : q) e = Ur(rng);
    for (auto& e : v) e = Ur(rng);
    for (auto& e : alpha) e = Ua(rng);
    for (auto& e : h_prev) e = Us(rng);
    // beta = sigmoid(.) in (0,1), write strength (matches engine)
    for (auto& e : beta) e = 1.f / (1.f + std::exp(-Ur(rng)));
    // k random then L2-NORMALIZED per (head,token) -- GatedDeltaNet precondition
    // that keeps (a - b||k||^2) a contraction (src/kernels/ssm_kernels.cuh).
    for (auto& e : k) e = Ur(rng);
    for (int h = 0; h < H; ++h) for (int t = 0; t < L; ++t) {
        float n = 0.f;
        for (int i = 0; i < Dk; ++i) n += k[idxKD(h,t,i)]*k[idxKD(h,t,i)];
        n = std::sqrt(n) + 1e-6f;
        for (int i = 0; i < Dk; ++i) k[idxKD(h,t,i)] /= n;
    }

    // ----------------------------------------------- CPU reference (ground truth)
    std::vector<float> O_ref((size_t)H*L*Dv, 0.f), S_ref((size_t)H*Dk*Dv, 0.f);
    cpu_reference(k, q, v, alpha, beta, h_prev, O_ref, S_ref);

    // ------------------------------------------------------------- GPU
    float *dk,*dq,*dv,*da,*db,*dhp,*doo,*dho,*du;
    CU(cudaMalloc(&dk,  k.size()*4));      CU(cudaMalloc(&dq,  q.size()*4));
    CU(cudaMalloc(&dv,  v.size()*4));      CU(cudaMalloc(&da,  alpha.size()*4));
    CU(cudaMalloc(&db,  beta.size()*4));   CU(cudaMalloc(&dhp, h_prev.size()*4));
    CU(cudaMalloc(&doo, (size_t)H*L*Dv*4));CU(cudaMalloc(&dho, (size_t)H*Dk*Dv*4));
    CU(cudaMalloc(&du,  (size_t)H*C*Dv*4));
    CU(cudaMemcpy(dk, k.data(), k.size()*4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dq, q.data(), q.size()*4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dv, v.data(), v.size()*4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(da, alpha.data(), alpha.size()*4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(db, beta.data(), beta.size()*4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dhp, h_prev.data(), h_prev.size()*4, cudaMemcpyHostToDevice));

    gated_delta_chunk_kernel<<<H, 256>>>(dk, dq, dv, da, db, dhp, doo, dho, du);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    std::vector<float> O_gpu((size_t)H*L*Dv), S_gpu((size_t)H*Dk*Dv);
    CU(cudaMemcpy(O_gpu.data(), doo, O_gpu.size()*4, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(S_gpu.data(), dho, S_gpu.size()*4, cudaMemcpyDeviceToHost));
    cudaFree(dk);cudaFree(dq);cudaFree(dv);cudaFree(da);cudaFree(db);
    cudaFree(dhp);cudaFree(doo);cudaFree(dho);cudaFree(du);

    // ------------------------------------------------------- verify
    const float dO = max_abs_diff(O_gpu, O_ref);
    const float dS = max_abs_diff(S_gpu, S_ref);

    printf("\n  per-token output o_t  (head 0, d=0), first 5 tokens:\n");
    printf("  t :        CPU              GPU\n");
    for (int t = 0; t < 5; ++t)
        printf("  %3d : %16.8f %16.8f\n", t, O_ref[idxVD(0,t,0)], O_gpu[idxVD(0,t,0)]);
    printf("  ... last chunk boundary (tokens %d..%d), d=0:\n", L-3, L-1);
    for (int t = L-3; t < L; ++t)
        printf("  %3d : %16.8f %16.8f\n", t, O_ref[idxVD(0,t,0)], O_gpu[idxVD(0,t,0)]);

    printf("\n  max|O_gpu - O_ref| over all %d token outputs = %.3e\n", H*L*Dv, dO);
    printf("  max|S_gpu - S_ref| over final state (%d elems) = %.3e\n", H*Dk*Dv, dS);

    const float eps = 1e-3f;
    bool ok = dO < eps && dS < eps;
    printf("\n==== %s (dO %.3e, dS %.3e, eps %.1e) ====\n",
           ok ? "ALL PASS" : "FAIL", dO, dS, eps);
    return ok ? 0 : 1;
}
