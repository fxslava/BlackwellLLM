// ============================================================================
// Standalone sandbox: BLOCK-level parallel associative scan for the SSM
// recurrence  h_t = A_t * h_{t-1} + Bx_t,  scaled to a full 1024-token chunk.
// ============================================================================
// Builds directly on test_ssm_warp_scan.cu. The warp scan proved the operator
// and the __shfl_up_sync exchange on 32 steps; a real prefill chunk is up to
// 1024 tokens, so we compose 32 warp scans into one block scan with the classic
// three-phase (scan / aggregate / re-scan-and-fixup) structure:
//
//   Phase 1  each of the 32 warps inclusive-scans its own 32 steps in registers
//            (__shfl_up_sync only). Lane 31 then holds that warp's full 32-step
//            aggregate (A_warp, Bx_warp).
//   Phase 2  lane 31 of every warp publishes its aggregate to shared memory,
//            32 pairs total. __syncthreads() makes all 32 visible.
//   Phase 3  warp 0 inclusive-scans those 32 warp-aggregates in shared memory
//            (same operator, same primitive), producing an exclusive prefix per
//            warp; __syncthreads(); then every thread in warp w>0 folds the
//            aggregate of warps [0..w-1] into its local partial.
//
// The associative combine  (A_c,Bx_c) (x) (A_p,Bx_p) = (A_c*A_p, A_c*Bx_p+Bx_c)
// is NOT commutative: _c is the later step, _p the earlier. Every fold below
// keeps that temporal order (local partial = curr, incoming prefix = prev).
//
// Build (x64 Native Tools prompt, CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 tests/standalone/test_ssm_block_scan.cu ^
//        -o test_ssm_block_scan.exe
//   ./test_ssm_block_scan.exe
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 99; } } while(0)

static constexpr int WARP  = 32;
static constexpr int BLOCK = 1024;          // 32 warps == 1024-token prefill tile
static constexpr int NWARPS = BLOCK / WARP; // 32
static constexpr unsigned FULL_MASK = 0xffffffffu;

// The associative SSM combine, applied in place: (this) := (this) (x) (prev).
// Bx must be updated BEFORE A is stomped (it reads the old A_c).
__device__ __forceinline__ void ssm_combine(float& A_c, float& Bx_c,
                                            float A_p, float Bx_p) {
    Bx_c = A_c * Bx_p + Bx_c;
    A_c  = A_c * A_p;
}

// Inclusive Hillis-Steele warp scan of the SSM operator over one warp's 32
// elements. Requires all 32 lanes of the warp to be active (FULL_MASK).
__device__ __forceinline__ void warp_scan(float& A, float& Bx, int lane) {
    #pragma unroll
    for (int offset = 1; offset < WARP; offset *= 2) {
        float A_p  = __shfl_up_sync(FULL_MASK, A,  offset);
        float Bx_p = __shfl_up_sync(FULL_MASK, Bx, offset);
        if (lane >= offset) ssm_combine(A, Bx, A_p, Bx_p);
    }
}

__global__ void block_ssm_scan_kernel(const float* __restrict__ A_in,
                                      const float* __restrict__ Bx_in,
                                      float* __restrict__ h_out) {
    __shared__ float smem_A[NWARPS];        // one warp aggregate per slot
    __shared__ float smem_Bx[NWARPS];

    const int tid     = threadIdx.x;        // == time step t
    const int lane    = tid & (WARP - 1);
    const int warp_id = tid >> 5;

    float A  = A_in[tid];
    float Bx = Bx_in[tid];

    // Phase 1 -- intra-warp inclusive scan (registers only).
    warp_scan(A, Bx, lane);

    // Phase 2 -- publish each warp's 32-step aggregate (lane 31 holds it).
    if (lane == WARP - 1) { smem_A[warp_id] = A; smem_Bx[warp_id] = Bx; }
    __syncthreads();                        // all NWARPS aggregates now visible

    // Phase 3a -- warp 0 scans the 32 aggregates in place. Uniform branch, so
    // all 32 lanes of warp 0 hit warp_scan together (FULL_MASK stays valid).
    if (warp_id == 0) {
        float aA  = smem_A[lane];
        float aBx = smem_Bx[lane];
        warp_scan(aA, aBx, lane);
        smem_A[lane]  = aA;                 // smem_*[w] := aggregate of warps 0..w
        smem_Bx[lane] = aBx;
    }
    __syncthreads();                        // scanned prefixes visible to all warps

    // Phase 3b -- fold the prefix (warps [0..warp_id-1]) into the local partial.
    // Prefix is the EARLIER state (prev); the local partial is later (curr).
    if (warp_id > 0) {
        float A_p  = smem_A[warp_id - 1];
        float Bx_p = smem_Bx[warp_id - 1];
        ssm_combine(A, Bx, A_p, Bx_p);
    }

    h_out[tid] = Bx;                        // Bx now carries h_t
}

int main() {
    int dev = 0;
    if (cudaSetDevice(dev) != cudaSuccess) { printf("No CUDA device.\n"); return 99; }
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // ------------------------------------------------------------- inputs
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> Ua(0.90f, 0.99f);   // stable, no underflow
    std::uniform_real_distribution<float> Ub(-1.0f, 1.0f);
    std::vector<float> A(BLOCK), Bx(BLOCK);
    for (auto& e : A)  e = Ua(rng);   // A in [0.90,0.99]: product over 1024 steps
    for (auto& e : Bx) e = Ub(rng);   // stays representable, no flush-to-zero

    // --------------------------------------------- sequential CPU reference
    std::vector<float> h_ref(BLOCK);
    h_ref[0] = Bx[0];
    for (int t = 1; t < BLOCK; ++t) h_ref[t] = A[t] * h_ref[t-1] + Bx[t];

    // ------------------------------------------------- GPU block scan
    float *dA, *dBx, *dh;
    CU(cudaMalloc(&dA,  BLOCK * sizeof(float)));
    CU(cudaMalloc(&dBx, BLOCK * sizeof(float)));
    CU(cudaMalloc(&dh,  BLOCK * sizeof(float)));
    CU(cudaMemcpy(dA,  A.data(),  BLOCK * sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dBx, Bx.data(), BLOCK * sizeof(float), cudaMemcpyHostToDevice));

    block_ssm_scan_kernel<<<1, BLOCK>>>(dA, dBx, dh);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    std::vector<float> h_gpu(BLOCK);
    CU(cudaMemcpy(h_gpu.data(), dh, BLOCK * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dBx); cudaFree(dh);

    // ------------------------------------------------------- verify
    float max_abs = 0.f; int worst_t = 0;
    for (int t = 0; t < BLOCK; ++t) {
        float d = std::fabs(h_gpu[t] - h_ref[t]);
        if (d > max_abs) { max_abs = d; worst_t = t; }
    }

    printf("\n  first 5 steps:\n  t :        CPU h_t          GPU h_t\n");
    for (int t = 0; t < 5; ++t)
        printf("  %4d : %16.8f %16.8f\n", t, h_ref[t], h_gpu[t]);
    printf("\n  last 5 steps (prove the prefix reached warp 31):\n");
    for (int t = BLOCK - 5; t < BLOCK; ++t)
        printf("  %4d : %16.8f %16.8f\n", t, h_ref[t], h_gpu[t]);
    printf("\n  max|gpu-cpu| over %d steps = %.3e  (worst at t=%d)\n",
           BLOCK, max_abs, worst_t);

    const float eps = 1e-3f;
    bool ok = max_abs < eps;
    for (float x : h_gpu) ok = ok && std::isfinite(x);
    printf("\n==== %s (max diff %.3e, eps %.1e) ====\n",
           ok ? "ALL PASS" : "FAIL", max_abs, eps);

    // ------------------------------------------- methodology & pitfalls report
    printf(
"\n"
"=================== Testing Methodology & Pitfalls ===================\n"
"\n"
"1. ORDER OF OPERATIONS (non-commutativity)\n"
"   The operator (A_c,Bx_c) (x) (A_p,Bx_p) = (A_c*A_p, A_c*Bx_p + Bx_c) is\n"
"   associative but NOT commutative -- swapping curr/prev gives a different\n"
"   Bx. The CPU reference walks strictly forward (h_t depends on h_{t-1}), so\n"
"   it is the ground truth for temporal order. In Phase 3b the shared-memory\n"
"   prefix is the EARLIER block of steps (prev) and the thread's own warp-local\n"
"   partial is the LATER step (curr); the fold is ssm_combine(local, prefix)\n"
"   i.e. local=curr, prefix=prev. Had we reversed the arguments, the tail\n"
"   steps (t=32..1023, every thread outside warp 0) would diverge immediately\n"
"   and max_abs would explode to O(1) -- so a passing 1e-3 check across ALL\n"
"   1024 positions, especially the last 5, is direct proof the order is right.\n"
"\n"
"2. CROSS-WARP SYNCHRONIZATION (no races between phases)\n"
"   Phase 2 writes 32 aggregates to shared memory from 32 different warps that\n"
"   the SM may run in any order. The first __syncthreads() is a block-wide\n"
"   barrier + memory fence: no thread reads smem in Phase 3a until every\n"
"   lane-31 write has landed. The second __syncthreads() guards the other\n"
"   direction -- warp 0's scanned prefixes must be fully written before warps\n"
"   1..31 read smem[warp_id-1] in Phase 3b. Intra-warp Phase 1 needs no barrier\n"
"   because __shfl_up_sync is warp-synchronous by construction. If either\n"
"   barrier were missing, results would be run-to-run nondeterministic; the\n"
"   fixed seed makes the CPU reference reproducible, so a stable pass across\n"
"   repeated runs is the evidence the ordering hazards are covered.\n"
"\n"
"3. FLOATING-POINT ACCUMULATION (why diff > machine epsilon)\n"
"   The CPU does 1024 strictly left-to-right fused steps; the GPU evaluates the\n"
"   SAME associative expression as a balanced tree (log2(32) warp levels x two\n"
"   warp scans + one cross-warp fold). Real fp32 add/mul are not associative,\n"
"   so re-parenthesizing reorders roundings and the two results differ by\n"
"   accumulated rounding -- not by a logic bug. With A in [0.90,0.99] the\n"
"   recurrence is a contraction, so early error is damped rather than amplified,\n"
"   keeping the observed drift near ~1e-4..1e-3 over 1024 steps. That is why\n"
"   the assertion tolerance is 1e-3 (not the ~1e-7 of the 32-step warp test),\n"
"   and why the actual max_abs is printed rather than asserting bit-exactness.\n"
"=====================================================================\n");

    return ok ? 0 : 1;
}
