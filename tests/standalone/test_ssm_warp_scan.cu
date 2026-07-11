// ============================================================================
// Standalone sandbox: warp-level parallel associative scan for the SSM
// recurrence  h_t = A_t * h_{t-1} + Bx_t.
// ============================================================================
// Proves, in isolation (no engine, no model weights), the one piece of math
// that True Batched Prefill for hybrid SSM models rests on: that the sequential
// linear recurrence can be turned into a parallel prefix sum over a warp using
// only __shfl_up_sync register exchanges — no shared memory, no barriers.
//
// The recurrence is a first-order linear scan. Package each step as the pair
// (A_t, Bx_t) and define the associative combine operator
//
//     (A_c, Bx_c) (x) (A_p, Bx_p) = (A_c * A_p,  A_c * Bx_p + Bx_c)
//
// where _c is the more-recent step and _p the earlier one. An INCLUSIVE scan of
// this operator over t = 0..31 yields the pair whose Bx component equals h_t for
// the recurrence seeded at h_0 = Bx_0. That identity is what we verify here
// against a scalar sequential CPU reference.
//
// One warp == 32 time steps: lane_id is the time index t. This is the inner
// primitive; a real prefill chunks the sequence into 32-step tiles and carries
// the last lane's (A,Bx) pair as the seed for the next tile.
//
// Build (x64 Native Tools prompt, CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 tests/standalone/test_ssm_warp_scan.cu ^
//        -o test_ssm_warp_scan.exe
//   ./test_ssm_warp_scan.exe
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 99; } } while(0)

static constexpr int WARP = 32;
static constexpr unsigned FULL_MASK = 0xffffffffu;

// The associative SSM combine, applied in place: (this) := (this) (x) (prev).
// Must exactly mirror the scalar recurrence the CPU reference walks.
__device__ __forceinline__ void ssm_combine(float& A_c, float& Bx_c,
                                            float A_p, float Bx_p) {
    Bx_c = A_c * Bx_p + Bx_c;   // order matters: read Bx_c before A_c is stomped
    A_c  = A_c * A_p;
}

// Single-warp inclusive Hillis-Steele scan of the SSM operator. Each lane holds
// step t's (A,Bx); after the loop its Bx register holds h_t. shfl_up pulls the
// partial from `offset` lanes earlier (the "prev" side of the operator); lanes
// with lane_id < offset have no earlier partner and are left unchanged.
__global__ void warp_ssm_scan_kernel(const float* __restrict__ A_in,
                                     const float* __restrict__ Bx_in,
                                     float* __restrict__ h_out) {
    const int lane = threadIdx.x;           // == time step t, block is one warp
    float A  = A_in[lane];
    float Bx = Bx_in[lane];

    #pragma unroll
    for (int offset = 1; offset < WARP; offset *= 2) {
        float A_p  = __shfl_up_sync(FULL_MASK, A,  offset);
        float Bx_p = __shfl_up_sync(FULL_MASK, Bx, offset);
        if (lane >= offset) ssm_combine(A, Bx, A_p, Bx_p);
    }
    h_out[lane] = Bx;                        // Bx now carries h_t
}

int main() {
    int dev = 0;
    if (cudaSetDevice(dev) != cudaSuccess) { printf("No CUDA device.\n"); return 99; }
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // ------------------------------------------------------------- inputs
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> A(WARP), Bx(WARP);
    // Keep |A| < 1 so the recurrence is a stable contraction (as it is after the
    // A = exp(dt * -exp(A_log)) reparam in the real kernel) and the 32-step
    // product doesn't blow past float precision.
    for (auto& e : A)  e = U(rng) * 0.9f;
    for (auto& e : Bx) e = U(rng);

    // --------------------------------------------- sequential CPU reference
    std::vector<float> h_ref(WARP);
    h_ref[0] = Bx[0];
    for (int t = 1; t < WARP; ++t) h_ref[t] = A[t] * h_ref[t-1] + Bx[t];

    // ------------------------------------------------- GPU warp scan
    float *dA, *dBx, *dh;
    CU(cudaMalloc(&dA,  WARP * sizeof(float)));
    CU(cudaMalloc(&dBx, WARP * sizeof(float)));
    CU(cudaMalloc(&dh,  WARP * sizeof(float)));
    CU(cudaMemcpy(dA,  A.data(),  WARP * sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dBx, Bx.data(), WARP * sizeof(float), cudaMemcpyHostToDevice));

    warp_ssm_scan_kernel<<<1, WARP>>>(dA, dBx, dh);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    std::vector<float> h_gpu(WARP);
    CU(cudaMemcpy(h_gpu.data(), dh, WARP * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dBx); cudaFree(dh);

    // ------------------------------------------------------- verify
    float max_abs = 0.f;
    for (int t = 0; t < WARP; ++t)
        max_abs = std::max(max_abs, std::fabs(h_gpu[t] - h_ref[t]));

    printf("\n  t :        CPU h_t          GPU h_t\n");
    for (int t = 0; t < 5; ++t)
        printf("  %d : %16.8f %16.8f\n", t, h_ref[t], h_gpu[t]);
    printf("  ...\n  max|gpu-cpu| over 32 steps = %.3e\n", max_abs);

    // Inclusive scan touches every step; the tail (t=31) folds a 32-term
    // A-product, so allow a little float slack rather than bit-exactness.
    const float eps = 1e-4f;
    bool ok = max_abs < eps;
    for (float x : h_gpu) ok = ok && std::isfinite(x);

    printf("\n==== %s (max diff %.3e, eps %.1e) ====\n",
           ok ? "ALL PASS" : "FAIL", max_abs, eps);
    return ok ? 0 : 1;
}
