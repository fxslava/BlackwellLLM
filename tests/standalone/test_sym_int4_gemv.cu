// ============================================================================
// Standalone self-consistency test for the symmetric INT4 (compressed-tensors
// pack-quantized) GEMV kernel.
// ============================================================================
// Proves the kernel correctly inverts the documented packing: we quantize a
// random fp weight to symmetric int4 per group of 32, pack 8 nibbles/int32 in
// the compressed-tensors layout (element t -> bits [4t,4t+4), two's complement),
// store bf16 group scales, and verify W·x on the GPU matches a CPU dequant-matmul.
//
// This pins the KERNEL math and our packing interpretation against each other; it
// does NOT prove the convention matches a real checkpoint — that is the job of the
// golden-dump integration test. Shapes here mirror the real Qwen3.5 in_proj_qkv
// (out=8192, in=4096, group_size=32).
//
// Build (x64 Native Tools prompt, CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 -I src ^
//        tests/standalone/test_sym_int4_gemv.cu src/kernels/sym_int4_linear.cu ^
//        -o test_sym_int4_gemv.exe
//   ./test_sym_int4_gemv.exe
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/sym_int4_linear.cuh"

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 99; } } while(0)

int main() {
    if (cudaSetDevice(0) != cudaSuccess) { printf("No CUDA device.\n"); return 99; }
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const int OUT = 8192, IN = 4096, GS = 32;     // Qwen3.5 in_proj_qkv geometry
    const int words = IN / 8, groups = IN / GS;
    std::mt19937 rng(7);
    std::normal_distribution<float> N(0.f, 0.1f);
    std::uniform_real_distribution<float> Ux(-1.f, 1.f);

    // Random activation.
    std::vector<float> x(IN); for (auto& e : x) e = Ux(rng);

    // Quantize a random fp weight to symmetric int4 per group of 32, and build the
    // packed int32 rows + bf16 scales using the SAME convention the kernel decodes.
    std::vector<int32_t>       packed((size_t)OUT * words, 0);
    std::vector<__nv_bfloat16> scales((size_t)OUT * groups);
    std::vector<float>         y_ref(OUT, 0.f);

    for (int o = 0; o < OUT; ++o) {
        for (int g = 0; g < groups; ++g) {
            // group of GS input weights
            float wq[GS]; float maxabs = 1e-8f;
            for (int j = 0; j < GS; ++j) { wq[j] = N(rng); maxabs = std::max(maxabs, std::fabs(wq[j])); }
            float scale = maxabs / 7.0f;                       // symmetric, levels [-7,7] (within [-8,7])
            scales[(size_t)o * groups + g] = __float2bfloat16(scale);
            float sbf = __bfloat162float(scales[(size_t)o * groups + g]);
            for (int j = 0; j < GS; ++j) {
                int q = (int)std::lround(wq[j] / scale);
                if (q < -8) q = -8; if (q > 7) q = 7;          // clamp to int4 range
                const int in_idx = g * GS + j;
                const int p = in_idx / 8, t = in_idx % 8;      // word + nibble slot
                packed[(size_t)o * words + p] |= (uint32_t)(q & 0xF) << (4 * t);
                y_ref[o] += x[in_idx] * (float)q * sbf;          // CPU dequant-matmul
            }
        }
    }

    int32_t* d_packed; __nv_bfloat16* d_scales; float *d_x, *d_y;
    CU(cudaMalloc(&d_packed, packed.size()*4)); CU(cudaMalloc(&d_scales, scales.size()*sizeof(__nv_bfloat16)));
    CU(cudaMalloc(&d_x, IN*4)); CU(cudaMalloc(&d_y, OUT*4));
    CU(cudaMemcpy(d_packed, packed.data(), packed.size()*4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_scales, scales.data(), scales.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_x, x.data(), IN*4, cudaMemcpyHostToDevice));

    launch_sym_int4_gemv(d_packed, d_scales, d_x, d_y, OUT, IN, GS, 0);
    CU(cudaDeviceSynchronize());
    std::vector<float> y_gpu(OUT);
    CU(cudaMemcpy(y_gpu.data(), d_y, OUT*4, cudaMemcpyDeviceToHost));

    // Compare GEMV vs CPU dequant-matmul.
    double dot=0, ng=0, na=0; float maxd=0; bool finite=true;
    for (int o = 0; o < OUT; ++o) {
        if (!std::isfinite(y_gpu[o])) finite=false;
        maxd = std::max(maxd, std::fabs(y_gpu[o]-y_ref[o]));
        dot += (double)y_ref[o]*y_gpu[o]; ng += (double)y_ref[o]*y_ref[o]; na += (double)y_gpu[o]*y_gpu[o];
    }
    double cos = dot / (std::sqrt(ng)*std::sqrt(na) + 1e-12);

    // Residual variant: y2 = base + W·x must equal base + y_ref.
    std::vector<float> base(OUT); for (auto& e : base) e = Ux(rng);
    CU(cudaMemcpy(d_y, base.data(), OUT*4, cudaMemcpyHostToDevice));
    launch_sym_int4_gemv_residual(d_packed, d_scales, d_x, d_y, OUT, IN, GS, 0);
    CU(cudaDeviceSynchronize());
    std::vector<float> y2(OUT); CU(cudaMemcpy(y2.data(), d_y, OUT*4, cudaMemcpyDeviceToHost));
    float maxd_res=0; for (int o=0;o<OUT;++o) maxd_res=std::max(maxd_res, std::fabs(y2[o]-(base[o]+y_ref[o])));

    int fails = 0;
    auto check=[&](const char*n,bool ok){ printf("  [%s] %s\n", ok?"PASS":"FAIL", n); if(!ok)++fails; };
    printf("\n[sym int4 GEMV] out=%d in=%d gs=%d\n", OUT, IN, GS);
    check("output finite", finite);
    check("GEMV matches CPU dequant-matmul (cos > 0.9999)", cos > 0.9999);
    check("GEMV max abs diff < 1e-3", maxd < 1e-3f);
    check("residual variant matches base + W*x (< 1e-3)", maxd_res < 1e-3f);
    printf("    cos=%.8f  max|y-ref|=%.2e  max|res-ref|=%.2e\n", cos, maxd, maxd_res);

    printf("\n==== %s (%d failure%s) ====\n", fails==0?"ALL PASS":"FAILURES", fails, fails==1?"":"s");
    return fails==0 ? 0 : 1;
}
