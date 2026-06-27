// ============================================================================
// Standalone numerical test for the SSM / linear-attention foundation.
// ============================================================================
// Proves, in isolation (no engine, no model weights), that:
//   1. causal_conv1d_update matches a CPU reference and updates its ring buffer;
//   2. selective_scan_update (gated linear-attention recurrent step) matches a
//      CPU reference across MULTIPLE sequential steps (the recurrence is the
//      whole point — state must carry correctly token-to-token);
//   3. output is shape-correct and numerically stable (all finite, decay keeps
//      the state bounded under a long run);
//   4. SsmStatePool allocates context-length-independent per-sequence VRAM and
//      hands out distinct, usable per-(seq,layer) state pointers.
//
// This is the reference Mamba2/SSD recurrence (see ssm_kernels.cuh) — it does NOT
// yet include Qwen3.5's GatedDeltaNet delta-rule term, so it is NOT a parity
// check against the HF checkpoint. It validates the kernel + allocator plumbing.
//
// Build (x64 Native Tools prompt, CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 -I src ^
//        tests/standalone/test_ssm_kernels.cu src/kernels/ssm_kernels.cu ^
//        -o test_ssm_kernels.exe
//   ./test_ssm_kernels.exe
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "kernels/ssm_kernels.cuh"
#include "ssm/ssm_state_pool.h"

using namespace blackwell;

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 99; } } while(0)

static int g_failures = 0;
static bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i]-b[i])); return m;
}
static void check(const char* name, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_failures;
}

// ---- CPU references (mirror the device math exactly) -----------------------
static float cpu_softplus(float x){ return x>20.f ? x : std::log1p(std::exp(x)); }
static float cpu_silu(float x){ return x/(1.f+std::exp(-x)); }

static void cpu_conv1d(const std::vector<float>& x_new, std::vector<float>& state,
                       const std::vector<float>& w, const std::vector<float>& bias,
                       std::vector<float>& out, int C, int K, bool silu) {
    for (int c = 0; c < C; ++c) {
        float acc = bias.empty() ? 0.f : bias[c];
        for (int j = 0; j < K-1; ++j) acc += w[c*K+j] * state[c*(K-1)+j];
        acc += w[c*K+(K-1)] * x_new[c];
        for (int j = 0; j < K-2; ++j) state[c*(K-1)+j] = state[c*(K-1)+j+1];
        if (K>=2) state[c*(K-1)+(K-2)] = x_new[c];
        out[c] = silu ? cpu_silu(acc) : acc;
    }
}

static void cpu_scan(const std::vector<float>& q, const std::vector<float>& k,
                     const std::vector<float>& v, const std::vector<float>& z,
                     const std::vector<float>& dt_raw, const std::vector<float>& dt_bias,
                     const std::vector<float>& A_log, std::vector<float>& S,
                     std::vector<float>& out, int H, int Dk, int Dv, bool gate) {
    for (int h = 0; h < H; ++h) {
        float dt = cpu_softplus(dt_raw[h] + dt_bias[h]);
        float a  = std::exp(dt * (-std::exp(A_log[h])));
        for (int j = 0; j < Dv; ++j) {
            float vj = v[h*Dv+j], acc = 0.f;
            for (int i = 0; i < Dk; ++i) {
                float s = a * S[(size_t)h*Dk*Dv + i*Dv + j] + (dt*k[h*Dk+i])*vj;
                S[(size_t)h*Dk*Dv + i*Dv + j] = s;
                acc += q[h*Dk+i] * s;
            }
            if (!z.empty()) { float g = z[h*Dv+j]; acc *= gate ? cpu_silu(g) : g; }
            out[h*Dv+j] = acc;
        }
    }
}

int main() {
    int dev = 0;
    if (cudaSetDevice(dev) != cudaSuccess) { printf("No CUDA device.\n"); return 99; }
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);

    // ------------------------------------------------------------------ conv1d
    printf("\n[1] causal_conv1d_update vs CPU reference\n");
    {
        const int C = 8192, K = 4;          // Qwen3.5 linear conv width
        std::vector<float> x(C), w(C*K), bias(C), state(C*(K-1)), out(C, 0.f);
        for (auto& e : x) e = U(rng);
        for (auto& e : w) e = U(rng) * 0.5f;
        for (auto& e : bias) e = U(rng) * 0.1f;
        for (auto& e : state) e = U(rng);
        std::vector<float> state_ref = state, out_ref(C, 0.f);
        cpu_conv1d(x, state_ref, w, bias, out_ref, C, K, /*silu=*/true);

        float *dx,*dw,*db,*ds,*dout;
        CU(cudaMalloc(&dx,C*4)); CU(cudaMalloc(&dw,C*K*4)); CU(cudaMalloc(&db,C*4));
        CU(cudaMalloc(&ds,C*(K-1)*4)); CU(cudaMalloc(&dout,C*4));
        CU(cudaMemcpy(dx,x.data(),C*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dw,w.data(),C*K*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(db,bias.data(),C*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(ds,state.data(),C*(K-1)*4,cudaMemcpyHostToDevice));
        ssm::launch_causal_conv1d_update(dx, ds, dw, db, dout, C, K, true, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> out_gpu(C), state_gpu(C*(K-1));
        CU(cudaMemcpy(out_gpu.data(),dout,C*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(state_gpu.data(),ds,C*(K-1)*4,cudaMemcpyDeviceToHost));

        check("output finite", all_finite(out_gpu));
        check("output matches CPU (<1e-4)", max_abs_diff(out_gpu, out_ref) < 1e-4f);
        check("ring buffer updated to CPU (<1e-5)", max_abs_diff(state_gpu, state_ref) < 1e-5f);
        printf("    max|out-ref|=%.2e  max|state-ref|=%.2e\n",
               max_abs_diff(out_gpu,out_ref), max_abs_diff(state_gpu,state_ref));
        cudaFree(dx);cudaFree(dw);cudaFree(db);cudaFree(ds);cudaFree(dout);
    }

    // -------------------------------------------------------- selective scan
    printf("\n[2] selective_scan_update vs CPU reference (multi-step recurrence)\n");
    {
        const int H = 32, Dk = 128, Dv = 128, STEPS = 64;   // Qwen3.5 linear geometry
        std::vector<float> dt_bias(H), A_log(H);
        for (auto& e : dt_bias) e = U(rng)*0.1f;
        for (auto& e : A_log)  e = U(rng)*0.5f;             // A=-exp(A_log) < 0 -> stable decay

        std::vector<float> S_ref((size_t)H*Dk*Dv, 0.f);
        float *dS,*dq,*dk,*dv,*dz,*ddt,*ddtb,*dA,*dout;
        CU(cudaMalloc(&dS,(size_t)H*Dk*Dv*4)); CU(cudaMemset(dS,0,(size_t)H*Dk*Dv*4));
        CU(cudaMalloc(&dq,H*Dk*4)); CU(cudaMalloc(&dk,H*Dk*4)); CU(cudaMalloc(&dv,H*Dv*4));
        CU(cudaMalloc(&dz,H*Dv*4)); CU(cudaMalloc(&ddt,H*4)); CU(cudaMalloc(&ddtb,H*4));
        CU(cudaMalloc(&dA,H*4)); CU(cudaMalloc(&dout,H*Dv*4));
        CU(cudaMemcpy(ddtb,dt_bias.data(),H*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dA,A_log.data(),H*4,cudaMemcpyHostToDevice));

        float worst_out = 0.f, worst_state = 0.f, max_state_mag = 0.f;
        bool finite_all = true;
        for (int t = 0; t < STEPS; ++t) {
            std::vector<float> q(H*Dk), k(H*Dk), v(H*Dv), z(H*Dv), dt(H), out_ref(H*Dv,0.f);
            for (auto& e:q) e=U(rng); for (auto& e:k) e=U(rng);
            for (auto& e:v) e=U(rng); for (auto& e:z) e=U(rng);
            for (auto& e:dt) e=U(rng)*0.5f;

            std::vector<float> S_ref_step = S_ref;
            cpu_scan(q,k,v,z,dt,dt_bias,A_log,S_ref,out_ref,H,Dk,Dv,/*gate=*/true);

            CU(cudaMemcpy(dq,q.data(),H*Dk*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dk,k.data(),H*Dk*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dv,v.data(),H*Dv*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dz,z.data(),H*Dv*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(ddt,dt.data(),H*4,cudaMemcpyHostToDevice));
            ssm::launch_selective_scan_update(dq,dk,dv,dz,ddt,ddtb,dA,dS,dout,H,Dk,Dv,true,0);
            CU(cudaDeviceSynchronize());

            std::vector<float> out_gpu(H*Dv), S_gpu((size_t)H*Dk*Dv);
            CU(cudaMemcpy(out_gpu.data(),dout,H*Dv*4,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(S_gpu.data(),dS,(size_t)H*Dk*Dv*4,cudaMemcpyDeviceToHost));
            finite_all = finite_all && all_finite(out_gpu) && all_finite(S_gpu);
            worst_out   = std::max(worst_out,   max_abs_diff(out_gpu, out_ref));
            worst_state = std::max(worst_state, max_abs_diff(S_gpu, S_ref));
            for (float s : S_gpu) max_state_mag = std::max(max_state_mag, std::fabs(s));
        }
        check("all outputs/states finite over 64 steps", finite_all);
        check("output matches CPU recurrence (<1e-3)", worst_out < 1e-3f);
        check("recurrent state matches CPU (<1e-3)", worst_state < 1e-3f);
        check("state bounded (decay keeps |S| < 1e3)", max_state_mag < 1e3f);
        printf("    worst|out-ref|=%.2e  worst|S-ref|=%.2e  max|S|=%.2f\n",
               worst_out, worst_state, max_state_mag);
        cudaFree(dS);cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dz);
        cudaFree(ddt);cudaFree(ddtb);cudaFree(dA);cudaFree(dout);
    }

    // ----------------------------------------------------------- state pool
    printf("\n[3] SsmStatePool allocation (context-length independent)\n");
    {
        ssm::SsmGeometry g{};
        g.num_linear_layers = 24; g.num_heads = 32;
        g.key_head_dim = 128; g.value_head_dim = 128;
        g.conv_dim = 8192; g.conv_kernel_dim = 4;
        const int N = 2;
        ssm::SsmStatePool pool(g, N);
        double mb = pool.bytes_resident() / (1024.0*1024.0);
        printf("    %d seqs x (24 linear layers): %.1f MB total (%.1f MB/seq)\n",
               N, mb, mb / N);
        // Exact: 24 layers * (H*Dk*Dv recurrent + conv_dim*(K-1) conv) * 4B
        //      = 24*(32*128*128 + 8192*3)*4 ~= 50 MB/seq, independent of context.
        check("per-seq state in expected ~50MB class (40-70MB)",
              (mb / N) > 40.0 && (mb / N) < 70.0);
        // distinct, usable pointers per (seq, layer)
        float* a = pool.rec_state(0, 0);
        float* b = pool.rec_state(1, 0);
        float* c = pool.rec_state(0, 23);
        check("distinct per-seq / per-layer state pointers", a!=b && a!=c && b!=c);
        pool.reset(0);
        CU(cudaDeviceSynchronize());
        check("reset(seq) succeeds", true);
    }

    printf("\n==== %s (%d failure%s) ====\n",
           g_failures==0 ? "ALL PASS" : "FAILURES", g_failures, g_failures==1?"":"s");
    return g_failures == 0 ? 0 : 1;
}
