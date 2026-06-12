// Validation for the GPU<->CPU offloading substrate: the pinned host pool and
// the event-ordered double-buffer handshake used by VRAMArena's layer
// streaming. No model checkpoint is required; these tests exercise the memory
// routing protocol itself (pinned allocations, non-blocking transfer stream,
// retire/ready event ordering, strided KV column spills).
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#include "memory_pool.h"
#include "common.h"

TEST(PinnedHostPool, ReserveAllocateAlignment) {
    PinnedHostPool pool(1 << 20);
    ASSERT_NE(pool.base(), nullptr);
    EXPECT_EQ(pool.capacity(), size_t(1) << 20);

    void* a = pool.allocate(3);   // deliberately unaligned size
    void* b = pool.allocate(64);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(a) % 16, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b) % 16, 0u);
    // 3 bytes round up to a 16-byte bump
    EXPECT_EQ(static_cast<uint8_t*>(b) - static_cast<uint8_t*>(a), 16);

    // The memory must be genuinely page-locked: cudaHostGetDevicePointer-level
    // checks are platform-dependent, but cudaPointerGetAttributes must classify
    // it as host-registered.
    cudaPointerAttributes attr{};
    CUDA_CHECK(cudaPointerGetAttributes(&attr, a));
    EXPECT_EQ(attr.type, cudaMemoryTypeHost);
}

TEST(PinnedHostPool, ExhaustionAndDoubleReserveThrow) {
    PinnedHostPool pool(256);
    EXPECT_THROW(pool.allocate(512), std::bad_alloc);
    EXPECT_THROW(pool.reserve(128), std::logic_error);

    void* p = pool.allocate(256); // exact fit still succeeds
    EXPECT_NE(p, nullptr);
    EXPECT_THROW(pool.allocate(1), std::bad_alloc);
}

TEST(PinnedHostPool, AsyncRoundTripOnNonBlockingStream) {
    constexpr size_t kFloats = 1 << 16;
    PinnedHostPool pool(2 * kFloats * sizeof(float));
    auto* h_src = static_cast<float*>(pool.allocate(kFloats * sizeof(float)));
    auto* h_dst = static_cast<float*>(pool.allocate(kFloats * sizeof(float)));
    for (size_t i = 0; i < kFloats; ++i) h_src[i] = static_cast<float>(i % 977) * 0.5f;
    std::memset(h_dst, 0, kFloats * sizeof(float));

    CudaVector<float> d_buf(kFloats);

    cudaStream_t transfer = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(d_buf.d_ptr, h_src, kFloats * sizeof(float),
                               cudaMemcpyHostToDevice, transfer));
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_buf.d_ptr, kFloats * sizeof(float),
                               cudaMemcpyDeviceToHost, transfer));
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    CUDA_CHECK(cudaStreamDestroy(transfer));

    EXPECT_EQ(std::memcmp(h_src, h_dst, kFloats * sizeof(float)), 0);
}

// Replays VRAMArena's weight-slot protocol: transfer stream stages "layer" i+1
// into slot (i+1)%2 while the legacy stream consumes slot i%2, ordered solely
// by ev_retire / ev_ready. Any handshake bug shows up as a torn or stale layer
// pattern in the per-layer outputs.
TEST(AsyncOffload, PingPongDoubleBufferConsistency) {
    constexpr int    kLayers     = 8;
    constexpr size_t kLayerFloats = 1 << 15;
    constexpr size_t kLayerBytes  = kLayerFloats * sizeof(float);

    PinnedHostPool pool((kLayers + kLayers) * kLayerBytes);
    float* h_layers[kLayers];
    for (int l = 0; l < kLayers; ++l) {
        h_layers[l] = static_cast<float*>(pool.allocate(kLayerBytes));
        for (size_t i = 0; i < kLayerFloats; ++i)
            h_layers[l][i] = static_cast<float>(l * 1000) + static_cast<float>(i % 251);
    }
    float* h_out[kLayers];
    for (int l = 0; l < kLayers; ++l) {
        h_out[l] = static_cast<float*>(pool.allocate(kLayerBytes));
        std::memset(h_out[l], 0, kLayerBytes);
    }

    CudaVector<float> d_slot0(kLayerFloats), d_slot1(kLayerFloats);
    float* d_slots[2] = {d_slot0.d_ptr, d_slot1.d_ptr};
    CudaVector<float> d_consume(kLayerFloats); // stand-in for compute reading the slot

    cudaStream_t transfer = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    cudaEvent_t ev_ready[2], ev_retire[2];
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_ready[s],  cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_retire[s], cudaEventDisableTiming));
    }

    auto stage = [&](int layer) {
        const int s = layer % 2;
        // Compute work consuming the slot's previous occupant is already
        // enqueued on stream 0; do not overwrite before it drains.
        CUDA_CHECK(cudaEventRecord(ev_retire[s], 0));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ev_retire[s], 0));
        CUDA_CHECK(cudaMemcpyAsync(d_slots[s], h_layers[layer], kLayerBytes,
                                   cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(ev_ready[s], transfer));
    };

    stage(0); // cold start
    for (int l = 0; l < kLayers; ++l) {
        const int s = l % 2;
        if (l + 1 < kLayers) stage(l + 1); // prefetch next layer concurrently

        // "Compute" on the legacy stream: wait for staging, read the slot
        // through a device-side hop, push the result back to pinned host.
        CUDA_CHECK(cudaStreamWaitEvent(0, ev_ready[s], 0));
        CUDA_CHECK(cudaMemcpyAsync(d_consume.d_ptr, d_slots[s], kLayerBytes,
                                   cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(h_out[l], d_consume.d_ptr, kLayerBytes,
                                   cudaMemcpyDeviceToHost, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int l = 0; l < kLayers; ++l)
        ASSERT_EQ(std::memcmp(h_out[l], h_layers[l], kLayerBytes), 0)
            << "Slot handshake corrupted layer " << l;

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaEventDestroy(ev_ready[s]));
        CUDA_CHECK(cudaEventDestroy(ev_retire[s]));
    }
    CUDA_CHECK(cudaStreamDestroy(transfer));
}

// The KV spill path copies one token column out of the strided
// [kv_heads, max_seq_len, head_dim] layout with cudaMemcpy2DAsync; verify the
// strided geometry (the exact call VRAMArena::commit_layer_kv makes).
TEST(AsyncOffload, StridedKvColumnSpill) {
    constexpr size_t kHeads = 4, kMaxSeq = 64, kHeadDim = 32;
    constexpr size_t kTotal = kHeads * kMaxSeq * kHeadDim;
    constexpr int    kPos = 17;

    std::vector<float> h_init(kTotal);
    for (size_t i = 0; i < kTotal; ++i) h_init[i] = static_cast<float>(i);

    CudaVector<float> d_cache(kTotal);
    d_cache.upload(h_init);

    PinnedHostPool pool(kTotal * sizeof(float));
    auto* h_mirror = static_cast<float*>(pool.allocate(kTotal * sizeof(float)));
    std::memset(h_mirror, 0, kTotal * sizeof(float));

    cudaStream_t transfer = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));

    const size_t pitch = kMaxSeq * kHeadDim * sizeof(float);
    const size_t col   = static_cast<size_t>(kPos) * kHeadDim;
    CUDA_CHECK(cudaMemcpy2DAsync(h_mirror + col, pitch, d_cache.d_ptr + col, pitch,
                                 kHeadDim * sizeof(float), kHeads,
                                 cudaMemcpyDeviceToHost, transfer));
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    CUDA_CHECK(cudaStreamDestroy(transfer));

    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t t = 0; t < kMaxSeq; ++t) {
            for (size_t d = 0; d < kHeadDim; ++d) {
                const size_t idx = h * kMaxSeq * kHeadDim + t * kHeadDim + d;
                const float expected = (t == kPos) ? h_init[idx] : 0.0f;
                ASSERT_EQ(h_mirror[idx], expected)
                    << "head " << h << " token " << t << " dim " << d;
            }
        }
    }
}
