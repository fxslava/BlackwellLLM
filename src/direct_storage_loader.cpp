#ifdef USE_DIRECT_STORAGE

// Batched asynchronous DirectStorage weight loader.
//
// Architecture (enqueue many, submit once):
//   load_to_vram() never touches the disk: it carves a sector-aligned region
//   out of the current pinned staging arena and records a pending request.
//   When an arena fills up, all of its requests are pushed to the
//   IDStorageQueue as ONE batch (single Submit + single fence signal) and
//   filling switches to the second arena. Completing an arena waits for its
//   fence once, performs the (<4KB) EOF residual reads, and drains every
//   region to VRAM with cudaMemcpyAsync on a dedicated stream. The result is
//   a pipeline where NVMe reads of batch N+1 overlap the PCIe copies of
//   batch N. flush() is the completion barrier for everything enqueued.
//
// Alignment & tail-read contract (BypassIO-safe):
//   - DS file offsets are rounded DOWN to the 4KB sector, DS sizes rounded UP
//     to a sector multiple, and the DS read never extends past
//     round_down(file_size, 4KB): the unaligned EOF tail ("residual") is read
//     with a positioned Win32 ReadFile into the same staging region instead,
//     so DirectStorage only ever sees fully aligned requests.
//   - Every DS destination pointer is 4KB-aligned by construction: staging
//     regions start on sector boundaries inside cudaHostAlloc'd (page-aligned)
//     arenas. Unaligned VRAM destinations are irrelevant — the final hop is a
//     cudaMemcpyAsync from pinned RAM, which has no alignment constraints.
//
// Failure handling:
//   - DirectStorage init failure  -> IWeightLoader::create() falls back to
//     the StandardLoader.
//   - A failed DS batch (error record) -> the whole batch is re-read through
//     Win32 ReadFile into the same pinned regions and DS is disabled for the
//     remainder of the load; correctness never depends on DS succeeding.

#include "blackwell/weight_loader.h"

#include <windows.h>
#include <d3d12.h>
#include <dstorage.h>
#include <wrl/client.h>
#include <comdef.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr size_t kSector     = 4096;          // BypassIO alignment quantum
constexpr size_t kArenaBytes = 128ull << 20;  // pinned staging arena, x2
constexpr size_t kMaxChunk   = 16ull << 20;   // split cap for huge tensors

constexpr size_t align_down(size_t v, size_t a) { return v & ~(a - 1); }
constexpr size_t align_up(size_t v, size_t a)   { return (v + a - 1) & ~(a - 1); }

std::string hr_str(HRESULT hr) {
    char hex[16];
    snprintf(hex, sizeof(hex), "0x%08X", static_cast<unsigned int>(hr));
    _com_error err(hr);
    return std::string(hex) + " (" + err.ErrorMessage() + ")";
}

void throw_hr(const char* what, HRESULT hr) {
    throw std::runtime_error(std::string("DirectStorage: ") + what + " failed - " + hr_str(hr));
}

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("DirectStorage: ") + what + " failed: " +
                                 cudaGetErrorString(err));
}

std::wstring utf8_to_utf16(const std::string& str) {
    if (str.empty()) return L"";
    int size = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size);
    return wstr;
}

} // namespace

class DirectStorageLoader : public IWeightLoader {
private:
    struct FileEntry {
        ComPtr<IDStorageFile> ds_file;
        HANDLE   win32 = INVALID_HANDLE_VALUE;  // residual / fallback reads
        uint64_t size  = 0;
        std::string path;
    };

    // One enqueued tensor (or tensor chunk) inside a staging arena.
    struct Pending {
        FileEntry* file;
        uint64_t ds_offset;     // sector-aligned file offset of the DS read
        size_t   ds_size;       // sector-multiple DS read size; 0 = no DS request
        uint64_t res_offset;    // file offset of the EOF residual ReadFile
        size_t   res_size;      // residual bytes (< 4KB); 0 = none
        size_t   stage_offset;  // arena-relative, sector-aligned (maps ds_offset)
        size_t   shift;         // payload start inside the region
        size_t   payload;       // bytes to copy to the device
        uint8_t* d_dst;
    };

    struct Arena {
        // Idle: reusable/being filled. DsInflight: batch submitted to the DS
        // queue. Draining: cudaMemcpyAsync issued, ev_drained pending.
        enum class State { Idle, DsInflight, Draining };
        uint8_t* base = nullptr;            // cudaHostAlloc'd, page-aligned
        size_t   used = 0;
        State    state = State::Idle;
        bool     ds_submitted = false;      // DS requests actually enqueued
        uint64_t fence_value = 0;
        cudaEvent_t ev_drained = nullptr;
        std::vector<Pending> pending;
    };

    ComPtr<ID3D12Device>     m_device;
    ComPtr<IDStorageFactory> m_factory;
    ComPtr<IDStorageQueue>   m_queue;
    ComPtr<ID3D12Fence>      m_fence;
    HANDLE   m_fence_event = nullptr;
    uint64_t m_fence_value = 0;

    std::unordered_map<std::string, FileEntry> m_files;

    Arena m_arenas[2];
    int   m_cur = 0;
    cudaStream_t m_stream = nullptr;

    // Set on the first failed batch: all subsequent batches bypass the DS
    // queue and read synchronously through Win32 (still pinned + async copy).
    bool m_ds_disabled = false;

    // Telemetry proving which path the bytes actually took.
    uint64_t m_stat_batches = 0, m_stat_ds_requests = 0, m_stat_ds_bytes = 0;
    uint64_t m_stat_residuals = 0, m_stat_residual_bytes = 0, m_stat_fallback_bytes = 0;

    Arena& current() { return m_arenas[m_cur]; }

    // ------------------------------------------------------------------
    // Files
    // ------------------------------------------------------------------
    FileEntry& get_or_open_file(const std::string& path) {
        auto it = m_files.find(path);
        if (it != m_files.end()) return it->second;

        std::wstring wpath = utf8_to_utf16(path);

        // Win32 handle first: it also resolves the authoritative file size and
        // serves residual/fallback reads.
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE)
            throw std::runtime_error("DirectStorage: CreateFileW failed for " + path +
                                     " (error " + std::to_string(GetLastError()) + ")");
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(h, &sz)) {
            CloseHandle(h);
            throw std::runtime_error("DirectStorage: GetFileSizeEx failed for " + path);
        }

        ComPtr<IDStorageFile> ds_file;
        HRESULT hr = m_factory->OpenFile(wpath.c_str(), IID_PPV_ARGS(&ds_file));
        if (FAILED(hr)) {
            CloseHandle(h);
            throw std::runtime_error("DirectStorage: failed to open file " + path +
                                     " - " + hr_str(hr));
        }

        FileEntry& fe = m_files[path];
        fe.ds_file = std::move(ds_file);
        fe.win32   = h;
        fe.size    = static_cast<uint64_t>(sz.QuadPart);
        fe.path    = path;
        return fe;
    }

    // Positioned synchronous read; used for EOF residuals (<4KB) and as the
    // whole-batch fallback when a DS batch reports an error record.
    static void read_file_region(const FileEntry& fe, uint64_t offset, size_t size, void* dst) {
        uint8_t* out = static_cast<uint8_t*>(dst);
        size_t remaining = size;
        while (remaining > 0) {
            OVERLAPPED ov{};
            ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFull);
            ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
            const DWORD want = static_cast<DWORD>((std::min)(remaining, size_t(1) << 30));
            DWORD got = 0;
            if (!ReadFile(fe.win32, out, want, &got, &ov) || got == 0)
                throw std::runtime_error("DirectStorage: Win32 read of " +
                                         std::to_string(size) + " bytes @" +
                                         std::to_string(offset) + " failed in " + fe.path +
                                         " (error " + std::to_string(GetLastError()) + ")");
            out += got;
            offset += got;
            remaining -= got;
        }
    }

    void wait_fence(uint64_t value) {
        if (m_fence->GetCompletedValue() >= value) return;
        HRESULT hr = m_fence->SetEventOnCompletion(value, m_fence_event);
        if (FAILED(hr)) throw_hr("SetEventOnCompletion", hr);
        if (WaitForSingleObject(m_fence_event, INFINITE) != WAIT_OBJECT_0)
            throw std::runtime_error("DirectStorage: fence wait failed");
    }

    // ------------------------------------------------------------------
    // Batch pipeline
    // ------------------------------------------------------------------

    // Push every DS request of the arena to the queue and signal its fence.
    // One Submit per batch (plus defensive intermediate kicks well below the
    // queue capacity so EnqueueRequest can never block on a full, unsubmitted
    // queue).
    void submit_arena(Arena& a) {
        if (a.state != Arena::State::Idle || a.pending.empty()) return;

        a.ds_submitted = false;
        if (!m_ds_disabled) {
            uint32_t since_kick = 0;
            for (const Pending& p : a.pending) {
                if (p.ds_size == 0) continue;

                DSTORAGE_REQUEST r{};
                r.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                r.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                r.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                r.Source.File.Source        = p.file->ds_file.Get();
                r.Source.File.Offset        = p.ds_offset;
                r.Source.File.Size          = static_cast<UINT32>(p.ds_size);
                r.Destination.Memory.Buffer = a.base + p.stage_offset;
                r.Destination.Memory.Size   = static_cast<UINT32>(p.ds_size);

                m_queue->EnqueueRequest(&r);
                a.ds_submitted = true;
                ++m_stat_ds_requests;
                m_stat_ds_bytes += p.ds_size;
                if (++since_kick == 1024) { m_queue->Submit(); since_kick = 0; }
            }
            if (a.ds_submitted) {
                a.fence_value = ++m_fence_value;
                m_queue->EnqueueSignal(m_fence.Get(), a.fence_value);
                m_queue->Submit();
                ++m_stat_batches;
            }
        }
        a.state = Arena::State::DsInflight;
    }

    // Wait for the arena's DS batch, fix up residuals/failures on the host,
    // then drain every region to VRAM asynchronously.
    void complete_arena(Arena& a) {
        if (a.state != Arena::State::DsInflight) return;

        bool ds_ok = false;
        if (a.ds_submitted) {
            wait_fence(a.fence_value);

            DSTORAGE_ERROR_RECORD rec{};
            m_queue->RetrieveErrorRecord(&rec);
            if (FAILED(rec.FirstFailure.HResult)) {
                std::cerr << "[DirectStorage] batch of " << a.pending.size()
                          << " requests failed (" << hr_str(rec.FirstFailure.HResult)
                          << "); re-reading via Win32 and disabling DirectStorage "
                             "for the rest of this load.\n";
                m_ds_disabled = true;
            } else {
                ds_ok = true;
            }
        }

        for (const Pending& p : a.pending) {
            // Whole-span fallback when the DS batch failed or was never enqueued.
            if (!ds_ok && p.ds_size > 0) {
                read_file_region(*p.file, p.ds_offset, p.ds_size, a.base + p.stage_offset);
                m_stat_fallback_bytes += p.ds_size;
            }

            // EOF residual: the final partial sector DS is not allowed to touch.
            if (p.res_size > 0) {
                read_file_region(*p.file, p.res_offset, p.res_size,
                                 a.base + p.stage_offset + (p.res_offset - p.ds_offset));
                ++m_stat_residuals;
                m_stat_residual_bytes += p.res_size;
            }

            cuda_check(cudaMemcpyAsync(p.d_dst, a.base + p.stage_offset + p.shift,
                                       p.payload, cudaMemcpyHostToDevice, m_stream),
                       "staging cudaMemcpyAsync");
        }
        cuda_check(cudaEventRecord(a.ev_drained, m_stream), "cudaEventRecord");
        a.pending.clear();
        a.state = Arena::State::Draining;
    }

    // Bring an arena back to Idle so it can be refilled: finish its DS batch
    // if needed, then wait until the GPU has consumed its previous contents.
    void reclaim_arena(Arena& a) {
        if (a.state == Arena::State::DsInflight) complete_arena(a);
        if (a.state == Arena::State::Draining) {
            cuda_check(cudaEventSynchronize(a.ev_drained), "cudaEventSynchronize");
            a.state = Arena::State::Idle;
        }
        a.used = 0;
    }

    void rotate_arena() {
        submit_arena(current());
        m_cur ^= 1;
        reclaim_arena(current());
    }

    // Alignment math + staging placement for one tensor chunk (<= kMaxChunk).
    void enqueue_one(const std::string& filepath, uint64_t offset, size_t size, uint8_t* d_dst) {
        FileEntry& fe = get_or_open_file(filepath);
        if (offset + size > fe.size)
            throw std::runtime_error("DirectStorage: read [" + std::to_string(offset) + ", " +
                                     std::to_string(offset + size) + ") exceeds size " +
                                     std::to_string(fe.size) + " of " + filepath);

        const uint64_t aligned_begin = align_down(offset, kSector);
        const size_t   shift         = static_cast<size_t>(offset - aligned_begin);
        const uint64_t want_end      = offset + size;

        // DS may only read whole sectors that exist on disk. Clamp the aligned
        // end to the last full sector; whatever remains is the residual.
        uint64_t ds_end = align_up(want_end, kSector);
        const uint64_t last_full_sector = align_down(fe.size, kSector);
        if (ds_end > fe.size) {
            ds_end = (std::max)(last_full_sector, aligned_begin);
        }
        const size_t ds_size      = static_cast<size_t>(ds_end - aligned_begin);
        const size_t region_bytes = (std::max)(ds_size, shift + size);

        size_t stage_off = align_up(current().used, kSector);
        if (stage_off + region_bytes > kArenaBytes) {
            rotate_arena();
            stage_off = 0;  // freshly reclaimed arena; region always fits
        }
        Arena& a = current();
        a.used = stage_off + region_bytes;

        Pending p{};
        p.file         = &fe;
        p.ds_offset    = aligned_begin;
        p.ds_size      = ds_size;
        p.res_offset   = (std::max)(ds_end, offset);
        p.res_size     = want_end > p.res_offset ? static_cast<size_t>(want_end - p.res_offset) : 0;
        p.stage_offset = stage_off;
        p.shift        = shift;
        p.payload      = size;
        p.d_dst        = d_dst;
        a.pending.push_back(p);
    }

    // ------------------------------------------------------------------
    // Lifetime
    // ------------------------------------------------------------------

    // Shared by the destructor and constructor failure paths so a partially
    // initialized loader never leaks handles, pinned RAM or CUDA objects.
    // ComPtr members release themselves.
    void release_resources() noexcept {
        // Never free pinned memory while DirectStorage may still DMA into it.
        if (m_fence && m_fence_value > 0 &&
            m_fence->GetCompletedValue() < m_fence_value && m_fence_event) {
            if (SUCCEEDED(m_fence->SetEventOnCompletion(m_fence_value, m_fence_event)))
                WaitForSingleObject(m_fence_event, 10000);
        }
        if (m_stream) cudaStreamSynchronize(m_stream);

        for (Arena& a : m_arenas) {
            if (a.ev_drained) { cudaEventDestroy(a.ev_drained); a.ev_drained = nullptr; }
            if (a.base)       { cudaFreeHost(a.base);           a.base = nullptr; }
        }
        if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }

        for (auto& [path, fe] : m_files) {
            if (fe.win32 != INVALID_HANDLE_VALUE) CloseHandle(fe.win32);
        }
        m_files.clear();

        if (m_fence_event) { CloseHandle(m_fence_event); m_fence_event = nullptr; }
    }

public:
    DirectStorageLoader() {
        try {
            HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device));
            if (FAILED(hr)) throw_hr("D3D12CreateDevice", hr);

            hr = DStorageGetFactory(IID_PPV_ARGS(&m_factory));
            if (FAILED(hr)) throw_hr("DStorageGetFactory", hr);

            DSTORAGE_QUEUE_DESC queueDesc{};
            queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            queueDesc.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
            queueDesc.Priority   = DSTORAGE_PRIORITY_NORMAL;
            queueDesc.Name       = "BlackwellLLM weight batch queue";
            hr = m_factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&m_queue));
            if (FAILED(hr)) throw_hr("CreateQueue", hr);

            m_fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (!m_fence_event)
                throw std::runtime_error("DirectStorage: failed to create fence event");
            hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
            if (FAILED(hr)) throw_hr("CreateFence", hr);

            cuda_check(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking),
                       "cudaStreamCreate");
            for (Arena& a : m_arenas) {
                // cudaHostAlloc returns page-aligned pinned memory: with
                // sector-aligned staging offsets every DS destination pointer
                // is 4KB-aligned, and the H2D hop is a true async DMA.
                void* p = nullptr;
                cuda_check(cudaHostAlloc(&p, kArenaBytes, cudaHostAllocDefault),
                           "cudaHostAlloc(staging arena)");
                a.base = static_cast<uint8_t*>(p);
                cuda_check(cudaEventCreateWithFlags(&a.ev_drained, cudaEventDisableTiming),
                           "cudaEventCreate");
            }
        } catch (...) {
            // The destructor will not run for a throwing constructor.
            release_resources();
            throw;
        }
    }

    ~DirectStorageLoader() override {
        // Forgetting flush() must not corrupt memory: complete any in-flight
        // work (best effort) before tearing the staging arenas down.
        try {
            flush();
        } catch (const std::exception& e) {
            std::cerr << "[DirectStorage] flush during teardown failed: " << e.what() << "\n";
        }
        release_resources();
    }

    void load_to_vram(const std::string& filepath, size_t offset, size_t size, void* d_ptr) override {
        if (size == 0) return;
        if (!d_ptr) throw std::runtime_error("DirectStorage: null destination for " + filepath);

        // Oversized tensors are split so a single request can never exceed an
        // arena (and stays far below DS's per-request staging limits). Each
        // chunk re-derives its own sector alignment.
        uint8_t* dst = static_cast<uint8_t*>(d_ptr);
        while (size > 0) {
            const size_t chunk = (std::min)(size, kMaxChunk);
            enqueue_one(filepath, offset, chunk, dst);
            offset += chunk;
            size   -= chunk;
            dst    += chunk;
        }
    }

    // Completion barrier: submit the partial batch, retire both arenas, and
    // join the copy stream. After this every byte handed to load_to_vram() is
    // in VRAM.
    void flush() override {
        submit_arena(current());
        reclaim_arena(m_arenas[m_cur ^ 1]);  // older batch first
        reclaim_arena(m_arenas[m_cur]);
        cuda_check(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");

        if (m_stat_batches || m_stat_fallback_bytes) {
            std::cout << "[DirectStorage] flush: " << m_stat_batches << " batches, "
                      << m_stat_ds_requests << " aligned DS requests ("
                      << m_stat_ds_bytes / (1024.0 * 1024.0) << " MB), "
                      << m_stat_residuals << " tail reads ("
                      << m_stat_residual_bytes << " B), Win32 fallback: "
                      << m_stat_fallback_bytes / (1024.0 * 1024.0) << " MB\n";
            m_stat_batches = m_stat_ds_requests = m_stat_ds_bytes = 0;
            m_stat_residuals = m_stat_residual_bytes = m_stat_fallback_bytes = 0;
        }
    }
};

std::unique_ptr<IWeightLoader> IWeightLoader::create() {
    try {
        return std::make_unique<DirectStorageLoader>();
    } catch (const std::exception& e) {
        std::cerr << "[DirectStorage] init failed (" << e.what()
                  << "); falling back to standard I/O loader.\n";
        return create_standard_loader();
    }
}
#endif
