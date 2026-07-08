#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "common.h"  // CUDA_CHECK_THROW (Hybrid error doctrine)
#include "paged_kv_cache.h"
#include "tiered_memory_pager.h"

// ============================================================================
// CUDA / NVMe implementations of the TieredMemoryPager interfaces
// ============================================================================
//   SmVramPool      — VramPool over SequenceManager's PageAllocator.
//   CudaTierBackend — TierBackend over:
//       RAM tier : one cudaHostAlloc pinned arena, ram_slots fixed records
//       DISK tier: one preallocated spill file, disk_slots fixed records
//
// PAGE RECORD layout (both tiers): for layer in [0, L): { K blob, V blob },
// page_bytes each — byte-identical to one page of a .bkv payload
// (kv_branch_serializer.h). One record format across RAM, spill file and
// shipped .bkv archives means one restore path and, later, one DirectStorage
// request shape for both the spill file and warmup archives.
//
// Streams & the fence contract:
//   * Demotions (vram_to_ram) run on the internal transfer stream and are
//     HOST-SYNCHRONOUS on return (the pager may immediately reuse/persist the
//     slot). Demotion is a cold-path event; the sync is the correctness seam.
//   * Promotions (ram_to_vram / disk_to_vram) are ASYNC H2D enqueues on the
//     transfer stream. fence(cs) records one event after the batch: cs != null
//     => cudaStreamWaitEvent(cs) — the compute stream yields until the pages
//     land, the host keeps going; cs == null => host blocks.
//   * Disk reads/writes are host-blocking fstream I/O in v1 (sequential,
//     4 KiB-multiple records). The async upgrade (IO worker + DirectStorage)
//     is an implementation swap behind this interface — see
//     docs/TIERED_KV_AND_AOT.md.
//
// Offload interplay: read_page/write_page route resident layers through
// device<->pinned DMA and KV-offloaded layers through their pinned host
// mirror, so tiering composes with layer offloading with no special cases.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

class SmVramPool : public VramPool {
public:
    explicit SmVramPool(SequenceManager& sm) : m_sm(sm) {}
    PageId try_allocate() override {
        return m_sm.free_pages() > 0 ? m_sm.allocate_raw_page() : PageId(-1);
    }
    void retain(PageId p)  override { m_sm.retain_page(p); }
    void release(PageId p) override { m_sm.release_page(p); }
    int  free_pages() const override { return m_sm.free_pages(); }
private:
    SequenceManager& m_sm;
};

class CudaTierBackend : public TierBackend {
public:
    CudaTierBackend(SequenceManager& sm, int ram_slots, int disk_slots,
                    const std::string& spill_path)
        : m_sm(sm), m_ram_slots(ram_slots), m_disk_slots(disk_slots) {
        m_page_bytes   = m_sm.page_elems() * sizeof(kv_t);
        m_record_bytes = (size_t)m_sm.num_layers() * 2 * m_page_bytes;

        if (ram_slots > 0)
            CUDA_CHECK_THROW(cudaHostAlloc(&m_arena, (size_t)ram_slots * m_record_bytes,
                                     cudaHostAllocDefault));
        if (disk_slots > 0) {
            // One pinned staging record for DISK<->VRAM, plus the spill file
            // preallocated to full capacity (slot-addressed, never grows).
            CUDA_CHECK_THROW(cudaHostAlloc(&m_stage, m_record_bytes, cudaHostAllocDefault));
            m_spill.open(spill_path, std::ios::binary | std::ios::in |
                                     std::ios::out | std::ios::trunc);
            if (!m_spill)
                throw std::runtime_error("CudaTierBackend: cannot open spill file " +
                                         spill_path);
            m_spill.seekp((std::streamoff)disk_slots * m_record_bytes - 1);
            m_spill.put('\0');
            m_spill.flush();
        }
        CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&m_xfer, cudaStreamNonBlocking));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&m_evt, cudaEventDisableTiming));
    }
    ~CudaTierBackend() override {
        cudaEventDestroy(m_evt);
        cudaStreamDestroy(m_xfer);
        if (m_arena) cudaFreeHost(m_arena);
        if (m_stage) cudaFreeHost(m_stage);
    }
    CudaTierBackend(const CudaTierBackend&) = delete;
    CudaTierBackend& operator=(const CudaTierBackend&) = delete;

    // -- demotions (synchronous on return) -----------------------------------
    void vram_to_ram(PageId phys, int ram_slot) override {
        char* rec = ram_record(ram_slot);
        for (int l = 0; l < m_sm.num_layers(); ++l)
            m_sm.read_page(l, phys, rec_k(rec, l), rec_v(rec, l), m_xfer);
        CUDA_CHECK_THROW(cudaStreamSynchronize(m_xfer));   // D2H landed in the arena
    }
    void ram_to_disk(int ram_slot, int disk_slot) override {
        m_spill.seekp((std::streamoff)disk_slot * m_record_bytes);
        m_spill.write(ram_record(ram_slot), (std::streamsize)m_record_bytes);
        if (!m_spill)
            throw std::runtime_error("CudaTierBackend: spill write failed");
    }

    // -- promotions (async; fenced) -------------------------------------------
    void ram_to_vram(int ram_slot, PageId phys) override {
        char* rec = ram_record(ram_slot);
        for (int l = 0; l < m_sm.num_layers(); ++l)
            m_sm.write_page(l, phys, rec_k(rec, l), rec_v(rec, l), m_xfer);
        // NOTE: the arena slot is freed by the pager right after this call;
        // slot reuse is ordered because any later write into it happens via
        // vram_to_ram, which synchronizes m_xfer first.
    }
    void disk_to_vram(int disk_slot, PageId phys) override {
        m_spill.seekg((std::streamoff)disk_slot * m_record_bytes);
        m_spill.read(m_stage, (std::streamsize)m_record_bytes);
        if (!m_spill)
            throw std::runtime_error("CudaTierBackend: spill read failed");
        for (int l = 0; l < m_sm.num_layers(); ++l)
            m_sm.write_page(l, phys, rec_k(m_stage, l), rec_v(m_stage, l), m_xfer);
        // The single staging record is reused by the NEXT disk_to_vram in the
        // same fault batch: its H2D must have landed first.
        CUDA_CHECK_THROW(cudaStreamSynchronize(m_xfer));
    }

    void fence(void* compute_stream) override {
        CUDA_CHECK_THROW(cudaEventRecord(m_evt, m_xfer));
        if (compute_stream)
            CUDA_CHECK_THROW(cudaStreamWaitEvent((cudaStream_t)compute_stream, m_evt, 0));
        else
            CUDA_CHECK_THROW(cudaEventSynchronize(m_evt));
    }

private:
    char* ram_record(int slot) const { return m_arena + (size_t)slot * m_record_bytes; }
    kv_t* rec_k(char* rec, int layer) const {
        return (kv_t*)(rec + (size_t)layer * 2 * m_page_bytes);
    }
    kv_t* rec_v(char* rec, int layer) const {
        return (kv_t*)(rec + (size_t)layer * 2 * m_page_bytes + m_page_bytes);
    }

    SequenceManager& m_sm;
    int    m_ram_slots, m_disk_slots;
    size_t m_page_bytes = 0, m_record_bytes = 0;
    char*  m_arena = nullptr;   // pinned, ram_slots * record_bytes
    char*  m_stage = nullptr;   // pinned, one record (disk staging)
    std::fstream m_spill;
    cudaStream_t m_xfer{};
    cudaEvent_t  m_evt{};
};

}} // namespace blackwell::paging
