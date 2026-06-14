// ============================================================================
// Standalone state-machine test for the host-side paged-KV SequenceManager.
// ============================================================================
// Exercises the REAL SequenceManager / PageAllocator (src/paging/paged_kv_cache.h)
// — allocation, reference counting, fork (branch), lazy Copy-on-Write, and
// rewind (backtrack) — the invariants an autonomous agent's thought branches
// rely on to NOT corrupt each other.
//
// The actual device page-copy kernel is validated separately by
// test_paged_attention.cpp; here we link a host STUB for launch_cow_copy_page
// that merely records that CoW was triggered (and with which src/dst pages), so
// this binary needs no .cu and tests pure block-table / ref-count logic. The
// SequenceManager still allocates its device pools via the CUDA runtime, so a
// GPU must be present.
//
// Build (Windows: "x64 Native Tools" prompt with CUDA on PATH/INCLUDE):
//   nvcc -std=c++17 -arch=sm_120 tests/standalone/test_sequence_manager.cpp -o test_sequence_manager.exe
//   ./test_sequence_manager.exe
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../../src/paging/paged_kv_cache.h"

using namespace blackwell::paging;

// ----------------------------------------------------------------------------
// Recording stub for the CoW page-copy kernel launcher. Signature must match
// the declaration in paged_flash_attention.cuh (default arg lives there only).
// ----------------------------------------------------------------------------
static int g_cow_calls = 0;
static int g_cow_last_src = -1;
static int g_cow_last_dst = -1;

void launch_cow_copy_page(kv_t* /*k_pool*/, kv_t* /*v_pool*/,
                          int src_page, int dst_page,
                          int /*num_layers*/, int /*total_pages*/,
                          int /*num_kv_heads*/, int /*head_dim*/,
                          cudaStream_t /*stream*/) {
    ++g_cow_calls;
    g_cow_last_src = src_page;
    g_cow_last_dst = dst_page;
}

// ----------------------------------------------------------------------------
// Minimal assertion harness (records failures, keeps running, exit code != 0).
// ----------------------------------------------------------------------------
static int g_checks = 0, g_fail = 0;
#define EXPECT(cond)                                                            \
    do { ++g_checks; if (!(cond)) { ++g_fail;                                   \
        std::printf("  FAIL %s:%d  EXPECT(%s)\n", __FILE__, __LINE__, #cond);   \
    } } while (0)
#define EXPECT_EQ(a, b)                                                         \
    do { ++g_checks; long long _a = (long long)(a), _b = (long long)(b);        \
        if (_a != _b) { ++g_fail; std::printf(                                  \
            "  FAIL %s:%d  EXPECT_EQ(%s, %s)  got %lld vs %lld\n",              \
            __FILE__, __LINE__, #a, #b, _a, _b); } } while (0)

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return 2;
    }

    const int num_layers = 2, num_kv_heads = 2, head_dim = 16;
    const int total_pages = 8, max_blocks = 8;
    SequenceManager mgr(num_layers, num_kv_heads, head_dim, total_pages, max_blocks);
    EXPECT_EQ(mgr.free_pages(), total_pages);

    // ---------------------------------------------------------------------
    // 1. Basic allocation & append spanning two physical pages.
    // ---------------------------------------------------------------------
    std::printf("[1] basic allocation & append\n");
    const SeqId p = mgr.create_sequence();
    const int N = PAGE_SIZE + 2;                 // 18 tokens -> blocks 0 and 1
    for (int t = 0; t < N; ++t) {
        AppendSlot s = mgr.reserve_append_slot(p);
        EXPECT_EQ(s.slot, t % PAGE_SIZE);        // slot walks within the page
        EXPECT_EQ(mgr.num_blocks(p), t / PAGE_SIZE + 1);  // new page only on boundary
    }
    EXPECT_EQ(mgr.length(p), N);
    EXPECT_EQ(mgr.num_blocks(p), 2);
    EXPECT_EQ(mgr.free_pages(), total_pages - 2);         // two pages consumed
    const PageId p_b0 = mgr.block_page(p, 0);
    const PageId p_b1 = mgr.block_page(p, 1);
    EXPECT(p_b0 != p_b1);
    EXPECT_EQ(mgr.page_ref_count(p_b0), 1);
    EXPECT_EQ(mgr.page_ref_count(p_b1), 1);
    EXPECT_EQ(g_cow_calls, 0);                            // plain append never CoWs

    // ---------------------------------------------------------------------
    // 2. Fork: share every page, no allocation, ref counts -> 2.
    // ---------------------------------------------------------------------
    std::printf("[2] fork (branch)\n");
    const int free_before_fork = mgr.free_pages();
    const SeqId c = mgr.fork(p);
    EXPECT_EQ(mgr.free_pages(), free_before_fork);        // nothing allocated
    EXPECT_EQ(mgr.num_blocks(c), 2);
    EXPECT_EQ(mgr.length(c), N);
    EXPECT_EQ(mgr.block_page(c, 0), p_b0);                // child shares parent pages
    EXPECT_EQ(mgr.block_page(c, 1), p_b1);
    EXPECT_EQ(mgr.page_ref_count(p_b0), 2);
    EXPECT_EQ(mgr.page_ref_count(p_b1), 2);
    EXPECT_EQ(g_cow_calls, 0);                            // fork copies nothing

    // ---------------------------------------------------------------------
    // 3. Lazy CoW: child appends into the shared partial page (block 1).
    // ---------------------------------------------------------------------
    std::printf("[3] lazy copy-on-write\n");
    const int free_before_cow = mgr.free_pages();
    const AppendSlot cs = mgr.reserve_append_slot(c);     // pos 18 -> block 1, slot 2
    EXPECT_EQ(g_cow_calls, 1);                            // CoW fired exactly once
    EXPECT_EQ(cs.slot, N % PAGE_SIZE);                    // slot 2
    EXPECT_EQ(mgr.free_pages(), free_before_cow - 1);     // one fresh page allocated
    const PageId c_b1 = mgr.block_page(c, 1);
    EXPECT(c_b1 != p_b1);                                 // child repointed
    EXPECT_EQ(cs.page, c_b1);
    EXPECT_EQ(g_cow_last_src, p_b1);                      // copied FROM original shared page
    EXPECT_EQ(g_cow_last_dst, c_b1);                      // TO the child's new page
    EXPECT_EQ(mgr.page_ref_count(c_b1), 1);              // child's new page: sole owner
    EXPECT_EQ(mgr.page_ref_count(p_b1), 1);              // parent's original: back to 1
    EXPECT_EQ(mgr.block_page(c, 0), p_b0);               // block 0 still shared (no write)
    EXPECT_EQ(mgr.page_ref_count(p_b0), 2);
    // Parent must be completely untouched by the child's write.
    EXPECT_EQ(mgr.length(p), N);
    EXPECT_EQ(mgr.num_blocks(p), 2);
    EXPECT_EQ(mgr.block_page(p, 1), p_b1);
    EXPECT_EQ(mgr.length(c), N + 1);

    // ---------------------------------------------------------------------
    // 4. Rewind: free an owned page (ref->0) and decref a shared page (ref->1).
    // ---------------------------------------------------------------------
    std::printf("[4] rewind (backtrack)\n");
    // 4a: drop the child's OWN block 1 page -> ref 1->0 -> back to free-list.
    const int free_before_rw = mgr.free_pages();
    mgr.rewind(c, 10);                                    // keep ceil(10/16)=1 block
    EXPECT_EQ(mgr.length(c), 10);
    EXPECT_EQ(mgr.num_blocks(c), 1);
    EXPECT_EQ(mgr.free_pages(), free_before_rw + 1);      // c_b1 returned to allocator
    EXPECT_EQ(mgr.page_ref_count(c_b1), 0);              // truly freed
    EXPECT_EQ(mgr.page_ref_count(p_b1), 1);             // parent unaffected
    EXPECT_EQ(mgr.length(p), N);

    // 4b: drop the SHARED block 0 -> ref 2->1, NOT freed (parent still owns it).
    const int free_before_rw2 = mgr.free_pages();
    mgr.rewind(c, 0);
    EXPECT_EQ(mgr.length(c), 0);
    EXPECT_EQ(mgr.num_blocks(c), 0);
    EXPECT_EQ(mgr.free_pages(), free_before_rw2);         // shared page NOT freed
    EXPECT_EQ(mgr.page_ref_count(p_b0), 1);             // parent now sole owner
    EXPECT_EQ(mgr.block_page(p, 0), p_b0);              // parent still points there
    EXPECT_EQ(mgr.length(p), N);

    std::printf("\n%d checks, %d failures -> %s\n",
                g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
