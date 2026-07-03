// ============================================================================
// Standalone test for TieredMemoryPager (src/paging/tiered_memory_pager.h)
// ============================================================================
// Pure host — the pager is exercised against a fake VramPool and a fake
// TierBackend that carries page CONTENT (strings) through every tier, so the
// tests validate the demotion waterfall and page faults for data integrity,
// LRU victim order, pin protection, graceful truncation, zombie reaping and
// reference balance — the invariants PrefixCacheManager::acquire relies on to
// degrade (shorter cached prefix) instead of failing a request.
//
// Build (any C++17 compiler, no GPU needed):
//   g++ -std=c++17 tests/standalone/test_tiered_pager.cpp -o test_tiered_pager
//   cl  /std:c++17 /EHsc tests/standalone/test_tiered_pager.cpp
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../src/paging/tiered_memory_pager.h"

using namespace blackwell::paging;

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
#define EXPECT_STREQ(a, b)                                                      \
    do { ++g_checks; std::string _a = (a), _b = (b); if (_a != _b) { ++g_fail;  \
        std::printf("  FAIL %s:%d  EXPECT_STREQ(%s, %s)  got \"%s\" vs \"%s\"\n",\
            __FILE__, __LINE__, #a, #b, _a.c_str(), _b.c_str()); } } while (0)

// ----------------------------------------------------------------------------
// Fakes: pool with SequenceManager-like refcount semantics; backend that
// moves string "content" between tiers and logs operations.
// ----------------------------------------------------------------------------
struct FakePool : VramPool {
    std::vector<int>    refs;
    std::vector<PageId> free_list;
    explicit FakePool(int n) : refs(n, 0) {
        for (PageId p = n - 1; p >= 0; --p) free_list.push_back(p);
    }
    PageId try_allocate() override {
        if (free_list.empty()) return -1;
        PageId p = free_list.back(); free_list.pop_back();
        refs[p] = 1;
        return p;
    }
    void retain(PageId p) override  { ++refs[p]; }
    void release(PageId p) override { if (--refs[p] == 0) free_list.push_back(p); }
    int  free_pages() const override { return (int)free_list.size(); }
};

struct FakeBackend : TierBackend {
    std::map<PageId, std::string> vram;   // phys -> content
    std::vector<std::string> ram, disk;
    std::vector<std::string> ops;
    int   fences = 0;
    void* last_stream = nullptr;
    FakeBackend(int ram_slots, int disk_slots) : ram(ram_slots), disk(disk_slots) {}

    void vram_to_ram(PageId phys, int slot) override {
        ram[slot] = vram.at(phys); vram.erase(phys);
        ops.push_back("v2r:" + ram[slot]);
    }
    void ram_to_vram(int slot, PageId phys) override {
        vram[phys] = ram[slot];
        ops.push_back("r2v:" + ram[slot]);
    }
    void ram_to_disk(int slot, int dslot) override {
        disk[dslot] = ram[slot];
        ops.push_back("r2d:" + ram[slot]);
    }
    void disk_to_vram(int dslot, PageId phys) override {
        vram[phys] = disk[dslot];
        ops.push_back("d2v:" + disk[dslot]);
    }
    void fence(void* cs) override { ++fences; last_stream = cs; }
};

// Adopt a tree-only page carrying `content`, mirroring the manager's commit
// flow: allocate (caller ref) -> adopt (pager ref) -> tree retain -> drop the
// caller ref. Pager + tree are then the only owners.
static VPid make_tree_page(TieredMemoryPager& pg, FakePool& pool,
                           FakeBackend& io, const std::string& content) {
    PageId phys = pool.try_allocate();
    io.vram[phys] = content;
    VPid v = pg.adopt_vram_page(phys);
    pg.retain(v);
    pool.release(phys);
    return v;
}

static void test_lifecycle_and_gc() {
    std::printf("[ test_lifecycle_and_gc ]\n");
    FakePool pool(4);
    FakeBackend io(2, 2);
    TieredMemoryPager pg(pool, io, {2, 2});

    PageId phys = pool.try_allocate();
    VPid v = pg.adopt_vram_page(phys);
    EXPECT_EQ(pool.refs[phys], 2);            // caller + pager
    EXPECT_EQ((int)pg.tier_of(v), (int)Tier::VRAM);
    EXPECT_EQ(pg.phys_of(v), phys);

    // Never adopted by the tree -> gc reaps it (drops the pager ref).
    EXPECT(pg.gc(v));
    EXPECT_EQ(pool.refs[phys], 1);
    pool.release(phys);
    EXPECT_EQ(pool.free_pages(), 4);

    // Adopted -> gc refuses; release destroys.
    VPid w = make_tree_page(pg, pool, io, "W");
    EXPECT(!pg.gc(w));
    EXPECT_EQ(pg.ref_count(w), 1);
    pg.release(w);
    EXPECT_EQ(pool.free_pages(), 4);
    EXPECT_EQ(pg.stats().vram, 0);
}

static void test_waterfall_lru_order() {
    std::printf("[ test_waterfall_lru_order ]\n");
    FakePool pool(4);
    FakeBackend io(2, 4);
    TieredMemoryPager pg(pool, io, {2, 4});

    VPid A = make_tree_page(pg, pool, io, "A");   // oldest
    VPid B = make_tree_page(pg, pool, io, "B");
    VPid C = make_tree_page(pg, pool, io, "C");
    VPid D = make_tree_page(pg, pool, io, "D");   // newest
    EXPECT_EQ(pool.free_pages(), 0);

    // LRU demotion, one page at a time: A then B.
    EXPECT(pg.ensure_vram(1));
    EXPECT_STREQ(io.ops.back(), "v2r:A");
    EXPECT_EQ((int)pg.tier_of(A), (int)Tier::RAM);
    EXPECT(pg.ensure_vram(2));
    EXPECT_STREQ(io.ops.back(), "v2r:B");

    // RAM now full (A, B): further VRAM demotion cascades RAM's LRU to disk.
    EXPECT(pg.ensure_vram(4));
    EXPECT_EQ((int)pg.tier_of(A), (int)Tier::DISK);   // A spilled first
    EXPECT_EQ((int)pg.tier_of(B), (int)Tier::DISK);
    EXPECT_EQ((int)pg.tier_of(C), (int)Tier::RAM);
    EXPECT_EQ((int)pg.tier_of(D), (int)Tier::RAM);
    EXPECT_EQ(pool.free_pages(), 4);
    EXPECT_EQ(pg.stats().demotions_ram, 4);
    EXPECT_EQ(pg.stats().demotions_disk, 2);

    // Fault back from RAM and DISK; content survives the full round trip.
    void* fake_stream = (void*)0x1234;
    VPid chain1[] = {C};
    EXPECT_EQ(pg.fault_in(chain1, 1, fake_stream), 1);
    EXPECT_STREQ(io.vram.at(pg.phys_of(C)), "C");
    EXPECT_EQ(io.fences, 1);
    EXPECT(io.last_stream == fake_stream);

    VPid chain2[] = {A};
    EXPECT_EQ(pg.fault_in(chain2, 1, nullptr), 1);
    EXPECT_STREQ(io.vram.at(pg.phys_of(A)), "A");
    EXPECT_EQ(pg.stats().faults_ram, 1);
    EXPECT_EQ(pg.stats().faults_disk, 1);
    EXPECT(io.last_stream == nullptr);

    // Both faults pinned their page: demotion must skip them.
    pg.unpin_one(C);
    pg.unpin_one(A);
    for (VPid v : {A, B, C, D}) pg.release(v);
    EXPECT_EQ(pg.stats().vram + pg.stats().ram + pg.stats().disk, 0);
    EXPECT_EQ(pool.free_pages(), 4);
}

static void test_pin_blocks_demotion_and_truncation() {
    std::printf("[ test_pin_blocks_demotion_and_truncation ]\n");
    FakePool pool(2);
    FakeBackend io(1, 2);
    TieredMemoryPager pg(pool, io, {1, 2});

    VPid A = make_tree_page(pg, pool, io, "A");
    VPid B = make_tree_page(pg, pool, io, "B");
    // Spill A to disk (through RAM), then B to RAM: VRAM empty.
    EXPECT(pg.demote(A, Tier::DISK));
    EXPECT(pg.demote(B, Tier::RAM));
    EXPECT_EQ(pool.free_pages(), 2);

    // Fault both and PIN both (a live block table).
    VPid chain[] = {B, A};
    EXPECT_EQ(pg.fault_in(chain, 2), 2);
    EXPECT_EQ(pool.free_pages(), 0);

    // Everything pinned: no VRAM can be freed.
    EXPECT(!pg.ensure_vram(1));
    EXPECT(!pg.demote(A, Tier::RAM));   // pinned pages refuse hints too

    pg.unpin_one(B);
    pg.unpin_one(A);
    EXPECT(pg.ensure_vram(1));          // unpinned -> demotable again
    for (VPid v : {A, B}) pg.release(v);
}

static void test_partial_fault_prefix() {
    std::printf("[ test_partial_fault_prefix ]\n");
    FakePool pool(2);
    FakeBackend io(2, 4);
    TieredMemoryPager pg(pool, io, {2, 4});

    VPid A = make_tree_page(pg, pool, io, "A");
    VPid B = make_tree_page(pg, pool, io, "B");
    EXPECT(pg.demote(A, Tier::RAM));
    EXPECT(pg.demote(B, Tier::RAM));

    // An unrelated pinned pair hogs all of VRAM.
    VPid X = make_tree_page(pg, pool, io, "X");
    VPid Y = make_tree_page(pg, pool, io, "Y");
    VPid hog[] = {X, Y};
    EXPECT_EQ(pg.fault_in(hog, 2), 2);
    EXPECT_EQ(pool.free_pages(), 0);

    // Faulting the chain [A, B] can restore nothing: resident prefix == 0.
    VPid chain[] = {A, B};
    EXPECT_EQ(pg.fault_in(chain, 2), 0);

    // Free one hog page -> exactly one page of the chain fits: prefix == 1.
    pg.unpin_one(Y);
    EXPECT_EQ(pg.fault_in(chain, 2), 1);
    EXPECT_EQ((int)pg.tier_of(A), (int)Tier::VRAM);
    EXPECT_STREQ(io.vram.at(pg.phys_of(A)), "A");
    EXPECT_EQ((int)pg.tier_of(B), (int)Tier::RAM);   // untouched

    pg.unpin_one(A);
    pg.unpin_one(X);
    for (VPid v : {A, B, X, Y}) pg.release(v);
}

static void test_touch_changes_victim() {
    std::printf("[ test_touch_changes_victim ]\n");
    FakePool pool(2);
    FakeBackend io(2, 2);
    TieredMemoryPager pg(pool, io, {2, 2});

    VPid A = make_tree_page(pg, pool, io, "A");
    VPid B = make_tree_page(pg, pool, io, "B");
    pg.touch(A);                        // A becomes MRU -> B is the victim
    EXPECT(pg.ensure_vram(1));
    EXPECT_EQ((int)pg.tier_of(B), (int)Tier::RAM);
    EXPECT_EQ((int)pg.tier_of(A), (int)Tier::VRAM);
    pg.release(A); pg.release(B);
}

static void test_zombie_reaped_on_unpin() {
    std::printf("[ test_zombie_reaped_on_unpin ]\n");
    FakePool pool(2);
    FakeBackend io(1, 1);
    TieredMemoryPager pg(pool, io, {1, 1});

    VPid A = make_tree_page(pg, pool, io, "A");
    VPid chain[] = {A};
    EXPECT_EQ(pg.fault_in(chain, 1), 1);   // pin
    pg.release(A);                          // tree evicts while pinned
    EXPECT_EQ(pg.stats().vram, 1);          // zombie: backing survives the pin
    pg.unpin_one(A);                        // last pin reaps it
    EXPECT_EQ(pg.stats().vram, 0);
    EXPECT_EQ(pool.free_pages(), 2);
}

static void test_flat_config_degenerates() {
    std::printf("[ test_flat_config_degenerates ]\n");
    FakePool pool(2);
    FakeBackend io(0, 0);
    TieredMemoryPager pg(pool, io, {0, 0});   // Phase-1 behaviour

    VPid A = make_tree_page(pg, pool, io, "A");
    VPid B = make_tree_page(pg, pool, io, "B");
    EXPECT(!pg.ensure_vram(1));               // no lower tiers -> cannot demote
    VPid chain[] = {A, B};
    EXPECT_EQ(pg.fault_in(chain, 2), 2);      // resident: pure pin/translate
    EXPECT_EQ(io.ops.size(), 0u);             // no byte movement at all
    EXPECT_EQ(io.fences, 0);
    pg.unpin_one(A); pg.unpin_one(B);
    pg.release(A);  pg.release(B);
    EXPECT_EQ(pool.free_pages(), 2);
}

int main() {
    test_lifecycle_and_gc();
    test_waterfall_lru_order();
    test_pin_blocks_demotion_and_truncation();
    test_partial_fault_prefix();
    test_touch_changes_victim();
    test_zombie_reaped_on_unpin();
    test_flat_config_degenerates();
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
