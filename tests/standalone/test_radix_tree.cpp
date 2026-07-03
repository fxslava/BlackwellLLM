// ============================================================================
// Standalone test for the prefix-cache RadixTreeIndex (src/paging/radix_tree.h)
// ============================================================================
// Pure host-side control plane — no CUDA, no engine. Exercises page-aligned
// matching, insertion with page-boundary splits, duplicate-span rejection,
// LRU eviction under lock protection, and the retain/release page-reference
// contract the PrefixCacheManager builds on. page_size is shrunk to 4 so the
// scenarios stay readable.
//
// Build (any C++17 compiler, no GPU needed):
//   g++ -std=c++17 tests/standalone/test_radix_tree.cpp -o test_radix_tree
//   cl  /std:c++17 /EHsc tests/standalone/test_radix_tree.cpp
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <map>
#include <vector>

#include "../../src/paging/radix_tree.h"

using namespace blackwell::paging;

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

// ----------------------------------------------------------------------------
// Fake page-reference ledger standing in for PageAllocator incref/decref.
// ----------------------------------------------------------------------------
struct Ledger {
    std::map<PageId, int> refs;
    int retains = 0, releases = 0;
    RadixTreeIndex::PageFn retain_fn() {
        return [this](PageId p) { ++refs[p]; ++retains; };
    }
    RadixTreeIndex::PageFn release_fn() {
        return [this](PageId p) { --refs[p]; ++releases; };
    }
    int live() const {
        int n = 0;
        for (auto& kv : refs) if (kv.second > 0) ++n;
        return n;
    }
};

static constexpr int PS = 4;   // test page size

// tokens 0..n-1 with a per-branch offset so branches differ where intended
static std::vector<TokenId> toks(std::initializer_list<TokenId> l) { return l; }

static void test_empty_and_single_branch() {
    std::printf("[ test_empty_and_single_branch ]\n");
    Ledger led;
    RadixTreeIndex t(PS, led.retain_fn(), led.release_fn());

    auto q = toks({1,2,3,4, 5,6,7,8, 9,10,11,12});
    auto m0 = t.match_prefix(q.data(), (int)q.size());
    EXPECT_EQ(m0.matched_tokens, 0);
    EXPECT_EQ((int)m0.pages.size(), 0);

    // Insert 3 pages (ids 100,101,102).
    PageId pages[] = {100, 101, 102};
    auto r = t.insert(q.data(), (int)q.size(), pages);
    EXPECT_EQ(r.pages_adopted, 3);
    EXPECT_EQ((int)t.total_pages(), 3);
    EXPECT_EQ(led.retains, 3);
    EXPECT_EQ(led.refs[100], 1);

    // Exact re-match.
    auto m1 = t.match_prefix(q.data(), (int)q.size());
    EXPECT_EQ(m1.matched_tokens, 12);
    EXPECT_EQ((int)m1.pages.size(), 3);
    EXPECT_EQ(m1.pages[0], 100);
    EXPECT_EQ(m1.pages[2], 102);

    // Longer query: still only the indexed 3 pages.
    auto q2 = toks({1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16});
    auto m2 = t.match_prefix(q2.data(), (int)q2.size());
    EXPECT_EQ(m2.matched_tokens, 12);

    // Divergence inside page 1 (token 6 -> 60): only page 0 shareable.
    auto q3 = toks({1,2,3,4, 5,60,7,8, 9,10,11,12});
    auto m3 = t.match_prefix(q3.data(), (int)q3.size());
    EXPECT_EQ(m3.matched_tokens, 4);
    EXPECT_EQ((int)m3.pages.size(), 1);
    EXPECT_EQ(m3.pages[0], 100);

    // Query shorter than a page: nothing shareable.
    auto m4 = t.match_prefix(q.data(), PS - 1);
    EXPECT_EQ(m4.matched_tokens, 0);
}

static void test_split_and_duplicate_insert() {
    std::printf("[ test_split_and_duplicate_insert ]\n");
    Ledger led;
    RadixTreeIndex t(PS, led.retain_fn(), led.release_fn());

    auto a = toks({1,2,3,4, 5,6,7,8, 9,10,11,12});
    PageId pa[] = {100, 101, 102};
    t.insert(a.data(), (int)a.size(), pa);

    // Branch B shares pages 0-1, diverges in page 2 -> splits the leaf.
    auto b = toks({1,2,3,4, 5,6,7,8, 90,91,92,93});
    PageId pb[] = {200, 201, 202};   // 200/201 are duplicates of the shared span
    auto rb = t.insert(b.data(), (int)b.size(), pb);
    EXPECT_EQ(rb.pages_adopted, 1);          // only the diverging page
    EXPECT_EQ((int)t.total_pages(), 4);      // 100,101,102,202
    EXPECT_EQ(led.refs[200], 0);             // duplicates were NOT adopted
    EXPECT_EQ(led.refs[201], 0);
    EXPECT_EQ(led.refs[202], 1);

    // Both branches fully matchable, shared span serves tree pages.
    auto ma = t.match_prefix(a.data(), (int)a.size());
    EXPECT_EQ(ma.matched_tokens, 12);
    EXPECT_EQ(ma.pages[0], 100); EXPECT_EQ(ma.pages[1], 101);
    EXPECT_EQ(ma.pages[2], 102);
    auto mb = t.match_prefix(b.data(), (int)b.size());
    EXPECT_EQ(mb.matched_tokens, 12);
    EXPECT_EQ(mb.pages[0], 100); EXPECT_EQ(mb.pages[1], 101);
    EXPECT_EQ(mb.pages[2], 202);

    // Re-inserting an existing path adopts nothing.
    PageId pc[] = {300, 301, 302};
    auto rc = t.insert(a.data(), (int)a.size(), pc);
    EXPECT_EQ(rc.pages_adopted, 0);
    EXPECT_EQ((int)t.total_pages(), 4);

    // Branch C extends B by one page: only the extension is adopted.
    auto c = toks({1,2,3,4, 5,6,7,8, 90,91,92,93, 94,95,96,97});
    PageId pd[] = {400, 401, 402, 403};
    auto rd = t.insert(c.data(), (int)c.size(), pd);
    EXPECT_EQ(rd.pages_adopted, 1);
    EXPECT_EQ((int)t.total_pages(), 5);
    auto mc = t.match_prefix(c.data(), (int)c.size());
    EXPECT_EQ(mc.matched_tokens, 16);
    EXPECT_EQ(mc.pages[3], 403);
}

static void test_eviction_lru_and_locks() {
    std::printf("[ test_eviction_lru_and_locks ]\n");
    Ledger led;
    RadixTreeIndex t(PS, led.retain_fn(), led.release_fn());

    // Two branches sharing page 0: root -> [P0] -> {[A1 A2], [B1]}
    auto a = toks({1,2,3,4, 5,6,7,8, 9,10,11,12});
    auto b = toks({1,2,3,4, 50,51,52,53});
    PageId pa[] = {10, 11, 12};
    PageId pb[] = {10, 20};        // page 0 dedups on insert
    t.insert(a.data(), (int)a.size(), pa);
    auto rb = t.insert(b.data(), (int)b.size(), pb);
    EXPECT_EQ((int)t.total_pages(), 4);   // 10, 11, 12, 20

    // Touch branch A so B's leaf is the LRU victim; lock B -> A dies instead.
    (void)t.match_prefix(a.data(), (int)a.size());
    t.lock(rb.node);
    int freed = t.evict(1);
    EXPECT_EQ(freed, 2);                  // A's leaf held pages 11,12
    EXPECT_EQ(led.refs[11], 0);
    EXPECT_EQ(led.refs[12], 0);
    EXPECT_EQ(led.refs[10], 1);           // shared page survives (B locked)
    EXPECT_EQ((int)t.total_pages(), 2);
    auto ma = t.match_prefix(a.data(), (int)a.size());
    EXPECT_EQ(ma.matched_tokens, 4);      // only the shared page remains for A

    // Unlock B and drain: cascade frees B's leaf, then the shared parent.
    t.unlock(rb.node);
    freed = t.evict(100);
    EXPECT_EQ(freed, 2);
    EXPECT_EQ((int)t.total_pages(), 0);
    EXPECT_EQ(led.live(), 0);
    EXPECT_EQ(led.retains, led.releases);

    auto m0 = t.match_prefix(a.data(), (int)a.size());
    EXPECT_EQ(m0.matched_tokens, 0);
}

static void test_lock_survives_split() {
    std::printf("[ test_lock_survives_split ]\n");
    Ledger led;
    RadixTreeIndex t(PS, led.retain_fn(), led.release_fn());

    auto a = toks({1,2,3,4, 5,6,7,8, 9,10,11,12});
    PageId pa[] = {10, 11, 12};
    t.insert(a.data(), (int)a.size(), pa);

    // A "sequence" matches all of A and locks the (single) leaf node.
    auto ma = t.match_prefix(a.data(), (int)a.size());
    t.lock(ma.node);

    // A second branch splits that node after page 1. The lock stays with the
    // suffix node, and the split-off top must remain transitively protected.
    auto b = toks({1,2,3,4, 5,6,7,8, 90,91,92,93});
    PageId pb[] = {10, 11, 20};
    t.insert(b.data(), (int)b.size(), pb);
    EXPECT_EQ((int)t.total_pages(), 4);

    int freed = t.evict(100);      // only B's unlocked leaf may go
    EXPECT_EQ(freed, 1);
    EXPECT_EQ(led.refs[20], 0);
    EXPECT_EQ(led.refs[10], 1);    // top pages pinned via the locked suffix
    EXPECT_EQ(led.refs[11], 1);
    EXPECT_EQ(led.refs[12], 1);
    auto m2 = t.match_prefix(a.data(), (int)a.size());
    EXPECT_EQ(m2.matched_tokens, 12);   // A's chain intact across the split

    t.unlock(ma.node);
    freed = t.evict(100);
    EXPECT_EQ(freed, 3);
    EXPECT_EQ(led.live(), 0);
}

static void test_clear_releases_all() {
    std::printf("[ test_clear_releases_all ]\n");
    Ledger led;
    {
        RadixTreeIndex t(PS, led.retain_fn(), led.release_fn());
        auto a = toks({1,2,3,4, 5,6,7,8});
        auto b = toks({1,2,3,4, 50,51,52,53});
        PageId pa[] = {10, 11};
        PageId pb[] = {10, 20};
        t.insert(a.data(), (int)a.size(), pa);
        t.insert(b.data(), (int)b.size(), pb);
        EXPECT_EQ((int)t.total_pages(), 3);
    }   // destructor -> clear()
    EXPECT_EQ(led.live(), 0);
    EXPECT_EQ(led.retains, led.releases);
}

int main() {
    test_empty_and_single_branch();
    test_split_and_duplicate_insert();
    test_eviction_lru_and_locks();
    test_lock_survives_split();
    test_clear_releases_all();
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
