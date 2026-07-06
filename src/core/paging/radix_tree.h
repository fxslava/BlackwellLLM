#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

// ============================================================================
// RadixTreeIndex — page-aligned compressed prefix tree over token streams.
// ============================================================================
// The index that turns the paged KV cache into a PREFIX CACHE: it maps token
// prefixes to the physical pages that already hold their K/V, so a new request
// sharing a prefix (system prompt, few-shot header, ToT parent branch) reuses
// those pages instead of re-prefilling — zero tensor copies, zero VRAM
// duplication (SGLang-style RadixAttention, at BlackwellEngine's PAGE_SIZE
// granularity).
//
// Design rules:
//   * PAGE granularity. A page (page_size tokens) is shareable iff ALL its
//     tokens match; every edge label is a whole number of pages and splits
//     happen only at page boundaries. Partial-page tails stay private to their
//     sequence (the existing CoW machinery already guards those).
//   * Children are keyed by a 64-bit hash of the child's FIRST page of tokens
//     (O(1) branch lookup); candidates are verified token-exact, and hash
//     collisions chain in a small per-key vector — a collision can never cause
//     false sharing.
//   * The tree OWNS one reference on every page it holds, taken/dropped via
//     the retain/release callbacks (bound to PageAllocator incref/decref by
//     PrefixCacheManager). Sequences built from a match take their own ref.
//   * Locking: an active sequence (or an in-flight serializer) locks the
//     deepest node of its matched path. Eviction only removes CHILDLESS,
//     UNLOCKED nodes, so every ancestor of a locked node is transitively
//     protected — O(1) lock/unlock, no path walks.
//   * Eviction is LRU over evictable leaves; freeing a leaf may expose its
//     parent, which joins the candidate set with its own last-access tick.
//
// Pure host-side control plane: no CUDA includes, no engine types beyond the
// int32 page/token ids — unit-testable with any C++17 compiler (page_size is
// a constructor parameter precisely so tests can shrink it).
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

using PageId  = int32_t;   // same alias as paged_kv_cache.h (redeclaration OK)
using TokenId = int32_t;

// FNV-1a over one page worth of token ids.
inline uint64_t hash_page_tokens(const TokenId* t, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; ++i)
        h = (h ^ (uint64_t)(uint32_t)t[i]) * 1099511628211ull;
    return h;
}

class RadixTreeIndex {
public:
    struct Node {
        Node*                parent = nullptr;
        std::vector<TokenId> tokens;   // edge label; size % page_size == 0 (root: empty)
        std::vector<PageId>  pages;    // tokens.size() / page_size entries
        std::unordered_map<uint64_t,
            std::vector<std::unique_ptr<Node>>> children;  // first-page hash -> chain
        uint64_t last_access = 0;
        int      lock_count  = 0;

        int num_pages() const { return (int)pages.size(); }
        bool evictable() const {
            return parent != nullptr && children.empty() && lock_count == 0;
        }
    };

    struct Match {
        std::vector<PageId> pages;          // full pages covering matched_tokens
        int                 matched_tokens = 0;  // multiple of page_size
        Node*               node = nullptr; // deepest node touched (root if no match)
    };

    struct InsertResult {
        Node* node = nullptr;   // deepest node of the inserted path
        int   pages_adopted = 0;// pages the tree took ownership of (retained)
    };

    using PageFn = std::function<void(PageId)>;

    // retain / release: page reference hooks (PageAllocator incref / decref).
    RadixTreeIndex(int page_size, PageFn retain, PageFn release)
        : m_page_size(page_size),
          m_retain(std::move(retain)), m_release(std::move(release)) {
        assert(page_size > 0);
    }

    // Non-copyable (owns the node graph; callbacks bind external state).
    RadixTreeIndex(const RadixTreeIndex&) = delete;
    RadixTreeIndex& operator=(const RadixTreeIndex&) = delete;
    ~RadixTreeIndex() { clear(); }

    // -----------------------------------------------------------------------
    // Longest cached prefix of tokens[0..n). Read-only: touches LRU ticks but
    // moves no references — the caller increfs the returned pages when it
    // builds a sequence from them, and should lock() the returned node for the
    // sequence's lifetime.
    // -----------------------------------------------------------------------
    Match match_prefix(const TokenId* tokens, int n) {
        Match m;
        m.node = &m_root;
        Node* cur = &m_root;
        int   matched = 0;
        const uint64_t tick = ++m_tick;
        m_root.last_access = tick;
        while (n - matched >= m_page_size) {
            Node* child = find_child(cur, tokens + matched);
            if (!child) break;
            const int j = count_matching_pages(child, tokens + matched, n - matched);
            assert(j >= 1);   // find_child verified the first page
            m.pages.insert(m.pages.end(),
                           child->pages.begin(), child->pages.begin() + j);
            matched += j * m_page_size;
            child->last_access = tick;
            m.node = child;
            if (j < child->num_pages()) break;   // diverged inside this node
            cur = child;
        }
        m.matched_tokens = matched;
        return m;
    }

    // -----------------------------------------------------------------------
    // Publish tokens[0..n) backed by pages[0..n/page_size). n is rounded down
    // to a page multiple. Spans already present keep the EXISTING tree pages
    // (the corresponding entries of `pages` are ignored — the caller's
    // sequence simply keeps its own copies); spans that are new are ADOPTED:
    // the tree calls retain() on each and will release() them on eviction.
    // -----------------------------------------------------------------------
    InsertResult insert(const TokenId* tokens, int n, const PageId* pages) {
        InsertResult r;
        const int total_pages = n / m_page_size;
        const uint64_t tick = ++m_tick;
        Node* cur = &m_root;
        m_root.last_access = tick;
        int pi = 0;   // pages consumed
        while (pi < total_pages) {
            const TokenId* qt = tokens + (size_t)pi * m_page_size;
            const int remaining = n - pi * m_page_size;
            Node* child = find_child(cur, qt);
            if (!child) {
                // Attach the whole remaining suffix as one new leaf.
                auto leaf = std::make_unique<Node>();
                leaf->parent = cur;
                leaf->tokens.assign(qt, tokens + (size_t)total_pages * m_page_size);
                leaf->pages.assign(pages + pi, pages + total_pages);
                leaf->last_access = tick;
                for (PageId p : leaf->pages) { m_retain(p); ++m_total_pages; }
                r.pages_adopted += leaf->num_pages();
                Node* raw = leaf.get();
                cur->children[hash_page_tokens(qt, m_page_size)]
                   .push_back(std::move(leaf));
                cur = raw;
                pi = total_pages;
                break;
            }
            const int j = count_matching_pages(child, qt, remaining);
            child->last_access = tick;
            pi += j;
            if (j < child->num_pages()) {
                // Diverged (or query ended) inside `child`: split at page j so
                // the shared span becomes its own node. `child` keeps the
                // suffix (and its locks); the next loop iteration attaches the
                // query's tail under the new top node.
                cur = split_node(child, j);
                cur->last_access = tick;
            } else {
                cur = child;
            }
        }
        r.node = cur;
        return r;
    }

    // -----------------------------------------------------------------------
    // Pin / unpin the deepest node of a path (root is accepted and ignored —
    // it is never evicted anyway).
    // -----------------------------------------------------------------------
    void lock(Node* node)   { if (node && node != &m_root) ++node->lock_count; }
    void unlock(Node* node) {
        if (node && node != &m_root) {
            assert(node->lock_count > 0);
            --node->lock_count;
        }
    }

    // Refresh the LRU tick of a node's whole path without re-matching tokens
    // (fault-in / orchestrator "keep this branch hot" hint). One tick for the
    // path, root included.
    void touch(Node* node) {
        const uint64_t tick = ++m_tick;
        for (Node* p = node; p; p = p->parent) p->last_access = tick;
    }

    // -----------------------------------------------------------------------
    // Evict least-recently-used unlocked leaves until at least `want_pages`
    // page references were released (or nothing evictable remains). Returns
    // the number of pages actually released.
    // -----------------------------------------------------------------------
    int evict(int want_pages) {
        auto older = [](const Node* a, const Node* b) {
            return a->last_access > b->last_access;   // min-heap on tick
        };
        std::priority_queue<Node*, std::vector<Node*>, decltype(older)> pq(older);
        collect_evictable(&m_root, pq);
        int freed = 0;
        while (freed < want_pages && !pq.empty()) {
            Node* victim = pq.top(); pq.pop();
            if (!victim->evictable()) continue;   // state changed since collect
            Node* parent = victim->parent;
            for (PageId p : victim->pages) { m_release(p); --m_total_pages; }
            freed += victim->num_pages();
            detach_child(parent, victim);         // destroys victim
            if (parent->evictable()) pq.push(parent);
        }
        return freed;
    }

    // Drop everything (releases every held page reference).
    void clear() {
        release_subtree(&m_root);
        m_root.children.clear();
        m_total_pages = 0;
    }

    size_t total_pages() const { return m_total_pages; }
    int    page_size()   const { return m_page_size; }

private:
    // Child of `cur` whose first page's tokens equal qt[0..page_size). O(1)
    // hash lookup + exact verification of the (rare) chain.
    Node* find_child(Node* cur, const TokenId* qt) const {
        auto it = cur->children.find(hash_page_tokens(qt, m_page_size));
        if (it == cur->children.end()) return nullptr;
        for (auto& c : it->second)
            if (std::equal(qt, qt + m_page_size, c->tokens.begin()))
                return c.get();
        return nullptr;
    }

    // How many leading FULL pages of `node` match qt (bounded by remaining
    // query tokens). Compares whole pages only.
    int count_matching_pages(const Node* node, const TokenId* qt,
                             int remaining_tokens) const {
        const int limit = std::min(node->num_pages(),
                                   remaining_tokens / m_page_size);
        int j = 0;
        while (j < limit) {
            const TokenId* a = node->tokens.data() + (size_t)j * m_page_size;
            const TokenId* b = qt + (size_t)j * m_page_size;
            if (!std::equal(a, a + m_page_size, b)) break;
            ++j;
        }
        return j;
    }

    // Split `node` after its first `j` pages: a new TOP node takes pages
    // [0, j) and `node` keeps the suffix (preserving its identity, so held
    // locks and Match pointers stay valid). Returns the top node.
    Node* split_node(Node* node, int j) {
        assert(j >= 1 && j < node->num_pages());
        Node* parent = node->parent;
        auto top = std::make_unique<Node>();
        Node* top_raw = top.get();
        top->parent      = parent;
        top->last_access = node->last_access;
        top->tokens.assign(node->tokens.begin(),
                           node->tokens.begin() + (size_t)j * m_page_size);
        top->pages.assign(node->pages.begin(), node->pages.begin() + j);
        // Swap `top` into the parent slot `node` occupied (same first page ->
        // same hash key), then re-hang `node` (now the suffix) under `top`.
        std::unique_ptr<Node> owned = extract_child(parent, node);
        node->tokens.erase(node->tokens.begin(),
                           node->tokens.begin() + (size_t)j * m_page_size);
        node->pages.erase(node->pages.begin(), node->pages.begin() + j);
        node->parent = top_raw;
        top->children[hash_page_tokens(node->tokens.data(), m_page_size)]
           .push_back(std::move(owned));
        parent->children[hash_page_tokens(top->tokens.data(), m_page_size)]
              .push_back(std::move(top));
        return top_raw;
    }

    // Remove `node` from parent's child map and return ownership.
    std::unique_ptr<Node> extract_child(Node* parent, Node* node) {
        const uint64_t h = hash_page_tokens(node->tokens.data(), m_page_size);
        auto it = parent->children.find(h);
        assert(it != parent->children.end());
        auto& chain = it->second;
        for (size_t i = 0; i < chain.size(); ++i) {
            if (chain[i].get() == node) {
                std::unique_ptr<Node> owned = std::move(chain[i]);
                chain.erase(chain.begin() + i);
                if (chain.empty()) parent->children.erase(it);
                return owned;
            }
        }
        assert(false && "node not found under parent");
        return nullptr;
    }

    void detach_child(Node* parent, Node* node) { extract_child(parent, node); }

    template <class PQ>
    void collect_evictable(Node* n, PQ& pq) {
        for (auto& kv : n->children)
            for (auto& c : kv.second) {
                if (c->evictable()) pq.push(c.get());
                else collect_evictable(c.get(), pq);
            }
    }

    void release_subtree(Node* n) {
        for (auto& kv : n->children)
            for (auto& c : kv.second) {
                release_subtree(c.get());
                for (PageId p : c->pages) m_release(p);
            }
    }

    int      m_page_size;
    PageFn   m_retain, m_release;
    Node     m_root;
    uint64_t m_tick = 0;
    size_t   m_total_pages = 0;
};

}} // namespace blackwell::paging
