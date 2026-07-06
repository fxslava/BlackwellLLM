#pragma once
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "prefix_cache_manager.h"
#include "prefill_driver.h"
#include "warmup_spec.h"

// ============================================================================
// AOTCacheWarmer — Ahead-of-Time context compiler (Text-to-Cache pipeline)
// ============================================================================
// Compiles a WarmupSpec into a ready-to-ship directory of .bkv branches:
//
//   spec.json --parse--> WarmupSpec --DFS--> per node:
//       tokenize(parent path + node.text)
//       acquire()                // prefix-HITS the parent committed just above
//       prefill(tail only)      // engine pass over the node's own segment
//       commit() [+ dump()]     // index + emit <out_dir>/<id>.bkv
//       recurse children        // parent seq stays live => pages pinned
//       release()
//
// so total prefill cost is O(unique tokens in the tree), not O(sum of full
// path lengths) — the compiler eats its own prefix-caching dogfood. The
// emitted directory carries a manifest.json (spec name, model_hash, entries)
// consumed by warm_start() on the deploy side, which loads every branch with
// LoadPolicy::ColdRam by default: the whole prompt library sits in pinned RAM
// at startup and pages fault into VRAM on first use. Overlapping branches
// dedup on load via the radix tree, exactly like overlapping commits.
//
// The engine is reached only through IPrefillDriver (mirrors ILLMGenerator in
// the orchestrator; lives in prefill_driver.h so interface consumers skip this
// header's json/filesystem deps): production binds tokenizer + the paged
// prefill sweep; tests inject a mock and never link CUDA.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

class AOTCacheWarmer {
public:
    struct Options {
        std::string  out_dir;
        bool         add_bos = true;
        cudaStream_t compute_stream = nullptr;   // for acquire-time faults
    };
    struct Report {
        int       nodes = 0;             // spec nodes processed
        int       files = 0;             // .bkv branches emitted
        long long prefilled_tokens = 0;  // engine work actually done
        long long reused_tokens = 0;     // prefix hits inside the compile
        std::vector<std::string> skipped;// emit-requested but < one page
    };

    AOTCacheWarmer(PrefixCacheManager& pc, IPrefillDriver& driver)
        : m_pc(pc), m_driver(driver) {}

    Report compile(const WarmupSpec& spec, const Options& opt) {
        namespace fs = std::filesystem;
        fs::create_directories(opt.out_dir);
        Report rep;
        nlohmann::json manifest;
        manifest["name"]       = spec.name;
        manifest["model_hash"] = m_pc.model_hash();
        manifest["page_size"]  = PAGE_SIZE;
        manifest["entries"]    = nlohmann::json::array();
        for (const auto& root : spec.roots)
            warm_node(root, {}, /*is_root=*/true, spec, opt, rep, manifest);
        std::ofstream mf(opt.out_dir + "/manifest.json",
                         std::ios::binary | std::ios::trunc);
        mf << manifest.dump(2) << '\n';
        if (!mf) throw std::runtime_error("AOTCacheWarmer: manifest write failed");
        return rep;
    }

    // Deploy-side counterpart: restore every branch of a compiled directory.
    // ColdRam keeps VRAM free until branches are actually requested; returns
    // total tokens made zero-prefill-servable. Rejects a foreign model's
    // directory before touching any page.
    static int warm_start(PrefixCacheManager& pc, const std::string& dir,
                          PrefixCacheManager::LoadPolicy policy =
                              PrefixCacheManager::LoadPolicy::ColdRam) {
        std::ifstream mf(dir + "/manifest.json", std::ios::binary);
        if (!mf) throw std::runtime_error("warm_start: no manifest in " + dir);
        nlohmann::json manifest = nlohmann::json::parse(mf);
        if (manifest.value("model_hash", uint64_t(0)) != pc.model_hash())
            throw std::runtime_error("warm_start: manifest model_hash mismatch");
        int tokens = 0;
        for (const auto& e : manifest["entries"])
            tokens += pc.load(dir + "/" + e["file"].get<std::string>(), policy);
        return tokens;
    }

private:
    void warm_node(const WarmupNode& node, std::vector<TokenId> path,
                   bool is_root, const WarmupSpec& spec, const Options& opt,
                   Report& rep, nlohmann::json& manifest) {
        const auto seg = m_driver.tokenize(node.text, is_root && opt.add_bos);
        path.insert(path.end(), seg.begin(), seg.end());
        const int n = (int)path.size();

        auto a = m_pc.acquire(path.data(), n, opt.compute_stream);
        m_driver.prefill(a.seq, path.data(), n, a.cached_tokens);
        m_pc.commit(a.seq, path.data(), n);
        rep.reused_tokens    += a.cached_tokens;
        rep.prefilled_tokens += n - a.cached_tokens;
        ++rep.nodes;

        const bool emit = node.emit.value_or(
            spec.emit_mode == WarmupSpec::EmitMode::All || node.children.empty());
        if (emit) {
            if (n >= PAGE_SIZE) {
                const std::string file = sanitize(node.id) + ".bkv";
                m_pc.dump(path.data(), n, opt.out_dir + "/" + file);
                manifest["entries"].push_back({
                    {"id", node.id},
                    {"file", file},
                    {"tokens", n - n % PAGE_SIZE},
                });
                ++rep.files;
            } else {
                rep.skipped.push_back(node.id);   // shorter than one page
            }
        }
        // Children while the parent seq is live: its pins keep the shared
        // path VRAM-resident through every child's fault/prefill.
        for (const auto& child : node.children)
            warm_node(child, path, /*is_root=*/false, spec, opt, rep, manifest);
        m_pc.release(a.seq);
    }

    static std::string sanitize(const std::string& id) {
        std::string s = id;
        for (char& c : s)
            if (!isalnum((unsigned char)c) && c != '.' && c != '_' && c != '-')
                c = '_';
        return s;
    }

    PrefixCacheManager& m_pc;
    IPrefillDriver&     m_driver;
};

}} // namespace blackwell::paging
