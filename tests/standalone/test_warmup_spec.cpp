// ============================================================================
// Standalone test for the warmup DSL parser (src/paging/warmup_spec.h)
// ============================================================================
// Host-pure; needs only nlohmann/json on the include path (FetchContent'd by
// the main build):
//   cl /std:c++17 /EHsc /I<build>/_deps/nlohmann_json-src/include ^
//      tests/standalone/test_warmup_spec.cpp
// ----------------------------------------------------------------------------
#include <cstdio>
#include <stdexcept>
#include <string>

#include "../../src/paging/warmup_spec.h"

using namespace blackwell::paging;

static int g_checks = 0, g_fail = 0;
#define EXPECT(cond)                                                            \
    do { ++g_checks; if (!(cond)) { ++g_fail;                                   \
        std::printf("  FAIL %s:%d  EXPECT(%s)\n", __FILE__, __LINE__, #cond);   \
    } } while (0)
#define EXPECT_THROWS(expr)                                                     \
    do { ++g_checks; bool _thrown = false;                                      \
        try { (void)(expr); } catch (const std::runtime_error&) { _thrown = true; } \
        if (!_thrown) { ++g_fail;                                               \
            std::printf("  FAIL %s:%d  EXPECT_THROWS(%s)\n", __FILE__, __LINE__, #expr); } \
    } while (0)

static const char* kGood = R"JSON({
  "name": "support-agent",
  "emit": "leaves",
  "nodes": [
    { "id": "base", "text": "You are the BlackwellEngine assistant.",
      "children": [
        { "id": "base.en", "text": "Respond in English.",
          "children": [ { "id": "base.en.tools", "text": "## Tools" } ] },
        { "id": "base.ru", "text": "Otvechay po-russki.", "emit": true }
      ] }
  ]
})JSON";

static void test_parse_good() {
    std::printf("[ test_parse_good ]\n");
    WarmupSpec s = parse_warmup_spec(kGood);
    EXPECT(s.name == "support-agent");
    EXPECT(s.emit_mode == WarmupSpec::EmitMode::Leaves);
    EXPECT(s.roots.size() == 1);
    const WarmupNode& base = s.roots[0];
    EXPECT(base.id == "base");
    EXPECT(!base.emit.has_value());
    EXPECT(base.children.size() == 2);
    EXPECT(base.children[0].children.size() == 1);
    EXPECT(base.children[0].children[0].id == "base.en.tools");
    EXPECT(base.children[1].emit.has_value() && *base.children[1].emit);
}

static void test_parse_errors() {
    std::printf("[ test_parse_errors ]\n");
    // Malformed JSON.
    EXPECT_THROWS(parse_warmup_spec("{ nope"));
    // Empty / missing nodes.
    EXPECT_THROWS(parse_warmup_spec(R"({"nodes": []})"));
    EXPECT_THROWS(parse_warmup_spec(R"({"name": "x"})"));
    // Missing id.
    EXPECT_THROWS(parse_warmup_spec(R"({"nodes": [{"text": "t"}]})"));
    // Duplicate id across the tree.
    EXPECT_THROWS(parse_warmup_spec(
        R"({"nodes": [{"id": "a", "text": "t",
                       "children": [{"id": "a", "text": "u"}]}]})"));
    // Both text and text_file / neither.
    EXPECT_THROWS(parse_warmup_spec(
        R"({"nodes": [{"id": "a", "text": "t", "text_file": "f"}]})"));
    EXPECT_THROWS(parse_warmup_spec(R"({"nodes": [{"id": "a"}]})"));
    // Empty text.
    EXPECT_THROWS(parse_warmup_spec(R"({"nodes": [{"id": "a", "text": ""}]})"));
    // Bad emit mode.
    EXPECT_THROWS(parse_warmup_spec(
        R"({"emit": "some", "nodes": [{"id": "a", "text": "t"}]})"));
    // Missing text_file on disk.
    EXPECT_THROWS(parse_warmup_spec(
        R"({"nodes": [{"id": "a", "text_file": "no_such_file.md"}]})"));
}

int main() {
    test_parse_good();
    test_parse_errors();
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
