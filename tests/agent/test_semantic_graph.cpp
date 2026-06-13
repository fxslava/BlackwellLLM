// End-to-end check of the agent's shadow graph:
//   1. parse a dummy translation unit,
//   2. extract a function's SemanticNode and query its callers / dependencies,
//   3. perform a formatting-preserving "simulated edit" via CstManager and
//      confirm the source is reconstructed perfectly (only the edited node
//      changes; every comment and byte of whitespace elsewhere is preserved).
#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "cst_manager.h"
#include "semantic_engine.h"

namespace {

// Note the deliberately irregular spacing and the comments: a correct
// formatting-preserving edit must leave all of this untouched.
const char* kSource =
    "#include <string>\n"
    "\n"
    "// a type the agent should track as a dependency\n"
    "struct Widget {\n"
    "    int value;\n"
    "};\n"
    "\n"
    "// leaf function\n"
    "int helper(int x) {\n"
    "    return x * 2;\n"
    "}\n"
    "\n"
    "Widget make_widget(int seed) {\n"
    "    Widget w;\n"
    "    w.value =   helper(seed);   // odd spacing on purpose\n"
    "    return w;\n"
    "}\n"
    "\n"
    "int main() {\n"
    "    Widget w = make_widget(21);\n"
    "    return w.value;\n"
    "}\n";

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

TSNode find_function_definition(TSNode node, const agent::CstManager& cst,
                                const std::string& name) {
    if (std::string("function_definition") == ts_node_type(node)) {
        TSNode decl = ts_node_child_by_field_name(node, "declarator", 10);
        if (!ts_node_is_null(decl) &&
            cst.node_text(decl).rfind(name, 0) == 0) {
            return node;
        }
    }
    const uint32_t n = ts_node_child_count(node);
    for (uint32_t i = 0; i < n; ++i) {
        TSNode found = find_function_definition(ts_node_child(node, i), cst, name);
        if (!ts_node_is_null(found)) return found;
    }
    return TSNode{};
}

// --- Semantic graph -----------------------------------------------------------

TEST(ShadowGraph, BuildsCallGraphAndDependencies) {
    agent::TreeSitterSemanticProvider provider;
    ASSERT_TRUE(provider.analyze(kSource))
        << "tree-sitter failed to parse / grammar ABI mismatch";

    // All three free functions are discovered.
    auto syms = provider.symbols();
    EXPECT_TRUE(contains(syms, "helper"));
    EXPECT_TRUE(contains(syms, "make_widget"));
    EXPECT_TRUE(contains(syms, "main"));

    auto helper = provider.node_for("helper");
    ASSERT_TRUE(helper.has_value());
    EXPECT_EQ(helper->kind, agent::SemanticKind::Function);
    EXPECT_EQ(helper->cst_type, "function_definition");

    // Callers (who calls me) -- the heart of the shadow graph.
    EXPECT_TRUE(contains(provider.callers_of("helper"), "make_widget"));
    EXPECT_TRUE(contains(provider.callers_of("make_widget"), "main"));
    EXPECT_TRUE(provider.callers_of("main").empty());

    // Callees (what I call).
    EXPECT_TRUE(contains(provider.callees_of("make_widget"), "helper"));
    EXPECT_TRUE(contains(provider.callees_of("main"), "make_widget"));

    // Type dependency tracking.
    auto mw = provider.node_for("make_widget");
    ASSERT_TRUE(mw.has_value());
    EXPECT_TRUE(contains(mw->type_dependencies, "Widget"));

    // Scratchpad is writable for the LLM's own notes.
    mw->scratchpad = "candidate for inlining";
    EXPECT_EQ(mw->scratchpad, "candidate for inlining");
}

// --- Formatting-preserving edit ----------------------------------------------

TEST(ShadowGraph, AtomicEditPreservesEverythingElse) {
    agent::CstManager cst;
    ASSERT_TRUE(cst.parse(kSource));

    TSNode helper = find_function_definition(cst.root(), cst, "helper");
    ASSERT_FALSE(ts_node_is_null(helper));

    // Replace just the body node "{ return x * 2; }" with a new body.
    TSNode body = ts_node_child_by_field_name(helper, "body", 4);
    ASSERT_FALSE(ts_node_is_null(body));
    const std::string old_body = cst.node_text(body);
    const std::string new_body = "{\n    return x * 3;\n}";

    ASSERT_TRUE(cst.replace_node(body, new_body));

    // The exact reconstruction guarantee: the edited buffer must equal the
    // original with ONLY old_body -> new_body swapped. If any surrounding
    // comment/whitespace had been disturbed this comparison would fail.
    std::string expected = kSource;
    const auto pos = expected.find(old_body);
    ASSERT_NE(pos, std::string::npos);
    expected.replace(pos, old_body.size(), new_body);
    EXPECT_EQ(cst.source(), expected);

    // Sanity: comments and odd spacing survived verbatim.
    EXPECT_NE(cst.source().find("// leaf function"), std::string::npos);
    EXPECT_NE(cst.source().find("w.value =   helper(seed);"), std::string::npos);
    EXPECT_NE(cst.source().find("return x * 3;"), std::string::npos);

    // The reparse is still valid and the graph can be rebuilt on the new text.
    agent::TreeSitterSemanticProvider provider;
    ASSERT_TRUE(provider.analyze(cst.source()));
    EXPECT_TRUE(contains(provider.callers_of("helper"), "make_widget"));
}

}  // namespace
