// -----------------------------------------------------------------------------
// reply_split_test.cpp — the <voice>/<ui> contract, read back.
//
// THE TWO PROPERTIES WORTH TESTING, and everything here is one of them:
//
//   1. A TAG SPLIT ACROSS DELTAS STILL WORKS. The reply arrives in transport
//      chunks whose boundaries nobody chose -- an SSE text_delta, a token, a
//      simulated slice -- so "<voice>" routinely arrives as "<vo" + "ice>". A
//      splitter that only handled whole tags would work in every manual test and
//      fail against a real stream.
//
//   2. IT FAILS OPEN. A model that ignores the contract must still be HEARD.
//      This is the opposite of the commit gate's fail-closed rule, deliberately:
//      there, a mistake costs money; here, a mistake costs the user an assistant
//      that has silently stopped talking, which is indistinguishable from broken
//      speech output.
//
// CPU-only, no UI, no engine: the splitter is a header with two std::functions.
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#include "reply_split.hpp"

namespace {

// Collects the two streams so a test can assert on what each AUDIENCE received
// rather than on how many fragments it arrived in -- the fragmentation is the
// transport's business and no test should be pinned to it.
struct Split {
    std::string ui;
    std::string voice;
};

Split run(const std::vector<std::string>& deltas) {
    Split out;
    rt::ReplySplitter s([&out](std::string_view t) { out.ui.append(t); },
                        [&out](std::string_view t) { out.voice.append(t); });
    s.reset();
    for (const std::string& d : deltas) s.push(d);
    s.finish();
    return out;
}

// Every single-byte split of `reply`, which is the worst case a transport can
// produce and the one that breaks a naive implementation.
Split run_byte_by_byte(std::string_view reply) {
    std::vector<std::string> deltas;
    deltas.reserve(reply.size());
    for (const char c : reply) deltas.emplace_back(1, c);
    return run(deltas);
}

constexpr const char* kCompliant =
    "<voice>Готово, файл сохранён.</voice>"
    "<ui>**Готово.** Файл сохранён в `out/report.md`.</ui>";

TEST(ReplySplit, EachHalfReachesItsOwnAudience) {
    const Split s = run({kCompliant});
    EXPECT_EQ(s.voice, "Готово, файл сохранён.");
    EXPECT_EQ(s.ui, "**Готово.** Файл сохранён в `out/report.md`.");
}

TEST(ReplySplit, TagsSplitAcrossDeltasAreStillRecognised) {
    // The property that matters, asserted the hard way: 1-byte deltas put a
    // boundary inside every tag AND inside every multi-byte Cyrillic character.
    const Split s = run_byte_by_byte(kCompliant);
    EXPECT_EQ(s.voice, "Готово, файл сохранён.");
    EXPECT_EQ(s.ui, "**Готово.** Файл сохранён в `out/report.md`.");
}

TEST(ReplySplit, TagsNeverLeakIntoEitherStream) {
    const Split s = run({kCompliant});
    EXPECT_EQ(s.ui.find("<ui>"), std::string::npos);
    EXPECT_EQ(s.ui.find("</voice>"), std::string::npos);
    EXPECT_EQ(s.voice.find("<voice>"), std::string::npos);
}

// ---- fail-open --------------------------------------------------------------

TEST(ReplySplit, AnUntaggedReplyIsShownAndSpokenInFull) {
    // The model ignored the contract. The user must still see the answer AND
    // hear it -- speech arrives at the end rather than as it streams, which is
    // the documented cost of the fallback.
    const Split s = run({"Просто ответ", " без тегов."});
    EXPECT_EQ(s.ui, "Просто ответ без тегов.");
    EXPECT_EQ(s.voice, "Просто ответ без тегов.");
}

TEST(ReplySplit, AVoiceBlockRetiresTheFallbackSoNothingIsSaidTwice) {
    // A preamble before the first tag reaches the screen (it is part of the
    // answer) but must not be spoken on top of the real <voice> block.
    const Split s = run({"Конечно! <voice>Готово.</voice><ui>Детали.</ui>"});
    EXPECT_EQ(s.voice, "Готово.");
    EXPECT_EQ(s.ui, "Конечно! Детали.");
}

TEST(ReplySplit, AUiOnlyReplyIsStillSpoken) {
    // Half-compliance is the common failure and it is indistinguishable from
    // muted speech unless the fallback covers it.
    const Split s = run({"<ui>Только текст.</ui>"});
    EXPECT_EQ(s.ui, "Только текст.");
    EXPECT_EQ(s.voice, "Только текст.");
}

TEST(ReplySplit, TextOutsideTheBlocksIsKeptRatherThanDropped) {
    const Split s = run({"<voice>А.</voice> хвост <ui>Б.</ui>"});
    EXPECT_EQ(s.voice, "А.");
    EXPECT_EQ(s.ui, " хвост Б.");
}

// ---- literal '<' ------------------------------------------------------------

TEST(ReplySplit, ALiteralLessThanSurvivesInBothHalves) {
    const Split s = run({"<voice>a < b</voice><ui>`if (a < b)`</ui>"});
    EXPECT_EQ(s.voice, "a < b");
    EXPECT_EQ(s.ui, "`if (a < b)`");
}

TEST(ReplySplit, AReplyEndingMidTagSurrendersWhatItHeld) {
    // The tail looked like the start of a tag and turned out to be the end of
    // the answer. Holding it forever would silently truncate the reply.
    const Split s = run({"<ui>done <"});
    EXPECT_EQ(s.ui, "done <");
}

// ---- turn isolation ---------------------------------------------------------

TEST(ReplySplit, ResetDropsEverythingHeldFromThePreviousTurn) {
    Split out;
    rt::ReplySplitter s([&out](std::string_view t) { out.ui.append(t); },
                        [&out](std::string_view t) { out.voice.append(t); });

    s.push("<ui>first</u");   // ends mid-tag, still held
    s.reset();                // a new answer starts
    s.push("<voice>second</voice>");
    s.finish();

    EXPECT_EQ(s.saw_voice_block(), true);
    EXPECT_EQ(out.voice, "second");
    // "first" was emitted as it streamed (it was inside <ui>), but the HELD
    // fragment "</u" must not reappear spliced onto this turn.
    EXPECT_EQ(out.ui, "first");
}

TEST(ReplySplit, SawVoiceBlockReportsComplianceForTelemetry) {
    Split out;
    rt::ReplySplitter s([&out](std::string_view t) { out.ui.append(t); },
                        [&out](std::string_view t) { out.voice.append(t); });
    s.push("no tags here");
    s.finish();
    EXPECT_FALSE(s.saw_voice_block());
}

// ---- the instruction half of the contract -----------------------------------

TEST(ReplySplit, ComposeSystemPromptKeepsThePersonaFirst) {
    const std::string composed = rt::compose_system_prompt("Ты ассистент.");
    EXPECT_EQ(composed.rfind("Ты ассистент.", 0), 0u);
    // The tags the parser above looks for must be the tags the model is asked
    // for. This is the assertion that catches a rename in one file only.
    EXPECT_NE(composed.find("<voice>"), std::string::npos);
    EXPECT_NE(composed.find("<ui>"), std::string::npos);
}

}  // namespace
