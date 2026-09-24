// TiktokenTokenizer: base64 decoding, rank-file parsing, the rank-merge BPE,
// and the GLM-4 chat template.
//
// Three tiers, so most of the file runs anywhere:
//   1. base64 against the RFC 4648 §10 vectors -- hermetic.
//   2. a synthetic 258-entry vocabulary built in a temp file -- hermetic, and
//      the only place where the *merge order* can be asserted by construction
//      rather than by agreement with a reference.
//   3. the real GLM-4 checkpoint, with token ids pinned from Python `tiktoken`
//      (the reference ChatGLM4Tokenizer wraps). SKIPped when it is absent.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <process.h>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "blackwell/tokenizer.h"
#include "tokenizer/tiktoken_tokenizer.h"

using blackwell::base64_decode;
using blackwell::TiktokenModelData;
using blackwell::TiktokenTokenizer;

namespace {

std::string b64_encode(const std::string& in) {
    static const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < in.size(); i += 3) {
        const unsigned b0 = static_cast<unsigned char>(in[i]);
        const unsigned b1 = i + 1 < in.size() ? static_cast<unsigned char>(in[i + 1]) : 0u;
        const unsigned b2 = i + 2 < in.size() ? static_cast<unsigned char>(in[i + 2]) : 0u;
        const unsigned v = (b0 << 16) | (b1 << 8) | b2;
        out += A[(v >> 18) & 63];
        out += A[(v >> 12) & 63];
        out += i + 1 < in.size() ? A[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? A[v & 63] : '=';
    }
    return out;
}

// RAII temp file so a throwing expectation cannot leave one behind.
class TempFile {
public:
    explicit TempFile(const std::string& contents) {
        path_ = (std::filesystem::temp_directory_path() /
                 ("blackwell_tiktoken_" + std::to_string(++counter_) + "_" +
                  std::to_string(::_getpid()) + ".model")).string();
        std::ofstream f(path_, std::ios::binary | std::ios::trunc);
        f << contents;
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
    static inline int counter_ = 0;
};

// A vocabulary with all 256 single bytes (ranks 0..255, the ctor requires them)
// plus explicit multi-byte entries whose ranks are chosen so the merge ORDER is
// observable: "ab" merges before "bc" because 256 < 257.
//   256 "ab"   257 "bc"   258 "abc"   259 "xy"
std::string synthetic_rank_file() {
    std::string out;
    for (int b = 0; b < 256; ++b)
        out += b64_encode(std::string(1, static_cast<char>(b))) + " " + std::to_string(b) + "\n";
    out += b64_encode("ab") + " 256\n";
    out += b64_encode("bc") + " 257\n";
    out += b64_encode("abc") + " 258\n";
    out += b64_encode("xy") + " 259\n";
    return out;
}

std::unique_ptr<TiktokenTokenizer> make_synthetic(
    std::unordered_map<std::string, int> specials = {},
    std::vector<int> prefix = {}) {
    static TempFile file(synthetic_rank_file());
    TiktokenModelData data = TiktokenTokenizer::load_rank_file(file.path());
    data.specials = std::move(specials);
    data.prefix_ids = std::move(prefix);
    return std::make_unique<TiktokenTokenizer>(std::move(data), nullptr);
}

int rank_of_bytes(const TiktokenTokenizer& tok, const std::string& s) {
    return tok.token_to_id(s);
}

// ---- tier 3 plumbing --------------------------------------------------------

std::string glm4_dir() {
    if (const char* env = std::getenv("BLACKWELL_GLM4_DIR")) return env;
    return "F:/AI/models/GLM-4-9B-Chat-1M-hf";
}

bool glm4_available() {
    std::error_code ec;
    return std::filesystem::exists(glm4_dir() + "/tokenizer.model", ec);
}

// GLM-4 special ids, from the checkpoint's added_tokens_decoder.
constexpr int kEndOfText = 151329;
constexpr int kGMask     = 151331;
constexpr int kSop       = 151333;
constexpr int kSystem    = 151335;
constexpr int kUser      = 151336;
constexpr int kAssistant = 151337;
constexpr int kObserve   = 151338;

} // namespace

// ============================================================================
// 1. base64 (RFC 4648 §10 test vectors)
// ============================================================================

TEST(Base64Decode, Rfc4648Vectors) {
    EXPECT_EQ(base64_decode(""), "");
    EXPECT_EQ(base64_decode("Zg=="), "f");
    EXPECT_EQ(base64_decode("Zm8="), "fo");
    EXPECT_EQ(base64_decode("Zm9v"), "foo");
    EXPECT_EQ(base64_decode("Zm9vYg=="), "foob");
    EXPECT_EQ(base64_decode("Zm9vYmE="), "fooba");
    EXPECT_EQ(base64_decode("Zm9vYmFy"), "foobar");
}

TEST(Base64Decode, PaddingIsOptional) {
    // tiktoken rank files are padded, but nothing in the format requires it.
    EXPECT_EQ(base64_decode("Zg"), "f");
    EXPECT_EQ(base64_decode("Zm8"), "fo");
    EXPECT_EQ(base64_decode("Zm9vYg"), "foob");
}

TEST(Base64Decode, HandlesEveryByteValue) {
    std::string all;
    for (int b = 0; b < 256; ++b) all += static_cast<char>(b);
    EXPECT_EQ(base64_decode(b64_encode(all)), all);
}

TEST(Base64Decode, RejectsCorruptInput) {
    EXPECT_THROW(base64_decode("Zm9v!"), std::runtime_error);   // outside the alphabet
    EXPECT_THROW(base64_decode("Zm9vY"), std::runtime_error);   // n % 4 == 1
    EXPECT_THROW(base64_decode("Zg-="), std::runtime_error);    // URL-safe alphabet
    EXPECT_THROW(base64_decode("Z m9v"), std::runtime_error);   // embedded space
    // Trailing group carrying bits that cannot be placed in a whole byte.
    EXPECT_THROW(base64_decode("Zh=="), std::runtime_error);
}

// ============================================================================
// 2. rank-file parsing
// ============================================================================

TEST(TiktokenRankFile, ParsesRanksAndRoundTripsTokens) {
    TempFile f(synthetic_rank_file());
    const TiktokenModelData d = TiktokenTokenizer::load_rank_file(f.path());

    EXPECT_EQ(d.decoder.size(), 260u);
    EXPECT_EQ(d.encoder.size(), 260u);
    EXPECT_EQ(d.decoder[65], "A");           // rank == byte value for singles
    EXPECT_EQ(d.decoder[256], "ab");
    EXPECT_EQ(d.decoder[258], "abc");
    EXPECT_EQ(d.encoder.at("xy"), 259);
    EXPECT_EQ(d.max_digit_run, 3);           // tiktoken-family default
}

TEST(TiktokenRankFile, AcceptsTheFormatProbe) {
    TempFile good(synthetic_rank_file());
    EXPECT_TRUE(TiktokenTokenizer::looks_like_rank_file(good.path()));

    // A SentencePiece .model is a protobuf: binary from byte 0.
    TempFile sentencepiece(std::string("\n\x07\x08\x01\x12\x02\x00\x01", 8));
    EXPECT_FALSE(TiktokenTokenizer::looks_like_rank_file(sentencepiece.path()));

    TempFile json_file("{\"model\": {\"vocab\": {}}}\n");
    EXPECT_FALSE(TiktokenTokenizer::looks_like_rank_file(json_file.path()));

    EXPECT_FALSE(TiktokenTokenizer::looks_like_rank_file("no/such/file.model"));
}

TEST(TiktokenRankFile, RejectsCorruptFiles) {
    {   // no rank column
        TempFile f("Zm9v\n");
        EXPECT_THROW(TiktokenTokenizer::load_rank_file(f.path()), std::runtime_error);
    }
    {   // rank is not an integer
        TempFile f("Zm9v zero\n");
        EXPECT_THROW(TiktokenTokenizer::load_rank_file(f.path()), std::runtime_error);
    }
    {   // token payload is not base64
        TempFile f("!!!! 0\n");
        EXPECT_THROW(TiktokenTokenizer::load_rank_file(f.path()), std::runtime_error);
    }
    {   // two tokens claiming the same id
        TempFile f(b64_encode("a") + " 0\n" + b64_encode("b") + " 0\n");
        EXPECT_THROW(TiktokenTokenizer::load_rank_file(f.path()), std::runtime_error);
    }
    {   // rank 1 missing: ids would silently shift
        TempFile f(b64_encode("a") + " 0\n" + b64_encode("b") + " 2\n");
        EXPECT_THROW(TiktokenTokenizer::load_rank_file(f.path()), std::runtime_error);
    }
    {
        TempFile f("\n\n");
        EXPECT_THROW(TiktokenTokenizer::load_rank_file(f.path()), std::runtime_error);
    }
    EXPECT_THROW(TiktokenTokenizer::load_rank_file("no/such/file.model"), std::runtime_error);
}

TEST(TiktokenRankFile, RejectsAVocabularyMissingSingleBytes) {
    // Drop byte 0x41 ('A'): BPE could reach a symbol it cannot emit, and this
    // family has no UNK to fall back on, so the ctor must refuse.
    std::string text;
    int rank = 0;
    for (int b = 0; b < 256; ++b) {
        if (b == 0x41) continue;
        text += b64_encode(std::string(1, static_cast<char>(b))) + " " +
                std::to_string(rank++) + "\n";
    }
    TempFile f(text);
    TiktokenModelData d = TiktokenTokenizer::load_rank_file(f.path());
    EXPECT_THROW(TiktokenTokenizer(std::move(d), nullptr), std::runtime_error);
}

// ============================================================================
// 3. the rank-merge BPE, asserted by construction
// ============================================================================

TEST(TiktokenBpe, MergesByVocabularyRankNotByPosition) {
    const auto tok = make_synthetic();

    // "abc" is a vocabulary entry, so the whole pre-token collapses to one id.
    EXPECT_EQ(tok->encode("abc", false), std::vector<int>{258});

    // "abd": "abc" cannot apply. "ab"(256) beats "bd"(absent), leaving ab + d.
    EXPECT_EQ(tok->encode("abd", false),
              (std::vector<int>{256, static_cast<int>('d')}));

    // "zbc": "bc"(257) is the only applicable merge.
    EXPECT_EQ(tok->encode("zbc", false),
              (std::vector<int>{static_cast<int>('z'), 257}));

    // Nothing merges: one id per byte.
    EXPECT_EQ(tok->encode("qq", false),
              (std::vector<int>{static_cast<int>('q'), static_cast<int>('q')}));
}

TEST(TiktokenBpe, LowestRankWinsWhenPairsOverlap) {
    // "abc" as bytes offers both "ab"(256) and "bc"(257). Lower rank merges
    // first, so a vocabulary WITHOUT the "abc" entry must yield ab + c -- this
    // is precisely what a merges-list BPE would get wrong.
    TempFile f([] {
        std::string out;
        for (int b = 0; b < 256; ++b)
            out += b64_encode(std::string(1, static_cast<char>(b))) + " " +
                   std::to_string(b) + "\n";
        out += b64_encode("ab") + " 256\n";
        out += b64_encode("bc") + " 257\n";
        return out;
    }());
    TiktokenModelData d = TiktokenTokenizer::load_rank_file(f.path());
    const TiktokenTokenizer tok(std::move(d), nullptr);

    EXPECT_EQ(tok.encode("abc", false), (std::vector<int>{256, static_cast<int>('c')}));
}

TEST(TiktokenBpe, WholePieceWinsEvenWhenNoPairMerges) {
    // The fork between "run BPE" and "look the pre-token up whole". With "xyz"
    // in the vocabulary but NEITHER "xy" nor "yz", a pure merge loop halts at
    // three bytes -- but the reference checks the whole piece first
    // (_encode_ordinary_native, ahead of byte_pair_encode) and emits one token.
    // Confirmed against Python tiktoken with exactly this vocabulary.
    TempFile f([] {
        std::string out;
        for (int b = 0; b < 256; ++b)
            out += b64_encode(std::string(1, static_cast<char>(b))) + " " +
                   std::to_string(b) + "\n";
        out += b64_encode("xyz") + " 256\n";
        return out;
    }());
    TiktokenModelData d = TiktokenTokenizer::load_rank_file(f.path());
    const TiktokenTokenizer tok(std::move(d), nullptr);

    EXPECT_EQ(tok.encode("xyz", false), std::vector<int>{256});
    // A substring of it still has no applicable merge and stays two bytes.
    EXPECT_EQ(tok.encode("xy", false),
              (std::vector<int>{static_cast<int>('x'), static_cast<int>('y')}));
}

TEST(TiktokenBpe, RoundTripsArbitraryBytes) {
    const auto tok = make_synthetic();
    // Byte-level BPE is lossless by construction; prove it over every byte
    // value, including the ones that are not valid UTF-8 on their own.
    std::string all;
    for (int b = 1; b < 256; ++b) all += static_cast<char>(b);
    EXPECT_EQ(tok->decode(tok->encode(all, false), true), all);
}

TEST(TiktokenBpe, MatchesSpecialTokensBeforeBpeAndLongestFirst) {
    // "<|a|>" is a prefix-sharing shorter literal: the longest match must win.
    const auto tok = make_synthetic({{"<|a|>", 300}, {"<|ab|>", 301}}, {300});

    EXPECT_EQ(tok->encode("<|ab|>", false), std::vector<int>{301});
    EXPECT_EQ(tok->encode("<|a|>", false), std::vector<int>{300});

    const auto ids = tok->encode("x<|ab|>y", false);
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], static_cast<int>('x'));
    EXPECT_EQ(ids[1], 301);
    EXPECT_EQ(ids[2], static_cast<int>('y'));

    // A '<' that starts no special literal is ordinary text, not a false hit.
    EXPECT_EQ(tok->encode("<|z|>", false).front(), static_cast<int>('<'));

    EXPECT_TRUE(tok->is_special(301));
    EXPECT_FALSE(tok->is_special(static_cast<int>('x')));
}

TEST(TiktokenBpe, AddSpecialTokensPrependsThePrefix) {
    const auto tok = make_synthetic({{"<|a|>", 300}}, {300, 299});
    const auto with = tok->encode("abc", true);
    ASSERT_EQ(with.size(), 3u);
    EXPECT_EQ(with[0], 300);
    EXPECT_EQ(with[1], 299);
    EXPECT_EQ(with[2], 258);
    EXPECT_EQ(tok->encode("abc", false), std::vector<int>{258});
}

TEST(TiktokenBpe, DecodeObeysRenderSpecial) {
    const auto tok = make_synthetic({{"<|a|>", 300}});
    const std::vector<int> ids{static_cast<int>('x'), 300, static_cast<int>('y')};
    EXPECT_EQ(tok->decode(ids, /*render_special=*/false), "xy");
    EXPECT_EQ(tok->decode(ids, /*render_special=*/true), "x<|a|>y");
    EXPECT_EQ(tok->decode(999999, true), "");  // unknown id: silent, as in the BPE path
}

TEST(TiktokenBpe, TokenToIdCoversBothTables) {
    const auto tok = make_synthetic({{"<|a|>", 300}});
    EXPECT_EQ(rank_of_bytes(*tok, "abc"), 258);
    EXPECT_EQ(rank_of_bytes(*tok, "<|a|>"), 300);
    EXPECT_EQ(rank_of_bytes(*tok, "nope"), -1);
    EXPECT_EQ(tok->vocab_size(), 261u); // 260 ranks + 1 special
}

// ============================================================================
// 4. the real GLM-4 checkpoint
// ============================================================================

class Glm4Tokenizer : public ::testing::Test {
protected:
    void SetUp() override {
        if (!glm4_available())
            GTEST_SKIP() << "GLM-4 checkpoint not found at " << glm4_dir()
                         << " (set BLACKWELL_GLM4_DIR)";
        tok_ = blackwell::TokenizerFactory::create(glm4_dir());
        ASSERT_NE(tok_, nullptr);
    }
    std::unique_ptr<blackwell::ITokenizer> tok_;
};

TEST_F(Glm4Tokenizer, FactoryBuildsTheTiktokenPathFromTokenizerDotModel) {
    // The checkpoint ships NO tokenizer.json -- this is exactly the case the
    // factory used to reject.
    std::error_code ec;
    EXPECT_FALSE(std::filesystem::exists(glm4_dir() + "/tokenizer.json", ec));

    EXPECT_EQ(tok_->vocab_size(), 151329u + 14u);
    EXPECT_NE(tok_->chat_template(), nullptr);
    EXPECT_EQ(tok_->chat_template()->name(), "glm4");
}

TEST_F(Glm4Tokenizer, ResolvesSpecialTokensAndTheStopSet) {
    EXPECT_EQ(tok_->token_to_id("[gMASK]"), kGMask);
    EXPECT_EQ(tok_->token_to_id("<sop>"), kSop);
    EXPECT_EQ(tok_->token_to_id("<|system|>"), kSystem);
    EXPECT_EQ(tok_->token_to_id("<|user|>"), kUser);
    EXPECT_EQ(tok_->token_to_id("<|assistant|>"), kAssistant);
    EXPECT_EQ(tok_->token_to_id("<|observation|>"), kObserve);
    EXPECT_EQ(tok_->token_to_id("<|endoftext|>"), kEndOfText);

    const auto& st = tok_->special_tokens();
    EXPECT_EQ(st.eos, kEndOfText);
    EXPECT_EQ(st.pad, kEndOfText);
    // generation_config.json: a turn also ends when the model hands the floor
    // back, not only at EOS.
    EXPECT_EQ(st.stop_ids, (std::vector<int>{kEndOfText, kUser, kObserve}));
    EXPECT_TRUE(tok_->is_stop(kUser));
    EXPECT_FALSE(tok_->is_stop(kAssistant));
}

TEST_F(Glm4Tokenizer, MatchesReferenceTiktokenIds) {
    // Pinned from Python `tiktoken` driven with GLM-4's own pat_str and
    // mergeable_ranks -- the exact encoder ChatGLM4Tokenizer wraps.
    struct Case { const char* text; std::vector<int> ids; };
    const std::vector<Case> cases{
        {"Hello, world!", {9703, 11, 1879, 0}},
        {"The quick brown fox jumps over the lazy dog 1234567890.",
         {785, 3974, 13867, 38627, 34041, 916, 279, 15666, 5562, 220, 108714,
          100461, 21, 100928, 24, 15, 13}},
        {"\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82, \xD0\xBC\xD0\xB8\xD1\x80!",
         {148366, 8177, 11, 130422, 0}},                       // "Привет, мир!"
        {"\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C",
         {109377, 3837, 99011}},                               // "你好，世界"
        {"Emoji: \xF0\x9F\x9A\x80\xF0\x9F\x98\x80",
         {91952, 25, 11157, 248, 222, 74764, 222}},            // rocket + grin
        {"def f(x):\n    return x * 2\n",
         {750, 282, 2075, 982, 262, 470, 856, 353, 220, 17, 198}},
        {"  leading and  double  spaces\t\ttabs\n\nblank\n",
         {220, 6388, 323, 220, 1990, 220, 12615, 197, 3244, 3435, 271, 10184, 198}},
        {"I'm don't we've they'll it's",
         {40, 2776, 1513, 944, 582, 3003, 807, 3278, 432, 594}},
        {"What is 2+2?", {3838, 374, 220, 17, 10, 17, 30}},
    };

    for (const auto& c : cases) {
        EXPECT_EQ(tok_->encode(c.text, false), c.ids) << "encoding \"" << c.text << "\"";
        EXPECT_EQ(tok_->decode(c.ids, false), c.text) << "decoding \"" << c.text << "\"";
    }
}

TEST_F(Glm4Tokenizer, RoundTripsMultiByteText) {
    const std::vector<std::string> texts{
        "plain ascii",
        "\xD0\xA0\xD1\x83\xD1\x81\xD1\x81\xD0\xBA\xD0\xB8\xD0\xB9 \xD1\x82\xD0\xB5\xD0\xBA"
        "\xD1\x81\xD1\x82 \xD1\x81 \xD1\x86\xD0\xB8\xD1\x84\xD1\x80\xD0\xB0\xD0\xBC\xD0\xB8 42",
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xA8\xE4\xB8\xAD\xE6\x96\x87",
        "\xF0\x9F\x8E\xAF\xF0\x9F\x9A\x80\xF0\x9F\x98\x80 mixed with ascii",
        "template<typename T>\n  auto f(T&& x) -> decltype(x) { return x; }\n",
        "line1\r\nline2\r\n\r\nline3",
        "trailing spaces   ",
        "\t\ttabs and \x0b vertical \x0c form feed",
    };
    for (const auto& t : texts)
        EXPECT_EQ(tok_->decode(tok_->encode(t, false), false), t) << "round-tripping \"" << t << "\"";
}

TEST_F(Glm4Tokenizer, EncodesSpecialTokenLiteralsInText) {
    const auto ids = tok_->encode("[gMASK]<sop><|user|>\nhi<|assistant|>", false);
    ASSERT_GE(ids.size(), 5u);
    EXPECT_EQ(ids[0], kGMask);
    EXPECT_EQ(ids[1], kSop);
    EXPECT_EQ(ids[2], kUser);
    EXPECT_EQ(ids.back(), kAssistant);

    // render_special round-trips the markers; the default suppresses them.
    EXPECT_EQ(tok_->decode(ids, true), "[gMASK]<sop><|user|>\nhi<|assistant|>");
    EXPECT_EQ(tok_->decode(ids, false), "\nhi");
}

TEST_F(Glm4Tokenizer, AddSpecialTokensPrependsGMaskAndSop) {
    const auto with    = tok_->encode("hi", true);
    const auto without = tok_->encode("hi", false);
    ASSERT_EQ(with.size(), without.size() + 2);
    EXPECT_EQ(with[0], kGMask);   // derived from the template's leading literal
    EXPECT_EQ(with[1], kSop);
}

// ---- the chat template, bit-for-bit against HuggingFace ---------------------

TEST_F(Glm4Tokenizer, ChatTemplateMatchesHuggingFaceRendering) {
    // Expected strings produced by rendering the checkpoint's own Jinja
    // chat_template with jinja2 (see docs/GLM4_TURBOQUANT_INTEGRATION.md).
    const std::vector<blackwell::ChatMessage> msgs{
        {"system", "You are a helpful assistant."},
        {"user", "What is 2+2?"},
        {"assistant", "4"},
        {"user", "And 3+3?"},
    };
    const blackwell::IChatTemplate& t = *tok_->chat_template();

    EXPECT_EQ(t.render_conversation(msgs, true),
              "[gMASK]<sop><|system|>\nYou are a helpful assistant."
              "<|user|>\nWhat is 2+2?<|assistant|>\n4<|user|>\nAnd 3+3?<|assistant|>");

    EXPECT_EQ(t.render_conversation(msgs, false),
              "[gMASK]<sop><|system|>\nYou are a helpful assistant."
              "<|user|>\nWhat is 2+2?<|assistant|>\n4<|user|>\nAnd 3+3?");

    EXPECT_EQ(t.render_conversation({{"user", "Hi"}}, true),
              "[gMASK]<sop><|user|>\nHi<|assistant|>");

    // The generation cue carries NO trailing newline -- adding one shifts every
    // generation by a token.
    EXPECT_EQ(t.render_generation_prompt(), "<|assistant|>");
}

TEST_F(Glm4Tokenizer, IncrementalChatEncodingEqualsWholeConversation) {
    // The streaming path (prelude once, then a turn at a time) must produce the
    // same ids as apply_chat_template, or a persistent KV cache diverges from a
    // re-prefilled one.
    const std::vector<blackwell::ChatMessage> msgs{
        {"system", "Be brief."},
        {"user", "Ping?"},
    };

    std::vector<int> incremental = tok_->encode_chat_prelude("Be brief.");
    for (size_t i = 1; i < msgs.size(); ++i) {
        const auto part = tok_->encode_chat_message(msgs[i]);
        incremental.insert(incremental.end(), part.begin(), part.end());
    }
    const auto gen = tok_->encode_generation_prompt();
    incremental.insert(incremental.end(), gen.begin(), gen.end());

    EXPECT_EQ(incremental, tok_->apply_chat_template(msgs, true));
}

TEST_F(Glm4Tokenizer, ApplyChatTemplateDoesNotDoublePrefix) {
    // The rendered text already contains "[gMASK]<sop>"; the numeric prefix
    // must not be prepended on top of it.
    const auto ids = tok_->apply_chat_template({{"user", "Hi"}}, true);
    ASSERT_GE(ids.size(), 3u);
    EXPECT_EQ(ids[0], kGMask);
    EXPECT_EQ(ids[1], kSop);
    EXPECT_NE(ids[2], kGMask);
    EXPECT_EQ(ids.back(), kAssistant);
}
