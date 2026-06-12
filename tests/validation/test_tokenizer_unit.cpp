// Unit tests for the tokenizer subsystem that need no model checkpoint: a
// synthetic HuggingFace-layout directory is written to a temp folder and the
// full TokenizerFactory path is exercised against it. Golden comparisons vs
// the real Llama-3 / Qwen2.5 tokenizers live in
// tests/integration/test_tokenizer_golden.cpp.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "blackwell/tokenizer.h"

namespace fs = std::filesystem;
using blackwell::ChatMessage;
using blackwell::ITokenizer;
using blackwell::TokenizerFactory;

namespace {

// Single-character vocab (no merges needed to encode the chat-template words)
// plus one mergeable pair to prove rank-based merging. Ġ/Ċ are the byte-level
// images of ' ' and '\n'.
constexpr const char* kTokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 100, "content": "<s>", "special": true},
    {"id": 101, "content": "<|im_start|>", "special": true},
    {"id": 102, "content": "<|im_end|>", "special": true},
    {"id": 103, "content": "<plain>", "special": false}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split",
       "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
       "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
    ]
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ]
  },
  "model": {
    "type": "BPE",
    "ignore_merges": false,
    "vocab": {
      "a": 0, "b": 1, "c": 2, "ab": 3, "bc": 9,
      "s": 10, "y": 11, "t": 12, "e": 13, "m": 14, "u": 15, "r": 16,
      "n": 17, "i": 18, "h": 19, "d": 20, "f": 21, "l": 22, "o": 23, "g": 33,
      "x": 24, "D": 25, ".": 26, "0": 27, "1": 28, "2": 29, "3": 30,
      "Ġ": 31, "Ċ": 32
    },
    "merges": ["b c"]
  }
})json";

constexpr const char* kTokenizerConfigJson = R"json({
  "bos_token": "<s>",
  "eos_token": {"content": "<|im_end|>"},
  "chat_template": "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}{%- else %}{{- '<|im_start|>system\\nDefault sys.<|im_end|>\\n' }}{%- endif %}{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
})json";

constexpr const char* kGenerationConfigJson = R"json({
  "eos_token_id": [102, 100]
})json";

class TokenizerUnit : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        dir_ = fs::temp_directory_path() / "blackwell_tokenizer_unit";
        fs::create_directories(dir_);
        write_file("tokenizer.json", kTokenizerJson);
        write_file("tokenizer_config.json", kTokenizerConfigJson);
        write_file("generation_config.json", kGenerationConfigJson);
        tokenizer_ = TokenizerFactory::create(dir_.string()).release();
    }

    static void TearDownTestSuite() {
        delete tokenizer_;
        tokenizer_ = nullptr;
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    static void write_file(const char* name, const char* content) {
        std::ofstream f(dir_ / name, std::ios::binary);
        ASSERT_TRUE(f.is_open()) << "cannot write " << name;
        f << content;
    }

    static fs::path dir_;
    static ITokenizer* tokenizer_;
};

fs::path TokenizerUnit::dir_;
ITokenizer* TokenizerUnit::tokenizer_ = nullptr;

TEST_F(TokenizerUnit, FactoryResolvesVocabAndSpecialTokens) {
    // 29 vocab entries + 4 added tokens, all ids resolved from JSON by name.
    EXPECT_EQ(tokenizer_->vocab_size(), 33u);
    EXPECT_EQ(tokenizer_->token_to_id("ab"), 3);
    EXPECT_EQ(tokenizer_->token_to_id("<|im_start|>"), 101);
    EXPECT_EQ(tokenizer_->token_to_id("definitely-not-a-token"), -1);

    const auto& st = tokenizer_->special_tokens();
    EXPECT_EQ(st.bos, 100); // plain-string form
    EXPECT_EQ(st.eos, 102); // AddedToken-object form
    EXPECT_EQ(st.unk, -1);  // not declared by the checkpoint
    EXPECT_EQ(st.stop_ids, (std::vector<int>{102, 100})); // generation_config.json
    EXPECT_TRUE(tokenizer_->is_stop(100));
    EXPECT_FALSE(tokenizer_->is_stop(0));
}

TEST_F(TokenizerUnit, MergeRanksDecideNotVocabIds) {
    // Vocab holds both "ab" (id 3) and "bc" (id 9) but the merges list only
    // contains "b c". A vocab-id-priority implementation (the legacy one)
    // would produce [3, 2]; rank-based BPE must produce [0, 9].
    EXPECT_EQ(tokenizer_->encode("abc", false), (std::vector<int>{0, 9}));
}

TEST_F(TokenizerUnit, PostProcessorPrefixAppliedOnDemand) {
    EXPECT_EQ(tokenizer_->encode("abc", true), (std::vector<int>{100, 0, 9}));
}

TEST_F(TokenizerUnit, AddedTokensSplitRawText) {
    // Special and non-special added tokens are both matched literally.
    EXPECT_EQ(tokenizer_->encode("x<s>y", false), (std::vector<int>{24, 100, 11}));
    EXPECT_EQ(tokenizer_->encode("x<plain>y", false), (std::vector<int>{24, 103, 11}));
}

TEST_F(TokenizerUnit, DecodeSuppressesSpecialsUnlessAsked) {
    EXPECT_EQ(tokenizer_->decode(100), "");
    EXPECT_EQ(tokenizer_->decode(100, /*render_special=*/true), "<s>");
    EXPECT_EQ(tokenizer_->decode(103), "<plain>"); // non-special added token
    EXPECT_EQ(tokenizer_->decode(31), " ");        // byte-alphabet reversal
    EXPECT_EQ(tokenizer_->decode(987654), "");     // unknown id is not an error
}

TEST_F(TokenizerUnit, DigitRunsSplitPerPreTokenizerRegex) {
    // \p{N}{1,3} from the synthetic regex: "0123" splits as "012" + "3";
    // no merges exist for digits so each char surfaces alone.
    EXPECT_EQ(tokenizer_->encode("0123", false), (std::vector<int>{27, 28, 29, 30}));
}

TEST_F(TokenizerUnit, ChatTemplateDetectedAsChatMLWithExtractedDefault) {
    ASSERT_NE(tokenizer_->chat_template(), nullptr);
    EXPECT_EQ(tokenizer_->chat_template()->name(), "chatml");
    EXPECT_EQ(tokenizer_->chat_template()->default_system_prompt(), "Default sys.");
}

TEST_F(TokenizerUnit, ChatTemplateRendersAndEncodesRoundTrip) {
    const std::vector<ChatMessage> chat = {{"user", "hi"}};
    const std::vector<int> ids = tokenizer_->apply_chat_template(chat, true);

    // Frame structure: starts with <|im_start|>, default system injected.
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids.front(), 101);
    EXPECT_EQ(std::count(ids.begin(), ids.end(), 102), 2); // system + user blocks

    // Stripping specials leaves exactly the rendered plain text.
    EXPECT_EQ(tokenizer_->decode(ids),
              "system\nDefault sys.\nuser\nhi\nassistant\n");
    // Rendering specials reproduces the full template text.
    EXPECT_EQ(tokenizer_->decode(ids, true),
              "<|im_start|>system\nDefault sys.<|im_end|>\n"
              "<|im_start|>user\nhi<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST_F(TokenizerUnit, IncrementalChatApiComposesToFullConversation) {
    const std::vector<ChatMessage> chat = {{"system", "sys msg"}, {"user", "hi"}};
    std::vector<int> incremental = tokenizer_->encode_chat_prelude("sys msg");
    const auto user = tokenizer_->encode_chat_message({"user", "hi"});
    const auto gen = tokenizer_->encode_generation_prompt();
    incremental.insert(incremental.end(), user.begin(), user.end());
    incremental.insert(incremental.end(), gen.begin(), gen.end());

    EXPECT_EQ(incremental, tokenizer_->apply_chat_template(chat, true));
}

TEST_F(TokenizerUnit, ByteLevelRoundTripOnUnknownBytesUsesUnkPolicy) {
    // 'z' has no vocab entry and the checkpoint declares no UNK: encode must
    // fail loudly instead of silently dropping or inventing tokens.
    EXPECT_THROW(tokenizer_->encode("z", false), std::runtime_error);
}

TEST(TokenizerFactoryErrors, MissingDirectoryThrows) {
    EXPECT_THROW(TokenizerFactory::create("Z:/definitely/not/a/model/dir"),
                 std::runtime_error);
}

} // namespace
