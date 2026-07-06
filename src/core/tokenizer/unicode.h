#pragma once
// UTF-8 and Unicode-class helpers for the byte-level BPE tokenizer. The
// classification tables are a pragmatic subset of \p{L}/\p{N}/\s covering the
// scripts the engine is validated on (Latin incl. extended, Cyrillic, Greek,
// CJK, common digit blocks); exotic scripts may pre-tokenize slightly
// differently from the reference implementation but still round-trip exactly,
// because byte-level BPE never loses bytes.
#include <cstdint>
#include <string>
#include <vector>

namespace blackwell::unicode {

// Decodes UTF-8 into codepoints. Invalid bytes decode as U+FFFD but never
// throw (byte-level BPE must accept arbitrary input). If `byte_offsets` is
// non-null it receives, per codepoint, the byte offset where it starts.
std::vector<uint32_t> decode_utf8(const std::string& text,
                                  std::vector<size_t>* byte_offsets = nullptr);

void append_utf8(std::string& out, uint32_t cp);

bool is_letter(uint32_t cp);
bool is_number(uint32_t cp);
bool is_whitespace(uint32_t cp);

// GPT-2 byte<->unicode alphabet: every raw byte maps to a printable
// codepoint so vocabulary entries are valid UTF-8 strings.
const uint32_t* byte_to_unicode_table(); // [256]
// Reverse lookup; returns -1 when `cp` is not part of the byte alphabet.
int unicode_to_byte(uint32_t cp);

} // namespace blackwell::unicode
