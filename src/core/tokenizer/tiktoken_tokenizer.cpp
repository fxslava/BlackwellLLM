#include "tokenizer/tiktoken_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "tokenizer/unicode.h"

namespace blackwell {

namespace {

// RFC 4648 §4 reverse alphabet: value 0..63, or -1 for any other byte. Built
// once at first use; 256 bytes of static data, no allocation.
const signed char* base64_reverse_table() {
    static const auto table = [] {
        std::array<signed char, 256> t{};
        t.fill(-1);
        const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (signed char i = 0; i < 64; ++i)
            t[static_cast<unsigned char>(alphabet[i])] = i;
        return t;
    }();
    return table.data();
}

[[noreturn]] void bad_base64(std::string_view in) {
    throw std::runtime_error("base64_decode: not valid base64: \"" + std::string(in) + "\"");
}

} // namespace

std::string base64_decode(std::string_view in) {
    const signed char* rev = base64_reverse_table();

    // Padding is optional on input; strip it and validate the remaining length.
    size_t n = in.size();
    while (n > 0 && in[n - 1] == '=') --n;
    // A 4-character group carries 3 bytes; a trailing group of 2 or 3 carries
    // 1 or 2. A trailing group of exactly 1 character is impossible.
    if (n % 4 == 1) bad_base64(in);

    std::string out;
    out.reserve(n / 4 * 3 + 2);

    uint32_t acc = 0;   // up to 4 six-bit groups, MSB-first
    int acc_bits = 0;
    for (size_t i = 0; i < n; ++i) {
        const signed char v = rev[static_cast<unsigned char>(in[i])];
        if (v < 0) bad_base64(in);
        acc = (acc << 6) | static_cast<uint32_t>(v);
        acc_bits += 6;
        if (acc_bits >= 8) {
            acc_bits -= 8;
            out.push_back(static_cast<char>((acc >> acc_bits) & 0xFF));
        }
    }
    // Whatever is left is < 8 bits of zero padding by construction; a non-zero
    // remainder means the encoder emitted bits the decoder cannot place.
    if (acc_bits > 0 && (acc & ((1u << acc_bits) - 1u)) != 0) bad_base64(in);
    return out;
}

// ============================================================================
// Rank-file parsing
// ============================================================================

bool TiktokenTokenizer::looks_like_rank_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Sampling the head is enough to tell a tiktoken rank file from a
    // SentencePiece .model (a protobuf, which starts with a non-printable tag)
    // or from anything else that happens to be called tokenizer.model.
    std::string line;
    int checked = 0;
    while (checked < 8 && std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;

        const size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size()) return false;
        for (size_t i = sp + 1; i < line.size(); ++i)
            if (line[i] < '0' || line[i] > '9') return false;
        try {
            (void)base64_decode(std::string_view(line).substr(0, sp));
        } catch (const std::runtime_error&) {
            return false;
        }
        ++checked;
    }
    return checked > 0;
}

TiktokenModelData TiktokenTokenizer::load_rank_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("TiktokenTokenizer: cannot open rank file " + path);

    TiktokenModelData data;
    std::string line;
    size_t lineno = 0;

    // Ranks are dense over [0, N) in every tiktoken file, but nothing in the
    // format guarantees they arrive in order -- place each by index and verify
    // density afterwards, so a gap is reported instead of silently shifting ids.
    std::vector<std::pair<std::string, int>> entries;
    int max_rank = -1;

    while (std::getline(f, line)) {
        ++lineno;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;

        const size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size())
            throw std::runtime_error("TiktokenTokenizer: " + path + ":" + std::to_string(lineno) +
                                     ": expected \"<base64> <rank>\", got \"" + line + "\"");

        const char* rank_begin = line.c_str() + sp + 1;
        char* rank_end = nullptr;
        const long long rank = std::strtoll(rank_begin, &rank_end, 10);
        if (rank_end == rank_begin || *rank_end != '\0' || rank < 0 ||
            rank > std::numeric_limits<int>::max())
            throw std::runtime_error("TiktokenTokenizer: " + path + ":" + std::to_string(lineno) +
                                     ": rank is not a non-negative integer: \"" +
                                     std::string(rank_begin) + "\"");

        std::string token;
        try {
            token = base64_decode(std::string_view(line).substr(0, sp));
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("TiktokenTokenizer: " + path + ":" + std::to_string(lineno) +
                                     ": " + e.what());
        }
        if (token.empty())
            throw std::runtime_error("TiktokenTokenizer: " + path + ":" + std::to_string(lineno) +
                                     ": empty token");

        max_rank = std::max(max_rank, static_cast<int>(rank));
        entries.emplace_back(std::move(token), static_cast<int>(rank));
    }

    if (entries.empty())
        throw std::runtime_error("TiktokenTokenizer: " + path + " contains no entries");

    data.decoder.assign(static_cast<size_t>(max_rank) + 1, std::string());
    data.encoder.reserve(entries.size() * 2);
    std::vector<bool> seen(data.decoder.size(), false);

    for (auto& [token, rank] : entries) {
        const size_t r = static_cast<size_t>(rank);
        if (seen[r])
            throw std::runtime_error("TiktokenTokenizer: " + path + ": duplicate rank " +
                                     std::to_string(rank));
        seen[r] = true;
        data.encoder.emplace(token, rank);
        data.decoder[r] = std::move(token);
    }
    for (size_t r = 0; r < seen.size(); ++r)
        if (!seen[r])
            throw std::runtime_error("TiktokenTokenizer: " + path + ": rank " +
                                     std::to_string(r) + " is missing (ranks must be dense)");

    return data;
}

// ============================================================================
// TiktokenTokenizer
// ============================================================================

TiktokenTokenizer::TiktokenTokenizer(TiktokenModelData data,
                                     std::unique_ptr<IChatTemplate> tmpl)
    : data_(std::move(data)), chat_template_(std::move(tmpl)) {
    if (data_.encoder.empty())
        throw std::runtime_error("TiktokenTokenizer: empty vocabulary");

    // Every raw byte must be a token of its own; without that, BPE can reach a
    // symbol it cannot emit and there is no UNK to fall back on in this family.
    for (int b = 0; b < 256; ++b) {
        const std::string one(1, static_cast<char>(b));
        if (data_.encoder.find(one) == data_.encoder.end())
            throw std::runtime_error(
                "TiktokenTokenizer: vocabulary does not cover single byte 0x" +
                std::string(1, "0123456789abcdef"[b >> 4]) +
                std::string(1, "0123456789abcdef"[b & 0xF]) +
                "; a tiktoken vocabulary must contain all 256");
    }

    for (const auto& [content, id] : data_.specials) {
        if (content.empty()) continue;
        special_first_byte_[static_cast<unsigned char>(content[0])] = true;
        max_special_len_ = std::max(max_special_len_, content.size());
        data_.special_by_id.emplace(id, content);
    }
}

int TiktokenTokenizer::rank_of(std::string_view piece) const {
    // Heterogeneous lookup would avoid this temporary, but it needs C++20
    // transparent hashing on the map; the pieces here are <= a few bytes and
    // this is a load-once/decode-rarely path, so the copy is not worth the
    // extra machinery.
    const auto it = data_.encoder.find(std::string(piece));
    return it == data_.encoder.end() ? -1 : it->second;
}

void TiktokenTokenizer::bpe_encode_piece(std::string_view piece, std::vector<int>& out) const {
    if (piece.empty()) return;
    if (piece.size() == 1) {
        out.push_back(rank_of(piece)); // guaranteed present by the ctor's check
        return;
    }

    // Whole-piece shortcut. This is NOT merely an optimization -- it is part of
    // the reference algorithm, and dropping it changes output. tiktoken applies
    // it in _encode_ordinary_native, *before* byte_pair_encode ever runs:
    //
    //     if let Some(token) = self.encoder.get(piece) { ret.push(*token); continue; }
    //
    // The two disagree whenever a vocabulary entry's own adjacent pairs are
    // absent: for ranks {all 256 bytes, "xyz"} with neither "xy" nor "yz"
    // present, the merge loop below halts at x,y,z while the reference emits
    // "xyz". Verified against Python tiktoken; pinned by
    // TiktokenBpe.WholePieceWinsEvenWhenNoPairMerges.
    if (const int whole = rank_of(piece); whole >= 0) {
        out.push_back(whole);
        return;
    }

    // Symbols as [begin, end) offsets into `piece`, one raw byte each to start.
    // THE tiktoken merge rule: the pair to merge is the one whose CONCATENATION
    // has the lowest rank in the vocabulary. (Byte-level HuggingFace BPE instead
    // consults a separate merges list -- see the header.)
    std::vector<size_t> bounds(piece.size() + 1);
    for (size_t i = 0; i <= piece.size(); ++i) bounds[i] = i;

    while (bounds.size() > 2) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best = 0;
        for (size_t k = 0; k + 2 < bounds.size(); ++k) {
            const int r = rank_of(piece.substr(bounds[k], bounds[k + 2] - bounds[k]));
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best = k;
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) break;
        bounds.erase(bounds.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }

    for (size_t k = 0; k + 1 < bounds.size(); ++k) {
        const int id = rank_of(piece.substr(bounds[k], bounds[k + 1] - bounds[k]));
        if (id < 0)
            throw std::runtime_error("TiktokenTokenizer: BPE produced a symbol outside the "
                                     "vocabulary (corrupt rank file?)");
        out.push_back(id);
    }
}

void TiktokenTokenizer::encode_plain_segment(std::string_view segment,
                                             std::vector<int>& out) const {
    if (segment.empty()) return;

    std::vector<size_t> off;
    const std::vector<uint32_t> cps = unicode::decode_utf8(std::string(segment), &off);
    off.push_back(segment.size());

    size_t i = 0;
    while (i < cps.size()) {
        const size_t end = unicode::match_pretoken(cps, i, data_.max_digit_run);
        bpe_encode_piece(segment.substr(off[i], off[end] - off[i]), out);
        i = end;
    }
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text,
                                           bool add_special_tokens) const {
    std::vector<int> out;
    // One id per ~3 bytes is a generous-but-sane first guess for mixed text.
    out.reserve(text.size() / 3 + data_.prefix_ids.size() + 4);

    if (add_special_tokens)
        out.insert(out.end(), data_.prefix_ids.begin(), data_.prefix_ids.end());

    // Special-token literals are matched in the raw text ahead of BPE and
    // longest-first, so "<|user|>" wins over any shorter literal sharing its
    // prefix. This mirrors HuggingFace's added-token behaviour, which is what
    // the chat template relies on: it renders role markers as *text*.
    size_t pos = 0, plain_begin = 0;
    while (pos < text.size()) {
        if (!special_first_byte_[static_cast<unsigned char>(text[pos])]) {
            ++pos;
            continue;
        }

        const size_t probe = std::min(max_special_len_, text.size() - pos);
        int hit_id = -1;
        size_t hit_len = 0;
        for (size_t len = probe; len >= 1; --len) {
            const auto it = data_.specials.find(text.substr(pos, len));
            if (it != data_.specials.end()) {
                hit_id = it->second;
                hit_len = len;
                break;
            }
        }
        if (hit_id < 0) {
            ++pos;
            continue;
        }

        encode_plain_segment(std::string_view(text).substr(plain_begin, pos - plain_begin), out);
        out.push_back(hit_id);
        pos += hit_len;
        plain_begin = pos;
    }
    encode_plain_segment(std::string_view(text).substr(plain_begin), out);

    return out;
}

std::string TiktokenTokenizer::decode(int token_id, bool render_special) const {
    if (const auto it = data_.special_by_id.find(token_id); it != data_.special_by_id.end())
        return render_special ? it->second : "";

    if (token_id < 0 || static_cast<size_t>(token_id) >= data_.decoder.size())
        return ""; // unknown id: same silent-skip contract as ByteLevelBpeTokenizer
    return data_.decoder[static_cast<size_t>(token_id)];
}

std::string TiktokenTokenizer::decode(const std::vector<int>& token_ids,
                                      bool render_special) const {
    // Raw bytes are concatenated verbatim, NOT sanitized into valid UTF-8: a
    // multi-byte codepoint may straddle two tokens, so a streaming caller that
    // appends per-token output must be able to complete it on the next token.
    std::string out;
    for (const int id : token_ids) out += decode(id, render_special);
    return out;
}

int TiktokenTokenizer::token_to_id(const std::string& token) const {
    if (const auto it = data_.specials.find(token); it != data_.specials.end())
        return it->second;
    if (const auto it = data_.encoder.find(token); it != data_.encoder.end())
        return it->second;
    return -1;
}

} // namespace blackwell
