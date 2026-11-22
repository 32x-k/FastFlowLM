/// \file plamo3_tokenizer.hpp
/// \brief PLaMo3 tokenizer (Aho-Corasick-free suffix Unigram + byte fallback)
/// \note Port of llama.cpp's llm_tokenizer_plamo2 (pre_segment = true) for the
///       plamo-3 tokenizer.jsonl vocabulary.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Plamo3Tokenizer {
public:
    /// \brief load tokenizer.jsonl from a model directory; returns false when absent
    static bool exists(const std::string& model_path);
    explicit Plamo3Tokenizer(const std::string& model_path);

    std::vector<int> encode(const std::string& text) const;
    /// \brief raw token text (byte tokens keep their "<0xNN>" spelling)
    const std::string& id_to_token(int id) const;
    int vocab_size() const { return static_cast<int>(tokens_.size()); }

private:
    std::vector<int> encode_cpts(const std::vector<uint32_t>& cpts) const;
    void build(const std::vector<float>& scores);

    static constexpr int32_t TABLE_PIECE_LENGTH = 0;
    static constexpr int32_t TABLE_TOKEN_ID     = 1;
    static constexpr int32_t TABLE_SCORE        = 2;
    static constexpr int32_t TABLE_PIECE_ID     = 3;

    static constexpr int32_t PATH_TOKEN_LENGTH = 0;
    static constexpr int32_t PATH_TOKEN_ID     = 1;
    static constexpr int32_t PATH_NUM_TOKENS   = 2;

    static constexpr int32_t INVALID_SCORE = -20000000;
    static constexpr int32_t UNKNOWN_SCORE = -10000000;

    std::vector<std::string> tokens_;
    std::vector<bool> is_byte_token_;
    std::vector<int32_t> bytes_;              // byte value -> token id
    std::unordered_map<int64_t, int32_t> to_suffix_id_;
    std::vector<std::vector<int32_t>> table_;
};
