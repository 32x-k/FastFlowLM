/// \file plamo3_tokenizer.cpp
/// \brief PLaMo3 tokenizer implementation
/// \note Port of llama.cpp's llm_tokenizer_plamo2 (pre_segment = true).
///       Reads the plamo-3 tokenizer.jsonl vocabulary, one ["text", score, "TYPE"]
///       JSON array per line where the line index is the token id.

#include "tokenizer/plamo3_tokenizer.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

static size_t unicode_len_utf8(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (u < 0x80) return 1;
    if ((u & 0xE0) == 0xC0) return 2;
    if ((u & 0xF0) == 0xE0) return 3;
    return 4;
}

static std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& text) {
    std::vector<uint32_t> cpts;
    cpts.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = text[i];
        uint32_t cp;
        size_t w;
        if (c < 0x80) { cp = c; w = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; w = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; w = 3; }
        else { cp = c & 0x07; w = 4; }
        for (size_t j = 1; j < w && i + j < text.size(); j++) {
            cp = (cp << 6) | (text[i + j] & 0x3F);
        }
        cpts.push_back(cp);
        i += w;
    }
    return cpts;
}

static uint32_t cpt_from_utf8(std::string_view s) {
    const std::vector<uint32_t> c = unicode_cpts_from_utf8(std::string(s));
    return c.empty() ? 0 : c[0];
}

static bool is_whitespace_cpt(uint32_t c) {
    // the same set Python's \s covers for our purposes (plamo3 pass-1 boundary check)
    if (c == ' ' || (c >= '\t' && c <= '\r') || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F || c == 0x85) return true;
    switch (c) {
        case 0x00A0: case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return c >= 0x2000 && c <= 0x200A;
    }
}

bool Plamo3Tokenizer::exists(const std::string& model_path) {
    return std::filesystem::exists(std::filesystem::path(model_path) / "tokenizer.jsonl");
}

Plamo3Tokenizer::Plamo3Tokenizer(const std::string& model_path) {
    const std::string path = model_path + "/tokenizer.jsonl";
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        throw std::runtime_error("Cannot open " + path);
    }
    tokens_.reserve(107520);
    is_byte_token_.reserve(107520);
    std::vector<float> scores;
    scores.reserve(107520);
    std::string line;
    while (std::getline(fs, line)) {
        if (line.empty()) continue;
        const nlohmann::json entry = nlohmann::json::parse(line);
        tokens_.push_back(entry[0].get<std::string>());
        scores.push_back(entry[1].get<float>());
        const std::string type = entry.size() > 2 ? entry[2].get<std::string>() : "NORMAL";
        is_byte_token_.push_back(type == "BYTE");
    }
    build(scores);
}

const std::string& Plamo3Tokenizer::id_to_token(int id) const {
    return tokens_.at(id);
}

void Plamo3Tokenizer::build(const std::vector<float>& scores) {
    bytes_.assign(256, 0);
    to_suffix_id_.clear();
    table_.clear();

    std::unordered_map<std::string_view, float> suffix_to_score;
    std::unordered_map<std::string_view, int32_t> token_to_id;
    suffix_to_score.reserve(tokens_.size() * 4);
    token_to_id.reserve(tokens_.size());

    auto next_cpt = [](std::string_view s, size_t offset) -> size_t {
        return offset + std::min(s.size() - offset, unicode_len_utf8(s[offset]));
    };

    for (size_t token_id = 0; token_id < tokens_.size(); ++token_id) {
        const std::string_view text = tokens_[token_id];
        token_to_id[text] = static_cast<int32_t>(token_id);

        if (is_byte_token_[token_id]) {
            if (text.length() == 6 && text.substr(0, 3) == "<0x" && text.back() == '>') {
                const std::string hex_str(text.substr(3, 2));
                if (std::isxdigit(static_cast<unsigned char>(hex_str[0])) && std::isxdigit(static_cast<unsigned char>(hex_str[1]))) {
                    bytes_[std::stoi(hex_str, nullptr, 16)] = static_cast<int32_t>(token_id);
                }
            }
            continue;
        }

        suffix_to_score[text] = scores[token_id];
        for (size_t i = text.empty() ? 0 : next_cpt(text, 0); i < text.size(); i = next_cpt(text, i)) {
            suffix_to_score.try_emplace(text.substr(i), std::numeric_limits<float>::quiet_NaN());
        }
    }

    for (int i = 0; i < 256; ++i) {
        if (bytes_[i] == 0) {
            throw std::runtime_error("Byte token for <0x" + std::to_string(i) + "> is not set");
        }
    }

    std::vector<std::string_view> suffixes;
    suffixes.reserve(suffix_to_score.size() + 1);
    for (const auto& pair : suffix_to_score) {
        suffixes.push_back(pair.first);
    }
    suffixes.push_back("");

    std::sort(suffixes.begin(), suffixes.end(), [](std::string_view a, std::string_view b) {
        return std::lexicographical_compare(a.rbegin(), a.rend(), b.rbegin(), b.rend(), [](char x, char y) {
            return static_cast<unsigned char>(x) < static_cast<unsigned char>(y);
        });
    });

    std::vector<size_t> prefix_ends;
    auto get_prefix_ends = [&](std::string_view suffix) {
        prefix_ends.clear();
        for (size_t i = 0; i < suffix.size();) {
            i = next_cpt(suffix, i);
            prefix_ends.push_back(i);
        }
    };

    std::unordered_map<std::string_view, int32_t> suffix_to_id;
    suffix_to_id.reserve(suffixes.size());
    int32_t num_pieces = 0;

    for (const auto& suffix : suffixes) {
        suffix_to_id[suffix] = num_pieces;
        if (!suffix.empty()) {
            get_prefix_ends(suffix);
            const uint32_t first_cpt = cpt_from_utf8(suffix.substr(0, prefix_ends[0]));
            const int64_t piece_code = (static_cast<int64_t>(first_cpt) << 32) | suffix_to_id[suffix.substr(prefix_ends[0])];
            to_suffix_id_[piece_code] = num_pieces;

            int32_t pieces_for_suffix = 1;
            for (size_t end : prefix_ends) {
                if (suffix_to_score.find(suffix.substr(0, end)) != suffix_to_score.end()) {
                    pieces_for_suffix++;
                }
            }
            num_pieces += pieces_for_suffix;
        } else {
            num_pieces++;
        }
    }

    table_.resize(num_pieces, std::vector<int32_t>(4, 0));
    int32_t table_idx = 0;

    for (const auto& suffix : suffixes) {
        get_prefix_ends(suffix);
        for (int32_t piece_length = static_cast<int32_t>(prefix_ends.size()); piece_length > 0; --piece_length) {
            const std::string_view piece = suffix.substr(0, prefix_ends[piece_length - 1]);
            auto score_it = suffix_to_score.find(piece);
            if (score_it == suffix_to_score.end()) {
                continue;
            }

            table_[table_idx][TABLE_PIECE_LENGTH] = piece_length;
            auto token_it = token_to_id.find(piece);
            table_[table_idx][TABLE_TOKEN_ID] = (token_it != token_to_id.end()) ? token_it->second : -1;

            const float score = score_it->second;
            table_[table_idx][TABLE_SCORE] = std::isfinite(score) ?
                static_cast<int32_t>(std::round(score * 1e4)) : INVALID_SCORE;
            table_[table_idx][TABLE_PIECE_ID] = suffix_to_id[piece];
            table_idx++;
        }

        table_[table_idx][TABLE_PIECE_LENGTH] = 1;
        table_[table_idx][TABLE_TOKEN_ID] = -1;
        table_[table_idx][TABLE_SCORE] = UNKNOWN_SCORE;
        table_idx++;
    }
}

std::vector<int> Plamo3Tokenizer::encode(const std::string& text) const {
    // plamo3 == pre_segment_ variant: cut on <|plamo:...|> markers and repeated runs
    const std::vector<uint32_t> unicode_data = unicode_cpts_from_utf8(text);
    const size_t n = unicode_data.size();
    std::vector<bool> cut(n + 1, false);

    // pass 1: <|plamo:...|>
    {
        static constexpr uint32_t prefix[] = { '<', '|', 'p', 'l', 'a', 'm', 'o', ':' };
        const size_t prefix_len = std::size(prefix);
        size_t i = 0;
        while (i + prefix_len <= n) {
            if (!std::equal(prefix, prefix + prefix_len, unicode_data.begin() + i)) {
                i++;
                continue;
            }
            size_t j = i + prefix_len;
            while (j < n && j - (i + prefix_len) < 64 && unicode_data[j] != '|' &&
                   (unicode_data[j] < 0x1C || unicode_data[j] > 0x1F) &&
                   !is_whitespace_cpt(unicode_data[j])) {
                j++;
            }
            if (j + 1 < n && unicode_data[j] == '|' && unicode_data[j + 1] == '>') {
                cut[i] = true;
                cut[j + 2] = true;
                i = j + 2;
            } else {
                i++;
            }
        }
    }

    // pass 2: runs of repeated characters / spaces
    {
        size_t i = 0;
        while (i < n) {
            const uint32_t c = unicode_data[i];
            size_t run = 1;
            while (i + run < n && unicode_data[i + run] == c && !cut[i + run]) {
                run++;
            }
            const bool is_repeated_chars = c != '\n' && run >= 4;
            const bool is_spaces = c == ' ' && run >= 2;
            if (is_repeated_chars || is_spaces) {
                cut[i] = true;
                cut[i + run] = true;
            }
            i += run;
        }
    }

    std::vector<int> output;
    size_t seg_start = 0;
    for (size_t seg_end = 0; seg_end <= n; ++seg_end) {
        const bool is_boundary = seg_end < n && unicode_data[seg_end] == 0xEE00;
        if (seg_end == n || cut[seg_end] || is_boundary) {
            if (seg_start < seg_end) {
                const std::vector<uint32_t> segment(unicode_data.begin() + seg_start,
                                                    unicode_data.begin() + seg_end);
                const std::vector<int> tokens = encode_cpts(segment);
                output.insert(output.end(), tokens.begin(), tokens.end());
            }
            seg_start = seg_end + is_boundary;
        }
    }
    return output;
}

std::vector<int> Plamo3Tokenizer::encode_cpts(const std::vector<uint32_t>& unicode_data) const {
    if (unicode_data.empty()) {
        return {};
    }
    const size_t data_len = unicode_data.size();

    std::vector<int64_t> scores(data_len + 1, static_cast<int64_t>(1) << 60);
    scores[data_len] = 0;
    std::vector<std::vector<int32_t>> path(data_len + 1, std::vector<int32_t>(3, 0));

    int32_t suffix_id = 0;

    for (int i = static_cast<int>(data_len) - 1; i >= 0; --i) {
        const uint32_t c = unicode_data[i];

        for (size_t p = suffix_id; p < table_.size(); ++p) {
            const int64_t piece_code = (static_cast<int64_t>(c) << 32) | table_[p][TABLE_PIECE_ID];
            auto it = to_suffix_id_.find(piece_code);
            suffix_id = (it != to_suffix_id_.end()) ? it->second : 0;
            if (suffix_id > 0 || table_[p][TABLE_SCORE] == UNKNOWN_SCORE) {
                break;
            }
        }

        for (size_t p = suffix_id; p < table_.size(); ++p) {
            const int32_t score = table_[p][TABLE_SCORE];
            if (score > INVALID_SCORE) {
                const int32_t piece_length = table_[p][TABLE_PIECE_LENGTH];
                const int64_t s = scores[i + piece_length] - score;
                if (s < scores[i]) {
                    scores[i] = s;
                    path[i][PATH_TOKEN_LENGTH] = piece_length;
                    path[i][PATH_TOKEN_ID] = table_[p][TABLE_TOKEN_ID];
                    path[i][PATH_NUM_TOKENS] = path[i + piece_length][PATH_NUM_TOKENS] + 1;
                    if (score == UNKNOWN_SCORE) {
                        path[i][PATH_NUM_TOKENS] += (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
                    }
                }
            }
            if (score == UNKNOWN_SCORE) {
                break;
            }
        }
    }

    std::vector<int> token_ids;
    token_ids.reserve(path[0][PATH_NUM_TOKENS]);

    int pos = 0;
    while (pos < static_cast<int>(data_len)) {
        if (path[pos][PATH_TOKEN_ID] >= 0) {
            token_ids.push_back(path[pos][PATH_TOKEN_ID]);
        } else {
            const uint32_t c = unicode_data[pos];
            const int s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
            for (int i = 0; i < s; ++i) {
                uint8_t b;
                if (s == 1) {
                    b = c;
                } else {
                    b = (i == 0) ? ((0xF00 >> s) & 0xFF) : 0x80;
                }
                token_ids.push_back(bytes_[b | ((c >> ((s - i - 1) * 6)) & 0x3F)]);
            }
        }
        assert(path[pos][PATH_TOKEN_LENGTH] > 0);
        pos += path[pos][PATH_TOKEN_LENGTH];
    }
    return token_ids;
}
