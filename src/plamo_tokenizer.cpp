// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "plamo_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "darts_clone/darts.h"
#include "utils.hpp"

namespace {
constexpr int32_t BYTE_FALLBACK_SCORE = -10000000;
constexpr int32_t INVALID_SCORE = -20000000;

struct Path {
    int64_t score = 0;
    size_t end = 0;
    int32_t id = -1;
};

// The string input is UTF-8. Invalid lead or continuation bytes advance by one byte.
size_t next_character(std::string_view text, size_t start) {
    const auto lead = static_cast<uint8_t>(text[start]);
    size_t width = lead >= 0xf0 && lead <= 0xf4 ? 4 :
                   lead >= 0xe0 && lead <= 0xef ? 3 :
                   lead >= 0xc2 && lead <= 0xdf ? 2 : 1;
    if (width > text.size() - start) {
        return start + 1;
    }
    for (size_t i = 1; i < width; ++i) {
        if ((static_cast<uint8_t>(text[start + i]) & 0xc0) != 0x80) {
            return start + 1;
        }
    }
    return start + width;
}
}  // namespace

class PlamoTokenizerImpl {
public:
    explicit PlamoTokenizerImpl(const ov::TensorVector& inputs) {
        const auto* begins = inputs[5].data<const int32_t>();
        const auto* ends = inputs[6].data<const int32_t>();
        const auto* pieces = inputs[7].data<const char>();
        const auto* scores = inputs[8].data<const int32_t>();
        const auto size = inputs[5].get_size();
        OPENVINO_ASSERT(inputs[6].get_size() == size && inputs[8].get_size() == size);
        OPENVINO_ASSERT(inputs[9].get_size() == m_byte_ids.size());
        m_scores.assign(scores, scores + size);
        std::copy_n(inputs[9].data<const int32_t>(), m_byte_ids.size(), m_byte_ids.begin());

        std::vector<std::pair<std::string_view, int32_t>> entries;
        for (size_t i = 0; i < size; ++i) {
            if (begins[i] != ends[i] && scores[i] > INVALID_SCORE) {
                entries.emplace_back(std::string_view(pieces + begins[i], ends[i] - begins[i]),
                                     static_cast<int32_t>(i));
            }
        }
        std::sort(entries.begin(), entries.end());
        std::vector<const char*> keys;
        std::vector<size_t> lengths;
        std::vector<int32_t> ids;
        for (const auto& [piece, id] : entries) {
            keys.push_back(piece.data());
            lengths.push_back(piece.size());
            ids.push_back(id);
        }
        OPENVINO_ASSERT(m_trie.build(keys.size(), keys.data(), lengths.data(), ids.data()) == 0,
                        "Failed to build the PLaMo vocabulary trie");
    }

    void tokenize(std::string_view text, std::vector<Path>& path, std::vector<int32_t>& tokens) const {
        path.resize(text.size() + 1);
        path.back() = {};
        for (size_t start = text.size(); start-- > 0;) {
            const auto next = next_character(text, start);
            path[start] = {BYTE_FALLBACK_SCORE + path[next].score, next, -1};
            size_t node = 0;
            size_t end = start;
            for (size_t limit = start + 1; limit <= text.size(); ++limit) {
                const auto id = m_trie.traverse(text.data(), node, end, limit);
                if (id == -2) {
                    break;
                }
                if (id >= 0) {
                    const auto score = m_scores[id] + path[end].score;
                    // Trie traversal finds shorter pieces first. On equal scores keep the longest
                    // prefix, and prefer a vocabulary piece to the single-character fallback.
                    if (score >= path[start].score) {
                        path[start] = {score, end, id};
                    }
                }
            }
        }

        for (size_t start = 0; start < text.size();) {
            const auto& choice = path[start];
            if (choice.id >= 0) {
                tokens.push_back(choice.id);
            } else {
                for (auto pos = start; pos < choice.end; ++pos) {
                    tokens.push_back(m_byte_ids[static_cast<uint8_t>(text[pos])]);
                }
            }
            start = choice.end;
        }
    }

private:
    Darts::DoubleArray m_trie;
    std::vector<int32_t> m_scores;
    std::array<int32_t, 256> m_byte_ids{};
};

void PlamoTokenizer::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 10, "Expected ragged strings, vocabulary, scores and byte IDs.");
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    for (size_t i : {8, 9}) {
        NODE_VALIDATION_CHECK(this, get_input_element_type(i) == ov::element::i32, "Expected an i32 tensor.");
    }
    set_ragged_output(this, 0, get_input_partial_shape(0), ov::element::i32);
}

bool PlamoTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    std::call_once(m_init, [&] { m_tokenizer = std::make_shared<PlamoTokenizerImpl>(inputs); });

    const auto* row_begins = inputs[0].data<const int32_t>();
    const auto* row_ends = inputs[1].data<const int32_t>();
    const auto* begins = inputs[2].data<const int32_t>();
    const auto* ends = inputs[3].data<const int32_t>();
    const auto* chars = inputs[4].data<const char>();

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    auto* output_begins = outputs[0].data<int32_t>();
    auto* output_ends = outputs[1].data<int32_t>();
    std::vector<int32_t> tokens;
    std::vector<Path> path;
    for (size_t row = 0; row < inputs[0].get_size(); ++row) {
        output_begins[row] = static_cast<int32_t>(tokens.size());
        for (auto i = row_begins[row]; i < row_ends[row]; ++i) {
            if (begins[i] != ends[i]) {
                m_tokenizer->tokenize(std::string_view(chars + begins[i], ends[i] - begins[i]), path, tokens);
            }
        }
        output_ends[row] = static_cast<int32_t>(tokens.size());
    }
    outputs[2].set_shape({tokens.size()});
    if (!tokens.empty()) {
        std::copy(tokens.begin(), tokens.end(), outputs[2].data<int32_t>());
    }
    return true;
}
