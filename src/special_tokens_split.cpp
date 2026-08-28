// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "special_tokens_split.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset13.hpp"
#include "darts_clone/darts.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <string_view>

using namespace ov;
using namespace ov::opset13;

namespace {

constexpr char WHITESPACE_PATTERN[] = "(?:\\s*)";

struct SpecialToken {
    std::string text;
    bool strip_left;
    bool strip_right;
};

struct TokenMetadata {
    uint32_t length;
    bool strip_left;
    bool strip_right;
};

struct Match {
    uint32_t whitespace_start;
    uint32_t token_start;
    uint32_t token_end;
    uint32_t token_id;
};

struct BuildNode {
    std::vector<std::pair<unsigned char, uint32_t>> children;
    std::vector<uint32_t> token_ids;
    uint32_t failure = 0;
    uint32_t dat_position = 0;
};

constexpr uint32_t NO_NODE = std::numeric_limits<uint32_t>::max();

uint32_t find_child(const BuildNode& node, unsigned char character) {
    const auto child = std::lower_bound(
        node.children.begin(), node.children.end(), character,
        [](const auto& edge, unsigned char value) { return edge.first < value; }
    );
    return child != node.children.end() && child->first == character ? child->second : NO_NODE;
}

bool consume(const std::string& pattern, size_t& position, const char* value) {
    const size_t length = std::char_traits<char>::length(value);
    if (pattern.compare(position, length, value) != 0) {
        return false;
    }
    position += length;
    return true;
}

std::vector<SpecialToken> parse_special_tokens_pattern(const std::string& pattern) {
    std::vector<SpecialToken> tokens;
    size_t position = 0;

    while (position < pattern.size()) {
        const bool strip_left = consume(pattern, position, WHITESPACE_PATTERN);
        OPENVINO_ASSERT(position < pattern.size() && pattern[position] == '(',
                        "[ SpecialTokensSplit ] Unsupported split pattern at byte ", position);
        ++position;

        std::string token;
        std::vector<std::string> group_tokens;
        bool group_closed = false;
        while (position < pattern.size()) {
            const char current = pattern[position++];
            if (current == '\\') {
                OPENVINO_ASSERT(position < pattern.size(),
                                "[ SpecialTokensSplit ] Invalid trailing escape in split pattern");
                token.push_back(pattern[position++]);
            } else if (current == '|') {
                group_tokens.push_back(std::move(token));
                token.clear();
            } else if (current == ')') {
                group_tokens.push_back(std::move(token));
                group_closed = true;
                break;
            } else {
                token.push_back(current);
            }
        }
        OPENVINO_ASSERT(group_closed, "[ SpecialTokensSplit ] Unterminated group in split pattern");

        const bool strip_right = consume(pattern, position, WHITESPACE_PATTERN);
        for (auto& group_token : group_tokens) {
            if (!group_token.empty()) {
                tokens.push_back({std::move(group_token), strip_left, strip_right});
            }
        }

        if (position < pattern.size()) {
            OPENVINO_ASSERT(pattern[position] == '|',
                            "[ SpecialTokensSplit ] Unsupported split pattern at byte ", position);
            ++position;
        }
    }

    OPENVINO_ASSERT(!tokens.empty(), "[ SpecialTokensSplit ] Split pattern contains no special tokens");
    return tokens;
}

size_t utf8_code_point_length(unsigned char lead) {
    if (lead < 0x80) {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0) {
        return 2;
    }
    if ((lead & 0xF0) == 0xE0) {
        return 3;
    }
    if ((lead & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

bool decode_code_point(std::string_view text, size_t position, uint32_t& code_point, size_t& length) {
    length = utf8_code_point_length(static_cast<unsigned char>(text[position]));
    if (position + length > text.size()) {
        length = 1;
        return false;
    }
    if (length == 1) {
        code_point = static_cast<unsigned char>(text[position]);
        return code_point < 0x80;
    }

    const unsigned char lead = static_cast<unsigned char>(text[position]);
    code_point = lead & ((1u << (7 - length)) - 1);
    for (size_t index = 1; index < length; ++index) {
        const unsigned char continuation = static_cast<unsigned char>(text[position + index]);
        if ((continuation & 0xC0) != 0x80) {
            length = 1;
            return false;
        }
        code_point = (code_point << 6) | (continuation & 0x3F);
    }
    return true;
}

bool is_unicode_whitespace(uint32_t code_point) {
    return (code_point >= 0x09 && code_point <= 0x0D) || code_point == 0x20 || code_point == 0x85 ||
           code_point == 0xA0 || code_point == 0x1680 || code_point == 0x180E ||
           (code_point >= 0x2000 && code_point <= 0x200A) ||
           code_point == 0x2028 || code_point == 0x2029 || code_point == 0x202F || code_point == 0x205F ||
           code_point == 0x3000;
}

size_t whitespace_end(std::string_view text, size_t position) {
    while (position < text.size()) {
        uint32_t code_point = 0;
        size_t length = 0;
        if (!decode_code_point(text, position, code_point, length) || !is_unicode_whitespace(code_point)) {
            break;
        }
        position += length;
    }
    return position;
}

size_t whitespace_start(std::string_view text, size_t position) {
    while (position > 0) {
        size_t previous = position - 1;
        while (previous > 0 && (static_cast<unsigned char>(text[previous]) & 0xC0) == 0x80) {
            --previous;
        }
        uint32_t code_point = 0;
        size_t length = 0;
        if (!decode_code_point(text, previous, code_point, length) || previous + length != position ||
            !is_unicode_whitespace(code_point)) {
            break;
        }
        position = previous;
    }
    return position;
}

}  // namespace

class SpecialTokensSplitImpl {
public:
    explicit SpecialTokensSplitImpl(const std::string& split_pattern)
        : m_tokens(parse_special_tokens_pattern(split_pattern)) {
        build();
    }

    size_t match_end(std::string_view text, const Match& match) const {
        return m_token_metadata[match.token_id].strip_right
            ? whitespace_end(text, match.token_end)
            : match.token_end;
    }

    void find_matches(std::string_view text, std::vector<Match>& matches) const {
        matches.clear();
        uint32_t state = 0;

        for (size_t position = 0; position < text.size(); ++position) {
            if (!step(state, static_cast<unsigned char>(text[position]))) {
                continue;
            }

            for (uint32_t output_state = state; output_state != 0;
                 output_state = m_output_link[output_state]) {
                for (uint32_t output_index = m_output_offsets[output_state];
                     output_index < m_output_offsets[output_state + 1];
                     ++output_index) {
                    const uint32_t token_id = m_output_token_ids[output_index];
                    const auto& token = m_token_metadata[token_id];
                    const size_t token_end = position + 1;
                    const size_t token_start = token_end - token.length;
                    matches.push_back({
                        static_cast<uint32_t>(token.strip_left ? whitespace_start(text, token_start) : token_start),
                        static_cast<uint32_t>(token_start),
                        static_cast<uint32_t>(token_end),
                        token_id
                    });
                }
            }
        }
    }

private:
    bool step(uint32_t& state, unsigned char character) const {
        if (state == 0 && !m_first_bytes.test(character)) {
            return false;
        }
        const char key = static_cast<char>(character);
        while (true) {
            size_t next_state = state;
            size_t key_position = 0;
            if (m_trie.traverse(&key, next_state, key_position, 1) != -2) {
                state = static_cast<uint32_t>(next_state);
                return true;
            }
            if (state == 0) {
                return false;
            }
            state = m_failure[state];
        }
    }

    void build() {
        std::vector<BuildNode> nodes(1);
        OPENVINO_ASSERT(m_tokens.size() <= std::numeric_limits<uint32_t>::max(),
                        "[ SpecialTokensSplit ] Too many special tokens");
        for (size_t token_index = 0; token_index < m_tokens.size(); ++token_index) {
            const uint32_t token_id = static_cast<uint32_t>(token_index);
            uint32_t node = 0;
            for (const unsigned char character : m_tokens[token_id].text) {
                const auto child = std::find_if(
                    nodes[node].children.begin(), nodes[node].children.end(),
                    [character](const auto& edge) { return edge.first == character; }
                );
                if (child == nodes[node].children.end()) {
                    OPENVINO_ASSERT(nodes.size() < std::numeric_limits<uint32_t>::max(),
                                    "[ SpecialTokensSplit ] Automaton is too large");
                    const uint32_t child_node = static_cast<uint32_t>(nodes.size());
                    nodes[node].children.emplace_back(character, child_node);
                    nodes.emplace_back();
                    node = child_node;
                } else {
                    node = child->second;
                }
            }
            nodes[node].token_ids.push_back(token_id);
        }
        for (auto& node : nodes) {
            std::sort(node.children.begin(), node.children.end());
        }

        std::vector<uint32_t> key_indices(m_tokens.size());
        std::iota(key_indices.begin(), key_indices.end(), 0);
        std::sort(key_indices.begin(), key_indices.end(), [this](uint32_t lhs, uint32_t rhs) {
            return m_tokens[lhs].text < m_tokens[rhs].text;
        });
        key_indices.erase(std::unique(key_indices.begin(), key_indices.end(), [this](uint32_t lhs, uint32_t rhs) {
            return m_tokens[lhs].text == m_tokens[rhs].text;
        }), key_indices.end());
        std::vector<const char*> key_pointers;
        std::vector<size_t> key_lengths;
        key_pointers.reserve(key_indices.size());
        key_lengths.reserve(key_indices.size());
        for (const uint32_t token_id : key_indices) {
            key_pointers.push_back(m_tokens[token_id].text.data());
            key_lengths.push_back(m_tokens[token_id].text.size());
        }
        const int build_result = m_trie.build(key_pointers.size(), key_pointers.data(), key_lengths.data());
        OPENVINO_ASSERT(build_result == 0,
                        "[ SpecialTokensSplit ] Failed to build double-array trie");

        OPENVINO_ASSERT(m_trie.size() < std::numeric_limits<uint32_t>::max(),
                "[ SpecialTokensSplit ] Double-array trie is too large");

        std::queue<uint32_t> queue;
        for (const auto& [character, child] : nodes[0].children) {
            m_first_bytes.set(character);
            map_dat_position(nodes, 0, child, character);
            queue.push(child);
        }
        while (!queue.empty()) {
            const uint32_t node = queue.front();
            queue.pop();
            for (const auto& [character, child] : nodes[node].children) {
                map_dat_position(nodes, node, child, character);
                uint32_t failure = nodes[node].failure;
                uint32_t failure_child = find_child(nodes[failure], character);
                while (failure != 0 && failure_child == NO_NODE) {
                    failure = nodes[failure].failure;
                    failure_child = find_child(nodes[failure], character);
                }
                if (failure_child != NO_NODE && failure_child != child) {
                    nodes[child].failure = failure_child;
                }
                queue.push(child);
            }
        }

        m_failure.assign(m_trie.size(), 0);
        m_output_link.assign(m_trie.size(), 0);
        for (const auto& node : nodes) {
            m_failure[node.dat_position] = nodes[node.failure].dat_position;
            uint32_t output = node.failure;
            while (output != 0 && nodes[output].token_ids.empty()) {
                output = nodes[output].failure;
            }
            m_output_link[node.dat_position] = nodes[output].dat_position;
        }

        m_output_offsets.assign(m_trie.size() + 1, 0);
        for (const auto& node : nodes) {
            m_output_offsets[node.dat_position + 1] = static_cast<uint32_t>(node.token_ids.size());
        }
        std::partial_sum(m_output_offsets.begin(), m_output_offsets.end(), m_output_offsets.begin());
        m_output_token_ids.resize(m_output_offsets.back());
        auto output_positions = m_output_offsets;
        for (const auto& node : nodes) {
            for (const uint32_t token_id : node.token_ids) {
                m_output_token_ids[output_positions[node.dat_position]++] = token_id;
            }
        }

        m_token_metadata.reserve(m_tokens.size());
        for (const auto& token : m_tokens) {
            OPENVINO_ASSERT(token.text.size() <= std::numeric_limits<uint32_t>::max(),
                            "[ SpecialTokensSplit ] Special token is too long");
            m_token_metadata.push_back({static_cast<uint32_t>(token.text.size()), token.strip_left, token.strip_right});
        }
        std::vector<SpecialToken>().swap(m_tokens);
    }

    void map_dat_position(std::vector<BuildNode>& nodes, uint32_t parent, uint32_t child, unsigned char character) {
        size_t key_position = 0;
        size_t dat_position = nodes[parent].dat_position;
        const char key = static_cast<char>(character);
        const auto traverse_result = m_trie.traverse(&key, dat_position, key_position, 1);
        OPENVINO_ASSERT(traverse_result != -2,
                        "[ SpecialTokensSplit ] Failed to map double-array trie state");
        nodes[child].dat_position = static_cast<uint32_t>(dat_position);
    }

    Darts::DoubleArray m_trie;
    std::vector<SpecialToken> m_tokens;
    std::vector<uint32_t> m_failure;
    std::vector<uint32_t> m_output_link;
    std::vector<uint32_t> m_output_offsets;
    std::vector<uint32_t> m_output_token_ids;
    std::vector<TokenMetadata> m_token_metadata;
    std::bitset<256> m_first_bytes;
};


SpecialTokensSplit::SpecialTokensSplit(const ov::OutputVector& arguments) :
    ov::op::Op(arguments) {
    constructor_validate_and_infer_types();
}


SpecialTokensSplit::SpecialTokensSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<SpecialTokensSplitImpl>& splitter
) :
    ov::op::Op(arguments),
    m_splitter(splitter) {

    constructor_validate_and_infer_types();
}


void SpecialTokensSplit::validate_and_infer_types() {
    auto input_size = get_input_size();
    const bool has_skips = input_size == 7;

    OPENVINO_ASSERT(input_size == 6 || input_size == 7, "Incorrect number of inputs passed to SpecialTokensSplit: " + std::to_string(input_size) +  "; try to reconvert tokenizer with newer version of OpenVINO Tokenizers");
    // input strings
    check_ragged_string_input(this, 0);
    // split pattern
    check_string_scalar_input(this, 5 + has_skips);

    set_ragged_string_output(this, 0, get_input_partial_shape(0));
    if (has_skips) {
        this->set_output_type(5, get_input_element_type(5), PartialShape{Dimension::dynamic()});
    } else {
        this->set_output_type(5, ov::element::boolean, get_input_partial_shape(2));
    };
}

bool SpecialTokensSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto input_size = get_input_size();
    const bool has_skips = (input_size == 7);

    std::call_once(m_init_flag, [this, &inputs, has_skips]() {
        if (!m_splitter) {
            const auto& pattern_input = inputs[5 + has_skips];
            const std::string split_pattern(pattern_input.data<const char>(), pattern_input.get_size());
            m_splitter = std::make_shared<SpecialTokensSplitImpl>(split_pattern);
        }
    });

    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    const size_t batch_size = inputs[0].get_size();
    const size_t num_chars = inputs[4].get_size();

    Tensor skips_alternative;
    const bool *skips;
    if (has_skips) {
        skips = inputs[5].data<bool>();
        outputs[5].set_shape(Shape{num_chars});
    } else {
        outputs[5].set_shape(Shape{num_chars});
        skips_alternative = Tensor(element::boolean, Shape{batch_size});
        skips = std::fill_n(skips_alternative.data<bool>(), batch_size, false) -
                batch_size;
    };

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    outputs[2].set_shape(Shape{num_chars});
    outputs[3].set_shape(Shape{num_chars});
    outputs[4] = inputs[4];

    // Get pointers in the output tensors
    auto new_ragged_begins = outputs[0].data<int32_t>();
    auto new_ragged_ends   = outputs[1].data<int32_t>();
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();
    auto new_skips = outputs[5].data<bool>();

    int32_t ragged_offset = 0;
    std::vector<Match> matches;

    for(size_t seq = 0; seq < batch_size; ++seq) {
        new_ragged_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            if (has_skips && skips[ragged_col]) {
                if (begins[ragged_col] < ends[ragged_col]) {
                    new_begins[ragged_offset] = begins[ragged_col];
                    new_skips[ragged_offset] = true;
                    new_ends[ragged_offset++] = ends[ragged_col];
                }
            } else {
                const std::string_view str(
                    reinterpret_cast<const char*>(chars + begins[ragged_col]),
                    ends[ragged_col] - begins[ragged_col]
                );
                size_t curr_start = 0;
                m_splitter->find_matches(str, matches);
                if (matches.empty()) {
                    if (!str.empty()) {
                        new_begins[ragged_offset] = begins[ragged_col];
                        new_skips[ragged_offset] = false;
                        new_ends[ragged_offset++] = ends[ragged_col];
                    }
                    continue;
                }
                auto compare_matches = [](const Match& lhs, const Match& rhs) {
                    if (lhs.whitespace_start != rhs.whitespace_start) {
                        return lhs.whitespace_start > rhs.whitespace_start;
                    }
                    if (lhs.token_id != rhs.token_id) {
                        return lhs.token_id > rhs.token_id;
                    }
                    return lhs.token_start > rhs.token_start;
                };
                std::priority_queue<Match, std::vector<Match>, decltype(compare_matches)> pending(compare_matches);
                std::sort(matches.begin(), matches.end(), [](const Match& lhs, const Match& rhs) {
                    return lhs.whitespace_start < rhs.whitespace_start;
                });
                size_t match_index = 0;
                while (match_index < matches.size() || !pending.empty()) {
                    if (pending.empty() && matches[match_index].whitespace_start > curr_start) {
                        new_begins[ragged_offset] = begins[ragged_col] + curr_start;
                        new_skips[ragged_offset] = false;
                        new_ends[ragged_offset++] = begins[ragged_col] + matches[match_index].whitespace_start;
                        curr_start = matches[match_index].whitespace_start;
                    }
                    while (match_index < matches.size() && matches[match_index].whitespace_start <= curr_start) {
                        pending.push(matches[match_index++]);
                    }
                    while (!pending.empty() && pending.top().token_start < curr_start) {
                        pending.pop();
                    }
                    if (pending.empty()) {
                        continue;
                    }
                    const Match match = pending.top();
                    pending.pop();
                    const size_t match_start = std::max(curr_start, static_cast<size_t>(match.whitespace_start));
                    if (curr_start < match_start) {
                        new_begins[ragged_offset] = begins[ragged_col] + curr_start;
                        new_skips[ragged_offset] = false;
                        new_ends[ragged_offset++] = begins[ragged_col] + match_start;
                    }
                    new_begins[ragged_offset] = begins[ragged_col] + match.token_start;
                    new_skips[ragged_offset] = true;
                    new_ends[ragged_offset++] = begins[ragged_col] + match.token_end;
                    curr_start = m_splitter->match_end(str, match);
                }
                if (curr_start < str.length()) {
                    new_begins[ragged_offset] = begins[ragged_col] + curr_start;
                    new_skips[ragged_offset] = false;
                    new_ends[ragged_offset++] = begins[ragged_col] + str.length();
                }
            }
        }

        new_ragged_ends[seq] = ragged_offset;
    }

    // Fix real shape based on collected results
    outputs[2].set_shape({size_t(ragged_offset)});
    outputs[3].set_shape({size_t(ragged_offset)});
    outputs[5].set_shape({size_t(ragged_offset)});

    return true;
}
