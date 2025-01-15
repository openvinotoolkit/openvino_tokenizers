// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset13.hpp"
#include <optional>
#include "regex_split.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::opset13;

namespace {

const std::map<std::string, RegexSplit::SplitMode> split_modes_map = {
    {"remove", RegexSplit::SplitMode::REMOVED},
    {"isolate", RegexSplit::SplitMode::ISOLATED},
    {"contiguous", RegexSplit::SplitMode::ISOLATED},
    {"mergedwithprevious", RegexSplit::SplitMode::MERGED_WITH_PREVIOUS},
    {"mergedwithnext", RegexSplit::SplitMode::MERGED_WITH_NEXT}
};

} // namespace

void RegexSplit::compile_pattern_if_necessary(std::string split_pattern) const {
    m_split_mode = split_modes_map.at(m_behaviour);
    
    if (m_search_pattern_pcre2) {
        return;
    }
    
    if (m_behaviour == "contiguous" && split_pattern[split_pattern.length() - 1] != '+') {
        std::stringstream tmp_stream;
        tmp_stream << "(" << split_pattern << ")+";
        split_pattern = tmp_stream.str();
    }
    m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(split_pattern);
}


RegexSplit::RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour, bool invert) :
    ov::op::Op(arguments),
    m_behaviour(behaviour),
    m_invert(invert) {
    constructor_validate_and_infer_types();
}


RegexSplit::RegexSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
    const std::string& behaviour,
    bool invert,
    int max_splits
) :
    ov::op::Op(arguments),
    m_search_pattern_pcre2(search_pattern_pcre2),
    m_behaviour(behaviour),
    m_invert(invert),
    m_max_splits(max_splits) {

    const bool has_skips = get_input_size() == 7;

    auto split_pattern_const = as_type_ptr<Constant>(arguments[5 + has_skips].get_node_shared_ptr());
    auto split_pattern_buf = static_cast<const char*>(split_pattern_const->get_data_ptr());
    auto split_pattern = std::string(split_pattern_buf, split_pattern_const->get_byte_size());
    compile_pattern_if_necessary(split_pattern);
    constructor_validate_and_infer_types();
}


RegexSplit::RegexSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
    const std::shared_ptr<std::set<std::string>>& skip_tokens,
    const std::string& behaviour,
    bool invert,
    int max_splits
) :
    ov::op::Op(arguments),
    m_search_pattern_pcre2(search_pattern_pcre2),
    m_skip_tokens(skip_tokens),
    m_behaviour(behaviour),
    m_invert(invert),
    m_max_splits(max_splits) {

    const bool has_skips = get_input_size() == 7;

    auto split_pattern_const = as_type_ptr<Constant>(arguments[5 + has_skips].get_node_shared_ptr());
    auto split_pattern_buf = static_cast<const char*>(split_pattern_const->get_data_ptr());
    auto split_pattern = std::string(split_pattern_buf, split_pattern_const->get_byte_size());
    compile_pattern_if_necessary(split_pattern);
    constructor_validate_and_infer_types();
}


void RegexSplit::validate_and_infer_types() {
    auto input_size = get_input_size();
    const bool has_skips = input_size == 7;

    OPENVINO_ASSERT(input_size == 6 || input_size == 7 || input_size == 9, "Incorrect number of inputs passed to RegexSplit: " + std::to_string(input_size) +  "; try to reconvert tokenizer with newer version of OpenVINO Tokenizers");

    // input strings
    check_ragged_string_input(this, 0);
    // split pattern
    check_string_scalar_input(this, 5 + has_skips);

    //skip regex
    if (input_size == 9) {
        check_string_input(this, 6);
    }
    OPENVINO_ASSERT(split_modes_map.find(m_behaviour) != split_modes_map.end(), "RegexSplit doesn't support unknown split mode: " + m_behaviour);
    OPENVINO_ASSERT(
        m_max_splits == -1 || m_max_splits > 0,
        "RegexSplit max_splits attribute must be greater then `0` or equal to `-1`, got ", m_max_splits
    );
    set_ragged_string_output(this, 0, get_input_partial_shape(0));
    if (has_skips) {
        this->set_output_type(5, get_input_element_type(5),  get_input_partial_shape(5));
    };
}

bool RegexSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto input_size = get_input_size();
    const bool has_skips = (input_size == 7);

    std::string split_pattern = std::string(inputs[5 + has_skips].data<const char>(), inputs[5 + has_skips].get_size());
    auto pattern_size = inputs[5 + has_skips].get_size();
    
    // Write to common trie structures should be protected to prevent race conditions.
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        compile_pattern_if_necessary(split_pattern);
    }
    
    auto get_next_match = [this](const std::string& str, size_t curr_start) -> std::optional<std::pair<size_t, size_t>>{
        auto match = this->m_search_pattern_pcre2->match(str, curr_start);
        if (match.first != SIZE_MAX && match.first != match.second) {
            return match;
        } else {
            return std::nullopt;
        }
    };

    {
        // Write to common trie structures should be protected to prevent race conditions.
        std::lock_guard<std::mutex> lock(m_mutex);
        if (input_size == 9 && m_skip_tokens == nullptr && inputs[6].get_size() > 0) {
            // vocab string keys
            auto skip_tokens_begins = inputs[6].data<const int32_t>();
            auto skip_tokens_ends   = inputs[7].data<const int32_t>();
            auto skip_tokens_chars  = inputs[8].data<const uint8_t>();

            m_skip_tokens = std::make_shared<std::set<std::string>>();
            std::string skip_tokens_pattern;
            for (size_t i = 0; i < inputs[6].get_size(); ++i) {
                std::string token = std::string(skip_tokens_chars + skip_tokens_begins[i], skip_tokens_chars + skip_tokens_ends[i]);
                m_skip_tokens->insert(token);
            }
        }
    }

    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    const size_t num_rows = inputs[0].get_size();
    const size_t num_chars = inputs[4].get_size();

    bool * skips;
    bool init_skips = false;
    if (has_skips) {
        skips = inputs[5].data<bool>();
        outputs[5].set_shape(Shape{num_chars});
    } else {
        skips = new bool[num_rows];
        init_skips = true;
        std::fill(skips, skips + num_rows, false);
    };

    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    outputs[4] = inputs[4];
    if (num_chars == 0) {
        outputs[2] = inputs[2];
        outputs[3] = inputs[3];
        return true;
    }

    outputs[2].set_shape(Shape{num_chars});
    outputs[3].set_shape(Shape{num_chars});

    // Get pointers in the output tensors
    auto new_ragged_begins = outputs[0].data<int32_t>();
    auto new_ragged_ends   = outputs[1].data<int32_t>();
    auto new_begins = outputs[2].data<int32_t>();
    auto new_ends   = outputs[3].data<int32_t>();
    bool * new_skips;
    if (has_skips) {
        new_skips = outputs[5].data<bool>();
    } else {
        new_skips = new bool[num_chars];
    };
    int32_t ragged_offset = 0;

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_ragged_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            const auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);

            if (skips[ragged_col]) {
                new_begins[ragged_offset] = begins[ragged_col];
                new_skips[ragged_offset] = true;
                new_ends[ragged_offset++] = ends[ragged_col];
            } else if (m_skip_tokens != nullptr && m_skip_tokens->count(str) == 1) {
                // legacy skip mechanism
                new_begins[ragged_offset] = begins[ragged_col];
                new_ends[ragged_offset++] = ends[ragged_col];
            } else {
                size_t start = 0;
                uint32_t num_splits = 0;
               
                size_t last_begin = -1;
                auto add_split = [&](int begin, int end, bool invert) {
                    switch (m_split_mode) {
                        case (SplitMode::REMOVED):
                            if (invert) { return; }
                            break;
                        case (SplitMode::ISOLATED):
                            // Do nothing. Do not take invert into account, add split as is.
                            break;
                        case (SplitMode::CONTIGUOUS):
                            OPENVINO_THROW("Prior to evaluate 'contiguous' mode should've been replaced with 'isolated'.");
                            break;
                        case (SplitMode::MERGED_WITH_PREVIOUS):
                            if (invert == false && end != str.length()) {
                                last_begin = begin;
                                return;
                            } else if (invert == true) {
                                begin = last_begin;
                            }
                            break;
                        case (SplitMode::MERGED_WITH_NEXT):
                            if (invert == false) {
                                if (last_begin != -1) { begin = last_begin; }
                            } else {
                                last_begin = begin;
                                return;
                            }
                            break;
                    }

                    // Clamp begin and end to the string length
                    begin = std::max(0, begin);
                    end = std::min(static_cast<int>(str.length()), end);

                    new_begins[ragged_offset] = begins[ragged_col] + begin;
                    if (num_splits == m_max_splits) {
                        end = str.length();
                    };
                    new_ends[ragged_offset++] = begins[ragged_col] + end;
                    
                    ++num_splits;
                };

                std::optional<std::pair<size_t, size_t>> match;
                while ((match = get_next_match(str, start)) != std::nullopt) {
                    auto [curr_start, curr_end] = *match;
                    
                    if (curr_start != start) {
                        if (has_skips) {
                            new_skips[ragged_offset] = false;
                        };
                        add_split(start, curr_start, m_invert);
                    }
                    if (has_skips) {
                        new_skips[ragged_offset] = false;
                    };
                    add_split(curr_start, curr_end, !m_invert);
                    start = curr_end;
                }
                if (start < str.length()) {
                    if (has_skips) { new_skips[ragged_offset] = false; }
                    add_split(start, str.length(), m_invert);
                } else if (m_split_mode == SplitMode::MERGED_WITH_NEXT && last_begin != str.length()) {
                    // Add last split if the match was at the end of the string
                    if (has_skips) { new_skips[ragged_offset] = false; }
                    add_split(last_begin, str.length(), m_invert);
                }
            }
        }

        new_ragged_ends[seq] = ragged_offset;
    }

    // Fix real shape based on collected results
    outputs[2].set_shape({size_t(ragged_offset)});
    outputs[3].set_shape({size_t(ragged_offset)});
    if (has_skips) {
        outputs[5].set_shape({size_t(ragged_offset)});
    };
    if (init_skips) {
        delete[] skips;
        delete[] new_skips;
    };
    return true;
}
