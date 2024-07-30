// Copyright (C) 2018-2024 Intel Corporation
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

const std::vector<std::string> split_modes = {
    "remove",
    "isolate",
    "merge_with_previous",
    "merge_with_next",
    "contiguous",
};

}


RegexSplit::RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour, bool invert) :
    ov::op::Op(arguments),
    m_behaviour(behaviour),
    m_invert(invert) {
    constructor_validate_and_infer_types();
}


RegexSplit::RegexSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<re2::RE2>& search_pattern_re2,
    const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
    const std::string& behaviour,
    bool invert,
    int max_splits
) :
    ov::op::Op(arguments),
    m_search_pattern_re2(search_pattern_re2),
    m_search_pattern_pcre2(search_pattern_pcre2),
    m_behaviour(behaviour),
    m_invert(invert),
    m_max_splits(max_splits) {
    
    auto split_pattern_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr());
    auto split_pattern_buf = static_cast<const char*>(split_pattern_const->get_data_ptr());
    auto split_pattern = std::string(split_pattern_buf, split_pattern_const->get_byte_size());

    if (m_search_pattern_re2 == nullptr) {
        auto options = re2::RE2::Options();
        options.set_log_errors(false);  
        m_search_pattern_re2 = std::make_shared<re2::RE2>(split_pattern, options);
    }

    if (m_search_pattern_re2->NumberOfCapturingGroups() == -1) {
        // If RE2 was unable to process pattern.
        m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(split_pattern);
        m_search_pattern_re2 = nullptr;
    }

    constructor_validate_and_infer_types();
}


RegexSplit::RegexSplit(
    const ov::OutputVector& arguments,
    const std::shared_ptr<re2::RE2>& search_pattern_re2,
    const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
    const std::shared_ptr<std::set<std::string>>& skip_tokens,
    const std::string& behaviour,
    bool invert,
    int max_splits
) :
    ov::op::Op(arguments),
    m_search_pattern_re2(search_pattern_re2),
    m_search_pattern_pcre2(search_pattern_pcre2),
    m_skip_tokens(skip_tokens),
    m_behaviour(behaviour),
    m_invert(invert),
    m_max_splits(max_splits) {

    auto split_pattern_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr());
    auto split_pattern_buf = static_cast<const char*>(split_pattern_const->get_data_ptr());
    auto split_pattern = std::string(split_pattern_buf, split_pattern_const->get_byte_size());

    if (m_search_pattern_re2 == nullptr) {
        auto options = re2::RE2::Options();
        options.set_log_errors(false);    
        m_search_pattern_re2 = std::make_shared<re2::RE2>(split_pattern, options);
    }

    if (m_search_pattern_re2->NumberOfCapturingGroups() == -1) {
        // If RE2 was unable to process pattern.
        m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(split_pattern);
        m_search_pattern_re2 = nullptr;
    }

    constructor_validate_and_infer_types();
}


void RegexSplit::validate_and_infer_types() {
    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 6 || input_size == 9, "Incorrect number of inputs passed to RegexSplit: " + std::to_string(input_size) +  "; try to reconvert tokenizer with newer version of OpenVINO Tokenizers");

    // input strings
    check_ragged_string_input(this, 0);
   // split pattern
    check_string_scalar_input(this, 5);

    if (input_size == 9) {
        check_string_input(this, 6);
    }

    OPENVINO_ASSERT(std::find(split_modes.begin(), split_modes.end(), m_behaviour) != split_modes.end(), "RegexSplit doesn't support unknown split mode: " + m_behaviour); 
    OPENVINO_ASSERT(
        m_max_splits == -1 || m_max_splits > 0,
        "RegexSplit max_splits attribute must be greater then `0` or equal to `-1`, got ", m_max_splits
    );
    set_ragged_string_output(this, 0, get_input_partial_shape(0));
}

bool RegexSplit::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    std::string split_pattern;
    if (m_search_pattern_re2 == nullptr || m_search_pattern_pcre2 == nullptr) {
        split_pattern = std::string(inputs[5].data<const char>(), inputs[5].get_size());
    };

    if (m_search_pattern_re2 == nullptr) {
        auto options = re2::RE2::Options();
        options.set_log_errors(false);  
        m_search_pattern_re2 = std::make_shared<re2::RE2>(split_pattern, options);
    }

    if (m_search_pattern_re2->NumberOfCapturingGroups() == -1) {
        // If RE2 was unable to process pattern.
        m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(split_pattern);
        m_search_pattern_re2 = nullptr;
    }
    
    // If RE2 didn't compiled successfully fallback to PCRE2 matcher.
    std::function<std::optional<std::pair<size_t, size_t>>(const std::string&, size_t)> get_next_match;
    if (m_search_pattern_re2) {
        get_next_match = [this](const std::string& str, size_t curr_start) -> std::optional<std::pair<size_t, size_t>>{
            re2::StringPiece result;
            bool flag = this->m_search_pattern_re2->Match(str, curr_start, str.length(), RE2::UNANCHORED, &result, 1);
            if (flag) {
                size_t curr_start = result.data() - str.data();
                size_t curr_end = curr_start + result.length();
                return std::pair(curr_start, curr_end);
            } else {
                return std::nullopt;
            }
        };
    } else {
        get_next_match = [this](const std::string& str, size_t curr_start) -> std::optional<std::pair<size_t, size_t>>{
            auto match = this->m_search_pattern_pcre2->match(str, curr_start);
            if (match.first != SIZE_MAX) {
                return match;
            } else {
                return std::nullopt;
            }
        };
    }

    auto input_size = get_input_size();
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
        };
    };

    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    outputs[4] = inputs[4];
    const size_t num_rows = inputs[0].get_size();
    const size_t num_chars = inputs[4].get_size();

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
    int32_t ragged_offset = 0;

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_ragged_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);

            if (m_skip_tokens != nullptr && m_skip_tokens->count(str) == 1) {
                new_begins[ragged_offset] = begins[ragged_col];
                new_ends[ragged_offset++] = ends[ragged_col];
            } else {
                size_t start = 0;
                re2::StringPiece result;
                uint32_t num_splits = 0;
                bool is_remove = m_behaviour == std::string("remove");
                bool is_isolate = m_behaviour == std::string("isolate");
                bool is_merge_next = m_behaviour == std::string("merge_with_next");
                bool is_merge_previous = m_behaviour == std::string("merge_with_previous");
                bool is_contiguous = m_behaviour == std::string("contiguous");
               
                size_t last_begin = -1;
                auto add_split = [&](int begin, int end, bool invert) {
                    if (is_remove) {
                        if (invert) {return;}
                    } else if (is_isolate || is_contiguous) {
                        // Do nothing. Do not take inver into account add split as is.
                    } else if (is_merge_next) {
                        if (invert == false && end != str.length()) {
                            last_begin = begin;
                            return;
                        } else if (invert == true) {
                            begin = last_begin;
                        }
                    } else if (is_merge_previous) {
                        if (invert == false) {
                            if (last_begin != -1) { begin = last_begin; }
                        } else {
                            last_begin = begin;
                            return;
                        }                        
                    }

                    new_begins[ragged_offset] = begins[ragged_col] + begin;
                    if (num_splits == m_max_splits) {
                        end = str.length();
                    };
                    new_ends[ragged_offset++] = begins[ragged_col] + end;
                    
                    ++num_splits; // if (invert == false){ ++num_splits; } // todo: check this
                };

                std::optional<std::pair<size_t, size_t>> match;
                while ((match = get_next_match(str, start)) != std::nullopt) {
                    size_t curr_start = (*match).first;
                    size_t curr_end = (*match).second;
                    
                    if (curr_start != start) {
                        add_split(start, curr_start, m_invert);
                    }
                    add_split(curr_start, curr_end, !m_invert);
                    start = curr_end;
                }
                if (start < str.length()) {
                    add_split(start, str.length(), m_invert);
                }
            }
        }

        new_ragged_ends[seq] = ragged_offset;
    }

    // Fix real shape based on collected results
    outputs[2].set_shape({size_t(ragged_offset)});
    outputs[3].set_shape({size_t(ragged_offset)});

    return true;
}
