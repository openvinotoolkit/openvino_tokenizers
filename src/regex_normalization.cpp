// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "regex_normalization.hpp"
#include "utils.hpp"
#include <regex>


using namespace ov;

namespace {

/**
 * @brief Reformat replace pattern to be compatible with PCRE2
 *
 * @param replace_pattern Replace pattern to reformat
 * @return std::string Reformatted replace pattern
 */
std::string reformat_replace_pattern(std::string replace_pattern) {
    return std::regex_replace(replace_pattern, std::regex(R"((\\)([0-9]+))"), R"($$2)");

}

} // namespace


RegexNormalization::RegexNormalization(
    const ov::OutputVector& arguments,
    bool global_replace
) : ov::op::Op(arguments),
m_global_replace(global_replace) {
    const auto pattern_input = 3 + (arguments.size() == 6);

    auto search_pattern_const = as_type_ptr<Constant>(arguments[pattern_input].get_node_shared_ptr());
    auto replace_pattern_const = as_type_ptr<Constant>(arguments[pattern_input + 1].get_node_shared_ptr());
    auto search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
    auto replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
    auto search_pattern = std::string(search_pattern_buf, search_pattern_const->get_byte_size());
    m_replace_pattern = std::string(replace_pattern_buf, replace_pattern_const->get_byte_size());

    m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
    
    constructor_validate_and_infer_types();
}


RegexNormalization::RegexNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
        const std::string replace_pattern,
        bool global_replace
    ) : ov::op::Op(arguments),
        m_search_pattern_pcre2(search_pattern_pcre2),
        m_replace_pattern(replace_pattern),
        m_global_replace(global_replace) {

        const auto pattern_input = 3 + (arguments.size() == 6);

        auto search_pattern_const = as_type_ptr<Constant>(arguments[pattern_input].get_node_shared_ptr());
        auto replace_pattern_const = as_type_ptr<Constant>(arguments[pattern_input + 1].get_node_shared_ptr());
        const char* search_pattern_buf;
        const char* replace_pattern_buf;
        std::string search_pattern;

        if (m_search_pattern_pcre2 == nullptr) {
            search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
            replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
            search_pattern = std::string(search_pattern_buf, search_pattern_const->get_byte_size());
            m_replace_pattern = std::string(replace_pattern_buf, replace_pattern_const->get_byte_size());
            m_replace_pattern = reformat_replace_pattern(m_replace_pattern);
            m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
        }

        constructor_validate_and_infer_types();
    }


void RegexNormalization::validate_and_infer_types() {
    check_string_input(this, 0);

    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 5 || input_size == 6, "supported input sizes are 5 or 6, got", input_size);

    check_string_scalar_input(this, 3 + (input_size == 6));
    check_string_scalar_input(this, 4 + (input_size == 6));

    set_string_output(this, 0, get_input_partial_shape(0));
    if (input_size == 6) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}


bool RegexNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 6);
    const auto pattern_input = 3 + has_skips;
    
    {
        // Write to common trie structures should be protected to prevent race conditions.
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_search_pattern_pcre2 == nullptr) {
            std::string search_pattern = std::string(inputs[pattern_input].data<const char>(), inputs[pattern_input].get_size());
            m_replace_pattern = std::string(inputs[pattern_input + 1].data<const char>(), inputs[pattern_input + 1].get_size());
            m_replace_pattern = reformat_replace_pattern(m_replace_pattern);
            m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
        }
    }
    
    return evaluate_normalization_helper(
        outputs, inputs,
        [this](const std::string& str) -> std::string {
            if (m_search_pattern_pcre2) {
                return m_search_pattern_pcre2->substitute(str, m_replace_pattern, m_global_replace);
            } else {
                return str;
            }
    }, has_skips);
}
