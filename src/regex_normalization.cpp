// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "regex_normalization.hpp"
#include "utils.hpp"


using namespace ov;


RegexNormalization::RegexNormalization(
    const ov::OutputVector& arguments,
    bool global_replace
) : ov::op::Op(arguments),
m_global_replace(global_replace) {
    auto search_pattern_const = as_type_ptr<Constant>(arguments[3].get_node_shared_ptr());
    auto replace_pattern_const = as_type_ptr<Constant>(arguments[4].get_node_shared_ptr());
    auto search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
    auto replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
    auto search_pattern = absl::string_view(search_pattern_buf, search_pattern_const->get_byte_size());
    m_replace_pattern = absl::string_view(replace_pattern_buf, replace_pattern_const->get_byte_size());
    m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern);
    
    if (m_search_pattern_re->NumberOfCapturingGroups() == -1) {
        // If RE2 was unable to process pattern.
        m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
        m_search_pattern_re = nullptr;
    }
    
    constructor_validate_and_infer_types();
}


RegexNormalization::RegexNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<re2::RE2>& search_pattern_re,
        const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
        const absl::string_view replace_pattern,
        bool global_replace
    ) : ov::op::Op(arguments),
        m_search_pattern_re(search_pattern_re),
        m_search_pattern_pcre2(search_pattern_pcre2),
        m_replace_pattern(replace_pattern),
        m_global_replace(global_replace) {

        auto search_pattern_const = as_type_ptr<Constant>(arguments[3].get_node_shared_ptr());
        auto replace_pattern_const = as_type_ptr<Constant>(arguments[4].get_node_shared_ptr());
        const char* search_pattern_buf;
        const char* replace_pattern_buf;
        absl::string_view search_pattern;

        if (m_search_pattern_re == nullptr || m_search_pattern_pcre2 == nullptr) {
            search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
            replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
            search_pattern = absl::string_view(search_pattern_buf, search_pattern_const->get_byte_size());
            m_replace_pattern = absl::string_view(replace_pattern_buf, replace_pattern_const->get_byte_size());
        };

        if (m_search_pattern_re == nullptr)
            m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern);
        
        if (m_search_pattern_re->NumberOfCapturingGroups() == -1 && m_search_pattern_pcre2 == nullptr) {
            m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
            m_search_pattern_re = nullptr;
        }

        constructor_validate_and_infer_types();
    }


void RegexNormalization::validate_and_infer_types() {
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    check_string_scalar_input(this, 4);
    set_string_output(this, 0, get_input_partial_shape(0));
}


bool RegexNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    absl::string_view search_pattern;
    if (m_search_pattern_re == nullptr || m_search_pattern_pcre2 == nullptr) {
        search_pattern = absl::string_view(inputs[3].data<const char>(), inputs[3].get_size());
        m_replace_pattern = absl::string_view(inputs[4].data<const char>(), inputs[4].get_size());
    }

    if (m_search_pattern_re == nullptr && m_search_pattern_pcre2 == nullptr)
        m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern);

    if ((m_search_pattern_re == nullptr) || (m_search_pattern_re->NumberOfCapturingGroups() == -1 && m_search_pattern_pcre2 == nullptr)) {
        m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
        m_search_pattern_re = nullptr;
    }

    return evaluate_normalization_helper(
        outputs, inputs,
        [this](const std::string& str) -> std::string {
            std::string result = str;

            // Use RE2 where possible, and fallback to PCRE2 if RE2 was not able to process.
            if (m_search_pattern_re) {
                if (m_global_replace) {
                    re2::RE2::GlobalReplace(&result, *m_search_pattern_re, m_replace_pattern);
                } else {
                    re2::RE2::Replace(&result, *m_search_pattern_re, m_replace_pattern);
                };
                return result;
            } else if (m_search_pattern_pcre2) {
                return m_search_pattern_pcre2->substitute(result, m_replace_pattern, m_global_replace);
            } else {
                return result;
            }
    });
}
