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
    const auto pattern_input = 3 + (arguments.size() == 6);

    auto search_pattern_const = as_type_ptr<Constant>(arguments[pattern_input].get_node_shared_ptr());
    auto replace_pattern_const = as_type_ptr<Constant>(arguments[pattern_input + 1].get_node_shared_ptr());
    auto search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
    auto replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
    auto search_pattern = std::string(search_pattern_buf, search_pattern_const->get_byte_size());
    m_replace_pattern = std::string(replace_pattern_buf, replace_pattern_const->get_byte_size());

    auto options = re2::RE2::Options();
    options.set_log_errors(false);    
    m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern, options);

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
        const std::string replace_pattern,
        bool global_replace
    ) : ov::op::Op(arguments),
        m_search_pattern_re(search_pattern_re),
        m_search_pattern_pcre2(search_pattern_pcre2),
        m_replace_pattern(replace_pattern),
        m_global_replace(global_replace) {

        const auto pattern_input = 3 + (arguments.size() == 6);

        auto search_pattern_const = as_type_ptr<Constant>(arguments[pattern_input].get_node_shared_ptr());
        auto replace_pattern_const = as_type_ptr<Constant>(arguments[pattern_input + 1].get_node_shared_ptr());
        const char* search_pattern_buf;
        const char* replace_pattern_buf;
        absl::string_view search_pattern;

        if (m_search_pattern_re == nullptr || m_search_pattern_pcre2 == nullptr) {
            search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
            replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
            search_pattern = std::string(search_pattern_buf, search_pattern_const->get_byte_size());
            m_replace_pattern = std::string(replace_pattern_buf, replace_pattern_const->get_byte_size());
        };
        
        auto options = re2::RE2::Options();
        options.set_log_errors(false);
        if (m_search_pattern_re == nullptr) {
            auto options = re2::RE2::Options();
            options.set_log_errors(false);
            m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern, options);
        }
        
        if (m_search_pattern_re->NumberOfCapturingGroups() == -1 && m_search_pattern_pcre2 == nullptr) {
            m_search_pattern_pcre2 = std::make_shared<PCRE2Wrapper>(search_pattern);
            m_search_pattern_re = nullptr;
        }

        constructor_validate_and_infer_types();
    }


void RegexNormalization::validate_and_infer_types() {
    check_string_input(this, 0);

    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 5 || input_size == 6, "supported input sizes are 5 or 6, got", input_size);

    check_string_scalar_input(this, 3 + (input_size == 6));
    check_string_scalar_input(this, 4 + (input_size == 6));

    auto search_pattern_const = dynamic_cast<Constant*>(get_input_node_ptr(3 + (input_size == 6)));
    auto search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
    auto search_pattern = absl::string_view(search_pattern_buf, search_pattern_const->get_byte_size());

    set_string_output(this, 0, get_input_partial_shape(0));
    if (input_size == 6) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}


bool RegexNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 6);
    const auto pattern_input = 3 + has_skips;

    absl::string_view search_pattern;
    if (m_search_pattern_re == nullptr || m_search_pattern_pcre2 == nullptr) {
        search_pattern = std::string(inputs[pattern_input].data<const char>(), inputs[pattern_input].get_size());
        m_replace_pattern = std::string(inputs[pattern_input + 1].data<const char>(), inputs[pattern_input + 1].get_size());

        auto options = re2::RE2::Options();
        options.set_log_errors(false);
        m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern, options);
    }

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
                return str;
            }
    }, has_skips);
}
