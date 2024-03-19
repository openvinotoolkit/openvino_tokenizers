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
    auto search_pattern = absl::string_view((const char*)search_pattern_buf, search_pattern_const->get_byte_size());
    m_replace_pattern = absl::string_view((const char*)replace_pattern_buf, replace_pattern_const->get_byte_size());
    m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern);
    constructor_validate_and_infer_types();
}


RegexNormalization::RegexNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<re2::RE2>& search_pattern_re,
        const absl::string_view replace_pattern,
        bool global_replace
    ) : ov::op::Op(arguments),
        m_search_pattern_re(search_pattern_re),
        m_replace_pattern(replace_pattern),
        m_global_replace(global_replace) {

        if (m_search_pattern_re == nullptr) {
            auto search_pattern_const = as_type_ptr<Constant>(arguments[3].get_node_shared_ptr());
            auto replace_pattern_const = as_type_ptr<Constant>(arguments[4].get_node_shared_ptr());
            auto search_pattern_buf = static_cast<const char*>(search_pattern_const->get_data_ptr());
            auto replace_pattern_buf = static_cast<const char*>(replace_pattern_const->get_data_ptr());
            auto search_pattern = absl::string_view((const char*)search_pattern_buf, search_pattern_const->get_byte_size());
            m_replace_pattern = absl::string_view((const char*)replace_pattern_buf, replace_pattern_const->get_byte_size());
            m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern);
        };
        constructor_validate_and_infer_types();
    }


void RegexNormalization::validate_and_infer_types() {
    check_string_input(this, 0);
    check_string_scalar_input(this, 3);
    check_string_scalar_input(this, 4);
    set_string_output(this, 0, get_input_partial_shape(0));
}


bool RegexNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (m_search_pattern_re == nullptr) {
        auto search_pattern = absl::string_view(inputs[3].data<const char>(), inputs[3].get_size());
        m_replace_pattern = absl::string_view(inputs[4].data<const char>(), inputs[4].get_size());
        m_search_pattern_re = std::make_shared<re2::RE2>(search_pattern);
    };
    return evaluate_normalization_helper(
        outputs, inputs,
        [this](const std::string& str) {
            // FIXME: if regex is not valid re2, return string without changing (use another regex engine)
            if (m_search_pattern_re->NumberOfCapturingGroups() == -1)
                return str;

            std::string result = str;
            if (m_global_replace) {
                re2::RE2::GlobalReplace(&result, *m_search_pattern_re, m_replace_pattern);
            } else {
                re2::RE2::Replace(&result, *m_search_pattern_re, m_replace_pattern);
            };
            return result;
    });
}
