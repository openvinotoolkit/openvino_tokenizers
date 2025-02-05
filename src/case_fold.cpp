// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "case_fold.hpp"
#include "utils.hpp"
#include "builder.h"  // for making normalizer spec

using namespace ov;


void CaseFold::validate_and_infer_types() {
    check_string_input(this, 0);
    OPENVINO_ASSERT(
        m_encoding == "" || m_encoding == "utf-8",
        "CaseFold operation `encoding` attribute must be one of [\"\", \"utf-8\"], got `", m_encoding, "`."
    );
    set_string_output(this, 0, get_input_partial_shape(0));

    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 3 || input_size == 4, "supported input sizes are 3 or 4");

    if (input_size == 4) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}

bool CaseFold::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 4);

    if (m_normalizer == nullptr && m_encoding == "utf-8") {
        std::call_once(m_init_flag, [&]() {
            sentencepiece::logging::SetMinLogLevel(1);

            m_spec = std::make_shared<sentencepiece::NormalizerSpec>();
            m_spec->set_add_dummy_prefix(false);
            m_spec->set_remove_extra_whitespaces(false);
            m_spec->set_escape_whitespaces(false);

            sentencepiece::normalizer::Builder::CharsMap chars_map;
            sentencepiece::normalizer::Builder::MergeUnicodeCaseFoldMap(&chars_map);
            std::string precompiled_charsmap;
            sentencepiece::normalizer::Builder::CompileCharsMap(chars_map, &precompiled_charsmap);
            m_spec->set_precompiled_charsmap(precompiled_charsmap);

            m_normalizer = std::make_shared<sentencepiece::normalizer::Normalizer>(*m_spec);
        });
    }

    if (m_encoding.empty()) {
        return evaluate_normalization_helper(
            outputs, inputs,
            [](const std::string& str) {
                std::string result = "";
                for (unsigned char ch : str) {
                    result += ('A' <= ch && ch <= 'Z' ) ? ch + 32 : ch;
                };
                return result;
            });
    } else {
        return evaluate_normalization_helper(
            outputs,
            inputs,
            [&](const std::string& str) {
                return m_normalizer->Normalize(str);;
            },
            has_skips);
        }
}
