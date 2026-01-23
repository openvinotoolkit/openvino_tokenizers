// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "charsmap_normalization.hpp"
#include "utils.hpp"
#include "precompiled_charsmap.hpp"  // generated header with precompiled charsmaps
#include "absl/strings/str_format.h"

using namespace ov;


void CharsMapNormalization::validate_and_infer_types() {
    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 3 || input_size == 4 || input_size == 5, "CharsMapNormalization supports input sizes 3, 4 or 5.");

    bool has_skips;
    if (input_size == 3) {
        has_skips = false;
    } else if (input_size == 4) {
        has_skips = (get_input_element_type(3) == element::boolean);
    } else if (input_size == 5) {
        has_skips = true;
    };

    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));
    if (has_skips) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}


bool CharsMapNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 5) || (m_normalization_form != "" && inputs.size() == 4);

    if (m_normalizer == nullptr) {
        std::call_once(m_init_flag, [&]() {
            sentencepiece::logging::SetMinLogLevel(1);

            m_spec = std::make_shared<sentencepiece::NormalizerSpec>();
            m_spec->set_add_dummy_prefix(m_add_dummy_prefix);
            m_spec->set_remove_extra_whitespaces(m_remove_extra_whitespaces);
            m_spec->set_escape_whitespaces(m_escape_whitespaces);

            std::string precompiled_charsmap;
            if (m_normalization_form != "") {
                // Use precompiled charsmap from generated header
                precompiled_charsmap = get_precompiled_charsmap(m_normalization_form, m_case_fold);
                OPENVINO_ASSERT(!precompiled_charsmap.empty(), 
                    "Unsupported normalization form: `", m_normalization_form, 
                    "` with case_fold=", m_case_fold);
            } else {
                // Use custom precompiled charsmap from input tensor
                precompiled_charsmap = std::string(inputs[3 + has_skips].data<const char>(), inputs[3 + has_skips].get_size());
            };
            m_spec->set_precompiled_charsmap(precompiled_charsmap);

            m_normalizer = std::make_shared<sentencepiece::normalizer::Normalizer>(*m_spec);
        });
    }

    return evaluate_normalization_helper(
        outputs,
        inputs,
        [&](const std::string& str) {
            return m_normalizer->Normalize(str);
        },
        has_skips
    );
}
