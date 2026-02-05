// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include "normalize_unicode.hpp"
#include "utils.hpp"
#include "builder.h"  // for making normalizer spec

using namespace ov;


inline void init_unicode_normalizer_chars_map(
    const std::string& normalization_form,
    sentencepiece::normalizer::Builder::CharsMap& chars_map
) {
    if (normalization_form == "NFC") {
        sentencepiece::normalizer::Builder::BuildNFCMap(&chars_map);
    } else if (normalization_form == "NFD") {
        sentencepiece::normalizer::Builder::BuildNFDMap(&chars_map);
    } else if (normalization_form == "NFKC") {
        sentencepiece::normalizer::Builder::BuildNFKCMap(&chars_map);
    } else if (normalization_form == "NFKD") {
        sentencepiece::normalizer::Builder::BuildNFKDMap(&chars_map);
    } else {
        OPENVINO_THROW("Unsupported normalization form: `", normalization_form, "`");
    };
}


void NormalizeUnicode::validate_and_infer_types() {
    check_string_input(this, 0);
    OPENVINO_ASSERT(
        m_normalization_form == "NFC" || m_normalization_form == "NFD" || m_normalization_form == "NFKC" || m_normalization_form == "NFKD",
        "NormalizeUnicode doesn't know normalization form ", m_normalization_form);
    set_string_output(this, 0, get_input_partial_shape(0));

    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 3 || input_size == 4, "supported input sizes are 5 or 6");

    if (input_size == 4) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}

bool NormalizeUnicode::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 4);


    if (m_normalizer == nullptr) {
        std::call_once(m_init_flag, [&]() {
            sentencepiece::logging::SetMinLogLevel(1);

            m_spec = std::make_shared<sentencepiece::NormalizerSpec>();
            m_spec->set_add_dummy_prefix(false);
            m_spec->set_remove_extra_whitespaces(false);
            m_spec->set_escape_whitespaces(false);

            sentencepiece::normalizer::Builder::CharsMap chars_map;
            init_unicode_normalizer_chars_map(m_normalization_form, chars_map);
            std::string precompiled_charsmap;
            sentencepiece::normalizer::Builder::CompileCharsMap(chars_map, &precompiled_charsmap);
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
