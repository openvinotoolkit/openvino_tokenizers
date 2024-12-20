// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "charsmap_normalization.hpp"
#include "utils.hpp"
#include "builder.h"  // for making normalizer spec
#include "absl/strings/str_format.h"

using namespace ov;


void CharsMapNormalization::validate_and_infer_types() {
    auto input_size = get_input_size();
    bool has_skips;
    if (m_normalization_form == "") {
        OPENVINO_ASSERT(input_size == 4 || input_size == 5, "supported input sizes are 4 or 5 with input spec");
        has_skips = (input_size == 5);
        OPENVINO_ASSERT(get_input_element_type(3 + has_skips) == element::u8, "Charsmap normalizer accepts precompiled mapping and it should be of type u8 tensor");
    } else {
        OPENVINO_ASSERT(input_size == 3 || input_size == 4, "supported input sizes are 3 or 4 without input spec");
        has_skips = (input_size == 4);
    }

    check_string_input(this, 0);
    set_string_output(this, 0, get_input_partial_shape(0));

    if (has_skips) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}

bool CharsMapNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 5) || (m_normalization_form != "" && inputs.size() == 4);
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_normalizer == nullptr) {
            m_spec = std::make_shared<sentencepiece::NormalizerSpec>();
            m_spec->set_add_dummy_prefix(m_add_dummy_prefix);
            m_spec->set_escape_whitespaces(m_escape_whitespaces);

            std::string precompiled_charsmap;
            if (m_normalization_form == "") {
                precompiled_charsmap = std::string(inputs[3 + has_skips].data<const char>(), inputs[3 + has_skips].get_size());
            } else if (m_normalization_form == "nfc") {
                sentencepiece::normalizer::Builder::CharsMap chars_map;
                sentencepiece::normalizer::Builder::BuildNFCMap(&chars_map);
                sentencepiece::normalizer::Builder::CompileCharsMap(chars_map, &precompiled_charsmap);
            } else if (m_normalization_form == "nfd") {
                sentencepiece::normalizer::Builder::CharsMap chars_map;
                sentencepiece::normalizer::Builder::BuildNFDMap(&chars_map);
                sentencepiece::normalizer::Builder::CompileCharsMap(chars_map, &precompiled_charsmap);
            } else if (m_normalization_form == "nfkc") {
                sentencepiece::normalizer::Builder::CharsMap chars_map;
                sentencepiece::normalizer::Builder::BuildNFKCMap(&chars_map);
                sentencepiece::normalizer::Builder::CompileCharsMap(chars_map, &precompiled_charsmap);
            } else if (m_normalization_form == "nfkd") {
                sentencepiece::normalizer::Builder::CharsMap chars_map;
                sentencepiece::normalizer::Builder::BuildNFKDMap(&chars_map);
                sentencepiece::normalizer::Builder::CompileCharsMap(chars_map, &precompiled_charsmap);
            } else {
                OPENVINO_ASSERT(false, "Unsupported normalization form: " + m_normalization_form);
            };

            std::cerr << "CharsMapNormalization: precompiled_charsmap.size() = " << precompiled_charsmap.size() << std::endl;
            std::cerr << "CharsMapNormalization: precompiled_charsmap first 100 chars = " << precompiled_charsmap.substr(0, 100) << std::endl;

            m_spec->set_precompiled_charsmap(precompiled_charsmap);

            m_normalizer = std::make_shared<sentencepiece::normalizer::Normalizer>(*m_spec);
        }
    }

    return evaluate_normalization_helper(
        outputs,
        inputs,
        [&](const std::string& str) {
            auto norm = m_normalizer->Normalize(str);
            return norm;
        },
        has_skips
    );
}
