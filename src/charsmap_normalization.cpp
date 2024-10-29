// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "charsmap_normalization.hpp"
#include "utils.hpp"
#include "sentencepiece_trainer.h"  // for making normalizer spec
#include "absl/strings/str_format.h"

using namespace ov;

namespace {

std::shared_ptr<sentencepiece::NormalizerSpec> make_identity_spec() {
    auto spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("identity");
    return std::make_shared<sentencepiece::NormalizerSpec>(spec);
}

}  // namespace


void CharsMapNormalization::validate_and_infer_types() {
    auto input_size = get_input_size();
    OPENVINO_ASSERT(input_size == 4 || input_size == 5, "supported input sizes are 4 or 5");

    const bool has_skips = (input_size == 5);

    check_string_input(this, 0);
    OPENVINO_ASSERT(get_input_element_type(3 + has_skips) == element::u8, "Charsmap normalizer accepts precompiled mapping and it should be of type u8 tensor");
    set_string_output(this, 0, get_input_partial_shape(0));

    if (has_skips) {
        this->set_output_type(3, get_input_element_type(3),  get_input_partial_shape(3));
    };
}

bool CharsMapNormalization::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const bool has_skips = (inputs.size() == 5);
    {            
        // Write to common trie structures should be protected to prevent race conditions.
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_normalizer == nullptr) {
            const std::string precompiled_charsmap = std::string(inputs[3 + has_skips].data<const char>(), inputs[3 + has_skips].get_size());
            m_spec = make_identity_spec();
            m_spec->set_add_dummy_prefix(m_add_dummy_prefix);
            m_spec->set_escape_whitespaces(m_escape_whitespaces);
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
