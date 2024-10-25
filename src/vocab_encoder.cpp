// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include <openvino/core/parallel.hpp>

#include "vocab_encoder.hpp"
#include "utils.hpp"

using namespace ov;


void VocabEncoder::validate_and_infer_types() {
    // main string input
    check_string_input(this, 0);
    // vocab keys
    check_string_input(this, 3);
    // vocab values
    OPENVINO_ASSERT(this->get_input_element_type(6) == element::i32, "Expected an i32 tensor for VocabEncode values.");
    // vocab.size == vocab_values.size when vocab is static
    OPENVINO_ASSERT(
        this->get_input_partial_shape(3).is_dynamic() || this->get_input_partial_shape(3) == this->get_input_partial_shape(6),
        "Expected equal number of vocab keys and values."
    );
    // Default value is compatible to vocab values
    OPENVINO_ASSERT(get_input_element_type(6).compatible(get_input_element_type(7)));
    // one data output, reuse ragged dimensions from split
    this->set_output_type(0, element::i32, get_input_partial_shape(0));
}


bool VocabEncoder::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // string inputs
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();
    {
        // Write to common trie structures should be protected to prevent race conditions.
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_vocab == nullptr) {
            // vocab string keys
            auto vocab_begins = inputs[3].data<const int32_t>();
            auto vocab_ends   = inputs[4].data<const int32_t>();
            auto vocab_chars  = inputs[5].data<const uint8_t>();

            auto vocab_values = inputs[6].data<const int32_t>();
            const auto vocab_size = inputs[6].get_size();

            m_vocab = std::make_shared<absl::flat_hash_map<std::string, int32_t>>();
            for (size_t i = 0; i < vocab_size; ++i) {
                auto token = std::string(vocab_chars + vocab_begins[i], vocab_chars + vocab_ends[i]);
                m_vocab->insert(std::pair{token, vocab_values[i]});
            };
        }
    }
    
    auto default_value = *inputs[7].data<const int32_t>();
    const size_t num_elements = inputs[0].get_size();

    // Set output shape
    outputs[0].set_shape({num_elements});
    auto token_ids = outputs[0].data<int32_t>();

    ov::parallel_for(num_elements, [&](size_t element_idx){
        const auto element = m_vocab->find(std::string(chars + begins[element_idx], chars + ends[element_idx]));
        token_ids[element_idx] = element == m_vocab->end() ? default_value : element->second;
    });

    return true;
}
