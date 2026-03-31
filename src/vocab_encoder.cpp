// Copyright (C) 2018-2026 Intel Corporation
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

    auto input_val_type = this->get_input_element_type(6);
    // vocab values
    OPENVINO_ASSERT(input_val_type == element::i32 || input_val_type == element::i64, "Expected an int32 or int64 tensor for VocabEncode values.");
    // vocab.size == vocab_values.size when vocab is static
    OPENVINO_ASSERT(
        this->get_input_partial_shape(3).is_dynamic() || this->get_input_partial_shape(3) == this->get_input_partial_shape(6),
        "Expected equal number of vocab keys and values."
    );
    // Default value is compatible to vocab values
    OPENVINO_ASSERT(get_input_element_type(6).compatible(get_input_element_type(7)));
    // one data output, reuse ragged dimensions from split
    this->set_output_type(0, input_val_type, get_input_partial_shape(0));
}

bool VocabEncoder::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto& input = inputs[0];

    switch (input.get_element_type()) {
    case ov::element::i32:
        evaluate_impl<int32_t>(outputs, inputs);
        break;
    case ov::element::i64:
        evaluate_impl<int64_t>(outputs, inputs);
        break;
    default:
        OPENVINO_THROW("VocabEncoder: unsupported element type: ",
            input.get_element_type());
    }
    return true;
}

template <typename T>
bool VocabEncoder::evaluate_impl(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // string inputs
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    if (!m_vocab.has_value()) {
        std::call_once(m_init_flag, [&]() {
            auto vocab_begins = inputs[3].data<const int32_t>();
            auto vocab_ends   = inputs[4].data<const int32_t>();
            auto vocab_chars  = inputs[5].data<const uint8_t>();

            auto vocab_values = inputs[6].data<const T>();
            const auto vocab_size = inputs[6].get_size();

            m_vocab = std::make_shared<absl::flat_hash_map<std::string, T>>();
            auto vocab = std::any_cast<std::shared_ptr<absl::flat_hash_map<std::string, T>>>(m_vocab);
      
            for (size_t i = 0; i < vocab_size; ++i) {
                auto token = std::string(vocab_chars + vocab_begins[i], vocab_chars + vocab_ends[i]);
                vocab->insert(std::pair{token, vocab_values[i]});
            };
        });
    }
    
    auto default_value = *inputs[7].data<const T>();
    const size_t num_elements = inputs[0].get_size();

    // Set output shape
    outputs[0].set_shape({num_elements});
    auto token_ids = outputs[0].data<T>();
    auto vocab = std::any_cast<std::shared_ptr<absl::flat_hash_map<std::string, T>>>(m_vocab);
    ov::parallel_for(num_elements, [&](size_t element_idx){
        const auto element = vocab->find(std::string(chars + begins[element_idx], chars + ends[element_idx]));
        token_ids[element_idx] = element == vocab->end() ? default_value : element->second;
    });

    return true;
}
