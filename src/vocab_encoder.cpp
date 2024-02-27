// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include <algorithm>

#include "vocab_encoder.hpp"
#include "utils.hpp"

using namespace ov;


VocabEncoder::VocabEncoder (
    const ov::OutputVector& arguments,
    std::shared_ptr<std::map<std::vector<uint8_t>, int>> vocab,
    int default_value
) :
    ov::op::Op(arguments), m_vocab(), m_default_value(default_value) {
    if (m_vocab == nullptr) {
        auto packed_vocab_const = as_type_ptr<Constant>(arguments[3].get_node_shared_ptr()->get_input_node_shared_ptr(0));
        auto packed_vocab_buf = static_cast<const uint8_t*>(packed_vocab_const->get_data_ptr());
        auto vocab_size = *reinterpret_cast<const int32_t*>(packed_vocab_buf + 0);
        auto vocab_begins = reinterpret_cast<const int32_t*>(packed_vocab_buf + 4);
        auto vocab_ends = reinterpret_cast<const int32_t*>(packed_vocab_buf + 4 + 4);
        auto vocab_chars = packed_vocab_buf + 4 + 4 + 4 * vocab_size;

        auto values_const = as_type_ptr<Constant>(arguments[4].get_node_shared_ptr()->get_input_node_shared_ptr(0));
        auto values = static_cast<const int32_t*>(packed_vocab_const->get_data_ptr());

        m_vocab = std::make_shared<std::map<std::vector<uint8_t>, int>>();

        std::cerr << "[ VocabEncoder ] vocab_size: " << vocab_size << "\n";

        for (size_t id = 0; id < vocab_size; ++id) {
            std::cerr << "[ VocabEncoder ] iter: " << id << "\n";
            std::vector<uint8_t> token = std::vector(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);

            (*m_vocab)[token] = values[id];

        };

        std::cerr << "[ VocabEncoder ] After Iter "<< "\n";
        auto default_value_const = as_type_ptr<Constant>(arguments[5].get_node_shared_ptr());
        std::cerr << "[ VocabEncoder ] After Iter Const "<< "\n";
        auto graph_default_value  = static_cast<const int32_t*>(default_value_const->get_data_ptr());
        std::cerr << "[ VocabEncoder ] After Iter Cast: " << graph_default_value << "\n";
        m_default_value = *graph_default_value;
        std::cerr << "[ VocabEncoder ] After Iter ="<< "\n";
    };

    constructor_validate_and_infer_types();
}


void VocabEncoder::validate_and_infer_types() {
    // main string input
    check_string_input(this, 0);
    // vocab input
    check_string_input(this, 3);
    // vocab values
    FRONT_END_GENERAL_CHECK(this->get_input_element_type(6) == element::i32, "Expected an i32 tensor for VocabEncode values.");
    // default value
    FRONT_END_GENERAL_CHECK(this->get_input_element_type(7) == element::i32, "Expected an i32 scalar for VocabEncode default value.");
    // one data output, reuse ragged dimensions from split
    this->set_output_type(0, element::i32, get_input_partial_shape(0));
}


bool VocabEncoder::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // string inputs
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

    std::cerr << "[ VocabEncoder ] Arguments1: " << "\n";

    const size_t num_elements = inputs[0].get_size();

    std::cerr << "[ VocabEncoder ] num_elements: " << num_elements << "\n";
    // Set output shape
    outputs[0].set_shape({num_elements});
    auto token_ids = outputs[0].data<int32_t>();

    for (size_t element_idx = 0; element_idx < num_elements; ++element_idx) {
        std::cerr << "[ VocabEncoder ] Arguments4: " << "\n";
        auto element = m_vocab->find(std::vector(chars + begins[element_idx], chars + ends[element_idx]));
        if (element == m_vocab->end()) {
            token_ids[element_idx] = m_default_value;
        } else {
            token_ids[element_idx] = element->second;
        };
    };

    return true;
}
