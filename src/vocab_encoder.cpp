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

        for (size_t id = 0; id < vocab_size; ++id) {
            std::vector<uint8_t> token = std::vector(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
            (*m_vocab)[token] = values[id];
        };
    };

    constructor_validate_and_infer_types();
}


void VocabEncoder::validate_and_infer_types() {
    // main string input
    check_string_input(this, 0);
    // 3 - vocab packed strings keys
    // vocab values
    FRONT_END_GENERAL_CHECK(this->get_input_element_type(4) == element::i32, "Expected an i32 tensor for VocabEncode values.");
    // default value
    FRONT_END_GENERAL_CHECK(this->get_input_element_type(5) == element::i32, "Expected an i32 scalar for VocabEncode defaule value.");
    // one data output, reuse ragged dimensions from split
    this->set_output_type(0, element::i32, get_input_partial_shape(0));
}


bool VocabEncoder::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // string inputs
    auto begins = inputs[0].data<const int32_t>();
    auto ends   = inputs[1].data<const int32_t>();
    auto chars  = inputs[2].data<const uint8_t>();

//    auto vocab_begins = inputs[3].data<const int32_t>();
//    auto vocab_ends   = inputs[4].data<const int32_t>();
//    auto vocab_chars  = inputs[5].data<const uint8_t>();
//    auto vocab_size   = inputs[3].get_size();

//    auto vocab_values = inputs[6].data<const int32_t>;
//    auto default_value = inputs[7].data<const int32_t>;


//
//    std::vector<std::vector<uint8_t>> vocab;
//    vocab.resize(vocab_size);
//
//    std::vector<uint8_t> empty = {};
//
//    OPENVINO_ASSERT(inputs.size() == 4, "Too few inputs passed to VocabEncoder, it means it is not converted properly or it is not used in the supported pattern");
//
//    for(size_t id = 0; id < vocab_size; ++id) {
//        vocab[id] = std::vector<uint8_t>(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
//    }
//    // Set output shapes
//    outputs[0].set_shape({batch_size});
//    outputs[1].set_shape({batch_size});
//    outputs[2].set_shape({batch_size * seq_len});
//    outputs[3].set_shape({batch_size * seq_len});
//    outputs[4].set_shape({batch_size * seq_len * 100});  // 100 chars - max token length
//    const size_t num_rows = inputs[0].get_size();
//
//    // Get pointers in the output tensors
//    auto new_ragged_begins = outputs[0].data<int32_t>();
//    auto new_ragged_ends = outputs[1].data<int32_t>();
//    auto new_begins = outputs[2].data<int32_t>();
//    auto new_ends   = outputs[3].data<int32_t>();
//    auto new_chars  = outputs[4].data<uint8_t>();
//    uint32_t char_offset = 0;
//
//    for(size_t batch = 0; batch < batch_size; ++batch) {
//        new_ragged_begins[batch] = batch * seq_len;
//        new_ragged_ends[batch]   = new_ragged_begins[batch] + seq_len;
//
//        for(size_t seq = new_ragged_begins[batch]; seq < new_ragged_ends[batch]; ++seq) {
//            auto token_id = input_data[seq];
//            std::vector<uint8_t> token;
//            if (std::find(m_skip_tokens.begin(), m_skip_tokens.end(), token_id) == m_skip_tokens.end()) {
//                token = vocab[token_id];
//            } else {
//                token = empty;
//            }
//
//            std::copy(token.begin(), token.end(), &new_chars[char_offset]);
//
//            new_begins[seq] = char_offset;
//            char_offset += token.size();
//            new_ends[seq] = char_offset;
//        }
//    }
//    outputs[4].set_shape({char_offset});
    return true;
}
