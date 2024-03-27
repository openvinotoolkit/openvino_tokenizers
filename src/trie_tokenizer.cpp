// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "trie_tokenizer.hpp"
#include "utils.hpp"


using namespace ov;


void Trie::add(const std::vector<unsigned char>& str, const int value, int idx) {
    if (idx == str.size()) {
        m_value = value;
    } else {
        auto ch = str[idx];
        if (m_to.count(ch) == 0) {
            m_to[ch] = std::make_unique<Trie>();
        }
        m_to[ch]->add(str, value, idx + 1);
    }
}
int Trie::find_longest(const std::vector<unsigned char>& str, int& idx) {
    int token_id = -1;  // no token found
    Trie* current_node = this;

    uint8_t ch = str[idx];
    int end_idx = idx;

    while (current_node->m_to.count(ch)) {
        current_node = current_node->m_to[ch].get();
        idx++;
        if (current_node->m_value != -1) {
            token_id = current_node->m_value;
            end_idx = idx;
        };
        if (idx == str.size()) {
            break;
        }
        ch = str[idx];
    };
    idx = end_idx;
    return token_id;
}


void TrieTokenizer::validate_and_infer_types() {
    // ragged string inputs
    check_ragged_string_input(this, 0);
    // vocab
    check_string_input(this, 5);
    // indices
    OPENVINO_ASSERT(get_input_element_type(8) == element::i32, "Indices should be i32 type.");
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}


bool TrieTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (m_trie == nullptr) {
        m_trie = std::make_shared<Trie>();

        auto vocab_begins = inputs[5].data<const int32_t>();
        auto vocab_ends   = inputs[6].data<const int32_t>();
        auto vocab_chars  = inputs[7].data<const uint8_t>();
        auto vocab_size   = inputs[5].get_size();

        auto indices = inputs[8].data<const int32_t>();

        OPENVINO_ASSERT(inputs[5].get_size() == inputs[8].get_size(), "Vocab size must be equal to Indices size");

        for(size_t idx = 0; idx < vocab_size; ++idx) {
            const auto token = std::vector(vocab_chars + vocab_begins[idx], vocab_chars + vocab_ends[idx]);
            m_trie->add(token, indices[idx]);
        }
    }
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();
    auto batch_size = inputs[0].get_size();

    outputs[0].set_shape(inputs[2].get_shape());
    outputs[1].set_shape(inputs[3].get_shape());
    // FIXME: Not accurate estimation as there is theoretical possibility for re-use the same symbol area
    // to represent different elements in ragged tensor
    outputs[2].set_shape({inputs[4].get_size()});


    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_elems  = outputs[2].data<int32_t>();
    int32_t ragged_offset = 0;

    for (size_t seq = 0; seq < batch_size; ++seq) {
        new_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            auto str = std::vector<unsigned char>(chars + begins[ragged_col], chars + ends[ragged_col]);
            int idx = 0;
            while (idx < str.size()) {
                auto res = m_trie->find_longest(str, idx);
                new_elems[ragged_offset++] = res;
            };
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}
