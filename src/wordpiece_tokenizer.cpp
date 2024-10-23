// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "wordpiece_tokenizer.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset13.hpp"
#include <mutex>

using namespace ov;
using namespace ov::opset13;

WordpieceTokenizer::WordpieceTokenizer(
    const ov::OutputVector& arguments,
    const std::string& suffix_indicator,
    int max_bytes_per_word
) :
    ov::op::Op(arguments),
    m_suffix_indicator(suffix_indicator),
    m_max_bytes_per_word(max_bytes_per_word) {

    constructor_validate_and_infer_types();
}

WordpieceTokenizer::WordpieceTokenizer(
    const ov::OutputVector& arguments,
    const std::shared_ptr<Trie>& trie_root,
    const std::shared_ptr<Trie>& trie_subwords,
    const std::string& suffix_indicator,
    int max_bytes_per_word
) :
    ov::op::Op(arguments),
    m_trie_root(trie_root),
    m_trie_subwords(trie_subwords),
    m_suffix_indicator(suffix_indicator),
    m_max_bytes_per_word(max_bytes_per_word) {

    constructor_validate_and_infer_types();
}


void WordpieceTokenizer::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}


bool WordpieceTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // Write to common trie structures should be protected to prevent race conditions.
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_trie_root || !m_trie_subwords) {
            auto vocab_begins = inputs[5].data<const int32_t>();
            auto vocab_ends   = inputs[6].data<const int32_t>();
            auto vocab_chars  = inputs[7].data<const uint8_t>();
            auto vocab_size   = inputs[6].get_size();

            m_trie_root = std::make_unique<Trie>();
            m_trie_subwords = std::make_unique<Trie>();
            for(size_t id = 0; id < vocab_size; ++id) {
                auto word = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
                
                if (word.substr(0, m_suffix_indicator.size()) == m_suffix_indicator) {
                    const auto word_chars_vect = std::vector<unsigned char>(word.begin() + m_suffix_indicator.size(), word.end());
                    m_trie_subwords->add(word_chars_vect, int32_t(id));
                } else {
                    const auto word_chars_vect = std::vector<unsigned char>(word.begin(), word.end());
                    m_trie_root->add(word_chars_vect, int32_t(id));
                }
            }
        }
    }
    const auto unk_token_id = *inputs[8].data<const int32_t>();
    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    // Set output shapes
    outputs[0].set_shape(inputs[0].get_shape());
    outputs[1].set_shape(inputs[1].get_shape());
    const size_t num_rows = inputs[0].get_size();

    // FIXME: Not accurate estimation as there is theoretical possibility for re-use the same symbol area
    // to represent different elements in ragged tensor
    outputs[2].set_shape({inputs[4].get_size()});

    // Get pointers in the output tensors
    auto new_begins = outputs[0].data<int32_t>();
    auto new_ends   = outputs[1].data<int32_t>();
    auto new_elems  = outputs[2].data<int32_t>();
    int32_t ragged_offset = 0;

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            if (ends[ragged_col] - begins[ragged_col] > m_max_bytes_per_word) {
                new_elems[ragged_offset++] = unk_token_id;
                continue;
            }

            auto text_view = std::string_view(reinterpret_cast<const char*>(chars + begins[ragged_col]), ends[ragged_col] - begins[ragged_col]);
            int idx = 0;
            auto token_id = m_trie_root->find_longest(text_view, idx);
            int32_t beginning_offset = ragged_offset;
            if (token_id == -1) {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = unk_token_id;
            } else {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token_id;
                
                while (idx < text_view.size()) {
                    token_id = m_trie_subwords->find_longest(text_view, idx);
                    if (token_id == -1) {
                        OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                        new_elems[beginning_offset] = unk_token_id;
                        ragged_offset = beginning_offset + 1;
                        break;
                    }
                    OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                    new_elems[ragged_offset++] = token_id;
                }
            }
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}
