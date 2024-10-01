// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_FAST_TOKENIZERS

#include "wordpiece_tokenizer.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset13.hpp"

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
    const std::shared_ptr<models::FastWordPiece>& tokenizer,
    const std::string& suffix_indicator,
    int max_bytes_per_word
) :
    ov::op::Op(arguments),
    m_tokenizer(tokenizer),
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
    Vocab m_vocab;
    m_tokenizer = nullptr;
    if (m_tokenizer == nullptr) {
        auto vocab_begins = inputs[5].data<const int32_t>();
        auto vocab_ends   = inputs[6].data<const int32_t>();
        auto vocab_chars  = inputs[7].data<const uint8_t>();
        auto vocab_size   = inputs[6].get_size();

        auto unk_token_id = *inputs[8].data<const int32_t>();
        core::Vocab vocab;

        std::string unk_token;
        if(unk_token_id < 0)
            unk_token_id += vocab_size;

        for(size_t id = 0; id < vocab_size; ++id) {
            auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
            vocab[token] = int32_t(id); // TODO: Check range
            m_vocab[token] = int32_t(id);

            if (id == unk_token_id)
                unk_token = token;
        }
        m_unk_token_id = unk_token_id;
        m_tokenizer = std::make_shared<models::FastWordPiece>(vocab, unk_token, m_max_bytes_per_word, m_suffix_indicator, true);
    };

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

    m_trie_root = std::make_unique<Trie>();
    m_trie_subwords = std::make_unique<Trie>();
    for(const auto& word: m_vocab) {
        const auto token = std::vector<unsigned char>(word.first.begin(), word.first.end());
        if (word.first.substr(0, 2) == "##") {
            m_trie_subwords->add(token, word.second);
        } else {
            m_trie_root->add(token, word.second);
        }
    }

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;

        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            
            auto text_orig = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
            auto text = text_orig;
            // Tokens res;
            // res.reserve(text.length());

            auto text_vec = std::vector<unsigned char>(text.begin(), text.end());
            int idx = 0;
            auto token_id = m_trie_root->find_longest(text_vec, idx);
            int32_t beginning_offset = ragged_offset;
            if (token_id == -1) {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = m_unk_token_id;
            } else {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token_id;
                // res.emplace_back(token_id);
                
                // bool exit_flag = false;
                for(; idx < text_vec.size();) {
                    // std::cout << "still continuing\n" << std::flush;
                    text = m_suffix_indicator + text.substr(idx);
                    text_vec = std::vector<unsigned char>(text.begin(), text.end());
                    idx = 0;
                    token_id = m_trie_subwords->find_longest(text_vec, idx);
                    if (token_id == -1) {
                        OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                        new_elems[beginning_offset] = m_unk_token_id;
                        ragged_offset = beginning_offset + 1;
                        // exit_flag = true;
                        break;
                    }
                    OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                    new_elems[ragged_offset++] = token_id;
                    // res.emplace_back(token_id);
                }

                // if (!exit_flag) {
                //     for (const auto& token : res) {
                //             OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                //             new_elems[ragged_offset++] = token;
                //         }
                // }
            }
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}

#endif // ENABLE_FAST_TOKENIZERS
