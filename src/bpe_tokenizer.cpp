// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bpe_tokenizer.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace ov::opset13;

#undef tokenizer


void BPETokenizer::validate_and_infer_types() {
    auto input_size = get_input_size();

    OPENVINO_ASSERT(input_size == 11 || input_size == 15, "Incorrect number of inputs passed to BPETokenizer, try to reconvert tokenizer with newer version of OpenVINO Tokenizers");
    // main string input
    check_ragged_string_input(this, 0);
    // vocab
    check_string_input(this, 5);
    // merges
    check_string_input(this, 8);
    if (input_size == 15) {
        // added tokens
        check_string_input(this, 11);
        // added tokens indices
        OPENVINO_ASSERT(this->get_input_element_type(14) == element::i32, "Expected an i32 tensor for added tokens indices.");
        OPENVINO_ASSERT(
            this->get_input_partial_shape(11).is_dynamic() || this->get_input_partial_shape(11) == this->get_input_partial_shape(14),
            "Expected equal number of added tokens and added token indices."
        );
    };
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}

bool BPETokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (m_tokenizer == nullptr) {
        // cache tokenizer
        auto vocab_begins = inputs[5].data<const int32_t>();
        auto vocab_ends   = inputs[6].data<const int32_t>();
        auto vocab_chars  = inputs[7].data<const uint8_t>();
        auto vocab_size   = inputs[6].get_size();

        core::Vocab vocab;
        for(size_t id = 0; id < vocab_size; ++id) {
            auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
            vocab[token] = int32_t(id); // TODO: Check range
        }

        auto merges_begins = inputs[8].data<const int32_t>();
        auto merges_ends   = inputs[9].data<const int32_t>();
        auto merges_chars  = inputs[10].data<const uint8_t>();
        auto merges_size   = inputs[8].get_size();

        core::Merges merges;
        std::string delim = " ";
        for(size_t id = 0; id < merges_size; ++id) {
            auto merge = std::string(merges_chars + merges_begins[id], merges_chars + merges_ends[id]);
            const int delim_pos = merge.find(delim);

            std::pair<std::string, std::string> merge_pair = {
                merge.substr(0, delim_pos), merge.substr(delim_pos + 1)
            };
            merges.emplace_back(merge_pair);
        }

        std::vector<std::string> unk_token = {};
        if (m_unk_token.size() > 0) {
            unk_token.push_back(m_unk_token);
        };
        std::vector<std::string> suffix_indicator = {};
        if (m_suffix_indicator.size() > 0) {
            suffix_indicator.push_back(m_suffix_indicator);
        };
        std::vector<std::string> end_suffix = {};
        if (m_end_suffix.size() > 0) {
            end_suffix.push_back(m_end_suffix);
        };

        m_tokenizer = std::make_shared<models::BPE>(
            vocab,
            merges,
            10000 /* default cache size */,
            std::vector<float> {} /* dropout - don't use dropout for inference */,
            unk_token,
            suffix_indicator,
            end_suffix,
            m_fuse_unk
        );
    }

    auto input_size = get_input_size();
    if (input_size == 15) {
        auto added_tokens_size = inputs[14].get_size();
        if (m_added_tokens == nullptr) {
            // vocab string keys
            auto added_tokens_begins = inputs[11].data<const int32_t>();
            auto added_tokens_ends   = inputs[12].data<const int32_t>();
            auto added_tokens_chars  = inputs[13].data<const uint8_t>();

            auto added_tokens_values = inputs[14].data<const int32_t>();
            auto added_tokens_size = inputs[14].get_size();

            m_added_tokens = std::make_shared<std::map<std::string, int32_t>>();
            for (size_t i = 0; i < added_tokens_size; ++i) {
                std::string token = std::string(added_tokens_chars + added_tokens_begins[i], added_tokens_chars + added_tokens_ends[i]);
                m_added_tokens->insert(std::pair{token, added_tokens_values[i]});
            };
        };
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

    for(size_t seq = 0; seq < num_rows; ++seq) {
        new_begins[seq] = ragged_offset;
        for(size_t ragged_col = ragged_begins[seq]; ragged_col < ragged_ends[seq]; ++ragged_col) {
            auto str = std::string(chars + begins[ragged_col], chars + ends[ragged_col]);
            if (input_size == 15) {
                auto special = m_added_tokens->find(str);
                if (special != m_added_tokens->end()) {
                    new_elems[ragged_offset++] = special->second;
                } else {
                    std::vector<core::Token> results = m_tokenizer->Tokenize(str);
                    for (const core::Token& token : results) {
                        OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                        new_elems[ragged_offset++] = token.id_;
                    };
                }
            } else {
                std::vector<core::Token> results = m_tokenizer->Tokenize(str);
                for (const core::Token& token : results) {
                    OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                    new_elems[ragged_offset++] = token.id_;
                };
            }
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}