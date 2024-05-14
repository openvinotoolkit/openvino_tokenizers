// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bpe_tokenizer.hpp"
#include "utils.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace ov::opset13;

#undef tokenizer


BPETokenizer::BPETokenizer(
        const ov::OutputVector& arguments,
        const std::string& unk_token,
        bool fuse_unk,
        const std::string& suffix_indicator,
        const std::string& end_suffix,
        bool byte_fallback
) :
    ov::op::Op(arguments),
    m_unk_token(unk_token),
    m_fuse_unk(fuse_unk),
    m_suffix_indicator(suffix_indicator),
    m_end_suffix(end_suffix),
    m_byte_fallback(byte_fallback) {

    constructor_validate_and_infer_types();
}
BPETokenizer::BPETokenizer(
        const ov::OutputVector& arguments,
        const std::shared_ptr<models::BPE>& tokenizer,
        const std::string& unk_token,
        bool fuse_unk,
        const std::string& suffix_indicator,
        const std::string& end_suffix,
        bool byte_fallback
) :
    ov::op::Op(arguments),
    m_tokenizer(tokenizer),
    m_unk_token(unk_token),
    m_fuse_unk(fuse_unk),
    m_suffix_indicator(suffix_indicator),
    m_end_suffix(end_suffix),
    m_byte_fallback(byte_fallback) {

    constructor_validate_and_infer_types();
}


void BPETokenizer::validate_and_infer_types() {
    check_ragged_string_input(this, 0);
    check_string_input(this, 5);
    check_string_input(this, 8);
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
    };

    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const uint8_t>();

    OPENVINO_ASSERT(inputs.size() == 11, "Too few inputs passed to BPETokenizer, it means it is not converted properly or it is not used in the supported pattern");

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
            std::vector<core::Token> results = m_tokenizer->Tokenize(str);
            for (const core::Token& token : results) {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token.id_;
            };
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}
