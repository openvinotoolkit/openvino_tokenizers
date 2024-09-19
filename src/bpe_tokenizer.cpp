// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bpe_tokenizer.hpp"
#include "openvino/opsets/opset13.hpp"
#include "absl/strings/str_format.h"
using namespace ov;
using namespace ov::opset13;

#undef tokenizer


void BPETokenizer::validate_and_infer_types() {
    auto input_size = get_input_size();

    OPENVINO_ASSERT(
        input_size == 11 || input_size == 14 || input_size == 15 || input_size == 18,
        "Incorrect number of inputs passed to BPETokenizer, try to reconvert tokenizer with newer version of OpenVINO Tokenizers"
    );
    // main string input
    check_ragged_string_input(this, 0);

    // vocab
    check_string_input(this, 5);
    // merges
    check_string_input(this, 8);
    if (input_size == 14 || input_size == 18) {
        check_string_input(this, 11);
    };

    if (input_size == 15 || input_size == 18) {
        const size_t added_token_input = input_size - 4;

        // added tokens
        check_string_input(this, added_token_input);
        // added tokens indices
        OPENVINO_ASSERT(this->get_input_element_type(added_token_input + 3) == element::i32, "Expected an i32 tensor for added tokens indices.");
        OPENVINO_ASSERT(
            this->get_input_partial_shape(added_token_input).is_dynamic() || this->get_input_partial_shape(added_token_input) == this->get_input_partial_shape(added_token_input + 3),
            "Expected equal number of added tokens and added token indices."
        );
    };
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}

bool BPETokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto input_size = get_input_size();

    if (m_added_tokens == nullptr && (input_size == 15 || input_size == 18)) {
        const size_t added_token_input = input_size - 4;
        const size_t added_tokens_size = inputs[added_token_input + 3].get_size();

        // vocab string keys
        auto added_tokens_begins = inputs[added_token_input].data<const int32_t>();
        auto added_tokens_ends   = inputs[added_token_input + 1].data<const int32_t>();
        auto added_tokens_chars  = inputs[added_token_input + 2].data<const uint8_t>();
        // vocab indicies
        auto added_tokens_values = inputs[added_token_input + 3].data<const int32_t>();

        m_added_tokens = std::make_shared<std::map<std::string, int32_t>>();
        for (size_t i = 0; i < added_tokens_size; ++i) {
            std::string token = std::string(added_tokens_chars + added_tokens_begins[i], added_tokens_chars + added_tokens_ends[i]);
            m_added_tokens->insert(std::pair{token, added_tokens_values[i]});
        };
    };

    if (m_tokenizer == nullptr) {
        // cache tokenizer
        auto vocab_begins = inputs[5].data<const int32_t>();
        auto vocab_ends   = inputs[6].data<const int32_t>();
        auto vocab_chars  = inputs[7].data<const uint8_t>();
        auto vocab_size   = inputs[6].get_size();

        Vocab vocab;
        for(size_t id = 0; id < vocab_size; ++id) {
            auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
            vocab[token] = int32_t(id); // TODO: Check range
        }

        auto merges_begins = inputs[8].data<const int32_t>();
        auto merges_ends   = inputs[9].data<const int32_t>();
        auto merges_chars  = inputs[10].data<const uint8_t>();
        auto merges_size   = inputs[8].get_size();

        TextMerges merges;
        if (input_size == 11 || input_size == 15){
            std::string delim = " ";
            for(size_t id = 0; id < merges_size; ++id) {
                auto merge = std::string(merges_chars + merges_begins[id], merges_chars + merges_ends[id]);
                const int delim_pos = merge.find(delim);

                std::pair<std::string, std::string> merge_pair = {
                    merge.substr(0, delim_pos), merge.substr(delim_pos + 1)
                };
                merges.emplace_back(merge_pair);
            }
        } else {
            auto right_merges_begins = inputs[11].data<const int32_t>();
            auto right_merges_ends   = inputs[12].data<const int32_t>();
            auto right_merges_chars  = inputs[13].data<const uint8_t>();

            for(size_t id = 0; id < merges_size; ++id) {
                std::pair<const std::string, const std::string> merge_pair = {
                    std::string(merges_chars + merges_begins[id], merges_chars + merges_ends[id]),
                    std::string(right_merges_chars + right_merges_begins[id], right_merges_chars + right_merges_ends[id])
                };
                merges.emplace_back(merge_pair);
            };
        };

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

        if (m_added_tokens){
            for (const auto& added_token: *m_added_tokens) {
                vocab.insert(added_token);
            }
        }

        m_tokenizer = std::make_shared<BPETokenizerImpl>(
            vocab, merges, m_cache_capacity, m_unk_token, m_suffix_indicator, m_end_suffix, m_fuse_unk, m_byte_fallback
        );
    }

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
                auto results = m_tokenizer->tokenize(str);
                for (const auto& token : results) {
                    OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                    new_elems[ragged_offset++] = token;
                }
            } else {
                auto results = m_tokenizer->tokenize(str);
                for (const auto& token : results) {
                    OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                    new_elems[ragged_offset++] = token;
                };
            }
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}


std::pair<std::pair<int32_t, int32_t>, size_t> BPETokenizerImpl::get_min_rank_pair(Tokens tokens) {
    int min_rank = INT_MAX;
    std::pair<int32_t, int32_t> min_rank_pair = {tokens[0], tokens[1]};
    size_t position = -1;
    for (size_t i = 0; i < tokens.size() - 1; i++) {
        auto pair = std::pair(tokens[i], tokens[i + 1]);
        if (m_merges.count(pair) && m_merges.at(pair).first < min_rank) {
            min_rank = m_merges.at(pair).first;
            min_rank_pair = pair;
            position = i;
        }
    }
    return {min_rank_pair, position};
}


Tokens BPETokenizerImpl::tokenize(std::string& text) {
    if (m_cache.count(text)) {
        return m_cache.at(text);
    }
    // For models with end_suffix (e.g. </w>) need to add suffix before looking them up in the vocabulary/prefix tree.
    text += m_end_suffix;
    // TODO: CVS-150387 Implement suffix_indicator.

    // Initialize sequence of integer tokens by looking up
    // for the longest matching sequnce in the prefix tree.
    Tokens res;
    res.reserve(text.length());
    const auto text_vec = std::vector<unsigned char>(text.begin(), text.end());
    for(int idx = 0; idx < text.size(); ) {
        auto r = m_trie->find_longest(text_vec, idx);
        if (r != -1) {
            res.emplace_back(r);
        } else if (m_byte_fallback) {
            res.emplace_back(m_vocab.at(absl::StrFormat("<0x%02X>", static_cast<unsigned char>(text[idx]))));
            idx++;
        } else {
            if (!m_fuse_unk || res.back() != -1){
                res.emplace_back(m_unk_token_id);
            }
            idx++;
        }
    };
    size_t initial_num_tokens = res.size();

    while (res.size() >= 2) {
        auto [pair, idx] = get_min_rank_pair(res);
        if (idx == -1) {
            break;
        }
        res.erase(res.begin() + idx, res.begin() + idx + 2);
        res.insert(res.begin() + idx, m_merges.at(pair).second);
    }
    // TODO: Check if LRU Cache is more effective.
    if (m_cache.size() < m_cache_capacity && initial_num_tokens > 2) {
        m_cache.insert({text, res});
    }
    return res;
}

BPETokenizerImpl::BPETokenizerImpl(
        const Vocab& vocab, const TextMerges& merges, size_t cache_capacity,
        std::string unk_token,
        std::string suffix_indicator,
        std::string end_suffix,
        bool fuse_unk,
        bool byte_fallback
    ): m_cache_capacity(cache_capacity), m_suffix_indicator(suffix_indicator), m_end_suffix(end_suffix), m_byte_fallback(byte_fallback) {
    if (vocab.count(unk_token)) {
        m_unk_token_id = vocab.at(unk_token);
    }
    Merges new_merges;
    Vocab new_vocab = vocab;

    for (size_t i = 0; i < merges.size(); i++) {
        auto pair = merges.at(i);
        auto id_pair = std::make_pair(vocab.at(pair.first), vocab.at(pair.second));
        new_merges[id_pair] = {i, vocab.at(pair.first + pair.second)};
        new_vocab.erase(pair.first + pair.second);
    }

    this->m_vocab = new_vocab;
    this->m_merges = new_merges;

    m_trie = std::make_unique<Trie>();
    for(const auto& word: new_vocab) {
        const auto token = std::vector<unsigned char>(word.first.begin(), word.first.end());
        m_trie->add(token, word.second);
    }
    m_cache.reserve(cache_capacity);
}