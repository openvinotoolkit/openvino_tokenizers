// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bpe_tokenizer.hpp"
#include "openvino/opsets/opset13.hpp"

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
        
        m_tokenizer = std::make_shared<BPETokenizerImpl>(vocab, merges);

        // m_tokenizer = std::make_shared<BPETokenizerImpl>(
        //     vocab,
        //     new_merges
        //     10000 /* default cache size */,
        //     std::vector<float> {} /* dropout - don't use dropout for inference */,
        //     unk_token,
        //     suffix_indicator,
        //     end_suffix,
        //     m_fuse_unk
        // );
    }

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
            // std::cout << "[ BPE ] str=`" << str << "`, size=" << str.size() << "\n";
            if (input_size == 15) {
                auto special = m_added_tokens->find(str);
                if (special != m_added_tokens->end()) {
                    new_elems[ragged_offset++] = special->second;
                } else {
                    auto results = m_tokenizer->tokenize(str);
                    for (const auto& token : results) {
                        OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                        new_elems[ragged_offset++] = token;
                    };
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


std::pair<int64_t, int64_t> BPETokenizerImpl::get_min_rank_pair(Tokens tokens) {
    std::map<std::pair<int64_t, int64_t>, int64_t> count_map;
    // todo: assert tokens.size() is >= 2
    
    for (size_t i = 0; i < tokens.size() - 1; i++) {
        auto pair = std::pair(tokens[i], tokens[i + 1]);
        count_map[pair] += 1;
    }

    int min_rank = INT_MAX;
    std::pair<int64_t, int64_t> min_rank_pair = {tokens[0], tokens[1]};
    for (auto& [k, v]: count_map) {
        if (m_merges.count(k) > 0 && m_merges.at(k) < min_rank) {
            min_rank = m_merges.at(k);
            min_rank_pair = k;
        }
    }
    return min_rank_pair;
}


Tokens BPETokenizerImpl::tokenize(std::string& text) {
    // TODO: Check if code below is really bytes_to_chars/chars_to_bytes transformation agnostic.
    // Each character from string will be converted to string of characters
    // Prompt ' d' ->  'Ġd' = {{0xc4, 0xa0}, 0x64} = {{196, 160}, {100}}
    
    // TODO: Add comment on how and why prefix tree is used.
    Tokens res;
    res.reserve(text.length());
    for(int idx = 0; idx < text.size(); ) {
        const auto text_vec = std::vector<unsigned char>(text.begin(), text.end());
        // TODO: Add setting unk_token_id if returned -1.
        res.emplace_back(m_trie->find_longest(text_vec, idx));
    };

    while (res.size() >= 2) {
        auto pair = get_min_rank_pair(res);

        bool found = false;
        for (size_t idx = 0; idx < res.size(); ) {
            if (m_merges.count(pair) < 1) {
                idx += 1;
                continue;
            } else {
                found = true;
            }
            if (idx < res.size() - 1 && res[idx] == pair.first && res[idx + 1] == pair.second) {
                res.erase(res.begin() + idx, res.begin() + idx + 2);
                res.insert(res.begin() + idx, m_merges[pair]);
                idx += 2;
            } else {
                idx += 1;
            }
        }
        if (!found) {
            break;
        }
    }
    return res;
}

BPETokenizerImpl::BPETokenizerImpl(const Vocab& vocab, const TextMerges& merges) {
    Merges new_merges;
    Vocab new_vocab = vocab;
    for (const auto& pair : merges) {
        auto id_pair = std::make_pair(vocab.at(pair.first), vocab.at(pair.second));
        new_merges[id_pair] = vocab.at(pair.first + pair.second);
        new_vocab.erase(pair.first + pair.second);
    }
    this->m_vocab = new_vocab;
    this->m_merges = new_merges;
        
    m_trie = std::make_unique<Trie>();
    for(const auto& word: vocab) {
        const auto token = std::vector<unsigned char>(word.first.begin(), word.first.end());
        m_trie->add(token, word.second);
    }
}
