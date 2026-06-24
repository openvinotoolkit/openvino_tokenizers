// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bpe_tokenizer.hpp"
#include "openvino/opsets/opset13.hpp"
#include "absl/strings/str_format.h"
#include <queue>
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

    std::call_once(m_init_flag, [&]() {
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
                m_added_tokens->insert(std::pair{std::move(token), added_tokens_values[i]});
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
                const auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
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
    });

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
            auto results = m_tokenizer->tokenize(str);
            for (const auto& token : results) {
                OPENVINO_ASSERT(ragged_offset < outputs[2].get_size());
                new_elems[ragged_offset++] = token;
            }
        }
        new_ends[seq] = ragged_offset;
    }
    outputs[2].set_shape({size_t(ragged_offset)});
    return true;
}

struct CompareRank {
    // QueueEntry: (position in merges, merged token id, first symbol index,
    // second symbol index, replacement sequence number).
    using QueueEntry = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>;
    bool operator()(const QueueEntry& lhs, const QueueEntry& rhs) const {
        // Compare based on positions in merges, but if positions in merges match
        // prefer pairs which are closer to the beginning of the sequence.
        return (std::get<0>(lhs) != std::get<0>(rhs)) ? std::get<0>(lhs) > std::get<0>(rhs) : std::get<4>(lhs) > std::get<4>(rhs);
    }
};

std::vector<int32_t> BPETokenizerImpl::tokenize(std::string& text) {
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        const auto it = m_cache.find(text);
        if (it != m_cache.end()) {
            return it->second;
        }
    }
    std::string cache_key = text;

    // For models with end_suffix (e.g. </w>) need to add suffix before looking them up in the vocabulary/prefix tree.
    text += m_end_suffix;

    std::vector<BPESymbol> symbols;
    symbols.reserve(text.size() + 1);

    auto append_symbol = [&symbols](int32_t id) {
        const int32_t idx = static_cast<int32_t>(symbols.size());
        if (idx > 0) {
            symbols[idx - 1].next = idx;
        }
        symbols.push_back(BPESymbol{id, idx - 1, -1, true});
    };

    // Initialize sequence of integer tokens by looking up the longest match in the prefix tree.
    const auto text_view = std::string_view(text);
    for (int idx = 0; idx < text.size();) {
        auto r = m_trie->find_longest(text_view, idx);
        if (r != -1) {
            append_symbol(r);
            continue;
        }

        // No vocabulary match at this position. Resolve it the way HF tokenizers does:
        // prefer the byte-fallback token, then the unk token, otherwise drop the byte.
        // The byte token is looked up (not asserted) because some published vocabs set
        // byte_fallback=true yet ship no <0xNN> tokens; a hard map::at there would crash
        // the whole inference instead of degrading like HF.
        int32_t fallback_id = -1;
        if (m_byte_fallback) {
            const auto byte_it = m_vocab.find(absl::StrFormat("<0x%02X>", static_cast<unsigned char>(text_view[idx])));
            if (byte_it != m_vocab.end()) {
                fallback_id = static_cast<int32_t>(byte_it->second);
            }
        }

        if (fallback_id != -1) {
            append_symbol(fallback_id);
        } else if (m_unk_token_id != -1 && (!m_fuse_unk || symbols.empty() || symbols.back().id != -1)) {
            append_symbol(m_unk_token_id);
        }
        // else: unresolvable byte and no unk token -> skip it, matching HF tokenizers.
        idx++;
    }
    const size_t initial_num_tokens = symbols.size();
    size_t live_count = symbols.size();

    // Prepare priority queue to store pairs with their ranks.
    using QueueEntry = CompareRank::QueueEntry;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, CompareRank> pq;

    // replacement sequence number, is used in CompareRank.
    // When merges have the same position prefer replaces which occured earlier.
    int32_t i = 0;

    // Try to enqueue the pair (a, b) of adjacent symbol indices if it is a known merge.
    auto try_push = [&](int32_t a, int32_t b) {
        const auto it = m_merges.find(std::make_pair(symbols[a].id, symbols[b].id));
        if (it != m_merges.end()) {
            const auto [merge_idx, merged_id] = it->second;
            pq.emplace(merge_idx, merged_id, a, b, i);
        }
    };

    // Fill the priority queue with initial pairs. head tracks the index of the
    // first live symbol; it moves when a merge consumes the current head.
    int32_t head = symbols.empty() ? -1 : 0;
    for (int32_t a = head; a != -1 && symbols[a].next != -1; a = symbols[a].next) {
        try_push(a, symbols[a].next);
        i++;
    }

    while (!pq.empty() && live_count >= 2) {
        auto [merge_idx, merged_id, first, second, position] = pq.top();
        pq.pop();

        // Skip stale entries: a merge always consumes (kills) its two operands,
        // so two symbols that formed a pair stay adjacent while both are alive.
        if (!symbols[first].alive || !symbols[second].alive || symbols[first].next != second) {
            continue;
        }

        // Merge: append a new symbol that takes first's left neighbor and second's
        // right neighbor, then mark the operands dead.
        const int32_t prev = symbols[first].prev;
        const int32_t next = symbols[second].next;
        const int32_t merged = static_cast<int32_t>(symbols.size());
        symbols.push_back(BPESymbol{merged_id, prev, next, true});
        symbols[first].alive = false;
        symbols[second].alive = false;
        if (prev != -1) {
            symbols[prev].next = merged;
        } else {
            head = merged;
        }
        if (next != -1) {
            symbols[next].prev = merged;
        }
        live_count--;
        i++;

        // Enqueue the new pairs created on the left and right of the merged symbol.
        if (prev != -1) {
            try_push(prev, merged);
        }
        if (next != -1) {
            try_push(merged, next);
        }
    }

    std::vector<int32_t> res_vec;
    res_vec.reserve(live_count);
    for (int32_t idx = head; idx != -1; idx = symbols[idx].next) {
        res_vec.emplace_back(symbols[idx].id);
    }

    {
        // Cache writes take an exclusive lock; lookups above take a shared lock.
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        // TODO: Check if LRU Cache is more effective.
        if (m_cache.size() < m_cache_capacity && initial_num_tokens > 2) {
            m_cache.emplace(std::move(cache_key), res_vec);
        }
    }
    return res_vec;
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
        auto& pair = merges.at(i);
        auto id_pair = std::make_pair(vocab.at(pair.first), vocab.at(pair.second));
        new_merges[id_pair] = {i, vocab.at(pair.first + pair.second)};
        new_vocab.erase(pair.first + pair.second);
    }

    m_vocab = std::move(new_vocab);
    m_merges = std::move(new_merges);

    m_trie = std::make_unique<Trie>();
    for(const auto& word: m_vocab) {
        const auto token = std::vector<unsigned char>(word.first.begin(), word.first.end());
        m_trie->add(token, word.second);
    }
    m_cache.reserve(cache_capacity);
}
