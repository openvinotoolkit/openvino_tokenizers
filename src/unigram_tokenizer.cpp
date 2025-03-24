#include "unigram_tokenizer.hpp"


using namespace ov;


void UnigramTokenizer::validate_and_infer_types() {
    auto input_size = get_input_size();

    // main string input
    check_ragged_string_input(this, 0);
    // vocab
    check_string_input(this, 5);
    set_ragged_output(this, 0, get_input_partial_shape(0), element::i32);
}

bool UnigramTokenizer::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto input_size = get_input_size();


    if (m_tokenizer == nullptr) {
        std::call_once(m_init_flag, [&]() {
            auto vocab_begins = inputs[5].data<const int32_t>();
            auto vocab_ends   = inputs[6].data<const int32_t>();
            auto vocab_chars  = inputs[7].data<const uint8_t>();
            auto vocab_probs  = inputs[8].data<const float>();
            auto vocab_size   = inputs[6].get_size();

            unigram_impl::Vocab vocab(vocab_size);
            for(int32_t id = 0; id < vocab_size; ++id) {
                auto token = std::string(vocab_chars + vocab_begins[id], vocab_chars + vocab_ends[id]);
                vocab[id] = {token, vocab_probs[id]};
            }
            m_tokenizer = std::make_shared<UnigramTokenizerImpl>(vocab, m_unk_token_id, m_byte_fallback);
        });
    }

    auto ragged_begins = inputs[0].data<const int32_t>();
    auto ragged_ends   = inputs[1].data<const int32_t>();
    auto begins = inputs[2].data<const int32_t>();
    auto ends   = inputs[3].data<const int32_t>();
    auto chars  = inputs[4].data<const char>();  // use char for string_view

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
            auto str = absl::string_view(chars + begins[ragged_col], ends[ragged_col] - begins[ragged_col]);
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
};


namespace unigram_impl {
    constexpr float UNK_PENALTY = 10.0;

    struct BestPathNode {
        int token_id = 0;
        float best_score = 0;
        int starts_at = -1;

        BestPathNode(int token_id)
        : token_id(token_id), best_score(0), starts_at(-1) {};
    };

    // from https://github.com/google/sentencepiece/blob/d8f741853847553169444afc12c00f4bbff3e9ce/src/util.h#L151
    // Return length of a single UTF-8 source character
    inline size_t get_next_char_length(const char *src) {
      return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
    };

} // namespace unigram_impl


UnigramTokenizerImpl::UnigramTokenizerImpl(
    unigram_impl::Vocab& vocab,
    int32_t unk_token_id,
    bool byte_fallback
) : m_unk_token_id(unk_token_id), m_byte_fallback(byte_fallback) {

    struct VocabWithIdx {
        std::string token;
        float score;
        int idx;
    };

    m_scores.reserve(vocab.size());
    std::vector<VocabWithIdx> vocab_with_idx(vocab.size());
    for (int idx = 0; idx < vocab.size(); ++idx) {
        vocab_with_idx[idx] = {vocab[idx].first, vocab[idx].second, idx};
        m_scores[idx] = vocab[idx].second;
    };
    // DoubleArray::build() requires sorted string input
    std::sort(vocab_with_idx.begin(), vocab_with_idx.end(), [](const VocabWithIdx& a, const VocabWithIdx& b) {
        return a.token < b.token;
    });

    int idx = 0;
    std::vector<const char *> keys(vocab.size());
    std::vector<int> values(vocab.size());
    m_min_score = std::numeric_limits<float>::max();

    int new_idx = 0;
    for (const auto& [token, score, old_idx] : vocab_with_idx) {
        m_min_score = std::min<float>(m_min_score, score);
        keys[new_idx] = token.c_str();
        values[new_idx++] = old_idx;
    };

    if (m_trie.build(keys.size(), const_cast<char**>(keys.data()), nullptr, values.data()) != 0) {
        OPENVINO_THROW("[ UNIGRAM ] Failed to build trie");
    };
    return;
};


std::vector<int32_t> UnigramTokenizerImpl::tokenize(absl::string_view text) {
    if (text.empty()) {
        return {};
    }

    const int input_length = text.size();
    const float unk_score = m_min_score - unigram_impl::UNK_PENALTY;
    std::vector<unigram_impl::BestPathNode> best_path(input_length + 1, unigram_impl::BestPathNode(m_unk_token_id));
    int starts_at = 0;

    while (starts_at < input_length) {
        size_t node_pos = 0;
        size_t current_pos = starts_at;
        const auto best_score_so_far = best_path[starts_at].best_score;
        bool found_next_token = false;
        const int next_char_input_length = std::min<int>(
            unigram_impl::get_next_char_length(text.data() + starts_at),
            input_length - starts_at
        );

        while (current_pos < input_length) {
            const int token_id = m_trie.traverse(text.data(), node_pos, current_pos, current_pos + 1);
            if (token_id == -2) { break; };

            if (token_id >= 0) {
                auto &target_node = best_path[current_pos];
                const auto length = current_pos - starts_at;

                // scores for special tokens should be precomputed and stored in m_scores
                const auto score = m_scores[token_id];
                const auto candidate_best_score = score + best_score_so_far;

                if (target_node.starts_at == -1 || candidate_best_score > target_node.best_score) {
                    target_node.best_score = candidate_best_score;
                    target_node.starts_at = starts_at;
                    target_node.token_id = token_id;
                };

                if (!found_next_token && length == next_char_input_length) {
                    found_next_token = true;
                };
            }
        };

        if (!found_next_token) {
            auto &target_node = best_path[starts_at + next_char_input_length];
            const auto candidate_best_score = unk_score + best_score_so_far;
            if (target_node.starts_at == -1 || candidate_best_score > target_node.best_score) {
                target_node.best_score = candidate_best_score;
                target_node.starts_at = starts_at;
                target_node.token_id = m_unk_token_id;
            };
        };
        starts_at += next_char_input_length;
    };

    // backtrack to get the best path
    int ends_at = input_length;
    std::vector<int32_t> result;
    int prev_token_id = -1;
    while (ends_at > 0) {
        const auto &node = best_path[ends_at];
        ends_at = node.starts_at;
        if (node.token_id == m_unk_token_id && prev_token_id == m_unk_token_id) {
            // skip consecutive unk tokens
            continue;
        };
        result.push_back(node.token_id);
        prev_token_id = node.token_id;
    };
    std::reverse(result.begin(), result.end());
    return result;
}
