// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#include <openvino/op/op.hpp>
#include "utils.hpp"
#include "darts_clone/darts.h"


namespace unigram_impl {

using TokenMap = std::unordered_map<std::string, unsigned int>;
using VocabToken = std::pair<std::string, float>;
using Vocab = std::vector<VocabToken>;
using Scores = std::vector<float>;

}  // namespace unigram_impl

class UnigramTokenizerImpl {
private:
    std::shared_ptr<Darts::DoubleArray> m_trie;
    std::shared_ptr<unigram_impl::Scores> m_scores;
    double m_min_score;
    bool m_byte_fallback = false;
    const int m_unk_token_id = 0;
    bool m_fuse_unk = false;
//    std::unordered_map<std::string, std::vector<int32_t>> m_cache;
public:
    UnigramTokenizerImpl(
        unigram_impl::Vocab& vocab,
        int32_t unk_token_id,
        bool byte_fallback
    );

    std::vector<int32_t> tokenize(absl::string_view text);
};


/**
* @brief UnigramTokenizer operation
*
* The operation tokenizes input string into a list of tokens using language model.
* The trie is built from the vocabulary of the model.
* The operation takes a ragged string tensor and returns ragged int32 tensor.
*/
class UnigramTokenizer : public ov::op::Op {
public:
    OPENVINO_OP("UnigramTokenizer");

    UnigramTokenizer () = default;
    UnigramTokenizer(
        const ov::OutputVector& arguments,
        const std::shared_ptr<UnigramTokenizerImpl>& tokenizer
    ) : Op(arguments), m_tokenizer(tokenizer) {
        constructor_validate_and_infer_types();
    };
    UnigramTokenizer(
        const ov::OutputVector& arguments,
        const std::shared_ptr<UnigramTokenizerImpl>& tokenizer,
        bool byte_fallback,
        int unk_token_id,
        bool fuse_unk,
        double min_score
    ) : Op(arguments), m_tokenizer(tokenizer), m_byte_fallback(byte_fallback),
        m_unk_token_id(unk_token_id), m_fuse_unk(fuse_unk), m_min_score(min_score) {
        constructor_validate_and_infer_types();
    };

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<UnigramTokenizer>(inputs, m_tokenizer, m_byte_fallback, m_unk_token_id,
                                                  m_fuse_unk, m_min_score);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("byte_fallback", m_byte_fallback);
        visitor.on_attribute("unk_token_id", m_unk_token_id);
        visitor.on_attribute("fuse_unk", m_fuse_unk);
        visitor.on_attribute("min_score", m_min_score);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    mutable std::shared_ptr<UnigramTokenizerImpl> m_tokenizer;
    bool m_byte_fallback = false;
    int m_unk_token_id = 0;
    bool m_fuse_unk = false;
    float m_min_score = std::numeric_limits<float>::infinity();

    mutable std::once_flag m_init_flag;
};
