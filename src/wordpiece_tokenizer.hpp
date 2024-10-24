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

using Vocab = std::unordered_map<std::string, unsigned int>;
using Tokens = std::vector<int32_t>;

class WordpieceTokenizer : public ov::op::Op {
public:
    OPENVINO_OP("WordpieceTokenizer");

    WordpieceTokenizer () = default;
    WordpieceTokenizer(
        const ov::OutputVector& arguments,
        const std::string& suffix_indicator = "##",
        int max_bytes_per_word = 100
    );
    WordpieceTokenizer(
        const ov::OutputVector& arguments,
        const std::shared_ptr<Trie>& trie_root,
        const std::shared_ptr<Trie>& trie_subwords,
        const std::string& suffix_indicator = "##",
        int max_bytes_per_word = 100
    );
    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<WordpieceTokenizer>(inputs, m_trie_root, m_trie_subwords, m_suffix_indicator, m_max_bytes_per_word);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("suffix_indicator", m_suffix_indicator);
        visitor.on_attribute("max_bytes_per_word", m_max_bytes_per_word);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    mutable std::shared_ptr<Trie> m_trie_root;
    mutable std::shared_ptr<Trie> m_trie_subwords;
    std::string m_suffix_indicator = "##";
    int m_max_bytes_per_word = 100;   // TODO: Can it be done outside the op as preprocessing of the input?
    mutable std::mutex m_mutex;
};
