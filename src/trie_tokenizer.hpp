// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <openvino/op/op.hpp>


class Trie {
    public:
        Trie() = default;

        void add(const std::vector<unsigned char>& str, const int value, int idx = 0);
        int find_longest(const std::vector<unsigned char>& str, int& idx);

    private:
        std::unordered_map<unsigned char, std::unique_ptr<Trie>> m_to;
        int m_value = -1;  // -1 for unset value
};


class TrieTokenizer : public ov::op::Op {
public:
    OPENVINO_OP("TrieTokenizer");

    TrieTokenizer () = default;

    TrieTokenizer(const ov::OutputVector& arguments, std::shared_ptr<Trie> trie) :
        ov::op::Op(arguments), m_trie(trie) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<TrieTokenizer>(inputs, m_trie);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    mutable std::shared_ptr<Trie> m_trie;
};
