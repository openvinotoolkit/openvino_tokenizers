// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#pragma once
#include <vector>
#include <openvino/op/op.hpp>
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace ov::opset13;


class VocabEncoder : public ov::op::Op {
public:
    OPENVINO_OP("VocabEncoder");

    VocabEncoder () = default;
    VocabEncoder(
        const ov::OutputVector& arguments,
        std::shared_ptr<std::map<std::vector<uint8_t>, int>> vocab,
        int default_value = -1
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<VocabEncoder>(inputs, m_vocab, m_default_value);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("default_value", m_default_value);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    std::shared_ptr<std::map<std::vector<uint8_t>, int>> m_vocab;
    int m_default_value = -1;
};
