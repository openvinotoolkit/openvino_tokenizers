// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#pragma once
#include <vector>
#include <openvino/op/op.hpp>

class VocabEncoder : public ov::op::Op {
public:
    OPENVINO_OP("VocabEncoder");

    VocabEncoder () = default;
    VocabEncoder(const ov::OutputVector& arguments);

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<VocabEncoder>(inputs);
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
    std::map<const std::string, const int> m_vocab;
    int m_default_value = -1;
};
