// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// Takes a ragged tensor with one ragged right-most dimension and produces a normal tensor
class RaggedToDense : public ov::op::Op {
public:
    OPENVINO_OP("RaggedToDense");

    RaggedToDense () = default;

    RaggedToDense(
        const ov::OutputVector& arguments,
        const bool pad_right = true,
        const bool pad_max_length = false
    ) :
        ov::op::Op(arguments),
        m_pad_right(pad_right),
        m_pad_max_length(pad_max_length) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RaggedToDense>(inputs, m_pad_right, m_pad_max_length);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("pad_right", m_pad_right);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    bool m_pad_right;
    bool m_pad_max_length;
};
