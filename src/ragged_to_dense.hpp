// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>


class RaggedToDense : public ov::op::Op {
public:
    OPENVINO_OP("RaggedToDense");

    RaggedToDense () = default;

    /**
     * @class RaggedToDense
     * @brief Takes a ragged tensor with one ragged right-most dimension and produces a normal tensor.
     *
     * This operation takes a ragged tensor and produces a dense tensor by padding the ragged dimension.
     *
     * @param arguments is a vector containing the following inputs:
     * - beging The beginning indices of the ragged tensor.
     * - ends The ending indices of the ragged tensor.
     * - data The data of the ragged tensor.
     * - padding_size The size of the padding to be applied.
     * - value The value to be used for padding.
     * - pad_right  This input has priority over the attribute "padding_side". If true, padding is applied to the right side of the tensor.
     * @param pad_right If true, padding is applied to the right side of the tensor. Default is true.
     * @param pad_max_length If true, padding is applied to the maximum length of the tensor. Default is false.
     *
     * @note This class inherits from ov::op::Op and overrides necessary methods for validation, cloning, and evaluation.
     */
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
        visitor.on_attribute("m_pad_max_length", m_pad_max_length);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    bool m_pad_right = true;
    bool m_pad_max_length = false;
};
