// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// Operation that transforms ragged tensor from rowids format to begins-ends format
// value_rowids just defines to which row each value from values vector belongs
// for example, rowids = [0, 0, 2, 3, 3, 3] and first_dims_size = 5
// it corresponds to ragged tensor with 
// begins = [0, 2, 2, 3, 6]
// ends   = [2, 2, 3, 6, 6]
class RaggedToRagged : public ov::op::Op {
public:
    OPENVINO_OP("RaggedToRagged");

    RaggedToRagged() = default;

    RaggedToRagged(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RaggedToRagged>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
};
