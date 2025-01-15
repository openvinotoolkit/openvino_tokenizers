// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// EqualStr compares two unpacked string tensors and outputs 1D boolean tensor
// The operation is only applicable if output shape of string tensor corresponds to 1D tensor
// It outputs i32 tensor: 1 means elements to be equal, 0 - otherwise
// Op extension must not output boolean due to current limitation in plugins that is why i32 is selected
class EqualStr : public ov::op::Op {
public:
    OPENVINO_OP("EqualStr");

    EqualStr() = default;

    EqualStr(ov::OutputVector inputs)
        : ov::op::Op(inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<EqualStr>(inputs);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
};
