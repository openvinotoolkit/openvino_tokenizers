// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// Converts a numeric tensor to a string tensor by representing each element as
// its decimal string representation (e.g., int64 12345 -> "12345").
// This is used to translate TensorFlow's AsString op when the input type is
// a numeric type (int32, int64, float32, etc.).
//
// When the input is already element::string, the translator passes it through
// directly (identity), so this op is only instantiated for numeric inputs.
class NumericToString : public ov::op::Op {
public:
    OPENVINO_OP("NumericToString", "openvino_tokenizers");

    NumericToString() = default;

    NumericToString(const ov::Output<ov::Node>& input) : ov::op::Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<NumericToString>(new_args[0]);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
};
