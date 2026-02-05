// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>


class Truncate : public ov::op::Op {
public:
    OPENVINO_OP("Truncate");

    Truncate () = default;

    /**
     * @brief Truncates the inputs tensor to the specified max_length.
     * 
     * Inputs, are the following:
     * begin, end, data can be repeated 2 times, then max_length, trunc_side and trunc_mode.
     * 
     */
    Truncate(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<Truncate>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("m_num_inputs", m_num_inputs);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    size_t m_num_inputs = 0;
};
