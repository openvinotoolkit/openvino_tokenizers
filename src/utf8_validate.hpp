// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <openvino/op/op.hpp>
#include "utils.hpp"

/**
 * @class UTF8Validate
 * @brief Checks the input char sequence and replaces or skips invalid characters based on the mode.
 * 
 * If replace_mode is true then values are replaced wtih �, if false then invalid character are skipped.
 */
class UTF8Validate : public ov::op::Op {
private:
    bool m_replace_mode = false;
public:
    OPENVINO_OP("UTF8Validate");

    UTF8Validate () = default;

    UTF8Validate(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<UTF8Validate>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("replace_mode", m_replace_mode);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
};
