// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include "utils.hpp"
#include <re2/re2.h>

using namespace ov;

class SpecialTokensSplit : public ov::op::Op {
public:
    OPENVINO_OP("SpecialTokensSplit");

    SpecialTokensSplit () = default;
    SpecialTokensSplit(const ov::OutputVector& arguments);
    SpecialTokensSplit(
        const ov::OutputVector& arguments,
        const std::shared_ptr<re2::RE2>& split_pattern
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<SpecialTokensSplit>(inputs, std::move(m_split_pattern));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }


private:
    mutable std::shared_ptr<re2::RE2> m_split_pattern;

    void compile_pattern_if_necessary(std::string split_pattern) const;
};
