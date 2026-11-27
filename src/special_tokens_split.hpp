// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include <mutex>

using namespace ov;

class SpecialTokensSplitImpl;

class SpecialTokensSplit : public ov::op::Op {
public:
    OPENVINO_OP("SpecialTokensSplit");

    SpecialTokensSplit () = default;
    SpecialTokensSplit(const ov::OutputVector& arguments);
    SpecialTokensSplit(
        const ov::OutputVector& arguments,
        const std::shared_ptr<SpecialTokensSplitImpl>& splitter
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<SpecialTokensSplit>(inputs, m_splitter);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }


private:
    mutable std::shared_ptr<SpecialTokensSplitImpl> m_splitter;
    mutable std::once_flag m_init_flag;
};
