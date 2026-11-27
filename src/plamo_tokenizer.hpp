// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>

#include <openvino/op/op.hpp>

class PlamoTokenizerImpl;

class PlamoTokenizer : public ov::op::Op {
public:
    OPENVINO_OP("PlamoTokenizer");

    PlamoTokenizer() = default;
    explicit PlamoTokenizer(const ov::OutputVector& arguments) : Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<PlamoTokenizer>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor&) override {
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

private:
    mutable std::once_flag m_init;
    mutable std::shared_ptr<const PlamoTokenizerImpl> m_tokenizer;
};
