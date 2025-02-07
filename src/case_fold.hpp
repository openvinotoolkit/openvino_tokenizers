// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalizer.h"  // from sentencepiece
#include <openvino/op/op.hpp>

class CaseFold : public ov::op::Op {
public:
    OPENVINO_OP("CaseFold");

    CaseFold () = default;

    CaseFold (
        const ov::OutputVector& arguments,
        const std::string& encoding = "utf-8"
    ) : ov::op::Op(arguments), m_encoding(encoding), m_init_flag() {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CaseFold>(inputs, m_encoding);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("encoding", m_encoding);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

private:
    std::string m_encoding = "utf-8";
    mutable std::shared_ptr<sentencepiece::normalizer::Normalizer> m_normalizer;
    // spec should be preserved for the lifetime of the normalizer
    mutable std::shared_ptr<sentencepiece::NormalizerSpec> m_spec;
    mutable std::once_flag m_init_flag;
};

