// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalizer.h"  // from sentencepiece
#include <openvino/op/op.hpp>


using namespace ov;

class CharsMapNormalization : public ov::op::Op {
public:
    OPENVINO_OP("CharsMapNormalization");

    CharsMapNormalization () = default;
    CharsMapNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<sentencepiece::normalizer::Normalizer> normalizer,
        const std::shared_ptr<sentencepiece::NormalizerSpec> spec
    ): ov::op::Op(arguments), m_normalizer(normalizer), m_spec(spec) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CharsMapNormalization>(inputs, m_normalizer, m_spec);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    mutable std::shared_ptr<sentencepiece::normalizer::Normalizer> m_normalizer;

    // spec should be preserved for the lifetime of the normalizer
    mutable std::shared_ptr<sentencepiece::NormalizerSpec> m_spec;
};
