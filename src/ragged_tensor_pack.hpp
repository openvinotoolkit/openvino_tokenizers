// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

#include "ragged_tensor_format.hpp"

// Serializes the decomposed representation of a ragged tensor (begins, ends, values) into a single u8
// tensor, see ragged_tensor_format.hpp for the description of the format.
// RaggedTensorUnpack converts such a tensor back into the decomposed representation, therefore a ragged
// tensor can cross a graph boundary as a single tensor.
// The values tensor that a ragged tensor is built on can have any shape and any element type except
// string, this operation doesn't try to interpret it.
class RaggedTensorPack : public ov::op::Op {
public:
    OPENVINO_OP("RaggedTensorPack");

    RaggedTensorPack () = default;

    RaggedTensorPack(ov::OutputVector inputs)
        : ov::op::Op(inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<RaggedTensorPack>(inputs);
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