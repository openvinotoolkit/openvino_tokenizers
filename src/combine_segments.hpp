// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>


class CombineSegments : public ov::op::Op {
public:
    OPENVINO_OP("CombineSegments");

    CombineSegments () = default;

    /**
     * @brief Evaluates the combination of segments from input tensors and produces the output tensors.
     *
     * @param arguments A vector of input tensors. The inputs are expected to be organized in groups of three:
     *               - The first tensor in each group contains the begin indices.
     *               - The second tensor in each group contains the end indices.
     *               - The third tensor in each group contains the elements to be combined.
     *               The last tensor in the inputs vector contains the IDs.
     *
     * @note This function currently works for POD (Plain Old Data) types only and does not support strings.
     *       The function assumes that the input tensors are organized in a specific way and that the number
     *       of ragged tensors is (inputs.size() - 1) / 3.
     */
    CombineSegments(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CombineSegments>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
};
