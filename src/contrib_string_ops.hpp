// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// ai.onnx.contrib.StringJoin equivalent. Joins string elements of a decomposed
// (begins, ends, chars) string tensor along `axis` using a scalar `sep`.
//
// Inputs (in order):
//   0: input begins   (i32, tensor shape S)
//   1: input ends     (i32, tensor shape S)
//   2: input chars    (u8,  1D)
//   3: sep begins     (i32, 1D, expects single element)
//   4: sep ends       (i32, 1D, expects single element)
//   5: sep chars      (u8,  1D)
//   6: axis           (i64, 0-D or 1-D single element)
//
// Outputs:
//   0: out begins (i32, shape = S with axis removed; scalar shape {} if rank==1)
//   1: out ends   (i32, same shape)
//   2: out chars  (u8,  1D)
class ContribStringJoin : public ov::op::Op {
public:
    OPENVINO_OP("ContribStringJoin");

    ContribStringJoin() = default;

    ContribStringJoin(const ov::OutputVector& arguments) : ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<ContribStringJoin>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& /*visitor*/) override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override { return true; }
};


// ai.onnx.contrib.StringSplit equivalent. Splits each string of a decomposed
// (begins, ends, chars) string tensor by a scalar `delimiter`. Produces a
// sparse-tensor-like representation with rank = input_rank + 1, where the
// trailing axis is the position of the split token.
//
// Inputs (in order):
//   0: input begins     (i32, tensor shape S)
//   1: input ends       (i32, tensor shape S)
//   2: input chars      (u8,  1D)
//   3: delim begins     (i32, 1D, single element)
//   4: delim ends       (i32, 1D, single element)
//   5: delim chars      (u8,  1D)
//   6: skip_empty       (bool, 0-D or 1-D single element)
//
// Outputs:
//   0: sparse indices   (i64, shape [N, rank+1])
//   1: values begins    (i32, shape [N])
//   2: values ends      (i32, shape [N])
//   3: values chars     (u8,  1D)
//   4: dense shape      (i64, shape [rank+1])
class ContribStringSplit : public ov::op::Op {
public:
    OPENVINO_OP("ContribStringSplit");

    ContribStringSplit() = default;

    ContribStringSplit(const ov::OutputVector& arguments) : ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<ContribStringSplit>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& /*visitor*/) override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override { return true; }
};
