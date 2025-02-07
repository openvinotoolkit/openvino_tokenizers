// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#pragma once
#include <vector>
#include <openvino/op/op.hpp>

#include "absl/container/flat_hash_map.h"

using namespace ov;


class VocabEncoder : public ov::op::Op {
public:
    OPENVINO_OP("VocabEncoder");

    VocabEncoder () = default;

    VocabEncoder(const ov::OutputVector& arguments) :
        ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    VocabEncoder(const ov::OutputVector& arguments, std::shared_ptr<absl::flat_hash_map<std::string, int32_t>> vocab) :
        ov::op::Op(arguments), m_vocab(vocab) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<VocabEncoder>(inputs, m_vocab);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    mutable std::shared_ptr<absl::flat_hash_map<std::string, int32_t>> m_vocab;
    mutable std::once_flag m_init_flag;
};
