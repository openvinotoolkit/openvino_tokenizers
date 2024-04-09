// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

// StringToHashBucket computes a bucket index for each string element
class StringToHashBucket : public ov::op::Op {
public:
    OPENVINO_OP("StringToHashBucket");

    StringToHashBucket() = default;

    StringToHashBucket(ov::OutputVector inputs, int32_t num_buckets)
        : m_num_buckets(num_buckets), ov::op::Op(inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<StringToHashBucket>(inputs, m_num_buckets);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("num_buckets", m_num_buckets);
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

private:
    int32_t m_num_buckets;
};
