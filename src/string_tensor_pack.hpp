// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include <openvino/op/string_tensor_pack.hpp>

// THIS CLASS IS DEPRECATED: This class is deprecated and will it's left here only for purpose of backward compatibility with old IRs.
// Please use op::v15::StringTensorPack from official opset instead.
// Having a decomposed representation for a tensor, converts it to a single string tensor with element::string element type.
class StringTensorPack : public ov::op::v15::StringTensorPack {
public:
    OPENVINO_OP("StringTensorPack", "extension", ov::op::v15::StringTensorPack);

    StringTensorPack () = default;

    StringTensorPack(ov::OutputVector inputs, const std::string& mode = "begins_ends")
        : ov::op::v15::StringTensorPack(inputs[0], inputs[1], inputs[2]), m_mode(mode) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        OPENVINO_ASSERT(m_mode == "begins_ends", "StringTensorPack supports only 'begins_ends' mode, but get ", m_mode);
        ov::op::v15::StringTensorPack::validate_and_infer_types();
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<StringTensorPack>(inputs, m_mode);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("mode", m_mode);
        return true;
    }

private:

    std::string m_mode = "begins_ends";
};
