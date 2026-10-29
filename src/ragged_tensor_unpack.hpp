// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <openvino/op/op.hpp>

#include "ragged_tensor_format.hpp"

// Deserializes a single u8 tensor that holds a packed ragged tensor into its decomposed representation
// (begins, ends, values), see ragged_tensor_format.hpp for the description of the format.
// This operation is an inverse of RaggedTensorPack, together they allow a ragged tensor to cross
// a graph boundary as a single tensor.
//
// The element type of values is stored in the header of the packed tensor, that is in the data that this
// operation receives at runtime. As OpenVINO requires a static element type for every output of a model,
// the element type is taken either from the `value_type` attribute or, when the attribute is not set and
// the input is a constant, from the header of that constant.
class RaggedTensorUnpack : public ov::op::Op {
public:
    OPENVINO_OP("RaggedTensorUnpack");

    // The default value of the `value_type` attribute, i.e. the case when the element type of values is
    // not given explicitly and therefore has to be taken from the header of a constant input.
    static constexpr const char* AUTO_VALUE_TYPE = "";

    RaggedTensorUnpack () = default;

    explicit RaggedTensorUnpack(const ov::OutputVector& inputs, const std::string& value_type = AUTO_VALUE_TYPE)
        : ov::op::Op(inputs),
          m_value_type(value_type.empty() ? ov::element::dynamic : ov::element::Type(value_type)),
          m_value_type_name(value_type) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<RaggedTensorUnpack>(inputs, m_value_type_name);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("value_type", m_value_type_name);
        // A factory builds an operation first and assigns its attributes afterwards, so the element type
        // has to be refreshed here, validate_and_infer_types() runs again after the attributes are set.
        m_value_type = m_value_type_name.empty() ? ov::element::dynamic : ov::element::Type(m_value_type_name);
        return true;
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

private:
    // Returns the element type of the values tensor: the one from the `value_type` attribute when it is set,
    // otherwise the one stored in the header of a constant input, otherwise element::dynamic.
    ov::element::Type resolve_value_type() const;
    // element::dynamic means that the element type of values has to be taken from the header of the packed tensor
    ov::element::Type m_value_type = ov::element::dynamic;
    // The `value_type` attribute as it is stored in the model, an empty string means element::dynamic
    std::string m_value_type_name = AUTO_VALUE_TYPE;
};