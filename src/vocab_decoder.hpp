// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <openvino/op/op.hpp>

/**
 * @class VocabDecoder
 * @brief The VocabDecoder class is an OpenVINO operation that decodes vocabulary tokens.
 *
 * This class inherits from the ov::op::Op base class and provides functionality to decode
 * vocabulary tokens while skipping specified tokens.
 * @param input_data
 * @param vocab_begins
 * @param vocab_ends
 * @param vocab_chars
 * @param skip_tokens input has priority over the attribute.
 * @param skip_tokens attribute is used only when skip_tokens input is not provided.
 */
class VocabDecoder : public ov::op::Op {
public:
    OPENVINO_OP("VocabDecoder");

    VocabDecoder () = default;    
    VocabDecoder(
        const ov::OutputVector& arguments,
        std::vector<int> skip_tokens
    ) :
        ov::op::Op(arguments), m_skip_tokens(skip_tokens) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<VocabDecoder>(inputs, m_skip_tokens);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("skip_tokens", m_skip_tokens);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    // used std::unordered_set in the first draft, but there are no mapping and support for set attribute yet
    std::vector<int> m_skip_tokens;
};
