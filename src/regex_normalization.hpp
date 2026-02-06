// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "absl/strings/string_view.h"
#include "utils.hpp"

#include <openvino/op/op.hpp>
#include "openvino/opsets/opset13.hpp"
#include <pcre2.h>

using namespace ov;
using namespace ov::opset13;

class RegexNormalization : public ov::op::Op {
public:
    OPENVINO_OP("RegexNormalization");

    RegexNormalization () = default;
    RegexNormalization(
        const ov::OutputVector& arguments,
        bool global_replace = true
    );
    RegexNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<PCRE2Wrapper>& search_pattern_rcre2,
        const std::string replace_pattern,
        bool global_replace = true
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexNormalization>(
            inputs,
            m_search_pattern_pcre2,
            m_replace_pattern,
            m_global_replace
        );
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("global_replace", m_global_replace);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    mutable std::shared_ptr<PCRE2Wrapper> m_search_pattern_pcre2;
    mutable std::string m_replace_pattern;
    bool m_global_replace = true;
    mutable std::mutex m_mutex;
};
