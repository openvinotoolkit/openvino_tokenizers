// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "absl/strings/string_view.h"
#include "utils.hpp"

#include <openvino/op/op.hpp>
#include "openvino/opsets/opset13.hpp"
#include <re2/re2.h>
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
        const std::shared_ptr<re2::RE2>& search_pattern_re,
        const std::shared_ptr<PCRE2Wrapper>& search_pattern_rcre2,
        const absl::string_view replace_pattern,
        bool global_replace = true
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexNormalization>(
            std::move(inputs),
            std::move(m_search_pattern_re),
            std::move(m_search_pattern_pcre2),
            std::move(m_replace_pattern),
            std::move(m_global_replace)
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
    mutable std::shared_ptr<re2::RE2> m_search_pattern_re;
    mutable std::shared_ptr<PCRE2Wrapper> m_search_pattern_pcre2;
    mutable absl::string_view m_replace_pattern;
    bool m_global_replace = true;
};
