// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"
#include <re2/re2.h>

using namespace ov;

class RegexSplit : public ov::op::Op {
public:
    OPENVINO_OP("RegexSplit");

    RegexSplit () = default;
    RegexSplit(const ov::OutputVector& arguments, const std::string& behaviour = "remove", bool invert = false);
    RegexSplit(
        const ov::OutputVector& arguments,
        const std::shared_ptr<re2::RE2>& search_pattern_re2,
        const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
        const std::string& behaviour = "remove",
        bool invert = false,
        int max_splits = -1
    );
    RegexSplit(
        const ov::OutputVector& arguments,
        const std::shared_ptr<re2::RE2>& search_pattern_re2,
        const std::shared_ptr<PCRE2Wrapper>& search_pattern_pcre2,
        const std::shared_ptr<std::set<std::string>>& skip_tokens,
        const std::string& behaviour = "remove",
        bool invert = false,
        int max_splits = -1
    );

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<RegexSplit>(inputs, m_search_pattern_re2, m_search_pattern_pcre2, 
                                            m_skip_tokens, m_behaviour, m_invert, m_max_splits);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("behaviour", m_behaviour);
        visitor.on_attribute("invert", m_invert);
        visitor.on_attribute("max_splits", m_max_splits);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }

    enum SplitMode {
        REMOVED,
        ISOLATED,
        MERGED_WITH_PREVIOUS,
        MERGED_WITH_NEXT,
        CONTIGUOUS,  // Contiguous is not used during evaluate, replaced with isolated with patched pattern in ctor.
    };


private:
    mutable std::shared_ptr<re2::RE2> m_search_pattern_re2;
    mutable std::shared_ptr<PCRE2Wrapper> m_search_pattern_pcre2;
    mutable std::shared_ptr<std::set<std::string>> m_skip_tokens;
    mutable std::string m_behaviour = "remove";
    mutable SplitMode m_split_mode;
    bool m_invert = false;
    int m_max_splits = -1;

    void compile_pattern_if_necessary(std::string split_pattern) const;
};
