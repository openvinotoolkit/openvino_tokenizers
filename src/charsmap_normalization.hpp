// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalizer.h"  // from sentencepiece
#include <openvino/op/op.hpp>


using namespace ov;

/**
 * @class CharsMapNormalization
 * @brief A class performs normalization with chars map using SentencePiece normalizer.
 *
 * Node requires precompiled chars map from huggingface (or sentencepiece) tokenizer and
 * applies it using SentencePiece Normalizer class.
 */
class CharsMapNormalization : public ov::op::Op {
public:
    OPENVINO_OP("CharsMapNormalization");

    CharsMapNormalization () = default;
    CharsMapNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<sentencepiece::normalizer::Normalizer> normalizer,
        const std::shared_ptr<sentencepiece::NormalizerSpec> spec
    ): ov::op::Op(arguments), m_normalizer(normalizer), m_spec(spec), m_init_flag() {
        sentencepiece::logging::SetMinLogLevel(1);
        constructor_validate_and_infer_types();
    }
    CharsMapNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<sentencepiece::normalizer::Normalizer> normalizer,
        const std::shared_ptr<sentencepiece::NormalizerSpec> spec,
        bool add_dummy_prefix = false,
        bool remove_extra_whitespaces = false,
        bool escape_whitespaces = false,
        bool case_fold = false,
        const std::string& normalization_form = "",
        bool nmt = false
    ): ov::op::Op(arguments), m_normalizer(normalizer), m_spec(spec), m_add_dummy_prefix(add_dummy_prefix), m_remove_extra_whitespaces(remove_extra_whitespaces), m_escape_whitespaces(escape_whitespaces), m_case_fold(case_fold), m_normalization_form(normalization_form), m_nmt(nmt), m_init_flag(){
        sentencepiece::logging::SetMinLogLevel(1);
        constructor_validate_and_infer_types();
    }
    CharsMapNormalization(
        const ov::OutputVector& arguments,
        const std::shared_ptr<sentencepiece::normalizer::Normalizer> normalizer,
        const std::shared_ptr<sentencepiece::NormalizerSpec> spec,
        const std::string& normalization_form = ""
    ): ov::op::Op(arguments), m_normalizer(normalizer), m_spec(spec), m_normalization_form(normalization_form), m_init_flag() {
        sentencepiece::logging::SetMinLogLevel(1);
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CharsMapNormalization>(inputs, m_normalizer, m_spec, m_add_dummy_prefix, m_remove_extra_whitespaces, m_escape_whitespaces, m_case_fold, m_normalization_form, m_nmt);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("add_dummy_prefix", m_add_dummy_prefix);
        visitor.on_attribute("remove_extra_whitespaces", m_remove_extra_whitespaces);
        visitor.on_attribute("escape_whitespaces", m_escape_whitespaces);
        visitor.on_attribute("normalization_form", m_normalization_form);
        visitor.on_attribute("case_fold", m_case_fold);
        visitor.on_attribute("nmt", m_nmt);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool has_evaluate() const override {
        return true;
    }
private:
    mutable std::shared_ptr<sentencepiece::normalizer::Normalizer> m_normalizer;

    bool m_add_dummy_prefix = false;
    bool m_remove_extra_whitespaces = true;
    bool m_escape_whitespaces = false;
    bool m_case_fold = false;
    bool m_nmt = false;
    std::string m_normalization_form = "";

    // spec should be preserved for the lifetime of the normalizer
    mutable std::shared_ptr<sentencepiece::NormalizerSpec> m_spec;
    mutable std::once_flag m_init_flag;
};
